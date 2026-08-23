"""Explainability Agent for recruiter-friendly candidate match explanations."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from app.core.config import settings
from app.core.logging import get_logger
from app.models.candidate import CandidateProfile
from app.models.jd import ParsedJobDescription
from app.models.response import CandidateEvidence, ScoreBreakdown

logger = get_logger(__name__)


EXPLAINABILITY_SYSTEM_PROMPT = """You are an expert recruiter assistant.

For each candidate, write one recruiter-friendly explanation of exactly 3 to 4
sentences.

Rules:
- final score >= 60: start with "Strong fit because:"
- final score 40-59.99: start with "Moderate fit:"
- final score < 40: start with "Partial fit —"
- Mention real company names and role titles from the candidate data.
- Mention 2 to 3 verified skills when available.
- Mention missing or unverified required skills honestly.
- Never invent experience, skills, companies, titles, or evidence.
- Keep each explanation concise.
- Return plain text inside each explanation.
"""

BATCH_EXPLANATION_PROMPT = """Generate recruiter explanations for the supplied candidates.

Return ONLY valid JSON using this structure:

{
  "explanations": [
    {
      "candidate_id": "exact candidate id",
      "explanation": "3-4 sentence explanation"
    }
  ]
}

Rules:
- Return exactly one item for every supplied candidate.
- Preserve candidate_id exactly.
- Do not add commentary outside the JSON object.
- Keep explanations concise.
"""


class ExplainabilityAgent:
    """Generate recruiter explanations individually or in small reliable batches."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the OpenRouter client with bounded request time.

        Args:
            api_key: OpenRouter API key.
            base_url: OpenRouter-compatible API base URL.
            model: LLM model name.
        """
        self.api_key = api_key or settings.openrouter_api_key
        self.base_url = base_url or settings.openrouter_base_url
        self.model = model or settings.llm_model

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=30.0,
        )

    @staticmethod
    def _score_prefix(final_score: float) -> str:
        """Return the required explanation prefix for the supplied score."""
        if final_score >= 60.0:
            return "Strong fit because:"
        if final_score >= 40.0:
            return "Moderate fit:"
        return "Partial fit —"

    @staticmethod
    def _deterministic_explanation(
        candidate: CandidateProfile,
        score_breakdown: ScoreBreakdown,
        verified_skills: list[str],
        missing_skills: list[str],
    ) -> str:
        """Generate a grounded fallback when LLM explanation generation fails."""
        prefix = ExplainabilityAgent._score_prefix(
            score_breakdown.final_score
        )

        recent_title = candidate.most_recent_title or "professional"

        company = (
            candidate.experiences[0].company
            if candidate.experiences
            and candidate.experiences[0].company
            else "previous roles"
        )

        exp_years = candidate.total_years_experience or 0.0

        verified_text = (
            ", ".join(verified_skills[:3])
            if verified_skills
            else "general experience"
        )

        missing_text = ""

        if missing_skills:
            missing_text = (
                f" However, required skills such as "
                f"{', '.join(missing_skills[:2])} could not be fully verified "
                f"in the available experience history."
            )

        return (
            f"{prefix} {candidate.name} brings "
            f"{exp_years:.1f} years of professional experience as a "
            f"{recent_title} at {company}. "
            f"Verified skill evidence includes {verified_text}."
            f"{missing_text} "
            f"Overall match score is "
            f"{score_breakdown.final_score:.1f}/100."
        )

    @staticmethod
    def _prepare_candidate_payload(
        candidate: CandidateProfile,
        parsed_jd: ParsedJobDescription,
        score_breakdown: ScoreBreakdown,
        evidences: list[CandidateEvidence],
    ) -> dict[str, Any]:
        """Build a compact evidence-grounded payload for one candidate."""
        verified_skills: list[str] = []
        missing_skills: list[str] = []

        for evidence in evidences:
            is_verified = getattr(
                evidence,
                "verified",
                getattr(evidence, "is_verified", False),
            )

            skill_name = str(
                getattr(evidence, "skill_name", "")
            ).strip()

            if not skill_name:
                continue

            if is_verified:
                verified_skills.append(skill_name)
            else:
                missing_skills.append(skill_name)

        if not evidences:
            missing_skills = parsed_jd.required_skill_names[:5]

        experiences: list[dict[str, str]] = []

        for experience in candidate.experiences[:2]:
            experiences.append(
                {
                    "company": experience.company,
                    "title": experience.title,
                    "description": experience.description[:220],
                }
            )

        evidence_items: list[dict[str, Any]] = []

        for evidence in evidences[:6]:
            is_verified = getattr(
                evidence,
                "verified",
                getattr(evidence, "is_verified", False),
            )

            snippets = getattr(
                evidence,
                "evidence_snippets",
                None,
            )

            if snippets:
                snippet = str(snippets[0])[:160]
            else:
                snippet = str(
                    getattr(evidence, "source_snippet", "")
                )[:160]

            evidence_items.append(
                {
                    "skill": str(
                        getattr(evidence, "skill_name", "")
                    ),
                    "verified": bool(is_verified),
                    "evidence": snippet,
                }
            )

        return {
            "candidate_id": candidate.candidate_id,
            "candidate_name": candidate.name,
            "target_role": parsed_jd.title,
            "final_score": round(score_breakdown.final_score, 2),
            "years_of_experience": candidate.total_years_experience,
            "recent_title": candidate.most_recent_title,
            "career_history": experiences,
            "verified_skills": verified_skills[:5],
            "missing_skills": missing_skills[:5],
            "evidence": evidence_items,
        }

    @staticmethod
    def _parse_batch_response(content: str) -> dict[str, str]:
        """Parse a batch JSON response into candidate-id to explanation mappings."""
        text = (content or "").strip()

        if not text:
            raise ValueError("LLM returned an empty batch explanation.")

        if text.startswith("```"):
            lines = text.splitlines()

            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            text = "\n".join(lines).strip()

            if text.lower().startswith("json"):
                text = text[4:].lstrip()

        data = json.loads(text)

        if not isinstance(data, dict):
            raise ValueError(
                "Batch explanation response must be a JSON object."
            )

        raw_explanations = data.get("explanations")

        if not isinstance(raw_explanations, list):
            raise ValueError(
                "'explanations' must be a list."
            )

        results: dict[str, str] = {}

        for item in raw_explanations:
            if not isinstance(item, dict):
                continue

            candidate_id = str(
                item.get("candidate_id", "")
            ).strip()

            explanation = str(
                item.get("explanation", "")
            ).strip()

            if candidate_id and explanation:
                results[candidate_id] = explanation

        if not results:
            raise ValueError(
                "No valid candidate explanations were returned."
            )

        return results

    @staticmethod
    def _fallback_for_candidate(
        candidate: CandidateProfile,
        parsed_jd: ParsedJobDescription,
        score_breakdown: ScoreBreakdown,
        evidences: list[CandidateEvidence],
    ) -> str:
        """Create a deterministic explanation for one failed batch item."""
        verified_skills = [
            e.skill_name
            for e in evidences
            if getattr(
                e,
                "verified",
                getattr(e, "is_verified", False),
            )
        ]

        missing_skills = [
            e.skill_name
            for e in evidences
            if not getattr(
                e,
                "verified",
                getattr(e, "is_verified", False),
            )
        ]

        if not missing_skills:
            missing_skills = [
                skill
                for skill in parsed_jd.required_skill_names
                if skill not in verified_skills
            ]

        return ExplainabilityAgent._deterministic_explanation(
            candidate=candidate,
            score_breakdown=score_breakdown,
            verified_skills=verified_skills,
            missing_skills=missing_skills,
        )

    def _generate_single_batch(
        self,
        candidates: list[CandidateProfile],
        parsed_jd: ParsedJobDescription,
        score_breakdowns: dict[str, ScoreBreakdown],
        evidences_by_candidate: dict[str, list[CandidateEvidence]],
    ) -> dict[str, str]:
        """Generate explanations for a small candidate batch with one LLM request."""
        payloads: list[dict[str, Any]] = []

        for candidate in candidates:
            score_breakdown = score_breakdowns.get(
                candidate.candidate_id
            )

            if score_breakdown is None:
                logger.warning(
                    "No score breakdown found for candidate '{}'.",
                    candidate.candidate_id,
                )
                continue

            payloads.append(
                self._prepare_candidate_payload(
                    candidate=candidate,
                    parsed_jd=parsed_jd,
                    score_breakdown=score_breakdown,
                    evidences=evidences_by_candidate.get(
                        candidate.candidate_id,
                        [],
                    ),
                )
            )

        if not payloads:
            return {}

        user_prompt = (
            f"{BATCH_EXPLANATION_PROMPT}\n\n"
            f"Target role: {parsed_jd.title}\n"
            f"Required skills: "
            f"{', '.join(parsed_jd.required_skill_names[:10])}\n\n"
            f"Candidates:\n"
            f"{json.dumps(payloads, ensure_ascii=False)}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": EXPLAINABILITY_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=settings.llm_temperature,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )

        if not response.choices:
            raise ValueError(
                "LLM returned no explanation choices."
            )

        content = response.choices[0].message.content or ""

        return self._parse_batch_response(content)

    def generate_explanations_batch(
        self,
        candidates: list[CandidateProfile],
        parsed_jd: ParsedJobDescription,
        score_breakdowns: dict[str, ScoreBreakdown],
        evidences_by_candidate: dict[str, list[CandidateEvidence]] | None = None,
    ) -> dict[str, str]:
        """Generate top-candidate explanations using reliable batches of five.

        Args:
            candidates: Candidates requiring explanations.
            parsed_jd: Parsed job description.
            score_breakdowns: Candidate ID to deterministic score breakdown.
            evidences_by_candidate: Candidate ID to evidence list.
            """
        if not candidates or parsed_jd is None:
            return {}

        evidences_by_candidate = evidences_by_candidate or {}

        batch_size = 5
        result: dict[str, str] = {}

        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]

            batch_number = (start // batch_size) + 1
            total_batches = (
                (len(candidates) + batch_size - 1)
                // batch_size
            )

            try:
                batch_result = self._generate_single_batch(
                    candidates=batch,
                    parsed_jd=parsed_jd,
                    score_breakdowns=score_breakdowns,
                    evidences_by_candidate=evidences_by_candidate,
                )

                for candidate in batch:
                    explanation = batch_result.get(
                        candidate.candidate_id
                    )

                    if explanation:
                        result[candidate.candidate_id] = explanation
                    else:
                        score_breakdown = score_breakdowns.get(
                            candidate.candidate_id
                        )

                        if score_breakdown is None:
                            continue

                        result[candidate.candidate_id] = (
                            self._fallback_for_candidate(
                                candidate,
                                parsed_jd,
                                score_breakdown,
                                evidences_by_candidate.get(
                                    candidate.candidate_id,
                                    [],
                                ),
                            )
                        )

                logger.info(
                    "Explanation batch {}/{} completed for {} candidates.",
                    batch_number,
                    total_batches,
                    len(batch),
                )

            except Exception as exc:
                logger.warning(
                    "Explanation batch {}/{} failed: {}. "
                    "Using deterministic fallbacks for this batch.",
                    batch_number,
                    total_batches,
                    exc,
                )

                for candidate in batch:
                    score_breakdown = score_breakdowns.get(
                        candidate.candidate_id
                    )

                    if score_breakdown is None:
                        continue

                    result[candidate.candidate_id] = (
                        self._fallback_for_candidate(
                            candidate,
                            parsed_jd,
                            score_breakdown,
                            evidences_by_candidate.get(
                                candidate.candidate_id,
                                [],
                            ),
                        )
                    )

        logger.info(
            "Generated explanations for {} candidates using batches of {}.",
            len(result),
            batch_size,
        )

        return result

    def generate_explanation(
        self,
        candidate: CandidateProfile,
        parsed_jd: ParsedJobDescription,
        score_breakdown: ScoreBreakdown,
        evidences: list[CandidateEvidence] | None = None,
    ) -> str:
        """Generate one recruiter explanation while preserving the existing API.

        Args:
            candidate: CandidateProfile instance.
            parsed_jd: ParsedJobDescription instance.
            score_breakdown: Candidate score breakdown.
            evidences: Optional verified candidate evidence.

        Returns:
            Plain-text recruiter explanation.
        """
        evidences_list = evidences or []

        verified_skills = [
            e.skill_name
            for e in evidences_list
            if getattr(
                e,
                "verified",
                getattr(e, "is_verified", False),
            )
        ]

        missing_skills = [
            e.skill_name
            for e in evidences_list
            if not getattr(
                e,
                "verified",
                getattr(e, "is_verified", False),
            )
        ]

        if not missing_skills and parsed_jd.required_skill_names:
            missing_skills = [
                skill
                for skill in parsed_jd.required_skill_names
                if skill not in verified_skills
            ]

        try:
            exp_summary = "; ".join(
                f"{experience.title} at {experience.company}"
                for experience in candidate.experiences[:2]
                if experience.company or experience.title
            )

            snippets: list[str] = []

            for evidence in evidences_list:
                is_verified = getattr(
                    evidence,
                    "verified",
                    getattr(evidence, "is_verified", False),
                )

                if not is_verified:
                    continue

                snippet_list = getattr(
                    evidence,
                    "evidence_snippets",
                    None,
                )

                snippet = (
                    snippet_list[0]
                    if snippet_list
                    else getattr(
                        evidence,
                        "source_snippet",
                        "",
                    )
                )

                if snippet:
                    snippets.append(
                        f"{evidence.skill_name}: {snippet}"
                    )

            prompt = f"""Candidate Name: {candidate.name}
Most Recent Title: {candidate.most_recent_title}
Total Experience: {candidate.total_years_experience} years
Recent Experience: {exp_summary}
Target Job Title: {parsed_jd.title}
Final Score: {score_breakdown.final_score:.1f}
Verified Skills & Evidence: {"; ".join(snippets) if snippets else ", ".join(verified_skills)}
Missing/Unverified Skills: {", ".join(missing_skills)}

Generate exactly 3-4 sentences with the required score prefix."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": EXPLAINABILITY_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=settings.llm_temperature,
                max_tokens=1024,
            )

            if not response.choices:
                raise ValueError(
                    "LLM returned no explanation choices."
                )

            explanation = (
                response.choices[0].message.content or ""
            ).strip()

            if explanation and not explanation.startswith("```"):
                logger.info(
                    "Generated recruiter explanation via LLM for candidate '{}'.",
                    candidate.candidate_id,
                )
                return explanation

            raise ValueError(
                "LLM returned an invalid explanation."
            )

        except Exception as exc:
            logger.warning(
                "LLM explanation generation failed for candidate '{}': {}. "
                "Using fallback.",
                candidate.candidate_id,
                exc,
            )

            return self._deterministic_explanation(
                candidate,
                score_breakdown,
                verified_skills,
                missing_skills,
            )