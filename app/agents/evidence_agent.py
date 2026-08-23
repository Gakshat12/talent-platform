"""Evidence Verification Agent module cross-checking skill claims against candidate career history via LLM or deterministic fallback."""

import json
import re
from typing import Any

from openai import OpenAI

from app.core.config import settings
from app.core.logging import get_logger
from app.models.candidate import CandidateProfile
from app.models.jd import ParsedJobDescription
from app.models.response import CandidateEvidence

logger = get_logger(__name__)

EVIDENCE_SYSTEM_PROMPT = """You are an expert technical resume verifier.
Given a candidate profile (skills and work experience entries) and a list of REQUIRED Job Description skills, determine whether the candidate's career history proves each required skill.

For EVERY required skill, evaluate candidate evidence and return ONLY a JSON array of objects with the following schema:
[
  {
    "skill_name": "Skill Name",
    "is_verified": true/false,
    "source_snippet": "Concise quote from career history proving skill (under 100 characters)",
    "confidence": 0.0 to 1.0
  }
]

Rules:
- source_snippet MUST quote or reference actual career history/experience and be under 100 characters.
- If the skill is NOT proven in career history or listed skills:
  - set is_verified to false
  - set source_snippet to "Not found in career history"
  - set confidence to 0.0

Return ONLY the JSON array. No explanation, no markdown, no backticks."""


class EvidenceVerificationAgent:
    """Agent verifying candidate skill claims against career history using OpenRouter LLM or deterministic fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize EvidenceVerificationAgent with OpenRouter client credentials.

        Args:
            api_key: OpenRouter API key. Defaults to settings.openrouter_api_key.
            base_url: OpenRouter base URL. Defaults to settings.openrouter_base_url.
            model: Model name string. Defaults to settings.llm_model.
        """
        self.api_key = api_key or settings.openrouter_api_key
        self.base_url = base_url or settings.openrouter_base_url
        self.model = model or settings.llm_model
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def _build_prompt(self, candidate: CandidateProfile, parsed_jd: ParsedJobDescription) -> str:
        """Construct prompt detailing candidate profile and required JD skills for LLM verification.

        Args:
            candidate: CandidateProfile instance.
            parsed_jd: ParsedJobDescription instance.

        Returns:
            Formatted user prompt string.
        """
        exp_lines: list[str] = []
        for idx, exp in enumerate(candidate.experiences):
            title = exp.title or ""
            company = exp.company or ""
            techs = ", ".join(exp.technologies_used) if exp.technologies_used else ""
            desc = exp.description[:300] if exp.description else ""
            exp_lines.append(f"Role {idx+1}: {title} at {company} | Tech: {techs} | Desc: {desc}")

        exp_str = "\n".join(exp_lines)
        skills_str = ", ".join(candidate.skills)
        req_skills_str = ", ".join(parsed_jd.required_skill_names)

        return f"""Candidate Name: {candidate.name}
Listed Skills: {skills_str}
Career History:
{exp_str}

REQUIRED SKILLS TO VERIFY:
{req_skills_str}

Return JSON array of verification objects matching the requested schema."""

    def _deterministic_evidence_check(
        self, candidate: CandidateProfile, parsed_jd: ParsedJobDescription
    ) -> list[CandidateEvidence]:
        """Perform deterministic fallback check of candidate skills and experience entries.

        Args:
            candidate: CandidateProfile instance.
            parsed_jd: ParsedJobDescription instance.

        Returns:
            List of CandidateEvidence objects.
        """
        evidences: list[CandidateEvidence] = []
        for req_skill in parsed_jd.required_skill_names:
            skill_clean = req_skill.strip().lower()
            found = False
            snippet = "Not found in career history"
            exp_idx: int | None = None

            for idx, exp in enumerate(candidate.experiences):
                techs = [t.lower() for t in exp.technologies_used]
                if skill_clean in techs:
                    found = True
                    snippet = f"Used {req_skill} at {exp.company}"[:95]
                    exp_idx = idx
                    break

                if exp.description:
                    desc_lower = exp.description.lower()
                    pattern = r"\b" + re.escape(skill_clean) + r"\b"
                    if re.search(pattern, desc_lower) or skill_clean in desc_lower:
                        found = True
                        snippet = exp.description[:95]
                        exp_idx = idx
                        break

            if not found:
                for skill in candidate.skills:
                    if skill and skill.strip().lower() == skill_clean:
                        found = True
                        snippet = f"Listed in candidate skills: {req_skill}"[:95]
                        break

            evidences.append(
                CandidateEvidence(
                    candidate_id=candidate.candidate_id,
                    skill_name=req_skill,
                    verified=found,
                    evidence_snippets=[snippet] if found else [],
                    matching_experience_index=exp_idx,
                    confidence=0.9 if found else 0.0,
                    reasoning=f"Proved: {snippet}" if found else "Not found in career history",
                )
            )

        return evidences

    def verify_candidate_evidence(
        self, candidate: CandidateProfile, parsed_jd: ParsedJobDescription
    ) -> list[CandidateEvidence]:
        """Verify candidate required skills via OpenRouter LLM, falling back to deterministic check on error.

        Args:
            candidate: CandidateProfile instance.
            parsed_jd: ParsedJobDescription instance.

        Returns:
            List of CandidateEvidence objects. Never raises exceptions.
        """
        if not parsed_jd.required_skill_names:
            return []

        try:
            prompt = self._build_prompt(candidate, parsed_jd)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": EVIDENCE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
            content = response.choices[0].message.content or ""
            clean_content = content.strip()

            if clean_content.startswith("```"):
                lines = clean_content.splitlines()
                if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
                    clean_content = "\n".join(lines[1:-1]).strip()

            data = json.loads(clean_content)
            if isinstance(data, list) and len(data) > 0:
                evidences: list[CandidateEvidence] = []
                for item in data:
                    is_verified = bool(item.get("is_verified", False))
                    snippet = str(item.get("source_snippet", "")).strip()[:100]
                    if not is_verified or not snippet:
                        snippet = "Not found in career history"

                    conf = float(item.get("confidence", 0.0)) if is_verified else 0.0
                    skill_name = str(item.get("skill_name", ""))

                    evidences.append(
                        CandidateEvidence(
                            candidate_id=candidate.candidate_id,
                            skill_name=skill_name,
                            verified=is_verified,
                            evidence_snippets=[snippet] if is_verified else [],
                            confidence=conf,
                            reasoning=f"Proved in career history: {snippet}"
                            if is_verified
                            else "Not found in career history",
                        )
                    )

                logger.info(
                    "Verified evidence via LLM for candidate '{}': {}/{} skills verified.",
                    candidate.candidate_id,
                    sum(1 for e in evidences if e.verified),
                    len(evidences),
                )
                return evidences

            logger.warning(
                "LLM evidence response not valid JSON array for candidate '{}'. Using fallback.",
                candidate.candidate_id,
            )
            return self._deterministic_evidence_check(candidate, parsed_jd)

        except Exception as e:
            logger.warning(
                "LLM call failed for evidence verification on candidate '{}': {}. Using fallback.",
                candidate.candidate_id,
                e,
            )
            return self._deterministic_evidence_check(candidate, parsed_jd)
