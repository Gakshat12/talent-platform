"""Evidence alignment scoring module cross-checking required JD skills against candidate resume evidence."""

import re
from typing import Any

from app.core.logging import get_logger
from app.models.candidate import CandidateProfile
from app.models.jd import ParsedJobDescription

logger = get_logger(__name__)


class EvidenceAlignmentScorer:
    """Scorer that checks required skills from JD against candidate skills, technologies, and descriptions."""

    def _is_skill_evidenced(self, skill_name: str, candidate: CandidateProfile) -> bool:
        """Check if a required skill is present in candidate skills, technologies_used, or work descriptions.

        Args:
            skill_name: Name of the required skill.
            candidate: CandidateProfile object to search.

        Returns:
            True if evidence is found in profile or experiences; False otherwise.
        """
        skill_clean = skill_name.strip().lower()
        if not skill_clean:
            return False

        for skill in candidate.skills:
            if skill and skill.strip().lower() == skill_clean:
                return True

        for exp in candidate.experiences:
            for tech in exp.technologies_used:
                if tech and tech.strip().lower() == skill_clean:
                    return True

            if exp.description:
                desc_clean = exp.description.lower()
                pattern = r"\b" + re.escape(skill_clean) + r"\b"
                if re.search(pattern, desc_clean) or skill_clean in desc_clean:
                    return True

        return False

    def get_matched_and_missing_skills(
        self, candidate: CandidateProfile, parsed_jd: ParsedJobDescription
    ) -> tuple[list[str], list[str]]:
        """Identify which required skills are matched versus missing for a candidate.

        Args:
            candidate: CandidateProfile instance.
            parsed_jd: ParsedJobDescription containing required skills.

        Returns:
            Tuple of (matched_skill_names, missing_skill_names).
        """
        required_skill_names = parsed_jd.required_skill_names
        if not required_skill_names:
            return [], []

        matched: list[str] = []
        missing: list[str] = []

        for req_skill in required_skill_names:
            if self._is_skill_evidenced(req_skill, candidate):
                matched.append(req_skill)
            else:
                missing.append(req_skill)

        return matched, missing

    def score(self, candidate: CandidateProfile, parsed_jd: ParsedJobDescription) -> float:
        """Calculate evidence alignment score as matched_required_skills / total_required_skills.

        If no required skills exist in ParsedJobDescription, returns 0.5.

        Args:
            candidate: CandidateProfile instance.
            parsed_jd: ParsedJobDescription instance.

        Returns:
            Float score representing proportion of required skills evidenced.
        """
        required_skill_names = parsed_jd.required_skill_names
        if not required_skill_names:
            logger.debug(
                "No required skills specified in JD. Returning default evidence alignment score of 0.5."
            )
            return 0.5

        matched, _ = self.get_matched_and_missing_skills(candidate, parsed_jd)
        score = float(len(matched)) / float(len(required_skill_names))

        logger.debug(
            "Evidence alignment score for candidate '{}': matched={}/{}, score={:.4f}",
            candidate.candidate_id,
            len(matched),
            len(required_skill_names),
            score,
        )
        return score
