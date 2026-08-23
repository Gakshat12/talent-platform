"""Hireability scoring module evaluating candidate experience presence, title seniority alignment, education, and skill count."""

from typing import Any

from app.core.logging import get_logger
from app.models.candidate import CandidateProfile
from app.models.jd import ParsedJobDescription

logger = get_logger(__name__)

SENIORITY_LEVEL_KEYWORDS: dict[str, list[str]] = {
    "senior": ["senior", "sr", "lead", "principal", "staff", "architect", "head", "director", "vp", "chief", "manager"],
    "mid": ["mid", "intermediate", "engineer", "developer", "analyst", "specialist", "consultant"],
    "junior": ["junior", "jr", "associate", "intern", "trainee", "entry", "apprentice"],
}


class HireabilityScorer:
    """Scorer evaluating career progression, seniority alignment, and background completeness."""

    def _check_seniority_match(self, candidate_title: str, target_seniority: str | None) -> bool:
        """Check if candidate's most recent title matches target seniority level.

        Args:
            candidate_title: Job title string.
            target_seniority: Target seniority string from ParsedJobDescription.

        Returns:
            True if title matches target seniority level or keywords; False otherwise.
        """
        if not candidate_title or not target_seniority:
            return False

        title_clean = candidate_title.strip().lower()
        target_clean = target_seniority.strip().lower()

        if target_clean in title_clean:
            return True

        keywords = SENIORITY_LEVEL_KEYWORDS.get(target_clean, [])
        for kw in keywords:
            if kw in title_clean:
                return True

        return False

    def score(self, candidate: CandidateProfile, parsed_jd: ParsedJobDescription) -> float:
        """Calculate additive hireability score capped at 1.0.

        Additive Components:
        - +0.30 if experience exists
        - +0.30 if most_recent_title matches seniority from ParsedJobDescription
        - +0.20 if education exists
        - +0.20 if 3+ skills

        Args:
            candidate: CandidateProfile instance.
            parsed_jd: ParsedJobDescription containing experience seniority level.

        Returns:
            Float hireability score capped at 1.0.
        """
        score = 0.0

        if candidate.experiences and len(candidate.experiences) > 0:
            score += 0.30

        target_seniority = (
            parsed_jd.experience.seniority_level if parsed_jd.experience else "mid"
        )
        if self._check_seniority_match(candidate.most_recent_title, target_seniority):
            score += 0.30

        if candidate.education and len(candidate.education) > 0:
            score += 0.20

        skill_count = max(len(candidate.skills), len(candidate.all_technologies))
        if skill_count >= 3:
            score += 0.20

        final_score = min(1.0, score)
        logger.debug(
            "Hireability score for candidate '{}': raw={}, final={:.4f}",
            candidate.candidate_id,
            score,
            final_score,
        )
        return final_score
