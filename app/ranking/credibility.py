"""Credibility scoring module assessing candidate skill claims against work experience evidence."""

import re
from typing import Any

from app.core.logging import get_logger
from app.models.candidate import CandidateProfile

logger = get_logger(__name__)


class CredibilityScorer:
    """Scorer evaluating candidate skill-to-career-history credibility."""

    def _is_skill_in_experiences(self, skill_name: str, candidate: CandidateProfile) -> bool:
        """Check if a candidate skill claim is evidenced in work experience entries.

        Args:
            skill_name: Candidate skill string.
            candidate: CandidateProfile instance.

        Returns:
            True if skill is found in any work experience entry; False otherwise.
        """
        skill_clean = skill_name.strip().lower()
        if not skill_clean:
            return False

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

    def score(self, candidate: CandidateProfile) -> float:
        """Calculate credibility score.

        Rules:
        - no skills -> 0.2
        - base_score = evidenced_skills / total_skills
        - +0.10 bonus for 2+ distinct companies
        - score capped at 1.0

        Args:
            candidate: CandidateProfile instance.

        Returns:
            Float credibility score capped at 1.0.
        """
        skills = [s for s in candidate.skills if s and s.strip()]
        if not skills:
            logger.debug(
                "Candidate '{}' has no listed skills. Assigning default credibility score of 0.2.",
                candidate.candidate_id,
            )
            return 0.2

        evidenced_count = sum(
            1 for skill in skills if self._is_skill_in_experiences(skill, candidate)
        )
        base_ratio = float(evidenced_count) / float(len(skills))

        distinct_companies = {
            exp.company.strip().lower()
            for exp in candidate.experiences
            if exp.company and exp.company.strip()
        }
        company_bonus = 0.10 if len(distinct_companies) >= 2 else 0.0

        total_score = min(1.0, base_ratio + company_bonus)

        logger.debug(
            "Credibility score for candidate '{}': evidenced={}/{}, companies={}, score={:.4f}",
            candidate.candidate_id,
            evidenced_count,
            len(skills),
            len(distinct_companies),
            total_score,
        )
        return total_score
