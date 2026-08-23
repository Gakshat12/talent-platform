"""Experience fit scoring module evaluating candidate total years of experience against JD requirements."""

from typing import Any

from app.core.logging import get_logger
from app.models.candidate import CandidateProfile
from app.models.jd import ParsedJobDescription

logger = get_logger(__name__)


class ExperienceFitScorer:
    """Scorer that evaluates total years of experience fit against JD min and max requirements."""

    def score(self, candidate: CandidateProfile, parsed_jd: ParsedJobDescription) -> float:
        """Calculate experience fit score between 0.10 and 1.0.

        Scoring Rules:
        - min_years = 0 -> 1.0
        - candidate >= min_years and <= max_years + 5 -> 1.0
        - candidate > max_years + 5 -> 0.75
        - shortfall <= 1 -> 0.70
        - shortfall 1 to 2 -> 0.45
        - shortfall 2 to 3 -> 0.25
        - shortfall > 3 -> 0.10

        Args:
            candidate: CandidateProfile instance containing total_years_experience.
            parsed_jd: ParsedJobDescription containing min_years and max_years experience band.

        Returns:
            Float experience fit score.
        """
        cand_years = candidate.total_years_experience or 0.0
        min_years = parsed_jd.experience.min_years if parsed_jd.experience else 0.0
        max_years = parsed_jd.experience.max_years if parsed_jd.experience else None

        if min_years <= 0.0:
            score = 1.0
        elif cand_years >= min_years:
            if max_years is not None:
                if cand_years <= max_years + 5.0:
                    score = 1.0
                else:
                    score = 0.75
            else:
                score = 1.0
        else:
            shortfall = min_years - cand_years
            if shortfall <= 1.0:
                score = 0.70
            elif shortfall <= 2.0:
                score = 0.45
            elif shortfall <= 3.0:
                score = 0.25
            else:
                score = 0.10

        logger.debug(
            "Experience fit score for candidate '{}': cand_years={}, min_years={}, max_years={}, score={}",
            candidate.candidate_id,
            cand_years,
            min_years,
            max_years,
            score,
        )
        return score
