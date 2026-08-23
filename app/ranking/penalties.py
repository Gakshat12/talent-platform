"""Penalty calculation module computing deductions for candidate profile gaps and missing requirements."""

from typing import Any

from app.core.logging import get_logger
from app.models.candidate import CandidateProfile

logger = get_logger(__name__)


class PenaltyCalculator:
    """Calculates score deductions based on profile gaps and low skill evidence alignment."""

    def calculate_penalties(
        self, candidate: CandidateProfile, evidence_alignment_score: float
    ) -> float:
        """Calculate total penalty score deduction capped at 0.30.

        Penalties:
        - +0.10 if candidate has no listed skills
        - +0.10 if candidate has no work experience
        - +0.10 if evidence alignment score < 0.20

        Args:
            candidate: CandidateProfile instance.
            evidence_alignment_score: Calculated evidence alignment score.

        Returns:
            Float penalty deduction capped at 0.30.
        """
        penalties = 0.0

        if not candidate.skills or len(candidate.skills) == 0:
            penalties += 0.10

        if not candidate.experiences or len(candidate.experiences) == 0:
            penalties += 0.10

        if evidence_alignment_score < 0.20:
            penalties += 0.10

        total_penalty = min(0.30, penalties)
        logger.debug(
            "Penalties for candidate '{}': raw={}, total={:.4f}",
            candidate.candidate_id,
            penalties,
            total_penalty,
        )
        return total_penalty
