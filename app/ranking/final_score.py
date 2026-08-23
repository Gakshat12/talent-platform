"""Final score calculation module aggregating component scores into a final match score and ScoreBreakdown."""

from typing import Any

from app.core.logging import get_logger
from app.models.candidate import CandidateProfile
from app.models.jd import ParsedJobDescription
from app.models.response import ScoreBreakdown
from app.ranking.credibility import CredibilityScorer
from app.ranking.evidence_alignment import EvidenceAlignmentScorer
from app.ranking.experience_fit import ExperienceFitScorer
from app.ranking.hireability import HireabilityScorer
from app.ranking.penalties import PenaltyCalculator

logger = get_logger(__name__)


class FinalScoreCalculator:
    """Calculates final candidate match scores and produces structured ScoreBreakdown objects."""

    def __init__(self) -> None:
        """Initialize FinalScoreCalculator with all individual component scorers."""
        self.experience_scorer = ExperienceFitScorer()
        self.evidence_scorer = EvidenceAlignmentScorer()
        self.credibility_scorer = CredibilityScorer()
        self.hireability_scorer = HireabilityScorer()
        self.penalty_calculator = PenaltyCalculator()

    def calculate_score(
        self, candidate: CandidateProfile, parsed_jd: ParsedJobDescription
    ) -> ScoreBreakdown:
        """Calculate component scores, apply penalties, compute final weighted score, and return ScoreBreakdown.

        Weighted Raw Formula:
        raw = (evidence_alignment * 0.30
             + experience_fit * 0.25
             + credibility * 0.20
             + hireability * 0.15
             + 0.10)

        After Penalty:
        after_penalty = max(0.0, raw - penalties)

        Final Score:
        final_score = round(after_penalty * 100, 2)

        Args:
            candidate: CandidateProfile instance.
            parsed_jd: ParsedJobDescription instance.

        Returns:
            ScoreBreakdown object detailing all component scores and final match percentage.
        """
        evidence_alignment = self.evidence_scorer.score(candidate, parsed_jd)
        experience_fit = self.experience_scorer.score(candidate, parsed_jd)
        credibility = self.credibility_scorer.score(candidate)
        hireability = self.hireability_scorer.score(candidate, parsed_jd)

        penalties = self.penalty_calculator.calculate_penalties(candidate, evidence_alignment)

        raw = (
            evidence_alignment * 0.30
            + experience_fit * 0.25
            + credibility * 0.20
            + hireability * 0.15
            + 0.10
        )

        after_penalty = max(0.0, raw - penalties)
        final_score = round(after_penalty * 100, 2)

        logger.debug(
            "Candidate '{}' Score Breakdown: evidence={:.4f}, exp={:.4f}, cred={:.4f}, hire={:.4f}, raw={:.4f}, penalty={:.4f}, final={:.2f}",
            candidate.candidate_id,
            evidence_alignment,
            experience_fit,
            credibility,
            hireability,
            raw,
            penalties,
            final_score,
        )

        return ScoreBreakdown(
            evidence_alignment=round(evidence_alignment, 4),
            experience_fit=round(experience_fit, 4),
            credibility=round(credibility, 4),
            hireability=round(hireability, 4),
            penalty_deduction=round(penalties, 4),
            final_score=final_score,
        )
