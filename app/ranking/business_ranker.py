"""Business ranker module orchestrating candidate scoring, skill matching, and rank assignment."""

from typing import Any

from app.core.logging import get_logger
from app.models.candidate import CandidateProfile
from app.models.jd import ParsedJobDescription
from app.models.response import CandidateRankResult
from app.ranking.evidence_alignment import EvidenceAlignmentScorer
from app.ranking.final_score import FinalScoreCalculator

logger = get_logger(__name__)


class BusinessRanker:
    """Ranks candidate profiles using deterministic business scoring logic."""

    def __init__(self) -> None:
        """Initialize BusinessRanker with FinalScoreCalculator and EvidenceAlignmentScorer."""
        self.final_calculator = FinalScoreCalculator()
        self.evidence_scorer = EvidenceAlignmentScorer()

    def rank_candidates(
        self, candidates: list[CandidateProfile], parsed_jd: ParsedJobDescription
    ) -> list[CandidateRankResult]:
        """Calculate score breakdowns, skill matches, and assign 1-based ranks to candidates.

        Args:
            candidates: List of CandidateProfile instances to rank.
            parsed_jd: ParsedJobDescription detailing job requirements.

        Returns:
            List of CandidateRankResult objects ordered by rank 1..N.
        """
        if not candidates:
            logger.info("BusinessRanker called with empty candidate list.")
            return []

        results: list[CandidateRankResult] = []

        for cand in candidates:
            score_breakdown = self.final_calculator.calculate_score(cand, parsed_jd)
            matched, missing = self.evidence_scorer.get_matched_and_missing_skills(cand, parsed_jd)

            rank_result = CandidateRankResult(
                rank=0,
                candidate_id=cand.candidate_id,
                candidate_name=cand.name,
                score_breakdown=score_breakdown,
                explanation="",
                evidences=[],
                matched_skills=matched,
                missing_skills=missing,
            )
            results.append(rank_result)

        results.sort(key=lambda r: r.score_breakdown.final_score, reverse=True)

        for idx, res in enumerate(results, start=1):
            res.rank = idx

        top_3 = [(r.candidate_name, r.score_breakdown.final_score) for r in results[:3]]
        logger.info("BusinessRanker assigned ranks for {} candidates. Top 3: {}", len(results), top_3)

        return results
