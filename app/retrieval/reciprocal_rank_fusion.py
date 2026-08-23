"""Reciprocal Rank Fusion (RRF) module for combining multiple ranked candidate lists."""

from typing import Any

from app.core.logging import get_logger
from app.models.candidate import CandidateProfile

logger = get_logger(__name__)


class RRFRanker:
    """Combines multiple ranked lists using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, k_constant: int = 60) -> None:
        """Initialize RRFRanker with constant k smoothing factor.

        Args:
            k_constant: Smoothing constant for RRF formula. Defaults to 60.
        """
        self.k_constant = k_constant

    def _compute_rrf_scores(
        self, ranked_lists: list[list[tuple[CandidateProfile, float]]]
    ) -> dict[str, float]:
        """Aggregate candidate IDs across ranked lists and compute un-normalized RRF scores.

        RRF score formula: score(candidate) = sum(1.0 / (k_constant + rank))
        using 1-based rank indexing.

        Args:
            ranked_lists: List of ranked lists containing (CandidateProfile, score) tuples.

        Returns:
            Dictionary mapping candidate_id to calculated RRF score float.
        """
        rrf_scores: dict[str, float] = {}

        for ranked_list in ranked_lists:
            for rank, (candidate, _) in enumerate(ranked_list, start=1):
                cand_id = candidate.candidate_id
                score = 1.0 / (self.k_constant + rank)
                rrf_scores[cand_id] = rrf_scores.get(cand_id, 0.0) + score

        return rrf_scores

    def fuse_rankings(
        self,
        ranked_lists: list[list[tuple[CandidateProfile, float]]],
        top_k: int = 2000,
    ) -> list[tuple[CandidateProfile, float]]:
        """Fuse multiple ranked candidate lists, normalize RRF scores to 0-1 range, and return top_k candidates.

        Args:
            ranked_lists: List of ranked candidate lists from different retrieval paths.
            top_k: Maximum number of fused candidates to return. Defaults to 2000.

        Returns:
            List of (CandidateProfile, normalized_rrf_score) tuples ordered descending by score.
        """
        if not ranked_lists:
            return []

        cand_map: dict[str, CandidateProfile] = {}
        for ranked_list in ranked_lists:
            for candidate, _ in ranked_list:
                cand_map[candidate.candidate_id] = candidate

        if not cand_map:
            return []

        rrf_scores = self._compute_rrf_scores(ranked_lists)

        sorted_cand_ids = sorted(
            rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True
        )

        scores = [rrf_scores[cid] for cid in sorted_cand_ids]
        min_s = min(scores)
        max_s = max(scores)

        fused_results: list[tuple[CandidateProfile, float]] = []
        for cand_id in sorted_cand_ids:
            candidate = cand_map[cand_id]
            raw_s = rrf_scores[cand_id]

            if max_s > min_s:
                norm_s = (raw_s - min_s) / (max_s - min_s)
            else:
                norm_s = 1.0 if max_s > 0.0 else 0.0

            fused_results.append((candidate, norm_s))

        final_results = fused_results[:top_k]
        logger.info(
            "Fused {} ranked lists containing {} unique candidates into top {}.",
            len(ranked_lists),
            len(cand_map),
            len(final_results),
        )
        return final_results
