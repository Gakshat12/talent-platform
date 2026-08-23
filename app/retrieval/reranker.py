"""CrossEncoder reranking module for precision scoring of candidate profiles against job requirements."""

import time
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.core.logging import get_logger
from app.models.candidate import CandidateProfile
from app.models.jd import ParsedJobDescription

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Reranker using SentenceTransformers CrossEncoder model for fine-grained relevance scoring."""

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize CrossEncoderReranker with specified or default model.

        Args:
            model_name: Optional CrossEncoder model name. Defaults to settings.reranker_model.
        """
        self.model_name = model_name or settings.reranker_model
        self.model = CrossEncoder(self.model_name)

    def _build_query(self, parsed_jd: ParsedJobDescription) -> str:
        """Combine JD title, required skills, top domain keywords, and summary into concise query text.

        Args:
            parsed_jd: ParsedJobDescription object.

        Returns:
            Formatted query string truncated to approximately 300 characters.
        """
        title = parsed_jd.title
        req_skills = ", ".join(parsed_jd.required_skill_names)
        domain_kw = ", ".join(parsed_jd.domain_keywords[:3])
        summary_snippet = parsed_jd.summary[:200] if parsed_jd.summary else ""

        query_parts = [
            f"Title: {title}",
            f"Required Skills: {req_skills}" if req_skills else "",
            f"Keywords: {domain_kw}" if domain_kw else "",
            f"Summary: {summary_snippet}" if summary_snippet else "",
        ]

        full_query = " | ".join(p for p in query_parts if p)
        return full_query[:300]

    def _build_candidate_text(self, candidate: CandidateProfile) -> str:
        """Combine candidate most recent title, top 15 skills, and top 3 experience descriptions.

        Args:
            candidate: CandidateProfile instance.

        Returns:
            Concatenated candidate text representation for CrossEncoder scoring.
        """
        recent_title = candidate.most_recent_title
        top_skills = ", ".join(candidate.skills[:15]) if candidate.skills else ""

        exp_parts: list[str] = []
        for exp in candidate.experiences[:3]:
            title = exp.title or ""
            desc = exp.description[:200] if exp.description else ""
            if title or desc:
                exp_parts.append(f"{title}: {desc}".strip(": "))

        exp_str = "; ".join(exp_parts)

        text_parts = [
            f"Title: {recent_title}" if recent_title else "",
            f"Skills: {top_skills}" if top_skills else "",
            f"Experience: {exp_str}" if exp_str else "",
        ]

        return " | ".join(p for p in text_parts if p)

    def rerank(
        self,
        query_or_jd: str | ParsedJobDescription,
        candidates: list[CandidateProfile],
        top_k: int = 100,
    ) -> list[tuple[CandidateProfile, float]]:
        """Rerank candidates using CrossEncoder predictions, min-max normalize scores, and return top_k.

        Args:
            query_or_jd: Query string or ParsedJobDescription instance.
            candidates: List of CandidateProfile objects to rerank.
            top_k: Number of top ranked candidates to return. Defaults to 100.

        Returns:
            List of (CandidateProfile, normalized_rerank_score) tuples ordered descending.
        """
        if not candidates:
            return []

        start_time = time.perf_counter()

        if isinstance(query_or_jd, ParsedJobDescription):
            query_str = self._build_query(query_or_jd)
        else:
            query_str = str(query_or_jd)

        pairs = [(query_str, self._build_candidate_text(c)) for c in candidates]

        scores = self.model.predict(pairs, batch_size=32)

        min_s = float(np.min(scores))
        max_s = float(np.max(scores))

        if max_s > min_s:
            norm_scores = (scores - min_s) / (max_s - min_s)
        else:
            norm_scores = np.full_like(scores, 0.5, dtype=np.float32)

        results = [(c, float(sc)) for c, sc in zip(candidates, norm_scores)]
        results.sort(key=lambda x: x[1], reverse=True)

        elapsed = time.perf_counter() - start_time
        final_results = results[:top_k]

        logger.info(
            "Reranked {} candidates in {:.3f}s down to top {}.",
            len(candidates),
            elapsed,
            len(final_results),
        )
        return final_results
