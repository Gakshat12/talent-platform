"""SentenceTransformer embedding service module for generating candidate and query vector embeddings."""

from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logging import get_logger
from app.models.candidate import CandidateProfile

logger = get_logger(__name__)


class EmbeddingService:
    """Service to create dense vector representations for candidates and queries using SentenceTransformers."""

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize EmbeddingService with specified or default SentenceTransformer model.

        Args:
            model_name: Optional SentenceTransformer model identifier. Defaults to settings.embedding_model.
        """
        self.model_name = model_name or settings.embedding_model
        self.model = SentenceTransformer(self.model_name)

    def _build_candidate_text(self, candidate: CandidateProfile) -> str:
        """Build text representation incorporating candidate name, skills, title, experience, and tech.

        Args:
            candidate: CandidateProfile instance to format.

        Returns:
            Concatenated text string representation kept within target token limits.
        """
        skills_str = ", ".join(candidate.skills) if candidate.skills else ""
        recent_title = candidate.most_recent_title
        tech_str = ", ".join(candidate.all_technologies) if candidate.all_technologies else ""

        exp_parts: list[str] = []
        for exp in candidate.experiences:
            title = exp.title or ""
            desc = exp.description[:200] if exp.description else ""
            if title or desc:
                exp_parts.append(f"{title}: {desc}".strip(": "))

        exp_summary = "; ".join(exp_parts[:3])

        text_parts = [
            f"Candidate: {candidate.name}",
            f"Role: {recent_title}" if recent_title else "",
            f"Skills: {skills_str}" if skills_str else "",
            f"Technologies: {tech_str}" if tech_str else "",
            f"Experience: {exp_summary}" if exp_summary else "",
        ]

        full_text = " | ".join(p for p in text_parts if p)
        return full_text[:2000]

    def embed_candidates(
        self, candidates: list[CandidateProfile], batch_size: int = 64
    ) -> list[list[float]]:
        """Generate normalized vector embeddings for a list of candidate profiles.

        Args:
            candidates: List of CandidateProfile objects.
            batch_size: Batch size for model inference. Defaults to 64.

        Returns:
            List of normalized floating point embedding vectors.
        """
        if not candidates:
            return []

        texts = [self._build_candidate_text(c) for c in candidates]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        logger.info("Generated normalized embeddings for {} candidates.", len(candidates))
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Generate a single normalized vector embedding for a query string.

        Args:
            query: Input search query string.

        Returns:
            List of floats representing the query vector.
        """
        if not query or not query.strip():
            # Return zero vector matching model dimension if query is empty
            dim = self.model.get_sentence_embedding_dimension() or settings.embedding_dimension
            return [0.0] * dim

        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embedding.tolist()
