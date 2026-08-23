"""BM25 sparse retrieval module with persistent cached corpus support."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.core.logging import get_logger
from app.models.candidate import CandidateProfile

logger = get_logger(__name__)


STOPWORDS = {
    "the",
    "a",
    "an",
    "in",
    "of",
    "and",
    "or",
    "to",
    "for",
    "with",
    "is",
    "are",
    "was",
}


class BM25Retriever:
    """Provide sparse lexical candidate retrieval with reusable persisted corpus data."""

    DEFAULT_INDEX_PATH = "indexes/candidates.bm25.json"

    def __init__(
        self,
        candidates: list[CandidateProfile] | None = None,
    ) -> None:
        """Initialize BM25 storage and optionally index the supplied candidates.

        Args:
            candidates: Optional candidate profiles to index immediately.
        """
        self.candidates: list[CandidateProfile] = []
        self.bm25: BM25Okapi | None = None
        self.corpus: list[list[str]] = []

        if candidates:
            self.index_candidates(candidates)

    @staticmethod
    def _index_path() -> Path:
        """Return the configured path used to persist the BM25 corpus."""
        configured_path = getattr(
            settings,
            "bm25_index_path",
            None,
        )

        return Path(
            configured_path
            or BM25Retriever.DEFAULT_INDEX_PATH
        )

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase alphanumeric terms while removing stopwords.

        Args:
            text: Raw text to tokenize.

        Returns:
            Tokenized list of terms.
        """
        if not text:
            return []

        tokens = re.findall(
            r"\w+",
            text.lower(),
        )

        return [
            token
            for token in tokens
            if token not in STOPWORDS
        ]

    def _build_candidate_document(
        self,
        candidate: CandidateProfile,
    ) -> str:
        """Create a searchable text document from a candidate's career profile.

        Args:
            candidate: Candidate profile to convert into searchable text.

        Returns:
            Combined candidate document.
        """
        parts: list[str] = []

        if candidate.skills:
            parts.append(
                " ".join(candidate.skills)
            )

        technologies = candidate.all_technologies

        if technologies:
            parts.append(
                " ".join(technologies)
            )

        experience_titles = [
            experience.title
            for experience in candidate.experiences
            if experience.title
        ]

        if experience_titles:
            parts.append(
                " ".join(experience_titles)
            )

        experience_descriptions = [
            experience.description[:200]
            for experience in candidate.experiences
            if experience.description
        ]

        if experience_descriptions:
            parts.append(
                " ".join(experience_descriptions)
            )

        return " ".join(parts).strip()

    def index_candidates(
        self,
        candidates: list[CandidateProfile],
    ) -> None:
        """Build the BM25 index from candidate documents and store it in memory.

        Args:
            candidates: Candidate profiles to index.
        """
        self.candidates = list(candidates or [])

        if not self.candidates:
            self.corpus = []
            self.bm25 = None

            logger.warning(
                "BM25 index requested with an empty candidate list."
            )
            return

        documents = [
            self._build_candidate_document(candidate)
            for candidate in self.candidates
        ]

        self.corpus = [
            self._tokenize(document)
            for document in documents
        ]

        if self.corpus and any(
            len(document) > 0
            for document in self.corpus
        ):
            self.bm25 = BM25Okapi(
                self.corpus
            )
        else:
            self.bm25 = None

        logger.info(
            "Indexed {} candidates for BM25 search.",
            len(self.candidates),
        )

    def save_index(
        self,
        path: str | None = None,
    ) -> None:
        """Persist tokenized BM25 corpus metadata for faster application startup.

        Args:
            path: Optional destination path. Defaults to the configured BM25 path.
        """
        if not self.candidates or not self.corpus:
            logger.warning(
                "Cannot save BM25 index because no indexed candidates exist."
            )
            return

        output_path = Path(
            path
            if path
            else self._index_path()
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "version": 1,
            "candidate_ids": [
                candidate.candidate_id
                for candidate in self.candidates
            ],
            "corpus": self.corpus,
        }

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
            )

        logger.info(
            "Saved BM25 corpus for {} candidates to '{}'.",
            len(self.candidates),
            output_path,
        )

    def load_index(
        self,
        path: str | None = None,
        candidates: list[CandidateProfile] | None = None,
    ) -> bool:
        """Load a persisted BM25 corpus and reconstruct the searchable BM25 index.

        Args:
            path: Optional persisted index path.
            candidates: Current candidate objects used to restore candidate references.

        Returns:
            True when the persisted corpus is compatible and loaded successfully,
            otherwise False.
        """
        input_path = Path(
            path
            if path
            else self._index_path()
        )

        if not input_path.exists():
            logger.info(
                "Persisted BM25 index not found at '{}'.",
                input_path,
            )
            return False

        try:
            with input_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                payload = json.load(file)

            persisted_ids = payload.get(
                "candidate_ids",
                [],
            )

            persisted_corpus = payload.get(
                "corpus",
                [],
            )

            if (
                not isinstance(persisted_ids, list)
                or not isinstance(persisted_corpus, list)
                or len(persisted_ids) != len(persisted_corpus)
            ):
                logger.warning(
                    "Persisted BM25 index at '{}' has invalid structure.",
                    input_path,
                )
                return False

            candidate_pool = list(
                candidates or []
            )

            candidate_lookup = {
                candidate.candidate_id: candidate
                for candidate in candidate_pool
                if candidate.candidate_id
            }

            restored_candidates: list[CandidateProfile] = []

            for candidate_id in persisted_ids:
                candidate = candidate_lookup.get(
                    candidate_id
                )

                if candidate is None:
                    logger.warning(
                        "Candidate '{}' from BM25 index is missing from current dataset.",
                        candidate_id,
                    )
                    return False

                restored_candidates.append(candidate)

            corpus: list[list[str]] = []

            for document in persisted_corpus:
                if not isinstance(document, list):
                    return False

                corpus.append(
                    [
                        str(token)
                        for token in document
                    ]
                )

            if not corpus or not restored_candidates:
                return False

            self.candidates = restored_candidates
            self.corpus = corpus

            self.bm25 = BM25Okapi(
                self.corpus
            )

            logger.info(
                "Loaded persisted BM25 corpus with {} candidates from '{}'.",
                len(self.candidates),
                input_path,
            )

            return True

        except Exception as exc:
            logger.warning(
                "Failed to load persisted BM25 index from '{}': {}",
                input_path,
                exc,
            )

            self.candidates = []
            self.corpus = []
            self.bm25 = None

            return False

    def search(
        self,
        query_terms: list[str] | str,
        top_k: int = 500,
    ) -> list[tuple[CandidateProfile, float]]:
        """Search candidates with BM25 and normalize scores to the 0-1 range.

        Args:
            query_terms: Search query string or query terms.
            top_k: Maximum number of candidates to return.

        Returns:
            Candidate-score pairs sorted by descending normalized BM25 score.
        """
        if (
            self.bm25 is None
            or not self.candidates
        ):
            return []

        if isinstance(
            query_terms,
            str,
        ):
            tokenized_query = self._tokenize(
                query_terms
            )
        else:
            tokenized_query = self._tokenize(
                " ".join(query_terms)
            )

        if not tokenized_query:
            return []

        raw_scores = self.bm25.get_scores(
            tokenized_query
        )

        if len(raw_scores) == 0:
            return []

        min_score = float(
            np.min(raw_scores)
        )

        max_score = float(
            np.max(raw_scores)
        )

        if max_score > min_score:
            normalized_scores = (
                raw_scores - min_score
            ) / (
                max_score - min_score
            )
        elif max_score > 0.0:
            normalized_scores = np.ones_like(
                raw_scores,
                dtype=np.float32,
            )
        else:
            normalized_scores = np.zeros_like(
                raw_scores,
                dtype=np.float32,
            )

        scored_candidates = [
            (
                candidate,
                float(score),
            )
            for candidate, score in zip(
                self.candidates,
                normalized_scores,
            )
        ]

        scored_candidates.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        safe_top_k = max(
            int(top_k),
            0,
        )

        return scored_candidates[
            :safe_top_k
        ]