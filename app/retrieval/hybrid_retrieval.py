"""Hybrid retrieval pipeline with reusable persisted retrieval resources."""

from __future__ import annotations

import threading
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import IndexNotFoundError
from app.core.logging import get_logger
from app.models.candidate import CandidateProfile
from app.models.jd import ParsedJobDescription
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.faiss_index import FAISSIndexManager
from app.retrieval.metadata_filter import MetadataFilterEngine
from app.retrieval.reciprocal_rank_fusion import RRFRanker
from app.retrieval.reranker import CrossEncoderReranker

logger = get_logger(__name__)


class HybridRetriever:
    """Orchestrate the five-stage retrieval pipeline with reusable resources."""

    _cache_lock = threading.Lock()
    _shared_initialized = False
    _shared_candidate_fingerprint: int | None = None

    _shared_candidates: list[CandidateProfile] = []
    _shared_bm25_retriever: BM25Retriever | None = None
    _shared_embedding_service: EmbeddingService | None = None
    _shared_faiss_manager: FAISSIndexManager | None = None
    _shared_metadata_filter: MetadataFilterEngine | None = None
    _shared_rrf_ranker: RRFRanker | None = None
    _shared_reranker: CrossEncoderReranker | None = None

    def __init__(
        self,
        candidates: list[CandidateProfile] | None = None,
    ) -> None:
        """Initialize the retriever and reuse cached retrieval resources.

        Args:
            candidates: Optional candidate pool used during first initialization.
        """
        candidate_pool = list(candidates or [])

        self._initialize_shared_resources(candidate_pool)

        self.candidates = self._shared_candidates
        self.bm25_retriever = self._shared_bm25_retriever
        self.embedding_service = self._shared_embedding_service
        self.faiss_manager = self._shared_faiss_manager
        self.metadata_filter = self._shared_metadata_filter
        self.rrf_ranker = self._shared_rrf_ranker
        self.reranker = self._shared_reranker

        # Request-specific queries must never be stored in shared state.
        self.expanded_queries: list[str] = []

    @classmethod
    def _candidate_fingerprint(
        cls,
        candidates: list[CandidateProfile],
    ) -> int:
        """Create a lightweight fingerprint for the current candidate pool.

        Args:
            candidates: Candidate profiles being indexed.

        Returns:
            In-process hash based on candidate IDs.
        """
        candidate_ids = tuple(
            candidate.candidate_id
            for candidate in candidates
            if candidate.candidate_id
        )
        return hash(candidate_ids)

    @classmethod
    def _initialize_shared_resources(
        cls,
        candidates: list[CandidateProfile],
    ) -> None:
        """Initialize retrieval resources once and reuse them across requests.

        Args:
            candidates: Candidate pool used by the retrieval system.
        """
        if not candidates:
            logger.warning(
                "HybridRetriever initialized with an empty candidate pool."
            )

        fingerprint = cls._candidate_fingerprint(candidates)

        if (
            cls._shared_initialized
            and cls._shared_candidate_fingerprint == fingerprint
        ):
            logger.info(
                "Reusing cached hybrid retrieval resources for {} candidates.",
                len(cls._shared_candidates),
            )
            return

        with cls._cache_lock:
            if (
                cls._shared_initialized
                and cls._shared_candidate_fingerprint == fingerprint
            ):
                return

            logger.info(
                "Initializing hybrid retrieval resources for {} candidates.",
                len(candidates),
            )

            cls._shared_candidates = list(candidates)
            cls._shared_candidate_fingerprint = fingerprint

            cls._shared_bm25_retriever = BM25Retriever()
            cls._shared_embedding_service = EmbeddingService()
            cls._shared_faiss_manager = FAISSIndexManager()

            cls._shared_metadata_filter = MetadataFilterEngine()
            cls._shared_rrf_ranker = RRFRanker(k_constant=60)
            cls._shared_reranker = CrossEncoderReranker()

            if cls._shared_candidates:
                cls._build_shared_indexes(cls._shared_candidates)

            cls._shared_initialized = True

            logger.info(
                "HybridRetriever shared resources initialized successfully."
            )

    @classmethod
    def _build_shared_indexes(
        cls,
        candidates: list[CandidateProfile],
    ) -> None:
        """Load persisted BM25 and FAISS indexes, building missing artifacts safely.

        Args:
            candidates: Candidate pool represented by the persisted indexes.
        """
        if cls._shared_bm25_retriever is None:
            raise RuntimeError(
                "BM25 retriever was not initialized."
            )

        if cls._shared_faiss_manager is None:
            raise RuntimeError(
                "FAISS manager was not initialized."
            )

        if cls._shared_embedding_service is None:
            raise RuntimeError(
                "Embedding service was not initialized."
            )

        # --------------------------------------------------------------
        # BM25
        # --------------------------------------------------------------
        bm25_loaded = cls._shared_bm25_retriever.load_index(
            path=settings.bm25_index_path,
            candidates=candidates,
        )

        if bm25_loaded:
            logger.info(
                "Loaded persisted BM25 index from '{}' for {} candidates.",
                settings.bm25_index_path,
                len(candidates),
            )
        else:
            logger.warning(
                "Persisted BM25 index unavailable or incompatible at '{}'. "
                "Building a new BM25 index.",
                settings.bm25_index_path,
            )

            cls._shared_bm25_retriever.index_candidates(
                candidates
            )

            cls._shared_bm25_retriever.save_index(
                path=settings.bm25_index_path
            )

            logger.info(
                "Built and persisted BM25 index for {} candidates.",
                len(candidates),
            )

        # --------------------------------------------------------------
        # FAISS
        # --------------------------------------------------------------
        index_path = Path(
            settings.faiss_index_path
        )

        try:
            cls._shared_faiss_manager.load_index(
                str(index_path),
                candidates,
            )

            logger.info(
                "Loaded persisted FAISS index from '{}' for {} candidates.",
                index_path,
                len(candidates),
            )

            return

        except IndexNotFoundError:
            logger.warning(
                "Persisted FAISS index not found at '{}'. "
                "Building the index from candidate embeddings.",
                index_path,
            )

        except Exception as exc:
            logger.warning(
                "Failed to load persisted FAISS index from '{}': {}. "
                "Falling back to index construction.",
                index_path,
                exc,
            )

        embeddings = cls._shared_embedding_service.embed_candidates(
            candidates
        )

        if not embeddings:
            raise RuntimeError(
                "Embedding generation returned no vectors."
            )

        cls._shared_faiss_manager.build_index(
            candidates,
            embeddings,
        )

        index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        cls._shared_faiss_manager.save_index(
            str(index_path)
        )

        logger.info(
            "Built and persisted FAISS index for {} candidates.",
            len(candidates),
        )

    def set_expanded_queries(
        self,
        expanded_queries: list[str] | None,
    ) -> None:
        """Set request-specific expanded retrieval queries for BM25.

        Args:
            expanded_queries: Alternative queries generated from the JD.
        """
        cleaned_queries: list[str] = []

        for query in expanded_queries or []:
            if not isinstance(query, str):
                continue

            cleaned = query.strip()

            if cleaned and cleaned not in cleaned_queries:
                cleaned_queries.append(cleaned)

        self.expanded_queries = cleaned_queries

        logger.info(
            "HybridRetriever received {} expanded retrieval queries.",
            len(self.expanded_queries),
        )

    def index_candidates(
        self,
        candidates: list[CandidateProfile],
    ) -> None:
        """Replace the candidate pool and rebuild persisted retrieval resources.

        Args:
            candidates: Candidate profiles to index.
        """
        candidate_pool = list(candidates or [])
        fingerprint = self._candidate_fingerprint(candidate_pool)

        with self._cache_lock:
            type(self)._shared_initialized = False
            type(self)._shared_candidate_fingerprint = fingerprint

        self._initialize_shared_resources(
            candidate_pool
        )

        self.candidates = self._shared_candidates
        self.bm25_retriever = self._shared_bm25_retriever
        self.embedding_service = self._shared_embedding_service
        self.faiss_manager = self._shared_faiss_manager
        self.metadata_filter = self._shared_metadata_filter
        self.rrf_ranker = self._shared_rrf_ranker
        self.reranker = self._shared_reranker

        logger.info(
            "HybridRetriever indexed {} candidates.",
            len(self.candidates),
        )

    def _build_default_bm25_query(
        self,
        parsed_jd: ParsedJobDescription,
    ) -> str:
        """Build a deterministic fallback BM25 query from structured JD fields.

        Args:
            parsed_jd: Parsed job description.

        Returns:
            Fallback BM25 query.
        """
        return (
            f"{parsed_jd.title} "
            f"{' '.join(parsed_jd.required_skill_names)} "
            f"{' '.join(parsed_jd.domain_keywords)} "
            f"{parsed_jd.experience.seniority_level}"
        ).strip()

    def _search_bm25(
        self,
        parsed_jd: ParsedJobDescription,
    ) -> list[tuple[CandidateProfile, float]]:
        """Run BM25 across expanded queries and merge results by best score.

        Args:
            parsed_jd: Parsed job description used for fallback retrieval.

        Returns:
            Deduplicated BM25 candidate-score pairs.
        """
        if self.bm25_retriever is None:
            logger.error(
                "BM25 retriever is not initialized."
            )
            return []

        default_query = self._build_default_bm25_query(
            parsed_jd
        )

        queries = self.expanded_queries or [default_query]

        if not queries:
            logger.warning(
                "No BM25 queries available."
            )
            return []

        candidate_scores: dict[
            str,
            tuple[CandidateProfile, float],
        ] = {}

        for query_number, query in enumerate(
            queries,
            start=1,
        ):
            if not query.strip():
                continue

            results = self.bm25_retriever.search(
                query,
                top_k=settings.bm25_top_k,
            )

            logger.debug(
                "BM25 expanded query {} retrieved {} candidates.",
                query_number,
                len(results),
            )

            for candidate, score in results:
                candidate_id = candidate.candidate_id

                if not candidate_id:
                    continue

                existing = candidate_scores.get(
                    candidate_id
                )

                if (
                    existing is None
                    or score > existing[1]
                ):
                    candidate_scores[candidate_id] = (
                        candidate,
                        score,
                    )

        merged_results = sorted(
            candidate_scores.values(),
            key=lambda item: item[1],
            reverse=True,
        )

        return merged_results[
            : settings.bm25_top_k
        ]

    def retrieve(
        self,
        parsed_jd: ParsedJobDescription,
        top_k: int | None = None,
    ) -> list[tuple[CandidateProfile, float]]:
        """Execute BM25, FAISS, metadata filtering, RRF, and CrossEncoder retrieval.

        Args:
            parsed_jd: Structured job description used for retrieval.
            top_k: Optional final number of candidates.

        Returns:
            Final candidate-score pairs sorted by reranker score.
        """
        if parsed_jd is None:
            logger.warning(
                "HybridRetriever received a missing ParsedJobDescription."
            )
            return []

        if not self.candidates:
            logger.warning(
                "HybridRetriever search called with empty candidate pool."
            )
            return []

        if (
            self.bm25_retriever is None
            or self.embedding_service is None
            or self.faiss_manager is None
            or self.metadata_filter is None
            or self.rrf_ranker is None
            or self.reranker is None
        ):
            logger.error(
                "HybridRetriever resources are not initialized."
            )
            return []

        final_top_k = (
            top_k
            if top_k is not None
            else settings.final_top_k
        )

        # --------------------------------------------------------------
        # Stage 1: BM25 sparse retrieval.
        # --------------------------------------------------------------
        bm25_results = self._search_bm25(
            parsed_jd
        )

        logger.info(
            "Stage 1 - BM25 retrieved {} unique candidates from {} queries.",
            len(bm25_results),
            len(self.expanded_queries)
            if self.expanded_queries
            else 1,
        )

        # --------------------------------------------------------------
        # Stage 2: FAISS dense retrieval.
        # --------------------------------------------------------------
        dense_query = self.reranker._build_query(
            parsed_jd
        )

        query_embedding = self.embedding_service.embed_query(
            dense_query
        )

        faiss_results = self.faiss_manager.search_similar(
            query_embedding,
            top_k=settings.dense_top_k,
        )

        logger.info(
            "Stage 2 - FAISS retrieved {} candidates.",
            len(faiss_results),
        )

        # --------------------------------------------------------------
        # Stage 3: Union + metadata filtering.
        # --------------------------------------------------------------
        union_map: dict[
            str,
            CandidateProfile,
        ] = {}

        for candidate, _ in bm25_results:
            if candidate.candidate_id:
                union_map[
                    candidate.candidate_id
                ] = candidate

        for candidate, _ in faiss_results:
            if candidate.candidate_id:
                union_map[
                    candidate.candidate_id
                ] = candidate

        union_candidates = list(
            union_map.values()
        )

        filtered_candidates = (
            self.metadata_filter.apply_filters(
                union_candidates,
                parsed_jd,
            )
        )

        filtered_ids = {
            candidate.candidate_id
            for candidate in filtered_candidates
            if candidate.candidate_id
        }

        filtered_bm25 = [
            item
            for item in bm25_results
            if item[0].candidate_id in filtered_ids
        ]

        filtered_faiss = [
            item
            for item in faiss_results
            if item[0].candidate_id in filtered_ids
        ]

        logger.info(
            "Stage 3 - Union pool of {} candidates "
            "filtered down to {} candidates.",
            len(union_candidates),
            len(filtered_candidates),
        )

        if not filtered_candidates:
            logger.warning(
                "No candidates survived metadata filtering stage."
            )
            return []

        # --------------------------------------------------------------
        # Stage 4: RRF fusion.
        # --------------------------------------------------------------
        rrf_results = self.rrf_ranker.fuse_rankings(
            [
                filtered_bm25,
                filtered_faiss,
            ],
            top_k=settings.rrf_top_k,
        )

        logger.info(
            "Stage 4 - RRF fused candidates down to top {}.",
            len(rrf_results),
        )

        if not rrf_results:
            return []

        # --------------------------------------------------------------
        # Stage 5: CrossEncoder reranking.
        # --------------------------------------------------------------
        rrf_candidates = [
            candidate
            for candidate, _ in rrf_results
        ]

        final_results = self.reranker.rerank(
            parsed_jd,
            rrf_candidates,
            top_k=final_top_k,
        )

        logger.info(
            "Stage 5 - CrossEncoder reranked down to final top {} candidates.",
            len(final_results),
        )

        return final_results

    @classmethod
    def reset_cache(cls) -> None:
        """Clear all shared retrieval resources for a deliberate rebuild."""
        with cls._cache_lock:
            cls._shared_initialized = False
            cls._shared_candidate_fingerprint = None
            cls._shared_candidates = []
            cls._shared_bm25_retriever = None
            cls._shared_embedding_service = None
            cls._shared_faiss_manager = None
            cls._shared_metadata_filter = None
            cls._shared_rrf_ranker = None
            cls._shared_reranker = None

        logger.info(
            "HybridRetriever shared resource cache cleared."
        )