"""FAISS vector index manager module using IndexFlatIP for dense similarity search."""

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.core.config import settings
from app.core.exceptions import IndexNotFoundError
from app.core.logging import get_logger
from app.models.candidate import CandidateProfile

logger = get_logger(__name__)


class FAISSIndexManager:
    """Manager for building, searching, saving, and loading FAISS IndexFlatIP dense vector indexes."""

    def __init__(
        self,
        candidates: list[CandidateProfile] | None = None,
        embeddings: list[list[float]] | np.ndarray | None = None,
    ) -> None:
        """Initialize FAISSIndexManager with optional candidates and precomputed embeddings.

        Args:
            candidates: Optional list of CandidateProfile objects.
            embeddings: Optional list or numpy array of vector embeddings.
        """
        self.index: faiss.IndexFlatIP | None = None
        self.candidates: list[CandidateProfile] = []
        self.candidate_ids: list[str] = []
        self.candidate_map: dict[str, CandidateProfile] = {}

        if candidates is not None and embeddings is not None:
            self.build_index(candidates, embeddings)

    def build_index(
        self,
        candidates: list[CandidateProfile],
        embeddings: list[list[float]] | np.ndarray,
    ) -> None:
        """Build FAISS IndexFlatIP from normalized embeddings and store candidate mapping.

        Args:
            candidates: List of CandidateProfile instances.
            embeddings: Candidate embeddings as 2D list or numpy array.
        """
        if not candidates or len(embeddings) == 0:
            logger.warning("Empty candidates or embeddings passed to build_index.")
            return

        embeddings_array = np.array(embeddings, dtype=np.float32)
        if embeddings_array.ndim != 2:
            raise ValueError(f"Embeddings array must be 2D, got shape {embeddings_array.shape}")

        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings_array)

        self.candidates = list(candidates)
        self.candidate_ids = [c.candidate_id for c in candidates]
        self.candidate_map = {c.candidate_id: c for c in candidates}

        logger.info("Built FAISS IndexFlatIP with {} candidates.", self.index.ntotal)

    def search_similar(
        self, query_embedding: list[float] | np.ndarray, top_k: int = 500
    ) -> list[tuple[CandidateProfile, float]]:
        """Search for top_k most similar candidates given a query embedding vector.

        Args:
            query_embedding: Dense query vector as list or numpy array.
            top_k: Number of nearest neighbors to retrieve. Defaults to 500.

        Returns:
            List of (CandidateProfile, similarity_score) tuples ordered descending.
        """
        if self.index is None or self.index.ntotal == 0 or not self.candidates:
            return []

        query_array = np.array(query_embedding, dtype=np.float32)
        if query_array.ndim == 1:
            query_array = np.expand_dims(query_array, axis=0)

        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_array, k)

        results: list[tuple[CandidateProfile, float]] = []
        for idx, score in zip(indices[0], distances[0]):
            if idx == -1 or idx >= len(self.candidates):
                continue
            candidate = self.candidates[idx]
            results.append((candidate, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def save_index(self, index_file_path: str | Path | None = None) -> None:
        """Save FAISS index and candidate IDs sidecar file to specified path.

        Args:
            index_file_path: Path for saving index file. Defaults to settings.faiss_index_path.
        """
        if self.index is None:
            logger.warning("Attempted to save uninitialized FAISS index.")
            return

        target_path = Path(index_file_path or settings.faiss_index_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(target_path))

        ids_path = Path(str(target_path) + ".ids")
        ids_path.write_text(json.dumps(self.candidate_ids), encoding="utf-8")

        logger.info("Saved FAISS index and IDs sidecar to {}", target_path)

    def load_index(
        self,
        index_file_path: str | Path | None = None,
        candidates: list[CandidateProfile] | None = None,
    ) -> None:
        """Load FAISS index and candidate IDs sidecar file from disk.

        Args:
            index_file_path: Path to existing index file. Defaults to settings.faiss_index_path.
            candidates: Optional candidate list to map candidate IDs back to CandidateProfile objects.

        Raises:
            IndexNotFoundError: If specified index file does not exist on disk.
        """
        target_path = Path(index_file_path or settings.faiss_index_path)
        if not target_path.exists():
            logger.error("FAISS index file not found at path: {}", target_path)
            raise IndexNotFoundError(f"FAISS index file not found at {target_path}")

        self.index = faiss.read_index(str(target_path))

        ids_path = Path(str(target_path) + ".ids")
        if ids_path.exists():
            self.candidate_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        else:
            self.candidate_ids = []

        if candidates:
            cand_dict = {c.candidate_id: c for c in candidates}
            self.candidates = [cand_dict[cid] for cid in self.candidate_ids if cid in cand_dict]
            self.candidate_map = {c.candidate_id: c for c in self.candidates}

        logger.info("Loaded FAISS index with {} entries from {}", self.index.ntotal, target_path)
