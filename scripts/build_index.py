"""Build and persist BM25 and FAISS retrieval indexes for candidate search."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger
from app.models.candidate import CandidateProfile
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.faiss_index import FAISSIndexManager

logger = get_logger(__name__)


def load_candidates(path: Path) -> list[CandidateProfile]:
    """Load and validate candidate profiles from the canonical JSONL dataset.

    Args:
        path: Path to the normalized candidate JSONL file.

    Returns:
        Valid CandidateProfile objects. Invalid records are skipped.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Candidate data file not found: {path}"
        )

    candidates: list[CandidateProfile] = []
    invalid_count = 0

    logger.info(
        "Loading candidates from '{}'.",
        path,
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)
                candidate = CandidateProfile.model_validate(data)
                candidates.append(candidate)

            except Exception as exc:
                invalid_count += 1

                if invalid_count <= 10:
                    logger.warning(
                        "Skipping invalid candidate at line {}: {}",
                        line_number,
                        exc,
                    )

    logger.info(
        "Loaded {} valid candidates; skipped {} invalid records.",
        len(candidates),
        invalid_count,
    )

    return candidates


def build_and_save_bm25(
    candidates: list[CandidateProfile],
) -> None:
    """Build the BM25 index and persist its tokenized corpus to disk.

    Args:
        candidates: Candidate profiles used to construct the sparse index.
    """
    if not candidates:
        raise ValueError(
            "Cannot build BM25 index from an empty candidate list."
        )

    logger.info(
        "Building BM25 index for {} candidates.",
        len(candidates),
    )

    start_time = time.perf_counter()

    retriever = BM25Retriever()
    retriever.index_candidates(candidates)
    retriever.save_index(
        path=settings.bm25_index_path,
    )

    elapsed = time.perf_counter() - start_time

    logger.info(
        "BM25 index saved successfully in {:.2f} seconds to '{}'.",
        elapsed,
        settings.bm25_index_path,
    )


def ensure_faiss_index(
    candidates: list[CandidateProfile],
) -> None:
    """Validate the persisted FAISS index or build it when it is unavailable.

    Args:
        candidates: Candidate pool whose ordering must match the FAISS index.
    """
    if not candidates:
        raise ValueError(
            "Cannot prepare FAISS index from an empty candidate list."
        )

    index_path = Path(
        settings.faiss_index_path
    )

    manager = FAISSIndexManager()

    if index_path.exists():
        logger.info(
            "Existing FAISS index found at '{}'. Validating it.",
            index_path,
        )

        try:
            manager.load_index(
                str(index_path),
                candidates,
            )

            logger.info(
                "Existing FAISS index validated successfully with {} entries.",
                len(candidates),
            )
            return

        except Exception as exc:
            logger.warning(
                "Existing FAISS index could not be loaded: {}. "
                "Rebuilding it.",
                exc,
            )

    logger.info(
        "FAISS index is missing or invalid. Generating embeddings for {} candidates.",
        len(candidates),
    )

    start_time = time.perf_counter()

    embedding_service = EmbeddingService()

    embeddings = embedding_service.embed_candidates(
        candidates
    )

    if not embeddings:
        raise ValueError(
            "Embedding generation returned no vectors."
        )

    dimension = len(embeddings[0])

    if dimension <= 0:
        raise ValueError(
            "Generated embeddings have an invalid dimension."
        )

    for index, embedding in enumerate(embeddings):
        if len(embedding) != dimension:
            raise ValueError(
                "Embedding dimension mismatch at candidate {}: "
                "expected {}, got {}.",
                index,
                dimension,
                len(embedding),
            )

    manager.build_index(
        candidates,
        embeddings,
    )

    index_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manager.save_index(
        str(index_path)
    )

    elapsed = time.perf_counter() - start_time

    logger.info(
        "FAISS index built and saved successfully in {:.2f} seconds. "
        "Candidates={}, dimension={}, path='{}'.",
        elapsed,
        len(candidates),
        dimension,
        index_path,
    )


def write_index_metadata(
    candidates: list[CandidateProfile],
) -> None:
    """Write metadata describing the persisted BM25 and FAISS artifacts.

    Args:
        candidates: Candidate pool represented by the generated indexes.
    """
    faiss_path = Path(
        settings.faiss_index_path
    )

    metadata_path = faiss_path.with_suffix(
        ".metadata.json"
    )

    metadata = {
        "candidate_count": len(candidates),
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "faiss_index_type": "IndexFlatIP",
        "normalized_embeddings": True,
        "faiss_index_path": str(
            settings.faiss_index_path
        ),
        "bm25_index_path": str(
            settings.bm25_index_path
        ),
        "candidate_data_path": str(
            settings.candidates_data_path
        ),
    }

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    logger.info(
        "Index metadata saved to '{}'.",
        metadata_path,
    )


def main() -> None:
    """Build the reusable candidate retrieval artifacts from the canonical dataset."""
    start_time = time.perf_counter()

    try:
        data_path = Path(
            settings.candidates_data_path
        )

        logger.info(
            "Starting index build. data='{}', faiss='{}', bm25='{}'.",
            data_path,
            settings.faiss_index_path,
            settings.bm25_index_path,
        )

        candidates = load_candidates(
            data_path
        )

        if not candidates:
            logger.error(
                "No valid candidates available for indexing."
            )
            sys.exit(1)

        # Build/persist BM25 every time this script is intentionally run.
        build_and_save_bm25(
            candidates
        )

        # Reuse an existing FAISS index whenever it is valid.
        ensure_faiss_index(
            candidates
        )

        write_index_metadata(
            candidates
        )

        elapsed = time.perf_counter() - start_time

        logger.info(
            "Index build completed successfully in {:.2f} seconds.",
            elapsed,
        )

    except KeyboardInterrupt:
        logger.warning(
            "Index build interrupted by user."
        )
        sys.exit(130)

    except Exception as exc:
        logger.exception(
            "Index build failed: {}",
            exc,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()