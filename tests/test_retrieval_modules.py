"""Scratch unit tests for BM25Retriever, EmbeddingService, FAISSIndexManager, and MetadataFilterEngine."""

import tempfile
from pathlib import Path
import pytest

from app.core.exceptions import IndexNotFoundError
from app.models.candidate import CandidateProfile, WorkExperience
from app.models.jd import ParsedJobDescription, ExperienceRequirement
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.faiss_index import FAISSIndexManager
from app.retrieval.metadata_filter import MetadataFilterEngine


@pytest.fixture
def sample_candidates():
    c1 = CandidateProfile(
        candidate_id="cand_1",
        name="Alice Smith",
        skills=["Python", "FastAPI", "PostgreSQL"],
        experiences=[
            WorkExperience(
                company="TechCorp",
                title="Senior Backend Engineer",
                description="Built REST APIs using FastAPI and Python.",
                technologies_used=["Python", "FastAPI"]
            )
        ],
        total_years_experience=6.0,
        location="San Francisco, CA"
    )
    c2 = CandidateProfile(
        candidate_id="cand_2",
        name="Bob Jones",
        skills=["Java", "Spring Boot", "Docker"],
        experiences=[
            WorkExperience(
                company="DataSystems",
                title="Software Developer",
                description="Developed microservices in Java and Spring Boot.",
                technologies_used=["Java", "Spring Boot"]
            )
        ],
        total_years_experience=2.0,
        location="New York, NY"
    )
    c3 = CandidateProfile(
        candidate_id="cand_3",
        name="Charlie Brown",
        skills=["C++", "Qt", "Linux"],
        experiences=[
            WorkExperience(
                company="EmbeddedSys",
                title="Systems Programmer",
                description="Developed C++ desktop applications.",
                technologies_used=["C++", "Qt"]
            )
        ],
        total_years_experience=4.0,
        location="Austin, TX"
    )
    return [c1, c2, c3]


def test_bm25_retriever(sample_candidates):
    retriever = BM25Retriever(sample_candidates)
    
    # Test search with matching query
    results = retriever.search("Python FastAPI", top_k=10)
    assert len(results) == 3
    top_candidate, top_score = results[0]
    assert top_candidate.candidate_id == "cand_1"
    assert top_score == 1.0

    # Test search with empty query
    empty_results = retriever.search("")
    assert empty_results == []


def test_embedding_service(sample_candidates):
    service = EmbeddingService()
    
    # Test candidate embedding
    embeddings = service.embed_candidates(sample_candidates)
    assert len(embeddings) == 3
    assert len(embeddings[0]) > 0
    
    # Test query embedding
    query_emb = service.embed_query("Python developer")
    assert len(query_emb) == len(embeddings[0])


def test_faiss_index_manager(sample_candidates):
    service = EmbeddingService()
    embeddings = service.embed_candidates(sample_candidates)
    
    manager = FAISSIndexManager(sample_candidates, embeddings)
    query_emb = service.embed_query("Python FastAPI")
    
    results = manager.search_similar(query_emb, top_k=5)
    assert len(results) == 3
    assert results[0][0].candidate_id == "cand_1"
    assert results[0][1] > results[1][1]

    # Test save and load index
    with tempfile.TemporaryDirectory() as tmp_dir:
        index_path = Path(tmp_dir) / "test_index.faiss"
        manager.save_index(index_path)
        
        new_manager = FAISSIndexManager()
        new_manager.load_index(index_path, sample_candidates)
        assert new_manager.index is not None
        assert len(new_manager.candidates) == 3
        
        # Test loading non-existent index raises IndexNotFoundError
        with pytest.raises(IndexNotFoundError):
            new_manager.load_index(Path(tmp_dir) / "missing.faiss")


def test_metadata_filter_engine(sample_candidates):
    engine = MetadataFilterEngine()
    
    # JD requiring 5 years experience in San Francisco
    jd = ParsedJobDescription(
        title="Senior Python Engineer",
        experience=ExperienceRequirement(min_years=5.0),
        location="San Francisco"
    )
    
    filtered = engine.apply_filters(sample_candidates, jd)
    # Alice has 6.0 years experience (>= 5 - 3 = 2.0) and location San Francisco -> passes
    # Bob has 2.0 years experience but location New York != San Francisco -> fails
    # Charlie has 4.0 years experience but location Austin != San Francisco -> fails
    assert len(filtered) == 1
    assert filtered[0].candidate_id == "cand_1"

    # Test remote location matching
    remote_jd = ParsedJobDescription(
        title="Remote Engineer",
        experience=ExperienceRequirement(min_years=5.0),
        location="Remote"
    )
    remote_filtered = engine.apply_filters(sample_candidates, remote_jd)
    # Location requirement "Remote" allows all candidates
    assert len(remote_filtered) == 3
