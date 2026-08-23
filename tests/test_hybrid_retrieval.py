"""Scratch unit tests for RRFRanker, CrossEncoderReranker, and HybridRetriever."""

import pytest

from app.models.candidate import CandidateProfile, WorkExperience
from app.models.jd import ParsedJobDescription, SkillRequirement, ExperienceRequirement
from app.retrieval.hybrid_retrieval import HybridRetriever
from app.retrieval.reciprocal_rank_fusion import RRFRanker
from app.retrieval.reranker import CrossEncoderReranker


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
        skills=["Python", "PyTorch", "Deep Learning"],
        experiences=[
            WorkExperience(
                company="AI Lab",
                title="ML Engineer",
                description="Trained deep learning models using Python and PyTorch.",
                technologies_used=["Python", "PyTorch"]
            )
        ],
        total_years_experience=4.0,
        location="San Francisco, CA"
    )
    return [c1, c2, c3]


def test_rrf_ranker(sample_candidates):
    c1, c2, c3 = sample_candidates
    list1 = [(c1, 0.9), (c2, 0.8), (c3, 0.7)]
    list2 = [(c3, 0.95), (c1, 0.85)]
    
    rrf = RRFRanker(k_constant=60)
    fused = rrf.fuse_rankings([list1, list2], top_k=10)
    
    assert len(fused) == 3
    # c1 is rank 1 in list1 and rank 2 in list2 -> highest combined score
    assert fused[0][0].candidate_id == "cand_1"
    assert fused[0][1] == 1.0


def test_cross_encoder_reranker(sample_candidates):
    reranker = CrossEncoderReranker()
    jd = ParsedJobDescription(
        title="Senior Python Engineer",
        skills=[SkillRequirement(name="Python", is_required=True)],
        domain_keywords=["Backend", "APIs"],
        summary="Looking for a Python expert to build backend APIs."
    )
    
    reranked = reranker.rerank(jd, sample_candidates, top_k=10)
    assert len(reranked) == 3
    assert reranked[0][1] >= reranked[1][1] >= reranked[2][1]


def test_hybrid_retriever(sample_candidates):
    retriever = HybridRetriever(sample_candidates)
    
    jd = ParsedJobDescription(
        title="Senior Python Engineer",
        skills=[SkillRequirement(name="Python", is_required=True)],
        experience=ExperienceRequirement(min_years=5.0),
        domain_keywords=["Backend"],
        location="San Francisco",
        summary="Looking for a Senior Python developer in SF."
    )
    
    results = retriever.retrieve(jd, top_k=2)
    assert len(results) <= 2
    assert results[0][0].candidate_id == "cand_1"
    assert 0.0 <= results[0][1] <= 1.0
