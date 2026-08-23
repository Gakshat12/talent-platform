"""Scratch unit tests for BusinessRanker, JDUnderstandingAgent, EvidenceVerificationAgent, ExplainabilityAgent, and SupervisorAgent."""

from unittest.mock import MagicMock, patch
import pytest

from app.agents.evidence_agent import EvidenceVerificationAgent
from app.agents.explainability_agent import ExplainabilityAgent
from app.agents.jd_agent import JDUnderstandingAgent
from app.agents.supervisor import SupervisorAgent, should_retry_jd_parse, should_verify_evidence
from app.models.candidate import CandidateProfile, WorkExperience
from app.models.jd import ParsedJobDescription, SkillRequirement, ExperienceRequirement
from app.models.response import CandidateEvidence, CandidateRankResult, ScoreBreakdown
from app.ranking.business_ranker import BusinessRanker


@pytest.fixture
def sample_candidate():
    return CandidateProfile(
        candidate_id="cand_1",
        name="Alice Smith",
        skills=["Python", "FastAPI"],
        experiences=[
            WorkExperience(
                company="TechCorp",
                title="Senior Developer",
                description="Built APIs in Python and FastAPI.",
                technologies_used=["Python", "FastAPI"]
            )
        ],
        total_years_experience=5.0,
        location="San Francisco, CA"
    )


@pytest.fixture
def sample_jd():
    return ParsedJobDescription(
        title="Senior Python Engineer",
        skills=[SkillRequirement(name="Python", is_required=True)],
        experience=ExperienceRequirement(min_years=5.0, seniority_level="senior"),
        confidence_score=0.9
    )


def test_business_ranker(sample_candidate, sample_jd):
    ranker = BusinessRanker()
    ranked = ranker.rank_candidates([sample_candidate], sample_jd)
    
    assert len(ranked) == 1
    result = ranked[0]
    assert isinstance(result, CandidateRankResult)
    assert result.rank == 1
    assert result.candidate_id == "cand_1"
    assert result.matched_skills == ["Python"]


def test_jd_understanding_agent(sample_jd):
    mock_parser = MagicMock()
    mock_parser.parse.return_value = sample_jd
    
    agent = JDUnderstandingAgent(parser=mock_parser)
    result = agent.run("Raw JD text")
    
    assert result["status"] == "success"
    assert result["parsed_jd"] == sample_jd
    assert result["confidence"] == 0.9
    assert result["skill_count"] == 1


def test_evidence_verification_agent_fallback(sample_candidate, sample_jd):
    # Test deterministic fallback when LLM fails
    with patch("app.agents.evidence_agent.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API Error")
        mock_openai.return_value = mock_client
        
        agent = EvidenceVerificationAgent(api_key="test-key")
        evidences = agent.verify_candidate_evidence(sample_candidate, sample_jd)
        
        assert isinstance(evidences, list)
        assert len(evidences) == 1
        assert evidences[0].verified is True
        assert evidences[0].skill_name == "Python"


def test_explainability_agent_prefixes(sample_candidate, sample_jd):
    agent = ExplainabilityAgent()
    
    # Strong fit (score >= 60)
    score_strong = ScoreBreakdown(final_score=85.0)
    explanation_strong = agent._deterministic_explanation(sample_candidate, score_strong, ["Python"], [])
    assert explanation_strong.startswith("Strong fit because:")
    
    # Moderate fit (score 40-59.99)
    score_mod = ScoreBreakdown(final_score=45.0)
    explanation_mod = agent._deterministic_explanation(sample_candidate, score_mod, ["Python"], ["Kubernetes"])
    assert explanation_mod.startswith("Moderate fit:")
    
    # Partial fit (score < 40)
    score_low = ScoreBreakdown(final_score=30.0)
    explanation_low = agent._deterministic_explanation(sample_candidate, score_low, [], ["Python", "FastAPI"])
    assert explanation_low.startswith("Partial fit —")


def test_supervisor_agent(sample_jd):
    supervisor = SupervisorAgent()
    
    # test should_retry_jd_parse
    state_ok = {"parsed_jd": sample_jd, "retry_count": 0, "error": None}
    assert supervisor.should_retry_jd_parse(state_ok) == "continue"
    
    state_no_skills = {"parsed_jd": ParsedJobDescription(title="Dev", skills=[]), "retry_count": 0}
    assert supervisor.should_retry_jd_parse(state_no_skills) == "reparse"
    
    state_max_retry = {"parsed_jd": None, "retry_count": 2}
    assert supervisor.should_retry_jd_parse(state_max_retry) == "continue"

    # test should_verify_evidence
    state_cand_ok = {"retrieved_candidates": [1, 2, 3], "error": None}
    assert supervisor.should_verify_evidence(state_cand_ok) == "verify"
    
    state_too_many_cands = {"retrieved_candidates": [i for i in range(55)], "error": None}
    assert supervisor.should_verify_evidence(state_too_many_cands) == "skip_verify"
