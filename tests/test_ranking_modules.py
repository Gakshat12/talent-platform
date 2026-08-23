"""Scratch unit tests for ExperienceFitScorer, EvidenceAlignmentScorer, CredibilityScorer, HireabilityScorer, PenaltyCalculator, and FinalScoreCalculator."""

import pytest

from app.models.candidate import CandidateProfile, WorkExperience, EducationEntry
from app.models.jd import ParsedJobDescription, SkillRequirement, ExperienceRequirement
from app.models.response import ScoreBreakdown
from app.ranking.credibility import CredibilityScorer
from app.ranking.evidence_alignment import EvidenceAlignmentScorer
from app.ranking.experience_fit import ExperienceFitScorer
from app.ranking.final_score import FinalScoreCalculator
from app.ranking.hireability import HireabilityScorer
from app.ranking.penalties import PenaltyCalculator


@pytest.fixture
def sample_candidate():
    return CandidateProfile(
        candidate_id="cand_101",
        name="John Developer",
        skills=["Python", "FastAPI", "Docker", "PostgreSQL"],
        experiences=[
            WorkExperience(
                company="Company A",
                title="Senior Software Engineer",
                description="Developing backend services using Python and FastAPI.",
                technologies_used=["Python", "FastAPI"]
            ),
            WorkExperience(
                company="Company B",
                title="Backend Developer",
                description="Worked with PostgreSQL and Docker.",
                technologies_used=["PostgreSQL", "Docker"]
            )
        ],
        education=[
            EducationEntry(institution="State Univ", degree="BS", field="Computer Science", year=2018)
        ],
        total_years_experience=6.0,
        location="San Francisco, CA"
    )


@pytest.fixture
def sample_jd():
    return ParsedJobDescription(
        title="Senior Python Engineer",
        skills=[
            SkillRequirement(name="Python", is_required=True),
            SkillRequirement(name="FastAPI", is_required=True),
            SkillRequirement(name="Kubernetes", is_required=True)  # Missing from candidate
        ],
        experience=ExperienceRequirement(min_years=5.0, max_years=8.0, seniority_level="senior"),
        domain_keywords=["Backend"]
    )


def test_experience_fit_scorer(sample_candidate, sample_jd):
    scorer = ExperienceFitScorer()
    
    # 6 years experience for 5-8 year JD -> 1.0
    score = scorer.score(sample_candidate, sample_jd)
    assert score == 1.0

    # Shortfall test: min 8 years, candidate has 6 -> shortfall 2 -> 0.45
    jd_high_exp = ParsedJobDescription(
        title="Lead Engineer",
        experience=ExperienceRequirement(min_years=8.0)
    )
    score_shortfall = scorer.score(sample_candidate, jd_high_exp)
    assert score_shortfall == 0.45


def test_evidence_alignment_scorer(sample_candidate, sample_jd):
    scorer = EvidenceAlignmentScorer()
    
    # 2 out of 3 required skills (Python, FastAPI) matched -> 2/3 = 0.6667
    matched, missing = scorer.get_matched_and_missing_skills(sample_candidate, sample_jd)
    assert matched == ["Python", "FastAPI"]
    assert missing == ["Kubernetes"]
    
    score = scorer.score(sample_candidate, sample_jd)
    assert round(score, 4) == round(2.0 / 3.0, 4)


def test_credibility_scorer(sample_candidate):
    scorer = CredibilityScorer()
    
    # All 4 skills are in experiences + 2 distinct companies -> 1.0 + 0.10 capped at 1.0
    score = scorer.score(sample_candidate)
    assert score == 1.0


def test_hireability_scorer(sample_candidate, sample_jd):
    scorer = HireabilityScorer()
    
    # Experience exists (+0.30)
    # Title 'Senior Software Engineer' matches 'senior' (+0.30)
    # Education exists (+0.20)
    # 4 skills >= 3 (+0.20)
    # Total = 1.0
    score = scorer.score(sample_candidate, sample_jd)
    assert score == 1.0


def test_penalty_calculator(sample_candidate):
    calc = PenaltyCalculator()
    
    # Has skills, has experience, alignment score 0.66 >= 0.20 -> 0.0 penalty
    penalty = calc.calculate_penalties(sample_candidate, evidence_alignment_score=0.66)
    assert penalty == 0.0

    # Low alignment < 0.20 -> 0.10 penalty
    penalty_low = calc.calculate_penalties(sample_candidate, evidence_alignment_score=0.10)
    assert penalty_low == 0.10


def test_final_score_calculator(sample_candidate, sample_jd):
    calc = FinalScoreCalculator()
    breakdown = calc.calculate_score(sample_candidate, sample_jd)
    
    assert isinstance(breakdown, ScoreBreakdown)
    assert breakdown.experience_fit == 1.0
    assert round(breakdown.evidence_alignment, 4) == round(2.0 / 3.0, 4)
    assert breakdown.credibility == 1.0
    assert breakdown.hireability == 1.0
    assert breakdown.penalty_deduction == 0.0
    
    # Check formula calculation:
    # raw = (2/3 * 0.30) + (1.0 * 0.25) + (1.0 * 0.20) + (1.0 * 0.15) + 0.10
    # raw = 0.20 + 0.25 + 0.20 + 0.15 + 0.10 = 0.90
    # after_penalty = 0.90
    # final_score = 90.0
    assert breakdown.final_score == 90.0
