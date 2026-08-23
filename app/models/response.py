"""API response and scoring data models for the AI Talent Intelligence Platform."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.jd import ParsedJobDescription


class ScoreBreakdown(BaseModel):
    """Detailed breakdown of candidate ranking score components."""

    evidence_alignment: float = Field(
        default=0.0, description="Score for skill evidence alignment against JD requirements"
    )
    experience_fit: float = Field(
        default=0.0, description="Score for overall years of experience fit"
    )
    credibility: float = Field(
        default=0.0, description="Score for skill-to-career-history credibility"
    )
    hireability: float = Field(
        default=0.0, description="Score for career progression and stability"
    )
    penalty_deduction: float = Field(
        default=0.0, description="Total penalty score deducted (e.g. gaps, missing critical skills)"
    )
    final_score: float = Field(
        default=0.0, description="Final aggregated weighted match score (0.0 to 1.0 or 0 to 100)"
    )

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class CandidateEvidence(BaseModel):
    """Verified evidence snippet for a specific candidate skill claim."""

    candidate_id: str = Field(description="Unique candidate identifier")
    skill_name: str = Field(description="Skill name being verified")
    verified: bool = Field(default=False, description="Whether the skill was verified in career history")
    evidence_snippets: list[str] = Field(
        default_factory=list, description="Extracted resume text snippets supporting this skill claim"
    )
    matching_experience_index: int | None = Field(
        default=None, description="Index of matching work experience entry if found"
    )
    confidence: float = Field(default=0.0, description="Verification confidence score (0.0 to 1.0)")
    reasoning: str = Field(default="", description="Explanatory text for verification result")

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class CandidateRankResult(BaseModel):
    """Ranked candidate output object with detailed score breakdown and evidence."""

    rank: int = Field(description="Assigned 1-based rank position")
    candidate_id: str = Field(description="Unique candidate identifier")
    candidate_name: str = Field(default="", description="Candidate full name")
    score_breakdown: ScoreBreakdown = Field(description="Detailed component score breakdown")
    explanation: str = Field(default="", description="AI generated recruiter explanation for recommendation")
    evidences: list[CandidateEvidence] = Field(
        default_factory=list, description="List of verified skill evidences for candidate"
    )
    matched_skills: list[str] = Field(
        default_factory=list, description="List of candidate skills matching JD requirements"
    )
    missing_skills: list[str] = Field(
        default_factory=list, description="List of required JD skills missing from candidate profile"
    )

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    @property
    def name(self) -> str:
        """Alias property for candidate_name."""
        return self.candidate_name


class JDAnalysis(BaseModel):
    """Structured summary of the analyzed job description and retrieval stats."""

    parsed_jd: ParsedJobDescription = Field(description="Structured intent extracted from JD")
    expanded_queries: list[str] = Field(
        default_factory=list, description="Expanded queries generated for hybrid retrieval"
    )
    total_candidates_retrieved: int = Field(
        default=0, description="Number of candidate records initially retrieved"
    )

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class RankingResponse(BaseModel):
    """API response model for job candidate ranking operations."""

    job_title: str = Field(description="Target job title from analyzed JD")
    total_matching_candidates: int = Field(
        default=0, description="Total count of candidate results returned"
    )
    ranked_candidates: list[CandidateRankResult] = Field(
        default_factory=list, description="Ordered list of candidate rank results"
    )
    jd_analysis: JDAnalysis | None = Field(
        default=None, description="Details of JD analysis and retrieval execution"
    )

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class APIResponse(BaseModel):
    """Generic wrapper model for standard API endpoint responses."""

    success: bool = Field(default=True, description="Indicates if the operation succeeded")
    message: str = Field(default="Success", description="Human-readable response message")
    data: Any | None = Field(default=None, description="Response payload data")
    error: str | None = Field(default=None, description="Error message if operation failed")

    model_config = ConfigDict(populate_by_name=True, extra="ignore")
