"""Job Description data models for the AI Talent Intelligence Platform."""


from pydantic import BaseModel, ConfigDict, Field


class SkillRequirement(BaseModel):
    """Represents a specific skill requirement extracted from a Job Description."""

    name: str = Field(description="Name of the required or preferred skill")
    is_required: bool = Field(default=True, description="Whether the skill is strictly required")
    min_years: float = Field(default=0.0, description="Minimum years of experience requested for this skill")
    category: str | None = Field(default=None, description="Optional domain category (e.g. backend, cloud)")

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class ExperienceRequirement(BaseModel):
    """Represents experience and seniority expectations from a Job Description."""

    min_years: float = Field(default=0.0, description="Minimum overall years of professional experience required")
    max_years: float | None = Field(default=None, description="Optional maximum years of experience")
    seniority_level: str = Field(default="mid", description="Target seniority level (e.g. junior, mid, senior)")

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class ParsedJobDescription(BaseModel):
    """Represents the structured, parsed intent extracted from a raw Job Description."""

    title: str = Field(description="Extracted target job title")
    summary: str = Field(default="", description="Summary description of the role intent")
    skills: list[SkillRequirement] = Field(
        default_factory=list, description="List of skill requirements extracted from the JD"
    )
    experience: ExperienceRequirement = Field(
        default_factory=ExperienceRequirement, description="Experience requirements for the role"
    )
    domain_keywords: list[str] = Field(
        default_factory=list, description="Domain-specific keywords and technical concepts"
    )
    industry: str | None = Field(default=None, description="Target industry sector")
    location: str | None = Field(default=None, description="Target location or remote requirement")
    employment_type: str | None = Field(default=None, description="Employment type (e.g. full-time, contract)")
    responsibilities: list[str] = Field(
        default_factory=list, description="Key duties and responsibilities"
    )
    confidence_score: float = Field(
        default=1.0, description="Confidence score (0.0 to 1.0) of the JD parsing step"
    )

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    @property
    def required_skills(self) -> list[SkillRequirement]:
        """Returns the list of skills marked as required."""
        return [skill for skill in self.skills if skill.is_required]

    @property
    def preferred_skills(self) -> list[SkillRequirement]:
        """Returns the list of optional or preferred skills."""
        return [skill for skill in self.skills if not skill.is_required]

    @property
    def required_skill_names(self) -> list[str]:
        """Returns the names of all required skills as a list of strings."""
        return [skill.name for skill in self.skills if skill.is_required]


class JobDescriptionInput(BaseModel):
    """Input payload containing raw job description text to be processed."""

    raw_text: str = Field(description="Raw text content of the job description")
    source: str | None = Field(default=None, description="Optional source or origin of the job posting")

    model_config = ConfigDict(populate_by_name=True, extra="ignore")
