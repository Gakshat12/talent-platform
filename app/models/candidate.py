"""Candidate profile data models for the AI Talent Intelligence Platform."""

from datetime import datetime, timezone

from dateutil.parser import parse as dateutil_parse
from pydantic import BaseModel, ConfigDict, Field


class EducationEntry(BaseModel):
    """Represents an education record for a candidate."""

    institution: str = Field(description="Name of the educational institution")
    degree: str = Field(description="Degree or qualification earned")
    field: str = Field(description="Field of study or major")
    year: int | None = Field(default=None, description="Graduation year")

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class WorkExperience(BaseModel):
    """Represents a work experience entry for a candidate."""

    company: str = Field(description="Name of the company or organization")
    title: str = Field(description="Job title or role")
    start_date: str | None = Field(default=None, description="Start date of employment")
    end_date: str | None = Field(default=None, description="End date of employment, or 'Present'")
    description: str = Field(default="", description="Detailed description of responsibilities and achievements")
    technologies_used: list[str] = Field(
        default_factory=list, description="Technologies, tools, and skills used in this role"
    )

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    @property
    def duration_months(self) -> int:
        """Calculates the duration of the work experience entry in months.

        Parses start_date and end_date using dateutil. Defaults to current date
        if end_date is missing or represents present employment. Returns 0 if dates
        are missing or invalid.

        Returns:
            int: Calculated total duration in months.
        """
        if not self.start_date:
            return 0

        try:
            start_dt = dateutil_parse(str(self.start_date))
        except (ValueError, TypeError, OverflowError):
            return 0

        if not self.end_date or str(self.end_date).strip().lower() in ("present", "current", "now", ""):
            end_dt = datetime.now(timezone.utc)
        else:
            try:
                end_dt = dateutil_parse(str(self.end_date))
            except (ValueError, TypeError, OverflowError):
                end_dt = datetime.now(timezone.utc)

        months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
        return max(0, months)


class CandidateProfile(BaseModel):
    """Represents the complete parsed profile and history of a candidate."""

    candidate_id: str = Field(description="Unique identifier for the candidate")
    name: str = Field(description="Full name of the candidate")
    skills: list[str] = Field(default_factory=list, description="Explicit skills listed on candidate profile")
    experiences: list[WorkExperience] = Field(
        default_factory=list, description="List of work experience entries"
    )
    total_years_experience: float = Field(
        default=0.0, description="Total calculated years of professional experience"
    )
    education: list[EducationEntry] = Field(
        default_factory=list, description="List of educational entries"
    )
    raw_resume_text: str = Field(default="", description="Raw resume text or profile content")
    location: str | None = Field(default=None, description="Candidate location or country")
    notice_period_days: int | None = Field(default=None, description="Notice period in days")

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    @property
    def work_experience(self) -> list[WorkExperience]:
        """Alias for experiences property to ensure code compatibility."""
        return self.experiences

    @property
    def all_technologies(self) -> list[str]:
        """Returns a deduplicated list of all explicit skills and technologies used in work experiences."""
        tech_set = set()
        result = []

        for skill in self.skills:
            if skill and skill.strip() and skill.strip() not in tech_set:
                tech_set.add(skill.strip())
                result.append(skill.strip())

        for exp in self.experiences:
            for tech in exp.technologies_used:
                if tech and tech.strip() and tech.strip() not in tech_set:
                    tech_set.add(tech.strip())
                    result.append(tech.strip())

        return result

    @property
    def most_recent_title(self) -> str:
        """Returns the job title from the candidate's most recent work experience, or empty string."""
        if self.experiences and self.experiences[0].title:
            return self.experiences[0].title
        return ""
