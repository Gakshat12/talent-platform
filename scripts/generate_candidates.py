"""Normalize raw candidate JSONL data into the platform's canonical schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.models.candidate import CandidateProfile, EducationEntry, WorkExperience

logger = get_logger(__name__)


def _clean_string(value: Any, default: str = "") -> str:
    """Convert a value into a safe trimmed string for canonical candidate data."""
    if value is None:
        return default

    try:
        return str(value).strip()
    except Exception:
        return default


def _extract_skills(raw_skills: Any) -> list[str]:
    """Extract unique skill names from the source dataset for candidate matching."""
    if not isinstance(raw_skills, list):
        return []

    skills: list[str] = []
    seen: set[str] = set()

    for item in raw_skills:
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict):
            name = _clean_string(item.get("name"))
        else:
            name = ""

        if not name:
            continue

        normalized = name.casefold()
        if normalized in seen:
            continue

        seen.add(normalized)
        skills.append(name)

    return skills


def _extract_education(raw_education: Any) -> list[EducationEntry]:
    """Convert source education records into the platform education model."""
    if not isinstance(raw_education, list):
        return []

    education_entries: list[EducationEntry] = []

    for item in raw_education:
        if not isinstance(item, dict):
            continue

        institution = _clean_string(item.get("institution"))
        degree = _clean_string(item.get("degree"))
        field = _clean_string(
            item.get("field_of_study") or item.get("field")
        )

        year_value = item.get("end_year")
        if year_value is None:
            year_value = item.get("start_year")

        year: int | None = None
        if year_value is not None:
            try:
                year = int(year_value)
            except (TypeError, ValueError):
                year = None

        # Skip completely empty education records.
        if not institution and not degree and not field and year is None:
            continue

        education_entries.append(
            EducationEntry(
                institution=institution,
                degree=degree,
                field=field,
                year=year,
            )
        )

    return education_entries


def _extract_experiences(raw_history: Any) -> list[WorkExperience]:
    """Convert source career history into the platform work-experience model."""
    if not isinstance(raw_history, list):
        return []

    experiences: list[WorkExperience] = []

    for item in raw_history:
        if not isinstance(item, dict):
            continue

        company = _clean_string(item.get("company"))
        title = _clean_string(item.get("title"))
        start_date = _clean_string(item.get("start_date")) or None
        end_date = _clean_string(item.get("end_date")) or None
        description = _clean_string(item.get("description"))

        if not company and not title and not description:
            continue

        # The raw dataset contains technologies mostly inside free-text
        # descriptions, rather than a dedicated per-experience technology list.
        # Do not invent technologies; keep this field empty and preserve the
        # original evidence in description.
        technologies_used: list[str] = []

        experiences.append(
            WorkExperience(
                company=company,
                title=title,
                start_date=start_date,
                end_date=end_date,
                description=description,
                technologies_used=technologies_used,
            )
        )

    return experiences


def _build_raw_resume_text(profile: dict[str, Any], experiences: list[WorkExperience]) -> str:
    """Build searchable resume text from the source profile and career history."""
    parts: list[str] = []

    summary = _clean_string(profile.get("summary"))
    headline = _clean_string(profile.get("headline"))
    current_title = _clean_string(profile.get("current_title"))
    current_company = _clean_string(profile.get("current_company"))

    if headline:
        parts.append(headline)

    if summary:
        parts.append(summary)

    if current_title:
        parts.append(f"Current title: {current_title}")

    if current_company:
        parts.append(f"Current company: {current_company}")

    for experience in experiences:
        experience_parts = [
            experience.title,
            experience.company,
            experience.description,
        ]
        text = ". ".join(part for part in experience_parts if part)
        if text:
            parts.append(text)

    return "\n".join(parts).strip()


def _normalize_candidate(raw_candidate: dict[str, Any]) -> CandidateProfile:
    """Convert one raw candidate record into a validated CandidateProfile."""
    candidate_id = _clean_string(raw_candidate.get("candidate_id"))

    raw_profile = raw_candidate.get("profile")
    profile = raw_profile if isinstance(raw_profile, dict) else {}

    skills = _extract_skills(raw_candidate.get("skills"))
    experiences = _extract_experiences(raw_candidate.get("career_history"))
    education = _extract_education(raw_candidate.get("education"))

    years_of_experience = profile.get("years_of_experience", 0.0)
    try:
        total_years_experience = float(years_of_experience)
    except (TypeError, ValueError):
        total_years_experience = 0.0

    raw_resume_text = _build_raw_resume_text(profile, experiences)

    candidate = CandidateProfile(
        candidate_id=candidate_id,
        name=_clean_string(
            profile.get("anonymized_name"),
            default=candidate_id or "Unknown Candidate",
        ),
        skills=skills,
        experiences=experiences,
        total_years_experience=max(0.0, total_years_experience),
        education=education,
        raw_resume_text=raw_resume_text,
        location=_clean_string(profile.get("location")) or None,
        notice_period_days=None,
    )

    return candidate


def normalize_candidates(
    input_path: Path,
    output_path: Path,
) -> tuple[int, int]:
    """Stream-convert raw candidate JSONL into validated canonical JSONL data."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input candidate file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid_count = 0
    invalid_count = 0

    logger.info(
        "Starting candidate normalization. input='{}', output='{}'",
        input_path,
        output_path,
    )

    with (
        input_path.open("r", encoding="utf-8") as source,
        output_path.open("w", encoding="utf-8") as destination,
    ):
        for line_number, line in enumerate(source, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                raw_candidate = json.loads(line)

                if not isinstance(raw_candidate, dict):
                    raise ValueError("Candidate record is not a JSON object.")

                candidate = _normalize_candidate(raw_candidate)

                if not candidate.candidate_id:
                    raise ValueError("Candidate record has no candidate_id.")

                destination.write(
                    json.dumps(
                        candidate.model_dump(mode="json"),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                valid_count += 1

                if valid_count % 10000 == 0:
                    logger.info(
                        "Normalized {} candidates. Invalid so far: {}",
                        valid_count,
                        invalid_count,
                    )

            except Exception as exc:
                invalid_count += 1

                if invalid_count <= 20:
                    logger.warning(
                        "Skipping invalid candidate at line {}: {}",
                        line_number,
                        exc,
                    )

    logger.info(
        "Normalization complete. valid={}, invalid={}, output='{}'",
        valid_count,
        invalid_count,
        output_path,
    )

    return valid_count, invalid_count


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments so the normalization script is reusable."""
    parser = argparse.ArgumentParser(
        description="Normalize raw candidate JSONL into CandidateProfile JSONL."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/candidates_raw.jsonl"),
        help="Path to the raw candidate JSONL file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/candidates.jsonl"),
        help="Path to the normalized CandidateProfile JSONL file.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the candidate normalization workflow from the command line."""
    args = parse_args()

    try:
        valid_count, invalid_count = normalize_candidates(
            input_path=args.input,
            output_path=args.output,
        )

        logger.info(
            "Candidate conversion finished successfully: valid={}, invalid={}",
            valid_count,
            invalid_count,
        )

    except Exception as exc:
        logger.error("Candidate conversion failed: {}", exc)
        raise


if __name__ == "__main__":
    main()