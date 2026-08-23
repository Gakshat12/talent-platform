"""Metadata filtering engine module for conservative filtering of candidates based on experience and location."""

import re

from app.core.logging import get_logger
from app.models.candidate import CandidateProfile
from app.models.jd import ParsedJobDescription

logger = get_logger(__name__)

FLEXIBLE_LOCATION_KEYWORDS = {"remote", "anywhere", "worldwide", "any", "telecommute", "work from home"}


class MetadataFilterEngine:
    """Engine providing conservative metadata filtering based on candidate experience and location."""

    def _check_experience(
        self, candidate: CandidateProfile, min_years: float | None
    ) -> bool:
        """Verify candidate experience against job requirements with a conservative 3-year buffer.

        Args:
            candidate: CandidateProfile instance.
            min_years: Minimum years of experience requested by JD, or None.

        Returns:
            True if candidate passes experience check or experience is unknown/unspecified.
        """
        if min_years is None or min_years <= 0.0:
            return True

        cand_years = candidate.total_years_experience
        if cand_years is None or cand_years == 0.0:
            return True

        return cand_years >= (min_years - 3.0)

    def _check_location(
        self, candidate: CandidateProfile, location_req: str | None
    ) -> bool:
        """Verify candidate location against job location requirements using flexible matching.

        Args:
            candidate: CandidateProfile instance.
            location_req: Target location requested in JD, or None.

        Returns:
            True if location matches, is remote/flexible, or either location is unspecified.
        """
        if not location_req or not location_req.strip():
            return True

        req_clean = location_req.strip().lower()
        req_words = set(re.findall(r"\w+", req_clean))

        if FLEXIBLE_LOCATION_KEYWORDS.intersection(req_words):
            return True

        if not candidate.location or not candidate.location.strip():
            return True

        cand_clean = candidate.location.strip().lower()
        cand_words = set(re.findall(r"\w+", cand_clean))

        if FLEXIBLE_LOCATION_KEYWORDS.intersection(cand_words):
            return True

        common_words = cand_words.intersection(req_words)
        # Exclude common non-location words if any
        stop_loc = {"state", "city", "country", "region", "area"}
        meaningful_common = [w for w in common_words if w not in stop_loc]

        if meaningful_common:
            return True

        return req_clean in cand_clean or cand_clean in req_clean

    def apply_filters(
        self, candidates: list[CandidateProfile], parsed_jd: ParsedJobDescription
    ) -> list[CandidateProfile]:
        """Apply experience and location metadata filters conservatively to a list of candidates.

        If all candidates are filtered out, returns the original candidate list as a fallback.

        Args:
            candidates: Candidate list to filter.
            parsed_jd: ParsedJobDescription containing role requirements.

        Returns:
            Filtered list of CandidateProfile objects (or original list as fallback).
        """
        if not candidates:
            return []

        min_years = parsed_jd.experience.min_years if parsed_jd.experience else None
        location_req = parsed_jd.location

        filtered: list[CandidateProfile] = []
        for candidate in candidates:
            if self._check_experience(candidate, min_years) and self._check_location(
                candidate, location_req
            ):
                filtered.append(candidate)

        if not filtered and candidates:
            logger.warning(
                "All {} candidates were filtered out by metadata constraints. Returning original list as fallback.",
                len(candidates),
            )
            return candidates

        logger.debug(
            "Filtered {} candidates down to {} using metadata filters.",
            len(candidates),
            len(filtered),
        )
        return filtered
