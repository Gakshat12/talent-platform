"""State definition module for the LangGraph talent intelligence workflow."""

from typing import Dict, List, Optional, TypedDict
from app.models.candidate import CandidateProfile
from app.models.jd import ParsedJobDescription
from app.models.response import CandidateEvidence, CandidateRankResult


class TalentGraphState(TypedDict, total=False):
    """TypedDict defining state maintained across nodes in the LangGraph workflow pipeline."""

    raw_jd: str
    parsed_jd: Optional[ParsedJobDescription]
    expanded_queries: List[str]
    retrieved_candidates: List[CandidateProfile]
    verified_evidence: Dict[str, List[CandidateEvidence]]
    ranked_candidates: List[CandidateRankResult]
    error: Optional[str]
    next_step: str
    retry_count: int


def create_initial_state(raw_jd: str) -> TalentGraphState:
    """Creates initial state dictionary with safe default values for starting workflow execution.

    Args:
        raw_jd: The raw text string of the job description to be processed.

    Returns:
        TalentGraphState: Initialized state payload ready for the graph pipeline.
    """
    return {
        "raw_jd": raw_jd or "",
        "parsed_jd": None,
        "expanded_queries": [],
        "retrieved_candidates": [],
        "verified_evidence": {},
        "ranked_candidates": [],
        "error": None,
        "next_step": "parse_jd",
        "retry_count": 0,
    }
