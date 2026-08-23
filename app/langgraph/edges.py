"""LangGraph conditional edge routing functions for the Talent Intelligence pipeline."""

from typing import Any

from app.agents.supervisor import SupervisorAgent
from app.core.logging import get_logger
from app.langgraph.state import TalentGraphState

logger = get_logger(__name__)

_supervisor = SupervisorAgent()


def should_reparse_jd(state: TalentGraphState) -> str:
    """Determine if the JD parse output requires a retry or can proceed.

    Uses SupervisorAgent.should_retry_jd_parse() routing logic.

    Routing Outcomes:
    - "reparse": re-run parse_jd node
    - "continue": proceed to expand_query node
    - "error_exit": terminate at END

    Args:
        state: Current TalentGraphState.

    Returns:
        Routing decision string.
    """
    decision = _supervisor.should_retry_jd_parse(dict(state))
    logger.info("Edge 'should_reparse_jd' -> '{}'", decision)
    return decision


def should_verify_evidence(state: TalentGraphState) -> str:
    """Determine whether evidence verification should be performed on retrieved candidates.

    Uses SupervisorAgent.should_verify_evidence() routing logic.

    Routing Outcomes:
    - "verify": run verify_evidence node
    - "skip_verify": skip directly to rank node

    Args:
        state: Current TalentGraphState.

    Returns:
        Routing decision string.
    """
    decision = _supervisor.should_verify_evidence(dict(state))
    logger.info("Edge 'should_verify_evidence' -> '{}'", decision)
    return decision
