"""Supervisor Agent module executing deterministic workflow routing decisions without LLM calls."""

from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


def should_retry_jd_parse(state: dict[str, Any]) -> str:
    """Determine whether to reparse the job description or continue down the pipeline.

    Routing Rules:
    - error in state -> "error_exit"
    - retry_count >= 2 -> "continue"
    - parsed_jd missing -> "reparse"
    - zero skills -> "reparse"
    - confidence < 0.5 -> "reparse"
    - otherwise -> "continue"

    Args:
        state: Workflow state dictionary.

    Returns:
        Routing decision string ("error_exit", "reparse", or "continue").
    """
    if state.get("error"):
        logger.info("Supervisor decision (retry_jd_parse): error present -> 'error_exit'")
        return "error_exit"

    retry_count = state.get("retry_count", 0)
    if retry_count >= 2:
        logger.info("Supervisor decision (retry_jd_parse): max retries reached ({}) -> 'continue'", retry_count)
        return "continue"

    parsed_jd = state.get("parsed_jd")
    if parsed_jd is None:
        logger.info("Supervisor decision (retry_jd_parse): parsed_jd missing -> 'reparse'")
        return "reparse"

    skills = getattr(parsed_jd, "skills", []) or []
    if len(skills) == 0:
        logger.info("Supervisor decision (retry_jd_parse): zero skills extracted -> 'reparse'")
        return "reparse"

    confidence = getattr(parsed_jd, "confidence_score", 1.0)
    if confidence < 0.5:
        logger.info("Supervisor decision (retry_jd_parse): low confidence ({}) -> 'reparse'", confidence)
        return "reparse"

    logger.info("Supervisor decision (retry_jd_parse): valid JD parse -> 'continue'")
    return "continue"


def should_verify_evidence(state: dict[str, Any]) -> str:
    """Determine whether to execute evidence verification on retrieved candidates.

    Routing Rules:
    - error in state -> "skip_verify"
    - no candidates -> "skip_verify"
    - > 50 candidates -> "skip_verify"
    - otherwise -> "verify"

    Args:
        state: Workflow state dictionary.

    Returns:
        Routing decision string ("skip_verify" or "verify").
    """
    if state.get("error"):
        logger.info("Supervisor decision (verify_evidence): error present -> 'skip_verify'")
        return "skip_verify"

    candidates = state.get("retrieved_candidates", []) or []
    if len(candidates) == 0:
        logger.info("Supervisor decision (verify_evidence): no retrieved candidates -> 'skip_verify'")
        return "skip_verify"

    if len(candidates) > 50:
        logger.info(
            "Supervisor decision (verify_evidence): candidate count ({}) > 50 -> 'skip_verify'",
            len(candidates),
        )
        return "skip_verify"

    logger.info(
        "Supervisor decision (verify_evidence): candidate count ({}) <= 50 -> 'verify'",
        len(candidates),
    )
    return "verify"


def get_routing_decision(state: dict[str, Any], node_name: str) -> str:
    """Dispatch routing decisions based on current workflow node.

    Args:
        state: Workflow state dictionary.
        node_name: Current workflow node identifier string.

    Returns:
        Routing decision string.
    """
    node_clean = node_name.strip().lower()
    if node_clean in ("jd_parser", "parse_jd", "parse", "jd_agent"):
        return should_retry_jd_parse(state)
    elif node_clean in ("retrieval", "retrieve", "verify_evidence", "evidence_agent", "retrieved_candidates"):
        return should_verify_evidence(state)

    logger.info("Supervisor decision default for node '{}' -> 'continue'", node_name)
    return "continue"


class SupervisorAgent:
    """Agent encapsulating deterministic workflow routing decisions for LangGraph nodes."""

    def should_retry_jd_parse(self, state: dict[str, Any]) -> str:
        """Method wrapper for should_retry_jd_parse function."""
        return should_retry_jd_parse(state)

    def should_verify_evidence(self, state: dict[str, Any]) -> str:
        """Method wrapper for should_verify_evidence function."""
        return should_verify_evidence(state)

    def get_routing_decision(self, state: dict[str, Any], node_name: str) -> str:
        """Method wrapper for get_routing_decision function."""
        return get_routing_decision(state, node_name)
