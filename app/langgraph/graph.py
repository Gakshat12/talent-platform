"""LangGraph graph assembly module for the Talent Intelligence workflow."""

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.core.logging import get_logger
from app.langgraph.edges import should_reparse_jd, should_verify_evidence
from app.langgraph.nodes import (
    expand_query_node,
    generate_explanations_node,
    parse_jd_node,
    rank_candidates_node,
    retrieve_candidates_node,
    verify_evidence_node,
)
from app.langgraph.state import TalentGraphState

logger = get_logger(__name__)

_graph: Any = None


def get_graph() -> Any:
    """Return the compiled Talent Intelligence LangGraph, creating it once as a lazy singleton.

    Returns:
        Compiled LangGraph instance ready for invocation.
    """
    global _graph

    if _graph is not None:
        return _graph

    builder = StateGraph(TalentGraphState)

    # ── Register nodes ──────────────────────────────────────────────────────────
    builder.add_node("parse_jd", parse_jd_node)
    builder.add_node("expand_query", expand_query_node)
    builder.add_node("retrieve", retrieve_candidates_node)
    builder.add_node("verify_evidence", verify_evidence_node)
    builder.add_node("rank", rank_candidates_node)
    builder.add_node("explain", generate_explanations_node)

    # ── Entry edge ───────────────────────────────────────────────────────────────
    builder.add_edge(START, "parse_jd")

    # ── Conditional: parse_jd -> reparse | continue | error_exit ────────────────
    builder.add_conditional_edges(
        "parse_jd",
        should_reparse_jd,
        {
            "reparse": "parse_jd",
            "continue": "expand_query",
            "error_exit": END,
        },
    )

    # ── Linear: expand_query -> retrieve ────────────────────────────────────────
    builder.add_edge("expand_query", "retrieve")

    # ── Conditional: retrieve -> verify_evidence | rank ─────────────────────────
    builder.add_conditional_edges(
        "retrieve",
        should_verify_evidence,
        {
            "verify": "verify_evidence",
            "skip_verify": "rank",
        },
    )

    # ── Linear: verify_evidence -> rank -> explain -> END ───────────────────────
    builder.add_edge("verify_evidence", "rank")
    builder.add_edge("rank", "explain")
    builder.add_edge("explain", END)

    _graph = builder.compile()

    logger.info("Talent Intelligence Graph compiled")

    return _graph
