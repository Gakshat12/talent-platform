"""Scratch tests for LangGraph nodes, edges, and graph topology."""

import pytest
from unittest.mock import MagicMock, patch

from app.langgraph.edges import should_reparse_jd, should_verify_evidence
from app.langgraph.graph import get_graph
from app.langgraph.nodes import (
    expand_query_node,
    generate_explanations_node,
    parse_jd_node,
    rank_candidates_node,
    retrieve_candidates_node,
    verify_evidence_node,
)
from app.langgraph.state import create_initial_state, TalentGraphState
from app.models.jd import ParsedJobDescription, SkillRequirement, ExperienceRequirement


@pytest.fixture
def parsed_jd():
    return ParsedJobDescription(
        title="Senior Python Engineer",
        skills=[SkillRequirement(name="Python", is_required=True)],
        experience=ExperienceRequirement(min_years=5.0, seniority_level="senior"),
        confidence_score=0.9,
    )


def test_graph_compiles_and_is_singleton():
    g1 = get_graph()
    g2 = get_graph()
    assert g1 is g2, "get_graph() should return the same singleton instance."


def test_graph_nodes_registered():
    g = get_graph()
    nodes = list(g.nodes.keys())
    for expected in ["parse_jd", "expand_query", "retrieve", "verify_evidence", "rank", "explain"]:
        assert expected in nodes, f"Expected node '{expected}' missing from graph."


def test_parse_jd_node_empty_input():
    state = create_initial_state("")
    result = parse_jd_node(state)
    assert result["retry_count"] == 1
    # Empty JD should fail cleanly
    assert "parsed_jd" in result


def test_parse_jd_node_exception_handling():
    state = create_initial_state("Some JD text")
    with patch("app.langgraph.nodes.JDUnderstandingAgent") as mock_agent_cls:
        mock_agent_cls.return_value.run.side_effect = RuntimeError("LLM down")
        result = parse_jd_node(state)
    assert result["error"] is not None
    assert result["retry_count"] == 1
    assert result["parsed_jd"] is None


def test_expand_query_node_with_no_parsed_jd():
    state = create_initial_state("JD")
    state["parsed_jd"] = None
    result = expand_query_node(state)
    assert result["expanded_queries"] == []


def test_expand_query_node_success(parsed_jd):
    state = create_initial_state("JD")
    state["parsed_jd"] = parsed_jd
    with patch("app.langgraph.nodes.QueryExpander") as mock_expander_cls:
        mock_expander_cls.return_value.expand.return_value = ["query1", "query2"]
        result = expand_query_node(state)
    assert result["expanded_queries"] == ["query1", "query2"]


def test_retrieve_candidates_node_no_jd():
    state = create_initial_state("JD")
    state["parsed_jd"] = None
    result = retrieve_candidates_node(state)
    assert result["retrieved_candidates"] == []
    assert result["error"] is not None


def test_retrieve_candidates_node_empty_candidate_pool(parsed_jd):
    state = create_initial_state("JD")
    state["parsed_jd"] = parsed_jd
    with patch("app.langgraph.nodes._load_candidates_from_disk", return_value=[]):
        result = retrieve_candidates_node(state)
    assert result["retrieved_candidates"] == []
    assert result["error"] is None


def test_edge_should_reparse_jd_paths(parsed_jd):
    # error state
    state_err = {"error": "some error", "parsed_jd": parsed_jd, "retry_count": 0}
    assert should_reparse_jd(state_err) == "error_exit"

    # max retries
    state_max = {"error": None, "parsed_jd": None, "retry_count": 2}
    assert should_reparse_jd(state_max) == "continue"

    # missing parsed_jd
    state_missing = {"error": None, "parsed_jd": None, "retry_count": 0}
    assert should_reparse_jd(state_missing) == "reparse"

    # valid parsed_jd
    state_ok = {"error": None, "parsed_jd": parsed_jd, "retry_count": 0}
    assert should_reparse_jd(state_ok) == "continue"


def test_edge_should_verify_evidence():
    # error
    assert should_verify_evidence({"error": "bad", "retrieved_candidates": [1]}) == "skip_verify"
    # empty
    assert should_verify_evidence({"error": None, "retrieved_candidates": []}) == "skip_verify"
    # too many
    assert should_verify_evidence({"error": None, "retrieved_candidates": list(range(60))}) == "skip_verify"
    # 3 candidates
    assert should_verify_evidence({"error": None, "retrieved_candidates": [1, 2, 3]}) == "verify"
