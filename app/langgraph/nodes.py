"""LangGraph node functions for the Talent Intelligence pipeline workflow."""

from __future__ import annotations

import json
from pathlib import Path

from app.agents.evidence_agent import EvidenceVerificationAgent
from app.agents.explainability_agent import ExplainabilityAgent
from app.agents.jd_agent import JDUnderstandingAgent
from app.agents.supervisor import SupervisorAgent
from app.core.config import settings
from app.core.logging import get_logger
from app.langgraph.state import TalentGraphState
from app.models.candidate import CandidateProfile
from app.parser.query_expansion import QueryExpander
from app.ranking.business_ranker import BusinessRanker
from app.retrieval.hybrid_retrieval import HybridRetriever

logger = get_logger(__name__)


def _load_candidates_from_disk() -> list[CandidateProfile]:
    """Load all candidate profiles from the configured JSONL dataset.

    Returns:
        Valid CandidateProfile objects. Invalid or unreadable records are skipped
        so one malformed record does not stop the ranking pipeline.
    """
    data_path = Path(settings.candidates_data_path)

    if not data_path.exists():
        logger.warning(
            "Candidate data file not found at '{}'. No candidates loaded.",
            data_path,
        )
        return []

    candidates: list[CandidateProfile] = []

    try:
        with data_path.open("r", encoding="utf-8") as file:
            for line_num, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    data = json.loads(line)
                    candidate = CandidateProfile.model_validate(data)
                    candidates.append(candidate)
                except Exception as exc:
                    logger.warning(
                        "Skipping invalid candidate record at line {}: {}",
                        line_num,
                        exc,
                    )
    except OSError as exc:
        logger.error(
            "Unable to read candidate data file '{}': {}",
            data_path,
            exc,
        )
        return []

    logger.info(
        "Loaded {} candidate profiles from '{}'.",
        len(candidates),
        data_path,
    )

    return candidates


def parse_jd_node(state: TalentGraphState) -> TalentGraphState:
    """Parse the raw JD into structured requirements using the JD agent.

    Args:
        state: Current LangGraph state.

    Returns:
        Updated state containing parsed_jd, retry_count, and any parsing error.
    """
    logger.info(
        "Node 'parse_jd' entered. retry_count={}.",
        state.get("retry_count", 0),
    )

    retry_count = state.get("retry_count", 0) + 1

    try:
        agent = JDUnderstandingAgent()
        raw_jd = state.get("raw_jd", "")

        result = agent.run(raw_jd)

        if result.get("status") == "error":
            return {
                **state,
                "parsed_jd": None,
                "retry_count": retry_count,
                "error": result.get("error", "JD parsing failed."),
            }

        return {
            **state,
            "parsed_jd": result.get("parsed_jd"),
            "retry_count": retry_count,
            "error": None,
        }

    except Exception as exc:
        logger.error(
            "Unhandled exception in parse_jd_node: {}",
            exc,
        )

        return {
            **state,
            "parsed_jd": None,
            "retry_count": retry_count,
            "error": str(exc),
        }


def expand_query_node(state: TalentGraphState) -> TalentGraphState:
    """Generate alternative retrieval queries from the parsed JD.

    Args:
        state: Current LangGraph state.

    Returns:
        Updated state containing expanded_queries.
    """
    logger.info("Node 'expand_query' entered.")

    try:
        parsed_jd = state.get("parsed_jd")

        if parsed_jd is None:
            logger.warning(
                "expand_query_node: parsed_jd is None; skipping expansion."
            )
            return {
                **state,
                "expanded_queries": [],
            }

        expander = QueryExpander()
        queries = expander.expand(parsed_jd)

        clean_queries = [
            query.strip()
            for query in (queries or [])
            if isinstance(query, str) and query.strip()
        ]

        logger.info(
            "expand_query_node: generated {} expanded queries.",
            len(clean_queries),
        )

        return {
            **state,
            "expanded_queries": clean_queries,
            "error": None,
        }

    except Exception as exc:
        logger.error(
            "Unhandled exception in expand_query_node: {}",
            exc,
        )

        return {
            **state,
            "expanded_queries": [],
            "error": str(exc),
        }


def retrieve_candidates_node(state: TalentGraphState) -> TalentGraphState:
    """Load candidates and run cached hybrid retrieval using expanded JD queries.

    Args:
        state: Current LangGraph state.

    Returns:
        Updated state containing retrieved_candidates.
    """
    logger.info("Node 'retrieve' entered.")

    try:
        parsed_jd = state.get("parsed_jd")

        if parsed_jd is None:
            logger.warning(
                "retrieve_candidates_node: parsed_jd is None; cannot retrieve."
            )
            return {
                **state,
                "retrieved_candidates": [],
                "error": "parsed_jd is missing for retrieval.",
            }

        candidates = _load_candidates_from_disk()

        if not candidates:
            logger.warning(
                "retrieve_candidates_node: candidate pool is empty."
            )
            return {
                **state,
                "retrieved_candidates": [],
                "error": None,
            }

        retriever = HybridRetriever(candidates)

        # The query expansion node generates the alternatives. Pass them into
        # this request-scoped retriever so BM25 can actually use them.
        expanded_queries = state.get("expanded_queries", []) or []
        retriever.set_expanded_queries(expanded_queries)

        ranked_results = retriever.retrieve(
            parsed_jd,
            top_k=settings.final_top_k,
        )

        retrieved = [
            candidate
            for candidate, _ in ranked_results
        ]

        logger.info(
            "retrieve_candidates_node: retrieved {} candidates.",
            len(retrieved),
        )

        return {
            **state,
            "retrieved_candidates": retrieved,
            "error": None,
        }

    except Exception as exc:
        logger.error(
            "Unhandled exception in retrieve_candidates_node: {}",
            exc,
        )

        return {
            **state,
            "retrieved_candidates": [],
            "error": str(exc),
        }


def verify_evidence_node(state: TalentGraphState) -> TalentGraphState:
    """Verify retrieved candidate skill claims against their career evidence.

    Args:
        state: Current LangGraph state.

    Returns:
        Updated state containing verified_evidence keyed by candidate ID.
    """
    logger.info("Node 'verify_evidence' entered.")

    try:
        parsed_jd = state.get("parsed_jd")
        candidates = state.get("retrieved_candidates", []) or []

        if not candidates or parsed_jd is None:
            logger.warning(
                "verify_evidence_node: no candidates or missing JD. Skipping."
            )
            return {
                **state,
                "verified_evidence": {},
            }

        agent = EvidenceVerificationAgent()
        verified_evidence: dict = {}

        for candidate in candidates:
            evidences = agent.verify_candidate_evidence(
                candidate,
                parsed_jd,
            )

            verified_evidence[candidate.candidate_id] = evidences

        logger.info(
            "verify_evidence_node: verified evidence for {} candidates.",
            len(verified_evidence),
        )

        return {
            **state,
            "verified_evidence": verified_evidence,
            "error": None,
        }

    except Exception as exc:
        logger.error(
            "Unhandled exception in verify_evidence_node: {}",
            exc,
        )

        return {
            **state,
            "verified_evidence": {},
            "error": str(exc),
        }


def rank_candidates_node(state: TalentGraphState) -> TalentGraphState:
    """Rank retrieved candidates using deterministic business scoring.

    Args:
        state: Current LangGraph state.

    Returns:
        Updated state containing ranked_candidates.
    """
    logger.info("Node 'rank' entered.")

    try:
        parsed_jd = state.get("parsed_jd")
        candidates = state.get("retrieved_candidates", []) or []

        if not candidates or parsed_jd is None:
            logger.warning(
                "rank_candidates_node: no candidates or missing JD."
            )
            return {
                **state,
                "ranked_candidates": [],
            }

        ranker = BusinessRanker()

        ranked = ranker.rank_candidates(
            candidates,
            parsed_jd,
        )

        verified_evidence = state.get(
            "verified_evidence",
            {},
        ) or {}

        for rank_result in ranked:
            cand_evidences = verified_evidence.get(
                rank_result.candidate_id,
                [],
            )

            rank_result.evidences = cand_evidences

        logger.info(
            "rank_candidates_node: ranked {} candidates.",
            len(ranked),
        )

        return {
            **state,
            "ranked_candidates": ranked,
            "error": None,
        }

    except Exception as exc:
        logger.error(
            "Unhandled exception in rank_candidates_node: {}",
            exc,
        )

        return {
            **state,
            "ranked_candidates": [],
            "error": str(exc),
        }


def generate_explanations_node(state: TalentGraphState) -> TalentGraphState:
    """Generate deterministic explanations for all ranked candidates and optionally enhance the top five with an LLM.

    Args:
        state: Current LangGraph state.

    Returns:
        Updated LangGraph state with explanations populated for ranked candidates.
    """
    logger.info("Node 'explain' entered.")

    try:
        parsed_jd = state.get("parsed_jd")
        ranked_candidates = state.get("ranked_candidates", []) or []
        retrieved_candidates = state.get("retrieved_candidates", []) or []
        verified_evidence = state.get("verified_evidence", {}) or {}

        if not ranked_candidates or parsed_jd is None:
            logger.warning(
                "generate_explanations_node: no ranked candidates or missing JD."
            )
            return state

        agent = ExplainabilityAgent()

        candidate_lookup = {
            candidate.candidate_id: candidate
            for candidate in retrieved_candidates
            if candidate.candidate_id
        }

        # ------------------------------------------------------------------
        # Step 1: Generate deterministic explanations for every ranked
        # candidate. This guarantees that explanation generation never blocks
        # the ranking pipeline when the LLM is unavailable.
        # ------------------------------------------------------------------
        for rank_result in ranked_candidates:
            candidate = candidate_lookup.get(rank_result.candidate_id)

            if candidate is None:
                logger.warning(
                    "generate_explanations_node: candidate '{}' not found "
                    "in retrieved pool.",
                    rank_result.candidate_id,
                )
                continue

            evidences = verified_evidence.get(
                rank_result.candidate_id,
                rank_result.evidences or [],
            )

            verified_skills = [
                evidence.skill_name
                for evidence in evidences
                if getattr(
                    evidence,
                    "verified",
                    getattr(evidence, "is_verified", False),
                )
            ]

            missing_skills = [
                evidence.skill_name
                for evidence in evidences
                if not getattr(
                    evidence,
                    "verified",
                    getattr(evidence, "is_verified", False),
                )
            ]

            if not missing_skills:
                missing_skills = [
                    skill
                    for skill in parsed_jd.required_skill_names
                    if skill not in verified_skills
                ]

            rank_result.explanation = agent._deterministic_explanation(
                candidate=candidate,
                score_breakdown=rank_result.score_breakdown,
                verified_skills=verified_skills,
                missing_skills=missing_skills,
            )

        logger.info(
            "Generated deterministic explanations for {} ranked candidates.",
            len(ranked_candidates),
        )

        # ------------------------------------------------------------------
        # Step 2: Enhance only the top five candidates through the LLM.
        # If the LLM fails, deterministic explanations remain untouched.
        # ------------------------------------------------------------------
        top_candidates = ranked_candidates[:5]

        explanation_candidates = [
            candidate_lookup[result.candidate_id]
            for result in top_candidates
            if result.candidate_id in candidate_lookup
        ]

        score_breakdowns = {
            result.candidate_id: result.score_breakdown
            for result in top_candidates
            if result.candidate_id
            and result.score_breakdown is not None
        }

        evidences_by_candidate = {
            result.candidate_id: verified_evidence.get(
                result.candidate_id,
                result.evidences or [],
            )
            for result in top_candidates
            if result.candidate_id
        }

        if explanation_candidates:
            try:
                llm_explanations = agent.generate_explanations_batch(
                    candidates=explanation_candidates,
                    parsed_jd=parsed_jd,
                    score_breakdowns=score_breakdowns,
                    evidences_by_candidate=evidences_by_candidate,
                )

                enhanced_count = 0

                for rank_result in top_candidates:
                    explanation = llm_explanations.get(
                        rank_result.candidate_id
                    )

                    if explanation:
                        rank_result.explanation = explanation
                        enhanced_count += 1

                logger.info(
                    "LLM-enhanced explanations for {} of top {} candidates.",
                    enhanced_count,
                    len(explanation_candidates),
                )

            except Exception as exc:
                logger.warning(
                    "Top-five LLM explanation enhancement failed: {}. "
                    "Keeping deterministic explanations.",
                    exc,
                )

        return {
            **state,
            "ranked_candidates": ranked_candidates,
            "error": None,
        }

    except Exception as exc:
        logger.error(
            "Unhandled exception in generate_explanations_node: {}",
            exc,
        )

        return {
            **state,
            "error": str(exc),
        }