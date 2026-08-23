"""FastAPI route handlers for the Talent Intelligence Platform API."""

import time
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.core.exceptions import TalentPlatformError
from app.core.logging import get_logger
from app.langgraph.graph import get_graph
from app.langgraph.state import create_initial_state
from app.models.jd import JobDescriptionInput
from app.models.response import APIResponse, JDAnalysis, RankingResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["ranking"])


@router.post("/rank", response_model=APIResponse)
async def rank_candidates(payload: JobDescriptionInput) -> APIResponse:
    """Rank candidates against a provided Job Description using the full pipeline.

    Args:
        payload: JobDescriptionInput containing raw JD text.

    Returns:
        APIResponse containing RankingResponse data.

    Raises:
        HTTPException: 400 if JD text is empty or too short; 500 on pipeline failure.
    """
    raw_jd = (payload.raw_text or "").strip()

    if not raw_jd:
        raise HTTPException(status_code=400, detail="Job description text cannot be empty.")

    if len(raw_jd) < 50:
        raise HTTPException(
            status_code=400,
            detail=f"Job description must be at least 50 characters. Got {len(raw_jd)}.",
        )

    start_time = time.perf_counter()
    logger.info("POST /rank received: source='{}', jd_length={}", payload.source, len(raw_jd))

    try:
        graph = get_graph()
        initial_state = create_initial_state(raw_jd)
        final_state = graph.invoke(initial_state)
    except Exception as e:
        logger.error("Graph invocation failed: {}", e)
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {e}") from e

    if final_state.get("error"):
        logger.error("Graph pipeline returned error state: {}", final_state["error"])
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline processing error: {final_state['error']}",
        )

    parsed_jd = final_state.get("parsed_jd")
    if parsed_jd is None:
        raise HTTPException(
            status_code=422,
            detail="Failed to extract structured intent from the provided job description.",
        )

    ranked_candidates = final_state.get("ranked_candidates", []) or []
    expanded_queries = final_state.get("expanded_queries", []) or []

    jd_analysis = JDAnalysis(
        parsed_jd=parsed_jd,
        expanded_queries=expanded_queries,
        total_candidates_retrieved=len(final_state.get("retrieved_candidates", []) or []),
    )

    ranking_response = RankingResponse(
        job_title=parsed_jd.title,
        total_matching_candidates=len(ranked_candidates),
        ranked_candidates=ranked_candidates,
        jd_analysis=jd_analysis,
    )

    elapsed = time.perf_counter() - start_time
    logger.info(
        "POST /rank completed in {:.3f}s: title='{}', total_candidates={}",
        elapsed,
        parsed_jd.title,
        len(ranked_candidates),
    )

    return APIResponse(
        success=True,
        message=f"Successfully ranked {len(ranked_candidates)} candidates.",
        data=ranking_response.model_dump(),
    )


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Return service health status.

    Returns:
        Dictionary with status, service name, and version.
    """
    return {
        "status": "ok",
        "service": "Talent Intelligence Platform",
        "version": "1.0.0",
    }


@router.get("/graph-info")
async def graph_info() -> dict[str, Any]:
    """Return LangGraph node and edge information for debugging.

    Returns:
        Dictionary containing node names and edge details.
    """
    try:
        graph = get_graph()
        node_names = [n for n in graph.nodes.keys() if not n.startswith("__")]
        edges = []
        for source, targets in graph.edges.items():
            for target in targets:
                edges.append({"from": source, "to": target})

        return {
            "status": "ok",
            "nodes": node_names,
            "edges": edges,
        }
    except Exception as e:
        logger.error("Failed to retrieve graph info: {}", e)
        return {
            "status": "error",
            "error": str(e),
            "nodes": [],
            "edges": [],
        }
