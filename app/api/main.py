"""FastAPI application factory for the Talent Intelligence Platform."""

import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.exceptions import TalentPlatformError
from app.core.logging import get_logger
from app.models.response import APIResponse

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler managing startup and shutdown events.

    Args:
        app: The FastAPI application instance.
    """
    logger.info("Talent Intelligence Platform API starting up.")
    try:
        from app.langgraph.graph import get_graph
        get_graph()
        logger.info("LangGraph pre-warmed successfully.")
    except Exception as e:
        logger.error("LangGraph pre-warm failed (non-fatal): {}", e)

    yield

    logger.info("Talent Intelligence Platform API shutting down.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Talent Intelligence Platform",
        description=(
            "AI-powered candidate ranking system using hybrid retrieval "
            "and LangGraph orchestration"
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── CORS for development ─────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ──────────────────────────────────────────────────────────────────
    app.include_router(router)

    # ── Exception Handlers ───────────────────────────────────────────────────────
    @app.exception_handler(TalentPlatformError)
    async def talent_platform_error_handler(
        request: Request, exc: TalentPlatformError
    ) -> JSONResponse:
        """Handle domain-specific TalentPlatformError exceptions.

        Args:
            request: Incoming HTTP request.
            exc: TalentPlatformError instance.

        Returns:
            JSONResponse with failure APIResponse payload.
        """
        logger.error("TalentPlatformError on {}: {}", request.url, exc.message)
        payload = APIResponse(
            success=False,
            message="Request processing failed.",
            error=exc.message,
        )
        return JSONResponse(status_code=500, content=payload.model_dump())

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle Pydantic RequestValidationError with readable detail.

        Args:
            request: Incoming HTTP request.
            exc: RequestValidationError instance.

        Returns:
            JSONResponse with HTTP 422 and error detail.
        """
        details = "; ".join(
            f"{' -> '.join(str(loc) for loc in err['loc'])}: {err['msg']}"
            for err in exc.errors()
        )
        logger.warning("Validation error on {}: {}", request.url, details)
        payload = APIResponse(
            success=False,
            message="Request validation failed.",
            error=details,
        )
        return JSONResponse(status_code=422, content=payload.model_dump())

    return app


app = create_app()


if __name__ == "__main__":
    from app.core.config import settings

    uvicorn.run(
        "app.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
