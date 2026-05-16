"""
Agentic Research Assistant — FastAPI entry point.

Bootstraps logging, CORS, the /health liveness probe, and mounts the
/research router that drives both the baseline and LangGraph agent paths.
"""

import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.schemas import HealthResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure logging and emit startup / shutdown lifecycle log lines.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control to the running application.
    """
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(module)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("Starting Agentic Research Assistant v0.1.0 (env=%s)", settings.app_env)
    logger.info("Settings loaded successfully")
    yield
    logger.info("Shutting down Agentic Research Assistant")


app = FastAPI(title="Agentic Research Assistant", version="0.1.0", lifespan=lifespan)

# TODO: add Vercel production URL to allow_origins once frontend deploys (Day 26)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    # Matches any *.vercel.app origin (PR previews + production before custom domain is set)
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler: logs the full traceback, returns a safe generic 500.

    Args:
        request: The incoming HTTP request that triggered the exception.
        exc: The unhandled exception instance.

    Returns:
        A 500 JSONResponse with a generic error message — no internal details
        are leaked to the caller.
    """
    logger.error(
        "Unhandled exception for %s %s:\n%s",
        request.method,
        request.url,
        traceback.format_exc(),
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe — returns app version and current deployment environment.

    Returns:
        HealthResponse with status 'ok', the app version, and the active env.
    """
    settings = get_settings()
    return HealthResponse(status="ok", version="0.1.0", env=settings.app_env)


from app.api.research import router as research_router  # noqa: E402

app.include_router(research_router)
