"""FastAPI application factory for paperless-ngx-smart-ocr."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)
from starlette.responses import Response  # noqa: TC002

from paperless_ngx_smart_ocr.observability import (
    clear_request_context,
    get_logger,
    set_request_id,
)
from paperless_ngx_smart_ocr.paperless.exceptions import (
    PaperlessAuthenticationError,
    PaperlessConnectionError,
    PaperlessError,
    PaperlessNotFoundError,
    PaperlessRateLimitError,
    PaperlessServerError,
    PaperlessValidationError,
)
from paperless_ngx_smart_ocr.workers.exceptions import (
    JobAlreadyCancelledError,
    JobNotFoundError,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from paperless_ngx_smart_ocr.config import Settings
    from paperless_ngx_smart_ocr.paperless import PaperlessClient
    from paperless_ngx_smart_ocr.web.preview_store import PreviewStore
    from paperless_ngx_smart_ocr.workers import JobQueue


__all__ = [
    "create_app",
    "get_app_settings",
    "get_job_queue",
    "get_preview_store",
    "get_service_client",
]


_WEB_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _WEB_DIR / "static"
_TEMPLATES_DIR = _WEB_DIR / "templates"


# -------------------------------------------------------------------
# Lifespan
# -------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifespan (startup/shutdown).

    On startup:
    - Creates and starts the JobQueue.
    - Creates the PaperlessClient (lazy-initialised on first call).

    On shutdown:
    - Stops the JobQueue (waits for running jobs).
    - Closes the PaperlessClient.

    Args:
        app: The FastAPI application instance.

    Yields:
        None after startup completes.
    """
    from paperless_ngx_smart_ocr.paperless import (
        PaperlessClient as _PaperlessClient,
    )
    from paperless_ngx_smart_ocr.web.preview_store import (
        PreviewStore as _PreviewStore,
    )
    from paperless_ngx_smart_ocr.workers import (
        JobQueue as _JobQueue,
    )

    logger = get_logger(__name__)
    settings: Settings = app.state.settings

    # --- Startup ---
    logger.info(
        "app_starting",
        host=settings.web.host,
        port=settings.web.port,
    )

    # Create and start job queue
    job_queue = _JobQueue(config=settings.jobs)
    await job_queue.start()
    app.state.job_queue = job_queue

    # Create service client for health/readiness probes
    service_client = _PaperlessClient(
        base_url=settings.paperless.url,
        token=settings.paperless.token or "",
    )
    app.state.service_client = service_client

    # Create preview store for dry-run result caching
    preview_store = _PreviewStore()
    app.state.preview_store = preview_store

    # Start periodic cleanup task for expired previews
    async def _cleanup_loop() -> None:
        while True:
            await asyncio.sleep(60)
            removed = await preview_store.cleanup_expired()
            if removed:
                logger.debug(
                    "preview_store_cleanup",
                    removed=removed,
                )

    cleanup_task = asyncio.create_task(_cleanup_loop())

    logger.info(
        "app_started",
        workers=settings.jobs.workers,
        paperless_url=settings.paperless.url,
    )

    yield

    # --- Shutdown ---
    logger.info("app_shutting_down")

    cleanup_task.cancel()
    await job_queue.stop()
    await service_client.close()

    logger.info("app_shutdown_complete")


# -------------------------------------------------------------------
# Middleware
# -------------------------------------------------------------------


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assign and propagate a unique request ID per request.

    Checks for an incoming ``X-Request-ID`` header. If present, uses
    it; otherwise generates a new one. The ID is bound to the
    structlog context and included in the response headers.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process the request with request ID tracking."""
        clear_request_context()

        incoming_id = request.headers.get("x-request-id")
        request_id = set_request_id(incoming_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def _configure_middleware(app: FastAPI) -> None:
    """Configure application middleware.

    Middleware is added in reverse order: the *last* call to
    ``add_middleware`` becomes the outermost layer. We want
    ``RequestIDMiddleware`` outermost so every downstream handler
    has a request ID available.

    Args:
        app: The FastAPI application.
    """
    from paperless_ngx_smart_ocr.web.auth import AuthMiddleware

    # Innermost layer - Auth (checks cookie, sets request.state)
    app.add_middleware(AuthMiddleware)

    # Middle layer - CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Outer layer - Request ID (added last, runs first)
    app.add_middleware(RequestIDMiddleware)


# -------------------------------------------------------------------
# Exception handlers
# -------------------------------------------------------------------


def _paperless_error_status(exc: PaperlessError) -> int:
    """Map a PaperlessError subclass to an HTTP status code.

    Args:
        exc: The PaperlessError exception.

    Returns:
        Appropriate HTTP status code.
    """
    if isinstance(exc, PaperlessNotFoundError):
        return 404
    if isinstance(exc, PaperlessValidationError):
        return 400
    if isinstance(exc, PaperlessAuthenticationError):
        return 401
    if isinstance(
        exc,
        PaperlessConnectionError | PaperlessServerError,
    ):
        return 502
    if isinstance(exc, PaperlessRateLimitError):
        return 503
    return 500


def _register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers.

    Args:
        app: The FastAPI application.
    """
    logger = get_logger(__name__)

    @app.exception_handler(PaperlessError)
    async def paperless_error_handler(
        _request: Request,
        exc: PaperlessError,
    ) -> JSONResponse:
        logger.warning(
            "paperless_error",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return JSONResponse(
            status_code=_paperless_error_status(exc),
            content={
                "detail": str(exc),
                "error_type": type(exc).__name__,
            },
        )

    @app.exception_handler(JobNotFoundError)
    async def job_not_found_handler(
        _request: Request,
        exc: JobNotFoundError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={
                "detail": str(exc),
                "error_type": "JobNotFoundError",
            },
        )

    @app.exception_handler(JobAlreadyCancelledError)
    async def job_already_cancelled_handler(
        _request: Request,
        exc: JobAlreadyCancelledError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "detail": str(exc),
                "error_type": "JobAlreadyCancelledError",
            },
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(
        _request: Request,
        exc: Exception,
    ) -> JSONResponse:
        logger.exception(
            "unhandled_error",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "error_type": type(exc).__name__,
            },
        )


# -------------------------------------------------------------------
# Static files & templates
# -------------------------------------------------------------------


def _mount_static_files(app: FastAPI) -> None:
    """Mount the static files directory.

    Args:
        app: The FastAPI application.
    """
    if _STATIC_DIR.is_dir():
        app.mount(
            "/static",
            StaticFiles(directory=str(_STATIC_DIR)),
            name="static",
        )


def _configure_templates(app: FastAPI) -> None:
    """Configure Jinja2 template rendering.

    Sets up template directory and injects global context variables
    (``version``, ``theme_mode``) so they are available in every
    template without being passed explicitly by each route.

    Args:
        app: The FastAPI application.
    """
    from paperless_ngx_smart_ocr import __version__

    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    settings: Settings = app.state.settings
    templates.env.globals["version"] = __version__
    templates.env.globals["theme_mode"] = settings.web.theme.value

    def _format_datetime(value: object, fmt: str = "%Y-%m-%d %H:%M") -> str:
        """Format a datetime for display in templates."""
        from datetime import datetime

        if isinstance(value, datetime):
            return value.strftime(fmt)
        return str(value)

    templates.env.filters["datefmt"] = _format_datetime
    app.state.templates = templates


# -------------------------------------------------------------------
# Router registration
# -------------------------------------------------------------------


def _include_routers(app: FastAPI) -> None:
    """Include API route routers.

    Args:
        app: The FastAPI application.
    """
    from paperless_ngx_smart_ocr.web.routes.auth import (
        router as auth_router,
    )
    from paperless_ngx_smart_ocr.web.routes.documents import (
        router as documents_router,
    )
    from paperless_ngx_smart_ocr.web.routes.health import (
        router as health_router,
    )
    from paperless_ngx_smart_ocr.web.routes.jobs import (
        router as jobs_router,
    )
    from paperless_ngx_smart_ocr.web.routes.views import (
        router as views_router,
    )

    app.include_router(auth_router)
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(jobs_router)
    app.include_router(views_router)


# -------------------------------------------------------------------
# Dependency helpers
# -------------------------------------------------------------------


def get_app_settings(request: Request) -> Settings:
    """FastAPI dependency: get application settings.

    Args:
        request: The incoming HTTP request.

    Returns:
        The application Settings instance.
    """
    return request.app.state.settings  # type: ignore[no-any-return]


def get_job_queue(request: Request) -> JobQueue:
    """FastAPI dependency: get the job queue.

    Args:
        request: The incoming HTTP request.

    Returns:
        The JobQueue instance.
    """
    return request.app.state.job_queue  # type: ignore[no-any-return]


def get_service_client(request: Request) -> PaperlessClient:
    """FastAPI dependency: get the service-level paperless client.

    This client uses the server-configured token and is intended
    only for health/readiness probes.

    Args:
        request: The incoming HTTP request.

    Returns:
        The service PaperlessClient instance.
    """
    return request.app.state.service_client  # type: ignore[no-any-return]


def get_preview_store(request: Request) -> PreviewStore:
    """FastAPI dependency: get the preview store.

    Args:
        request: The incoming HTTP request.

    Returns:
        The PreviewStore instance.
    """
    return request.app.state.preview_store  # type: ignore[no-any-return]


# -------------------------------------------------------------------
# Application factory
# -------------------------------------------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Application settings. If None, loads from default
            configuration sources via ``get_settings()``.

    Returns:
        Configured FastAPI application instance.
    """
    from paperless_ngx_smart_ocr import __version__
    from paperless_ngx_smart_ocr.config import get_settings

    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="paperless-ngx-smart-ocr",
        version=__version__,
        description=("Intelligent OCR and Markdown conversion for paperless-ngx"),
        lifespan=_lifespan,
    )

    app.state.settings = settings

    _configure_middleware(app)
    _register_exception_handlers(app)
    _mount_static_files(app)
    _configure_templates(app)
    _include_routers(app)

    return app
