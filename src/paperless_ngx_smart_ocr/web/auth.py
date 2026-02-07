"""Authentication utilities for per-user cookie-based auth."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request  # noqa: TC002
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)
from starlette.responses import Response

from paperless_ngx_smart_ocr.observability import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Coroutine
    from typing import Any

    from paperless_ngx_smart_ocr.config import Settings
    from paperless_ngx_smart_ocr.paperless import PaperlessClient
    from paperless_ngx_smart_ocr.pipeline.models import PipelineResult
    from paperless_ngx_smart_ocr.web.preview_store import PreviewStore


__all__ = [
    "AUTH_COOKIE_MAX_AGE",
    "AUTH_COOKIE_NAME",
    "AuthMiddleware",
    "get_user_client",
    "make_job_coroutine",
    "make_preview_job_coroutine",
    "validate_token",
]

AUTH_COOKIE_NAME = "smartocr_token"
AUTH_COOKIE_MAX_AGE = 28800  # 8 hours

# Paths that don't require authentication
_PUBLIC_PREFIXES = ("/login", "/logout", "/api/health", "/api/ready")
_PUBLIC_EXACT = frozenset(("/login", "/logout"))

logger = get_logger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """Check for auth cookie on every request.

    Exempt paths: ``/login``, ``/logout``, ``/api/health``,
    ``/api/ready``, and ``/static/*``.

    For missing cookies:
    - HTML / htmx requests get a redirect to ``/login``.
    - API requests get a 401 JSON response.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process the request, enforcing authentication."""
        path = request.url.path

        # Allow public paths through
        if self._is_public(path):
            return await call_next(request)

        token = request.cookies.get(AUTH_COOKIE_NAME)
        if not token:
            return self._unauthenticated_response(request)

        # Store token on request state for downstream use
        request.state.paperless_token = token
        return await call_next(request)

    @staticmethod
    def _is_public(path: str) -> bool:
        """Check whether a path is exempt from authentication."""
        if path.startswith("/static"):
            return True
        return any(
            path == prefix or path.startswith(prefix + "/")
            for prefix in _PUBLIC_PREFIXES
        )

    @staticmethod
    def _unauthenticated_response(request: Request) -> Response:
        """Return the appropriate unauthenticated response."""
        # htmx requests: send HX-Redirect header
        if request.headers.get("HX-Request") == "true":
            response = Response(status_code=200)
            response.headers["HX-Redirect"] = "/login"
            return response

        # API requests: return 401 JSON
        if request.url.path.startswith("/api/"):
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Authentication required",
                    "error_type": "AuthenticationRequired",
                },
            )

        # Browser HTML requests: redirect
        return RedirectResponse(url="/login", status_code=302)


async def get_user_client(
    request: Request,
) -> AsyncIterator[PaperlessClient]:
    """FastAPI dependency: per-request PaperlessClient using user token.

    Creates a client with the user's cookie token, yields it,
    then closes it after the response.

    Args:
        request: The incoming HTTP request.

    Yields:
        A PaperlessClient configured with the user's token.
    """
    from paperless_ngx_smart_ocr.paperless import (
        PaperlessClient,
    )

    base_url: str = request.app.state.settings.paperless.url
    token: str = request.state.paperless_token

    client = PaperlessClient(base_url=base_url, token=token)
    try:
        yield client
    finally:
        await client.close()


async def validate_token(base_url: str, token: str) -> bool:
    """Validate a paperless-ngx API token.

    Creates a temporary client and calls ``health_check()``.

    Args:
        base_url: The paperless-ngx base URL.
        token: The API token to validate.

    Returns:
        True if the token is valid and the server is reachable.
    """
    from paperless_ngx_smart_ocr.paperless import (
        PaperlessClient,
    )

    client = PaperlessClient(base_url=base_url, token=token)
    try:
        return await client.health_check()
    except Exception:  # noqa: BLE001
        logger.warning(
            "token_validation_failed",
            base_url=base_url,
        )
        return False
    finally:
        await client.close()


def make_job_coroutine(
    document_id: int,
    *,
    settings: Settings,
    base_url: str,
    token: str,
) -> Coroutine[Any, Any, PipelineResult]:
    """Create a coroutine that processes a document with its own client.

    The returned coroutine creates and closes a ``PaperlessClient``
    internally, so it is safe for use in background jobs that outlive
    the HTTP request.

    Args:
        document_id: The paperless-ngx document ID.
        settings: Application settings.
        base_url: Paperless-ngx base URL.
        token: User's API token.

    Returns:
        A coroutine that, when awaited, processes the document.
    """

    async def _run() -> PipelineResult:
        from paperless_ngx_smart_ocr.paperless import (
            PaperlessClient,
        )
        from paperless_ngx_smart_ocr.pipeline.orchestrator import (
            process_document,
        )

        async with PaperlessClient(
            base_url=base_url,
            token=token,
        ) as client:
            return await process_document(
                document_id,
                settings=settings,
                client=client,
                force=True,
            )

    return _run()


def make_preview_job_coroutine(
    document_id: int,
    *,
    settings: Settings,
    base_url: str,
    token: str,
    preview_store: PreviewStore,
) -> Coroutine[Any, Any, PipelineResult]:
    """Create a coroutine that dry-runs a document and stores preview.

    Like ``make_job_coroutine`` but runs in ``dry_run=True`` mode and
    uses ``before_cleanup`` to capture OCR'd PDF bytes and markdown
    into the preview store.

    Args:
        document_id: The paperless-ngx document ID.
        settings: Application settings.
        base_url: Paperless-ngx base URL.
        token: User's API token.
        preview_store: The preview store for caching results.

    Returns:
        A coroutine that, when awaited, dry-runs and stores preview.
    """

    async def _run() -> PipelineResult:
        from paperless_ngx_smart_ocr.paperless import (
            PaperlessClient,
        )
        from paperless_ngx_smart_ocr.pipeline.orchestrator import (
            process_document,
        )
        from paperless_ngx_smart_ocr.web.preview_store import (
            PreviewEntry,
        )

        async def _capture(result: PipelineResult) -> None:
            markdown = ""
            ocr_pdf_bytes: bytes | None = None

            if (
                result.stage2_result
                and result.stage2_result.success
                and result.stage2_result.markdown
            ):
                markdown = result.stage2_result.markdown

            if (
                result.stage1_result
                and result.stage1_result.success
                and result.stage1_result.output_path
                and result.stage1_result.output_path.exists()
            ):
                ocr_pdf_bytes = result.stage1_result.output_path.read_bytes()

            pid = preview_store.generate_id()
            entry = PreviewEntry(
                preview_id=pid,
                document_id=document_id,
                pipeline_result=result,
                markdown=markdown,
                ocr_pdf_bytes=ocr_pdf_bytes,
            )
            await preview_store.store(entry)

        async with PaperlessClient(
            base_url=base_url,
            token=token,
        ) as client:
            return await process_document(
                document_id,
                settings=settings,
                client=client,
                force=True,
                dry_run=True,
                before_cleanup=_capture,
            )

    return _run()
