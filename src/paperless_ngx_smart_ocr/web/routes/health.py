"""Health check and readiness endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse


if TYPE_CHECKING:
    from paperless_ngx_smart_ocr.paperless import PaperlessClient
    from paperless_ngx_smart_ocr.workers import JobQueue


__all__ = ["router"]

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
async def health_check(_request: Request) -> dict[str, Any]:
    """Liveness probe.

    Returns 200 if the service is running. Does not check
    external dependencies.

    Args:
        request: The incoming HTTP request.

    Returns:
        Dictionary with status and version information.
    """
    from paperless_ngx_smart_ocr import __version__

    return {
        "status": "ok",
        "version": __version__,
    }


@router.get("/ready")
async def readiness_check(request: Request) -> JSONResponse:
    """Readiness probe.

    Checks that the service can communicate with paperless-ngx
    and the job queue is running.

    Args:
        request: The incoming HTTP request.

    Returns:
        200 with status details if ready, 503 if not.
    """
    checks: dict[str, bool] = {}

    # Check job queue
    job_queue: JobQueue = request.app.state.job_queue
    checks["job_queue"] = job_queue.is_running

    # Check paperless-ngx connectivity
    client: PaperlessClient = request.app.state.service_client
    try:
        checks["paperless"] = await client.health_check()
    except Exception:  # noqa: BLE001
        checks["paperless"] = False

    all_ready = all(checks.values())
    status_code = 200 if all_ready else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if all_ready else "not_ready",
            "checks": checks,
        },
    )
