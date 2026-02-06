"""Job listing, status, and cancellation endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Query, Request

from paperless_ngx_smart_ocr.workers.models import JobStatus  # noqa: TC001


if TYPE_CHECKING:
    from paperless_ngx_smart_ocr.workers import JobQueue


__all__ = ["router"]

router = APIRouter(prefix="/api", tags=["jobs"])


@router.get("/jobs")
async def list_jobs(
    request: Request,
    status: JobStatus | None = Query(default=None),  # noqa: B008
    document_id: int | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[dict[str, Any]]:
    """List background jobs with optional filtering.

    Args:
        request: The incoming HTTP request.
        status: Filter by job status (pending, running, completed,
            failed, cancelled).
        document_id: Filter by associated document ID.
        limit: Maximum number of jobs to return.

    Returns:
        List of job details, newest first.
    """
    job_queue: JobQueue = request.app.state.job_queue
    jobs = await job_queue.list_jobs(
        status=status,
        document_id=document_id,
        limit=limit,
    )
    return [job.to_dict() for job in jobs]


@router.get("/jobs/{job_id}")
async def get_job(
    request: Request,
    job_id: str,
) -> dict[str, Any]:
    """Get the status of a single job.

    Args:
        request: The incoming HTTP request.
        job_id: The job ID.

    Returns:
        Job details.
    """
    job_queue: JobQueue = request.app.state.job_queue
    job = await job_queue.get(job_id)
    return job.to_dict()


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    request: Request,
    job_id: str,
) -> dict[str, Any]:
    """Cancel a running or pending job.

    Args:
        request: The incoming HTTP request.
        job_id: The job ID to cancel.

    Returns:
        Updated job details.
    """
    job_queue: JobQueue = request.app.state.job_queue
    job = await job_queue.cancel(job_id)
    return job.to_dict()
