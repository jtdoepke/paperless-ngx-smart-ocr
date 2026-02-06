"""Document listing, detail, processing, and dry-run endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse


if TYPE_CHECKING:
    from paperless_ngx_smart_ocr.config import Settings
    from paperless_ngx_smart_ocr.paperless import PaperlessClient
    from paperless_ngx_smart_ocr.workers import JobQueue


__all__ = ["router"]

router = APIRouter(prefix="/api", tags=["documents"])


def _parse_tag_ids(value: str | None) -> list[int] | None:
    """Parse a comma-separated string of tag IDs into a list.

    Args:
        value: Comma-separated integer IDs, or None.

    Returns:
        List of integer tag IDs, or None if input is None.

    Raises:
        ValueError: If any value is not a valid integer.
    """
    if value is None:
        return None
    return [int(t.strip()) for t in value.split(",") if t.strip()]


@router.get("/documents")
async def list_documents(  # noqa: PLR0913
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
    query: str | None = Query(default=None),
    ordering: str | None = Query(default=None),
    tags_include: str | None = Query(default=None),
    tags_exclude: str | None = Query(default=None),
) -> dict[str, Any]:
    """List documents from paperless-ngx with optional filtering.

    Args:
        request: The incoming HTTP request.
        page: Page number (1-indexed).
        page_size: Number of results per page.
        query: Full-text search query.
        ordering: Field to order by (prefix with '-' for descending).
        tags_include: Comma-separated tag IDs to include.
        tags_exclude: Comma-separated tag IDs to exclude.

    Returns:
        Paginated document list.
    """
    client: PaperlessClient = request.app.state.client
    result = await client.list_documents(
        page=page,
        page_size=page_size,
        query=query,
        ordering=ordering,
        tags_include=_parse_tag_ids(tags_include),
        tags_exclude=_parse_tag_ids(tags_exclude),
        truncate_content=True,
    )
    return result.model_dump(mode="json")


@router.get("/documents/{document_id}")
async def get_document(
    request: Request,
    document_id: int,
) -> dict[str, Any]:
    """Get a single document from paperless-ngx.

    Args:
        request: The incoming HTTP request.
        document_id: The paperless-ngx document ID.

    Returns:
        Document detail.
    """
    client: PaperlessClient = request.app.state.client
    doc = await client.get_document(document_id)
    return doc.model_dump(mode="json")


@router.post("/documents/{document_id}/process")
async def process_document_endpoint(
    request: Request,
    document_id: int,
    force: bool = Query(default=False),  # noqa: FBT001
) -> JSONResponse:
    """Submit a document for background processing.

    Queues the document for asynchronous processing through the
    pipeline (Stage 1 OCR + Stage 2 Markdown). Returns immediately
    with job details; poll ``GET /api/jobs/{id}`` for status.

    Args:
        request: The incoming HTTP request.
        document_id: The paperless-ngx document ID.
        force: Force processing regardless of born-digital status.

    Returns:
        202 Accepted with job details.
    """
    from paperless_ngx_smart_ocr.pipeline.orchestrator import (
        process_document,
    )

    settings: Settings = request.app.state.settings
    client: PaperlessClient = request.app.state.client
    job_queue: JobQueue = request.app.state.job_queue

    coro = process_document(
        document_id,
        settings=settings,
        client=client,
        force=force,
    )
    job = await job_queue.submit(
        coro,
        name=f"Process document {document_id}",
        document_id=document_id,
    )
    return JSONResponse(status_code=202, content=job.to_dict())


@router.post("/documents/{document_id}/dry-run")
async def dry_run_document(
    request: Request,
    document_id: int,
    force: bool = Query(default=False),  # noqa: FBT001
) -> dict[str, Any]:
    """Run a dry-run preview of document processing.

    Processes the document synchronously through the pipeline but
    skips content updates and tag changes in paperless-ngx. Use
    this to preview what processing would produce.

    Args:
        request: The incoming HTTP request.
        document_id: The paperless-ngx document ID.
        force: Force processing regardless of born-digital status.

    Returns:
        Pipeline result with processing details.
    """
    from paperless_ngx_smart_ocr.pipeline.orchestrator import (
        PipelineOrchestrator,
    )

    settings: Settings = request.app.state.settings
    client: PaperlessClient = request.app.state.client

    orchestrator = PipelineOrchestrator(
        settings=settings,
        client=client,
    )
    result = await orchestrator.process(
        document_id,
        force=force,
        dry_run=True,
    )
    return result.to_dict()
