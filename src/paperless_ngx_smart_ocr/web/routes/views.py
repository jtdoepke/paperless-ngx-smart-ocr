"""HTML view routes for the web UI."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Form, Path, Query, Request
from fastapi.responses import HTMLResponse

from paperless_ngx_smart_ocr.paperless.exceptions import (
    PaperlessNotFoundError,
)
from paperless_ngx_smart_ocr.workers.exceptions import (
    JobAlreadyCancelledError,
    JobNotFoundError,
)
from paperless_ngx_smart_ocr.workers.models import JobStatus


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi.templating import Jinja2Templates
    from starlette.responses import Response

    from paperless_ngx_smart_ocr.config import Settings
    from paperless_ngx_smart_ocr.paperless import PaperlessClient
    from paperless_ngx_smart_ocr.workers import JobQueue


__all__ = ["router"]

router = APIRouter(tags=["ui"])


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _get_templates(request: Request) -> Jinja2Templates:
    """Get the Jinja2Templates instance from app state.

    Args:
        request: The incoming HTTP request.

    Returns:
        The configured Jinja2Templates instance.
    """
    return request.app.state.templates  # type: ignore[no-any-return]


def _is_htmx(request: Request) -> bool:
    """Check whether the request was made by htmx."""
    return request.headers.get("HX-Request") == "true"


def _ctx(request: Request, **kwargs: object) -> dict[str, object]:
    """Build a template context dict with the request included."""
    return {"request": request, **kwargs}


def _resolve_name(
    resolved: dict[str, Any],
    key: str,
) -> str | None:
    """Extract a name from a resolved lookup dict.

    Returns None if the key is missing or the result was an exception.
    """
    obj = resolved.get(key)
    if obj is None or isinstance(obj, BaseException):
        return None
    name = getattr(obj, "name", None)
    return name if isinstance(name, str) else None


def _render(
    request: Request,
    full_template: str,
    partial_template: str | None,
    **kwargs: object,
) -> Response:
    """Render a full page or htmx partial depending on request type.

    Args:
        request: The incoming HTTP request.
        full_template: Template name for full-page navigation.
        partial_template: Template name for htmx requests. If None,
            always renders the full template.
        **kwargs: Additional template context variables.

    Returns:
        Rendered template response.
    """
    templates = _get_templates(request)
    ctx = _ctx(request, **kwargs)
    if partial_template and _is_htmx(request):
        return templates.TemplateResponse(partial_template, ctx)
    return templates.TemplateResponse(full_template, ctx)


# -------------------------------------------------------------------
# Home
# -------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> Response:
    """Render the dashboard home page.

    Args:
        request: The incoming HTTP request.

    Returns:
        Rendered index page.
    """
    client: PaperlessClient = request.app.state.client
    try:
        ready = await client.health_check()
    except Exception:  # noqa: BLE001
        ready = False

    templates = _get_templates(request)
    return templates.TemplateResponse(
        "index.html",
        _ctx(request, ready=ready),
    )


# -------------------------------------------------------------------
# Documents
# -------------------------------------------------------------------


def _parse_int_list(value: str | None) -> list[int] | None:
    """Parse a comma-separated string of IDs into a list.

    Args:
        value: Comma-separated integer IDs, or None.

    Returns:
        List of integer IDs, or None if input is None/empty.
    """
    if not value:
        return None
    return [int(t.strip()) for t in value.split(",") if t.strip()]


@router.get("/documents", response_class=HTMLResponse)
async def document_list(  # noqa: PLR0913
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
    query: str | None = Query(default=None),
    ordering: str | None = Query(default=None),
    tags_include: str | None = Query(default=None),
    tags_exclude: str | None = Query(default=None),
    correspondent: int | None = Query(default=None),
    document_type: int | None = Query(default=None),
    created_from: str | None = Query(default=None),
    created_to: str | None = Query(default=None),
    added_from: str | None = Query(default=None),
    added_to: str | None = Query(default=None),
) -> Response:
    """Render the document list page.

    Args:
        request: The incoming HTTP request.
        page: Page number (1-indexed).
        page_size: Number of results per page.
        query: Full-text search query.
        ordering: Field to order by (prefix with ``-`` for desc).
        tags_include: Comma-separated tag IDs to include.
        tags_exclude: Comma-separated tag IDs to exclude.
        correspondent: Filter by correspondent ID.
        document_type: Filter by document type ID.
        created_from: Filter created date from (YYYY-MM-DD).
        created_to: Filter created date to (YYYY-MM-DD).
        added_from: Filter added date from (YYYY-MM-DD).
        added_to: Filter added date to (YYYY-MM-DD).

    Returns:
        Full page or htmx partial with document table.
    """
    client: PaperlessClient = request.app.state.client

    result = await client.list_documents(
        page=page,
        page_size=page_size,
        query=query,
        ordering=ordering,
        tags_include=_parse_int_list(tags_include),
        tags_exclude=_parse_int_list(tags_exclude),
        correspondent=correspondent,
        document_type=document_type,
        created_from=created_from,
        created_to=created_to,
        added_from=added_from,
        added_to=added_to,
        truncate_content=True,
    )

    total_pages = max(1, -(-result.count // page_size))

    # Build current filter state for template
    filters = {
        "query": query or "",
        "tags_include": tags_include or "",
        "tags_exclude": tags_exclude or "",
        "correspondent": correspondent,
        "document_type": document_type,
        "created_from": created_from or "",
        "created_to": created_to or "",
        "added_from": added_from or "",
        "added_to": added_to or "",
        "ordering": ordering or "",
    }

    # For htmx partial requests, skip fetching filter options
    if _is_htmx(request):
        templates = _get_templates(request)
        return templates.TemplateResponse(
            "partials/document_table.html",
            _ctx(
                request,
                documents=result,
                page=page,
                page_size=page_size,
                total_pages=total_pages,
                filters=filters,
            ),
        )

    # Full page: fetch filter options for dropdowns
    tags_resp = await client.list_tags(page_size=500)
    correspondents_resp = await client.list_correspondents(
        page_size=500,
    )
    doc_types_resp = await client.list_document_types(
        page_size=500,
    )

    templates = _get_templates(request)
    return templates.TemplateResponse(
        "documents/list.html",
        _ctx(
            request,
            documents=result,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            filters=filters,
            tags=tags_resp.results,
            correspondents=correspondents_resp.results,
            document_types=doc_types_resp.results,
        ),
    )


@router.get(
    "/documents/{document_id}",
    response_class=HTMLResponse,
)
async def document_detail(
    request: Request,
    document_id: int = Path(...),
) -> Response:
    """Render the document detail page.

    Fetches the document, metadata, and resolves tag/correspondent/
    document-type/storage-path names in parallel.

    Args:
        request: The incoming HTTP request.
        document_id: The paperless-ngx document ID.

    Returns:
        Rendered document detail page.
    """
    client: PaperlessClient = request.app.state.client
    settings: Settings = request.app.state.settings

    try:
        doc, metadata = await asyncio.gather(
            client.get_document(document_id),
            client.get_document_metadata(document_id),
        )
    except PaperlessNotFoundError:
        templates = _get_templates(request)
        return templates.TemplateResponse(
            "partials/error.html",
            _ctx(
                request,
                error=f"Document {document_id} not found.",
            ),
            status_code=404,
        )

    # Resolve names in parallel (graceful on errors)
    lookups: dict[str, Any] = {}
    if doc.correspondent is not None:
        lookups["correspondent"] = client.get_correspondent(
            doc.correspondent,
        )
    if doc.document_type is not None:
        lookups["document_type"] = client.get_document_type(
            doc.document_type,
        )
    if doc.storage_path is not None:
        lookups["storage_path"] = client.get_storage_path(
            doc.storage_path,
        )
    for tag_id in doc.tags:
        lookups[f"tag_{tag_id}"] = client.get_tag(tag_id)

    resolved: dict[str, Any] = {}
    if lookups:
        keys = list(lookups.keys())
        results = await asyncio.gather(
            *lookups.values(),
            return_exceptions=True,
        )
        resolved = dict(zip(keys, results, strict=True))

    correspondent_name = _resolve_name(
        resolved,
        "correspondent",
    )
    document_type_name = _resolve_name(
        resolved,
        "document_type",
    )
    storage_path_name = _resolve_name(
        resolved,
        "storage_path",
    )

    tag_names = []
    for tag_id in doc.tags:
        obj = resolved.get(f"tag_{tag_id}")
        if obj is None or isinstance(obj, BaseException):
            tag_names.append(
                {"id": tag_id, "name": f"#{tag_id}"},
            )
        else:
            tag_names.append(
                {
                    "id": obj.id,
                    "name": obj.name,
                    "color": obj.color,
                    "text_color": obj.text_color,
                }
            )

    return _render(
        request,
        "documents/detail.html",
        None,
        document=doc,
        metadata=metadata,
        correspondent_name=correspondent_name,
        document_type_name=document_type_name,
        storage_path_name=storage_path_name,
        tag_names=tag_names,
        stage1_enabled=settings.pipeline.stage1.enabled,
        stage2_enabled=settings.pipeline.stage2.enabled,
    )


@router.get("/documents/{document_id}/pdf")
async def document_pdf_proxy(
    request: Request,
    document_id: int = Path(...),
) -> Response:
    """Proxy PDF download from paperless-ngx.

    Streams the document PDF so the browser's built-in viewer
    can display it in an iframe without needing a paperless-ngx
    auth token.

    Args:
        request: The incoming HTTP request.
        document_id: The paperless-ngx document ID.

    Returns:
        Streaming PDF response.
    """
    from starlette.responses import (
        StreamingResponse,
    )

    client: PaperlessClient = request.app.state.client

    async def _stream() -> AsyncIterator[bytes]:
        async with client.download_document(
            document_id,
            original=True,
        ) as resp:
            async for chunk in resp.aiter_bytes(
                chunk_size=65536,
            ):
                yield chunk

    return StreamingResponse(
        _stream(),
        media_type="application/pdf",
        headers={"Content-Disposition": "inline"},
    )


@router.post(
    "/documents/{document_id}/process",
    response_class=HTMLResponse,
)
async def process_document_view(
    request: Request,
    document_id: int = Path(...),
    force: bool = Form(default=False),  # noqa: FBT001
) -> Response:
    """Submit a document for background processing (UI).

    Returns an htmx partial with the new job card that auto-polls
    for status updates.

    Args:
        request: The incoming HTTP request.
        document_id: The paperless-ngx document ID.
        force: Force processing regardless of born-digital status.

    Returns:
        Job card partial.
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

    templates = _get_templates(request)
    return templates.TemplateResponse(
        "partials/job_card.html",
        _ctx(request, job=job.to_dict()),
    )


@router.post(
    "/documents/{document_id}/dry-run",
    response_class=HTMLResponse,
)
async def dry_run_view(
    request: Request,
    document_id: int = Path(...),
    force: bool = Form(default=False),  # noqa: FBT001
) -> Response:
    """Run a dry-run preview of document processing (UI).

    Processes synchronously and returns the result as an htmx partial.

    Args:
        request: The incoming HTTP request.
        document_id: The paperless-ngx document ID.
        force: Force processing regardless of born-digital status.

    Returns:
        Process result partial.
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

    templates = _get_templates(request)
    return templates.TemplateResponse(
        "partials/process_result.html",
        _ctx(request, result=result.to_dict()),
    )


# -------------------------------------------------------------------
# Jobs
# -------------------------------------------------------------------


@router.get("/jobs", response_class=HTMLResponse)
async def job_list(
    request: Request,
    status: JobStatus | None = Query(default=None),  # noqa: B008
    document_id: int | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
) -> Response:
    """Render the job list page.

    Args:
        request: The incoming HTTP request.
        status: Filter by job status.
        document_id: Filter by associated document ID.
        limit: Maximum number of jobs to return.

    Returns:
        Full page or htmx partial with job list.
    """
    job_queue: JobQueue = request.app.state.job_queue
    jobs = await job_queue.list_jobs(
        status=status,
        document_id=document_id,
        limit=limit,
    )

    return _render(
        request,
        "jobs/list.html",
        "partials/job_list.html",
        jobs=[j.to_dict() for j in jobs],
        current_status=status.value if status else "",
        statuses=[s.value for s in JobStatus],
    )


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_detail(
    request: Request,
    job_id: str = Path(...),
) -> Response:
    """Render the job detail page.

    Args:
        request: The incoming HTTP request.
        job_id: The job ID.

    Returns:
        Full page or htmx partial with job card.
    """
    job_queue: JobQueue = request.app.state.job_queue
    try:
        job = await job_queue.get(job_id)
    except JobNotFoundError:
        templates = _get_templates(request)
        return templates.TemplateResponse(
            "partials/error.html",
            _ctx(request, error=f"Job {job_id} not found."),
            status_code=404,
        )

    return _render(
        request,
        "jobs/detail.html",
        "partials/job_card.html",
        job=job.to_dict(),
    )


@router.post("/jobs/{job_id}/cancel", response_class=HTMLResponse)
async def cancel_job_view(
    request: Request,
    job_id: str = Path(...),
) -> Response:
    """Cancel a running or pending job (UI).

    Returns the updated job card partial.

    Args:
        request: The incoming HTTP request.
        job_id: The job ID to cancel.

    Returns:
        Updated job card partial.
    """
    job_queue: JobQueue = request.app.state.job_queue
    templates = _get_templates(request)

    try:
        job = await job_queue.cancel(job_id)
    except JobNotFoundError:
        return templates.TemplateResponse(
            "partials/error.html",
            _ctx(request, error=f"Job {job_id} not found."),
            status_code=404,
        )
    except JobAlreadyCancelledError:
        return templates.TemplateResponse(
            "partials/error.html",
            _ctx(
                request,
                error=f"Job {job_id} is already cancelled.",
            ),
            status_code=409,
        )

    return templates.TemplateResponse(
        "partials/job_card.html",
        _ctx(request, job=job.to_dict()),
    )
