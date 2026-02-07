"""HTML view routes for the web UI."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, Form, Path, Query, Request
from fastapi.responses import HTMLResponse

from paperless_ngx_smart_ocr.paperless.exceptions import (
    PaperlessNotFoundError,
)
from paperless_ngx_smart_ocr.web.auth import (
    get_user_client,
    make_preview_job_coroutine,
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
    from paperless_ngx_smart_ocr.pipeline.models import PipelineResult
    from paperless_ngx_smart_ocr.web.preview_store import (
        BulkPreviewBatch,
        PreviewStore,
    )
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


def _check_archive_dir(
    settings: Settings,
) -> dict[str, object]:
    """Check if the archive directory is configured and readable.

    Returns:
        Dict with ``configured``, ``ok``, and ``message`` keys.
    """
    archive_dir = settings.paperless.archive_dir
    if archive_dir is None:
        return {
            "configured": False,
            "ok": False,
            "message": "Not configured",
        }
    try:
        if not archive_dir.exists():
            return {
                "configured": True,
                "ok": False,
                "message": f"Directory not found: {archive_dir}",
            }
        if not archive_dir.is_dir():
            return {
                "configured": True,
                "ok": False,
                "message": f"Not a directory: {archive_dir}",
            }
        # Check readability by listing contents
        next(archive_dir.iterdir(), None)
    except PermissionError:
        return {
            "configured": True,
            "ok": False,
            "message": f"Permission denied: {archive_dir}",
        }
    except OSError as exc:
        return {
            "configured": True,
            "ok": False,
            "message": str(exc),
        }
    else:
        return {
            "configured": True,
            "ok": True,
            "message": "Readable",
        }


async def _check_database(
    settings: Settings,
) -> dict[str, object]:
    """Check if the PostgreSQL database is configured and reachable.

    Returns:
        Dict with ``configured``, ``ok``, and ``message`` keys.
    """
    database_url = settings.paperless.database_url
    if database_url is None:
        return {
            "configured": False,
            "ok": False,
            "message": "Not configured",
        }
    try:
        import asyncpg

        conn = await asyncpg.connect(database_url)
        try:
            await conn.execute("SELECT 1")
        finally:
            await conn.close()
    except Exception as exc:  # noqa: BLE001
        return {
            "configured": True,
            "ok": False,
            "message": str(exc),
        }
    else:
        return {
            "configured": True,
            "ok": True,
            "message": "Connected",
        }


@router.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    client: PaperlessClient = Depends(get_user_client),  # noqa: B008
) -> Response:
    """Render the dashboard home page.

    Args:
        request: The incoming HTTP request.
        client: Per-request PaperlessClient from user cookie.

    Returns:
        Rendered index page.
    """
    try:
        ready = await client.health_check()
    except Exception:  # noqa: BLE001
        ready = False

    settings: Settings = request.app.state.settings
    archive_status = _check_archive_dir(settings)
    db_status = await _check_database(settings)

    templates = _get_templates(request)
    return templates.TemplateResponse(
        "index.html",
        _ctx(
            request,
            ready=ready,
            archive_status=archive_status,
            db_status=db_status,
        ),
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
    client: PaperlessClient = Depends(get_user_client),  # noqa: B008
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
        client: Per-request PaperlessClient from user cookie.
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
    client: PaperlessClient = Depends(get_user_client),  # noqa: B008
) -> Response:
    """Render the document detail page.

    Fetches the document, metadata, and resolves tag/correspondent/
    document-type/storage-path names in parallel.

    Args:
        request: The incoming HTTP request.
        document_id: The paperless-ngx document ID.
        client: Per-request PaperlessClient from user cookie.

    Returns:
        Rendered document detail page.
    """
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
        llm_enabled=settings.pipeline.stage2.marker.use_llm,
    )


@router.get("/documents/{document_id}/pdf")
async def document_pdf_proxy(
    _request: Request,
    document_id: int = Path(...),
    client: PaperlessClient = Depends(get_user_client),  # noqa: B008
) -> Response:
    """Proxy PDF download from paperless-ngx.

    Streams the document PDF so the browser's built-in viewer
    can display it in an iframe without needing a paperless-ngx
    auth token.

    Args:
        _request: The incoming HTTP request (unused).
        document_id: The paperless-ngx document ID.
        client: Per-request PaperlessClient from user cookie.

    Returns:
        Streaming PDF response.
    """
    from starlette.responses import (
        StreamingResponse,
    )

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


# -------------------------------------------------------------------
# Preview / Apply flow
# -------------------------------------------------------------------


def _render_markdown_to_html(markdown: str) -> str:
    """Render markdown to HTML for the preview modal.

    Uses a simple conversion; falls back to escaped pre block on
    error.

    Args:
        markdown: Raw markdown text.

    Returns:
        HTML string.
    """
    try:
        import markdown as md

        result: str = md.markdown(
            markdown,
            extensions=["tables", "fenced_code"],
        )
    except Exception:  # noqa: BLE001
        import html

        return f"<pre>{html.escape(markdown)}</pre>"
    else:
        return result


async def _build_preview_entry(
    document_id: int,
    result: PipelineResult,
    *,
    preview_store: PreviewStore,
) -> tuple[str, str, bytes | None]:
    """Read temp files and store a preview entry.

    Must be called from the ``before_cleanup`` callback while temp
    files still exist.

    Args:
        document_id: The document ID.
        result: Pipeline result (temp files still accessible).
        preview_store: The preview store instance.

    Returns:
        Tuple of (preview_id, markdown, ocr_pdf_bytes).
    """
    from paperless_ngx_smart_ocr.web.preview_store import (
        PreviewEntry,
    )

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

    preview_id = preview_store.generate_id()
    entry = PreviewEntry(
        preview_id=preview_id,
        document_id=document_id,
        pipeline_result=result,
        markdown=markdown,
        ocr_pdf_bytes=ocr_pdf_bytes,
    )
    await preview_store.store(entry)
    return preview_id, markdown, ocr_pdf_bytes


@router.post(
    "/documents/{document_id}/preview",
    response_class=HTMLResponse,
)
async def preview_document_view(
    request: Request,
    document_id: int = Path(...),
    use_llm: bool = Form(default=False),  # noqa: FBT001
    client: PaperlessClient = Depends(get_user_client),  # noqa: B008
) -> Response:
    """Run a dry-run and show results in the preview modal.

    Processes the document synchronously through the pipeline in
    dry-run mode, stores the result in the preview store, and
    returns the preview modal HTML.

    Args:
        request: The incoming HTTP request.
        document_id: The paperless-ngx document ID.
        use_llm: Whether to enable LLM for Stage 2 Markdown.
        client: Per-request PaperlessClient from user cookie.

    Returns:
        Preview modal partial.
    """
    from paperless_ngx_smart_ocr.pipeline.orchestrator import (
        PipelineOrchestrator,
    )

    settings: Settings = request.app.state.settings
    preview_store: PreviewStore = request.app.state.preview_store

    # Override use_llm if it differs from config
    if use_llm != settings.pipeline.stage2.marker.use_llm:
        marker = settings.pipeline.stage2.marker.model_copy(
            update={"use_llm": use_llm},
        )
        stage2 = settings.pipeline.stage2.model_copy(
            update={"marker": marker},
        )
        pipeline = settings.pipeline.model_copy(
            update={"stage2": stage2},
        )
        settings = settings.model_copy(update={"pipeline": pipeline})

    # Run dry-run with callback to capture temp files
    captured: dict[str, object] = {}

    async def _capture(result: PipelineResult) -> None:
        pid, md, pdf_bytes = await _build_preview_entry(
            document_id,
            result,
            preview_store=preview_store,
        )
        captured["preview_id"] = pid
        captured["markdown"] = md
        captured["ocr_pdf_bytes"] = pdf_bytes

    orchestrator = PipelineOrchestrator(
        settings=settings,
        client=client,
    )
    result = await orchestrator.process(
        document_id,
        force=True,
        dry_run=True,
        before_cleanup=_capture,
    )

    preview_id: str = captured.get("preview_id", "")  # type: ignore[assignment]
    markdown: str = captured.get("markdown", "")  # type: ignore[assignment]
    ocr_pdf_bytes: bytes | None = captured.get("ocr_pdf_bytes")  # type: ignore[assignment]

    # Get document title for the modal header
    try:
        doc = await client.get_document(document_id)
        document_title = doc.title
    except Exception:  # noqa: BLE001
        document_title = f"Document {document_id}"

    markdown_html = _render_markdown_to_html(markdown) if markdown else ""

    templates = _get_templates(request)
    return templates.TemplateResponse(
        "partials/preview_modal.html",
        _ctx(
            request,
            document_id=document_id,
            document_title=document_title,
            preview_id=preview_id,
            result=result,
            markdown=markdown,
            markdown_html=markdown_html,
            has_ocr_pdf=ocr_pdf_bytes is not None,
            archive_dir_configured=(settings.paperless.pdf_replacement_enabled),
        ),
    )


@router.post(
    "/documents/{document_id}/apply/{preview_id}",
    response_class=HTMLResponse,
)
async def apply_preview_view(
    request: Request,
    document_id: int = Path(...),
    preview_id: str = Path(...),
    replace_pdf: bool = Form(default=False),  # noqa: FBT001
    replace_content: bool = Form(default=False),  # noqa: FBT001
    client: PaperlessClient = Depends(get_user_client),  # noqa: B008
) -> Response:
    """Apply preview results to the document.

    Commits the selected actions (PDF replacement, content update,
    tag update) from a stored preview entry.

    Args:
        request: The incoming HTTP request.
        document_id: The paperless-ngx document ID.
        preview_id: The preview store entry ID.
        replace_pdf: Whether to replace the archive PDF.
        replace_content: Whether to replace document content.
        client: Per-request PaperlessClient from user cookie.

    Returns:
        Apply result partial.
    """
    from paperless_ngx_smart_ocr.paperless.models import (
        DocumentUpdate,
    )

    settings: Settings = request.app.state.settings
    preview_store: PreviewStore = request.app.state.preview_store
    templates = _get_templates(request)

    entry = await preview_store.get(preview_id)
    if entry is None or entry.document_id != document_id:
        return templates.TemplateResponse(
            "partials/apply_result.html",
            _ctx(
                request,
                success=False,
                error="Preview expired or not found. Please try again.",
                pdf_replaced=False,
                content_replaced=False,
                tags_updated=False,
            ),
        )

    pdf_replaced = False
    content_replaced = False
    tags_updated = False
    error: str | None = None

    try:
        # Replace archive PDF via shared filesystem
        if (
            replace_pdf
            and entry.ocr_pdf_bytes
            and settings.paperless.pdf_replacement_enabled
        ):
            archive_filename = await client.get_archive_filename(document_id)
            if archive_filename:
                from paperless_ngx_smart_ocr.web.archive import (
                    replace_archive_pdf,
                )

                await replace_archive_pdf(
                    archive_dir=settings.paperless.archive_dir,  # type: ignore[arg-type]
                    archive_media_filename=archive_filename,
                    pdf_bytes=entry.ocr_pdf_bytes,
                    database_url=settings.paperless.database_url,  # type: ignore[arg-type]
                    document_id=document_id,
                )
                pdf_replaced = True

        # Replace content via API
        if replace_content and entry.markdown:
            await client.update_document(
                document_id,
                DocumentUpdate(content=entry.markdown),
            )
            content_replaced = True

        # Clean up
        await preview_store.remove(preview_id)

    except Exception as exc:  # noqa: BLE001
        error = str(exc)

    return templates.TemplateResponse(
        "partials/apply_result.html",
        _ctx(
            request,
            success=error is None,
            error=error,
            pdf_replaced=pdf_replaced,
            content_replaced=content_replaced,
            tags_updated=tags_updated,
        ),
    )


@router.get("/documents/{document_id}/preview-pdf/{preview_id}")
async def preview_pdf_proxy(
    _request: Request,
    document_id: int = Path(...),
    preview_id: str = Path(...),
) -> Response:
    """Stream the OCR'd PDF from the preview store.

    Falls back to a 404 if the preview entry doesn't exist or has
    no OCR'd PDF bytes.

    Args:
        _request: The incoming HTTP request.
        document_id: The paperless-ngx document ID.
        preview_id: The preview store entry ID.

    Returns:
        PDF response or 404.
    """
    from starlette.responses import Response as StarletteResponse

    preview_store: PreviewStore = _request.app.state.preview_store
    entry = await preview_store.get(preview_id)

    if entry is None or entry.document_id != document_id or entry.ocr_pdf_bytes is None:
        return StarletteResponse(
            content=b"Preview not found",
            status_code=404,
            media_type="text/plain",
        )

    return StarletteResponse(
        content=entry.ocr_pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "inline"},
    )


# -------------------------------------------------------------------
# Bulk Preview / Review / Apply
# -------------------------------------------------------------------


@router.post(
    "/documents/bulk-preview",
    response_class=HTMLResponse,
)
async def bulk_preview_view(
    request: Request,
    document_ids: str = Form(default=""),
    filter_query: str = Form(default=""),
    client: PaperlessClient = Depends(get_user_client),  # noqa: B008
) -> Response:
    """Submit dry-run jobs for bulk preview review.

    Creates a batch in the preview store, submits dry-run jobs
    for each document, and returns the review table modal.

    Args:
        request: The incoming HTTP request.
        document_ids: Comma-separated document IDs.
        filter_query: URL filter query string for "all matching".
        client: Per-request PaperlessClient from user cookie.

    Returns:
        Bulk review table modal partial.
    """
    from paperless_ngx_smart_ocr.web.preview_store import (
        BulkPreviewBatch,
    )

    settings: Settings = request.app.state.settings
    job_queue: JobQueue = request.app.state.job_queue
    preview_store: PreviewStore = request.app.state.preview_store

    doc_ids = [int(x.strip()) for x in document_ids.split(",") if x.strip()]
    if not doc_ids and filter_query:
        doc_ids = await _collect_filtered_ids(
            client,
            filter_query,
        )

    batch_id = preview_store.generate_id()
    job_ids: dict[int, str] = {}

    for doc_id in doc_ids:
        coro = make_preview_job_coroutine(
            doc_id,
            settings=settings,
            base_url=settings.paperless.url,
            token=request.state.paperless_token,
            preview_store=preview_store,
        )
        job = await job_queue.submit(
            coro,
            name=f"Preview document {doc_id}",
            document_id=doc_id,
        )
        job_ids[doc_id] = job.id

    batch = BulkPreviewBatch(
        batch_id=batch_id,
        document_ids=doc_ids,
        job_ids=job_ids,
    )
    await preview_store.store_batch(batch)

    templates = _get_templates(request)
    return templates.TemplateResponse(
        "partials/bulk_review_table.html",
        _ctx(
            request,
            batch_id=batch_id,
            document_ids=doc_ids,
            total=len(doc_ids),
            # Initial rows all show "processing" status
            **{f"status_{did}": "pending" for did in doc_ids},
        ),
    )


async def _collect_batch_rows(
    batch: BulkPreviewBatch,
    job_queue: JobQueue,
    preview_store: PreviewStore,
) -> tuple[list[dict[str, object]], bool]:
    """Check job statuses for a batch and build row context dicts.

    Returns:
        Tuple of (rows_context, all_done).
    """
    all_done = True
    rows: list[dict[str, object]] = []

    for doc_id in batch.document_ids:
        job_id = batch.job_ids.get(doc_id)
        status = "pending"
        preview_id: str | None = None

        if job_id:
            try:
                job = await job_queue.get(job_id)
                if job.status == JobStatus.COMPLETED:
                    status = "completed"
                    preview_id = batch.preview_ids.get(doc_id)
                    if not preview_id:
                        preview_id = await _find_preview_for_doc(preview_store, doc_id)
                        if preview_id:
                            batch.preview_ids[doc_id] = preview_id
                elif job.status == JobStatus.FAILED:
                    status = "failed"
                else:
                    all_done = False
            except Exception:  # noqa: BLE001
                status = "failed"

        rows.append(
            {
                "doc_id": doc_id,
                "status": status,
                "preview_id": preview_id,
            }
        )

    return rows, all_done


@router.get(
    "/documents/bulk-review/{batch_id}",
    response_class=HTMLResponse,
)
async def bulk_review_poll(
    request: Request,
    batch_id: str = Path(...),
) -> Response:
    """Poll batch status and return updated review rows.

    Called by htmx every 3s to update the review table with
    completed/failed dry-run results.

    Args:
        request: The incoming HTTP request.
        batch_id: The batch identifier.

    Returns:
        Updated table body rows.
    """
    preview_store: PreviewStore = request.app.state.preview_store
    job_queue: JobQueue = request.app.state.job_queue
    templates = _get_templates(request)

    batch = await preview_store.get_batch(batch_id)
    if batch is None:
        return templates.TemplateResponse(
            "partials/error.html",
            _ctx(
                request,
                error="Batch expired or not found.",
            ),
        )

    # Check job statuses and collect preview IDs
    rows_context, all_done = await _collect_batch_rows(batch, job_queue, preview_store)

    # Build response - render table body rows
    html_parts = []
    for row in rows_context:
        resp = templates.TemplateResponse(
            "partials/bulk_review_row.html",
            _ctx(request, **row),
        )
        html_parts.append(bytes(resp.body).decode())

    # If all done, stop polling by not including hx-trigger
    body = "".join(html_parts)
    if all_done:
        # Wrap in table body that replaces the polling div
        body = (
            f'<table class="w-full text-sm">'
            f'<thead class="sticky top-0 bg-gray-50'
            f' dark:bg-gray-700"><tr>'
            f'<th class="px-4 py-2 text-left text-gray-500'
            f' dark:text-gray-400 font-medium">Include</th>'
            f'<th class="px-4 py-2 text-left text-gray-500'
            f' dark:text-gray-400 font-medium">ID</th>'
            f'<th class="px-4 py-2 text-left text-gray-500'
            f' dark:text-gray-400 font-medium">Status</th>'
            f'<th class="px-4 py-2 text-left text-gray-500'
            f' dark:text-gray-400 font-medium">Preview</th>'
            f"</tr></thead><tbody>{body}</tbody></table>"
        )

    return HTMLResponse(content=body)


async def _find_preview_for_doc(
    preview_store: PreviewStore,
    document_id: int,
) -> str | None:
    """Find a preview entry for a document in the store.

    Scans the preview store's entries for one matching the
    document ID.

    Args:
        preview_store: The preview store.
        document_id: The document ID to search for.

    Returns:
        Preview ID if found, None otherwise.
    """
    # Access internal entries under lock
    async with preview_store._lock:  # noqa: SLF001
        for pid, entry in preview_store._entries.items():  # noqa: SLF001
            if entry.document_id == document_id:
                return pid
    return None


@router.post(
    "/documents/bulk-apply/{batch_id}",
    response_class=HTMLResponse,
)
async def bulk_apply_view(
    request: Request,
    batch_id: str = Path(...),
    client: PaperlessClient = Depends(get_user_client),  # noqa: B008
) -> Response:
    """Apply all successful previews in a batch.

    Iterates over the batch's preview entries and applies content
    updates for each included document.

    Args:
        request: The incoming HTTP request.
        batch_id: The batch identifier.
        client: Per-request PaperlessClient from user cookie.

    Returns:
        Apply result partial.
    """
    from paperless_ngx_smart_ocr.paperless.models import (
        DocumentUpdate,
    )

    settings: Settings = request.app.state.settings
    preview_store: PreviewStore = request.app.state.preview_store
    templates = _get_templates(request)

    batch = await preview_store.get_batch(batch_id)
    if batch is None:
        return templates.TemplateResponse(
            "partials/apply_result.html",
            _ctx(
                request,
                success=False,
                error="Batch expired or not found.",
                pdf_replaced=False,
                content_replaced=False,
                tags_updated=False,
            ),
        )

    applied_count = 0
    errors: list[str] = []

    for doc_id in batch.document_ids:
        if doc_id in batch.excluded_ids:
            continue

        pid = batch.preview_ids.get(doc_id)
        if not pid:
            continue

        entry = await preview_store.get(pid)
        if entry is None or not entry.pipeline_result.success:
            continue

        try:
            # Replace archive PDF if configured
            if entry.ocr_pdf_bytes and settings.paperless.pdf_replacement_enabled:
                archive_filename = await client.get_archive_filename(doc_id)
                if archive_filename:
                    from paperless_ngx_smart_ocr.web.archive import (
                        replace_archive_pdf,
                    )

                    await replace_archive_pdf(
                        archive_dir=settings.paperless.archive_dir,  # type: ignore[arg-type]
                        archive_media_filename=archive_filename,
                        pdf_bytes=entry.ocr_pdf_bytes,
                        database_url=settings.paperless.database_url,  # type: ignore[arg-type]
                        document_id=doc_id,
                    )

            # Replace content
            if entry.markdown:
                await client.update_document(
                    doc_id,
                    DocumentUpdate(content=entry.markdown),
                )

            applied_count += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Doc {doc_id}: {exc}")

    # Clean up batch
    await preview_store.remove_batch(batch_id)

    error_msg = "; ".join(errors) if errors else None

    return templates.TemplateResponse(
        "partials/apply_result.html",
        _ctx(
            request,
            success=not errors,
            error=error_msg,
            pdf_replaced=applied_count > 0,
            content_replaced=applied_count > 0,
            tags_updated=False,
        ),
    )


async def _collect_filtered_ids(
    client: PaperlessClient,
    filter_query: str,
    *,
    max_docs: int = 500,
) -> list[int]:
    """Re-query paperless-ngx with filters to collect document IDs.

    Args:
        client: The paperless-ngx API client.
        filter_query: URL-encoded filter query string
            (e.g. ``&query=invoice&ordering=-created``).
        max_docs: Maximum number of document IDs to collect.

    Returns:
        List of matching document IDs (capped at *max_docs*).
    """
    from urllib.parse import parse_qs

    params = parse_qs(filter_query.lstrip("&?"))

    def _first(key: str) -> str | None:
        vals = params.get(key, [])
        return vals[0] if vals else None

    query = _first("query")
    ordering = _first("ordering")
    tags_include = _parse_int_list(_first("tags_include"))
    tags_exclude = _parse_int_list(_first("tags_exclude"))
    correspondent_str = _first("correspondent")
    correspondent = int(correspondent_str) if correspondent_str else None
    doc_type_str = _first("document_type")
    document_type = int(doc_type_str) if doc_type_str else None
    created_from = _first("created_from")
    created_to = _first("created_to")
    added_from = _first("added_from")
    added_to = _first("added_to")

    all_ids: list[int] = []
    pg = 1
    batch = 100
    while len(all_ids) < max_docs:
        result = await client.list_documents(
            page=pg,
            page_size=batch,
            query=query,
            ordering=ordering,
            tags_include=tags_include,
            tags_exclude=tags_exclude,
            correspondent=correspondent,
            document_type=document_type,
            created_from=created_from,
            created_to=created_to,
            added_from=added_from,
            added_to=added_to,
            truncate_content=True,
        )
        all_ids.extend(d.id for d in result.results)
        if pg * batch >= result.count:
            break
        pg += 1

    return all_ids[:max_docs]


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
