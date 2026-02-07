"""Unit tests for HTML view routes."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from paperless_ngx_smart_ocr.config import Settings
from paperless_ngx_smart_ocr.paperless.exceptions import (
    PaperlessNotFoundError,
)
from paperless_ngx_smart_ocr.paperless.models import (
    Correspondent,
    Document,
    DocumentMetadata,
    DocumentType,
    PaginatedResponse,
    Tag,
)
from paperless_ngx_smart_ocr.web import create_app
from paperless_ngx_smart_ocr.web.auth import AUTH_COOKIE_NAME, get_user_client
from paperless_ngx_smart_ocr.workers.exceptions import (
    JobAlreadyCancelledError,
    JobNotFoundError,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Generator

    from fastapi import FastAPI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)


def _make_settings(**overrides: object) -> Settings:
    """Create a Settings instance with sensible defaults for testing."""
    return Settings(
        paperless={
            "url": "http://localhost:8000",
            "token": "test-token",
        },
        **overrides,  # type: ignore[arg-type]
    )


def _make_mock_queue() -> MagicMock:
    """Create a mock JobQueue with sensible defaults."""
    queue = MagicMock()
    queue.is_running = True
    queue.start = AsyncMock()
    queue.stop = AsyncMock()
    queue.submit = AsyncMock()
    queue.list_jobs = AsyncMock(return_value=[])
    queue.get = AsyncMock()
    queue.cancel = AsyncMock()
    return queue


def _make_mock_client() -> MagicMock:
    """Create a mock PaperlessClient with sensible defaults."""
    client = MagicMock()
    client.health_check = AsyncMock(return_value=True)
    client.close = AsyncMock()
    client.list_documents = AsyncMock()
    client.get_document = AsyncMock()
    client.get_document_metadata = AsyncMock(
        return_value=DocumentMetadata(
            original_checksum="abc123",
            original_size=12345,
            original_mime_type="application/pdf",
            has_archive_version=False,
            original_metadata=[],
            archive_metadata=[],
        ),
    )
    client.get_tag = AsyncMock(
        side_effect=lambda tag_id: Tag(
            id=tag_id,
            slug=f"tag-{tag_id}",
            name=f"Tag {tag_id}",
        ),
    )
    client.get_correspondent = AsyncMock(
        return_value=Correspondent(
            id=1,
            slug="acme",
            name="ACME Corp",
        ),
    )
    client.get_document_type = AsyncMock(
        return_value=DocumentType(
            id=1,
            slug="invoice",
            name="Invoice",
        ),
    )
    client.get_storage_path = AsyncMock(
        return_value=MagicMock(
            id=1,
            name="Archive",
        ),
    )
    client.list_tags = AsyncMock(
        return_value=PaginatedResponse[Tag](count=0, results=[]),
    )
    client.list_correspondents = AsyncMock(
        return_value=PaginatedResponse[Correspondent](
            count=0,
            results=[],
        ),
    )
    client.list_document_types = AsyncMock(
        return_value=PaginatedResponse[DocumentType](
            count=0,
            results=[],
        ),
    )
    return client


def _make_document(
    document_id: int = 1,
    **overrides: object,
) -> Document:
    """Create a Document instance for testing."""
    defaults = {
        "id": document_id,
        "title": f"Test Document {document_id}",
        "content": "Sample content",
        "tags": [1, 2],
        "created": _NOW,
        "created_date": "2024-01-15",
        "modified": _NOW,
        "added": _NOW,
        "correspondent": 1,
        "document_type": 1,
        "archive_serial_number": 42,
    }
    defaults.update(overrides)
    return Document(**defaults)  # type: ignore[arg-type]


def _make_paginated_response(
    documents: list[Document] | None = None,
) -> PaginatedResponse[Document]:
    """Create a PaginatedResponse for testing."""
    docs = [_make_document()] if documents is None else documents
    return PaginatedResponse[Document](
        count=len(docs),
        results=docs,
    )


def _make_mock_job(
    job_id: str = "abc123def456",
    document_id: int = 1,
    status: str = "pending",
) -> MagicMock:
    """Create a mock Job with to_dict() support."""
    job = MagicMock()
    job.id = job_id
    job.to_dict.return_value = {
        "id": job_id,
        "name": f"Process document {document_id}",
        "status": status,
        "document_id": document_id,
        "created_at": _NOW.isoformat(),
        "started_at": None,
        "completed_at": None,
        "duration_seconds": None,
        "result": None,
        "metadata": {},
    }
    return job


def _mock_pdf_download(
    content: bytes,
) -> object:
    """Create a mock async context manager for PDF download."""

    @asynccontextmanager
    async def _download(*_args: object, **_kw: object) -> AsyncIterator[MagicMock]:
        mock_resp = MagicMock()

        async def aiter_bytes(
            chunk_size: int = 65536,
        ) -> AsyncIterator[bytes]:
            yield content

        mock_resp.aiter_bytes = aiter_bytes
        yield mock_resp

    return _download


def _make_mock_pipeline_result(
    document_id: int = 1,
) -> MagicMock:
    """Create a mock PipelineResult with to_dict() support."""
    result = MagicMock()
    result.to_dict.return_value = {
        "document_id": document_id,
        "success": True,
        "dry_run": True,
        "stage1_result": None,
        "stage2_result": None,
        "stage1_skipped_by_config": False,
        "stage2_skipped_by_config": False,
        "tags_updated": False,
        "content_updated": False,
        "document_uploaded": False,
        "error": None,
        "processing_time_seconds": 1.5,
        "created_at": _NOW.isoformat(),
    }
    return result


@pytest.fixture
def mock_user_client() -> MagicMock:
    """Create a mock user client for dependency injection."""
    return _make_mock_client()


@pytest.fixture
def test_app(mock_user_client: MagicMock) -> Generator[FastAPI, None, None]:
    """Create a FastAPI app with mocked dependencies."""
    settings = _make_settings()
    mock_queue = _make_mock_queue()
    mock_service_client = MagicMock()
    mock_service_client.health_check = AsyncMock(return_value=True)
    mock_service_client.close = AsyncMock()

    with (
        patch(
            "paperless_ngx_smart_ocr.workers.JobQueue",
            return_value=mock_queue,
        ),
        patch(
            "paperless_ngx_smart_ocr.paperless.PaperlessClient",
            return_value=mock_service_client,
        ),
    ):
        app = create_app(settings=settings)

        # Override get_user_client to inject mock
        async def _override() -> MagicMock:  # type: ignore[misc]
            return mock_user_client

        app.dependency_overrides[get_user_client] = _override
        yield app
        app.dependency_overrides.clear()


@pytest.fixture
def client(test_app: FastAPI) -> Generator[TestClient, None, None]:
    """Create a test client with lifespan management and auth cookie."""
    with TestClient(test_app) as c:
        c.cookies.set(AUTH_COOKIE_NAME, "test-token")
        yield c


# ---------------------------------------------------------------------------
# TestIndexView
# ---------------------------------------------------------------------------


class TestIndexView:
    """Tests for GET /."""

    def test_returns_200(self, test_app: FastAPI, client: TestClient) -> None:
        """Returns 200 with HTML content."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_contains_title(self, test_app: FastAPI, client: TestClient) -> None:
        """Page contains the app title."""
        response = client.get("/")
        assert "Smart OCR" in response.text

    def test_shows_connected_status(
        self,
        test_app: FastAPI,
        client: TestClient,
    ) -> None:
        """Shows connected status when health check passes."""
        response = client.get("/")
        assert "Connected to paperless-ngx" in response.text

    def test_shows_disconnected_status(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Shows disconnected when health check fails."""
        mock_user_client.health_check = AsyncMock(
            side_effect=Exception("refused"),
        )
        response = client.get("/")
        assert "Cannot reach paperless-ngx" in response.text

    def test_contains_nav_links(self, test_app: FastAPI, client: TestClient) -> None:
        """Page contains navigation links."""
        response = client.get("/")
        assert 'href="/documents"' in response.text
        assert 'href="/jobs"' in response.text

    def test_contains_version(self, test_app: FastAPI, client: TestClient) -> None:
        """Page footer contains version string."""
        response = client.get("/")
        assert "v0.1.0" in response.text

    def test_contains_dark_mode_toggle(
        self, test_app: FastAPI, client: TestClient
    ) -> None:
        """Page contains dark mode toggle button."""
        response = client.get("/")
        assert 'id="theme-toggle"' in response.text

    def test_theme_mode_attribute(self, test_app: FastAPI, client: TestClient) -> None:
        """HTML tag includes theme mode data attribute."""
        response = client.get("/")
        assert 'data-theme-mode="auto"' in response.text

    def test_contains_logout_button(
        self, test_app: FastAPI, client: TestClient
    ) -> None:
        """Page contains logout button."""
        response = client.get("/")
        assert "Logout" in response.text


# ---------------------------------------------------------------------------
# TestDocumentListView
# ---------------------------------------------------------------------------


class TestDocumentListView:
    """Tests for GET /documents."""

    def test_returns_html(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Returns 200 with HTML content."""
        mock_user_client.list_documents = AsyncMock(
            return_value=_make_paginated_response(),
        )
        response = client.get("/documents")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_contains_document_title(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Page contains the document title."""
        mock_user_client.list_documents = AsyncMock(
            return_value=_make_paginated_response(),
        )
        response = client.get("/documents")
        assert "Test Document 1" in response.text

    def test_htmx_request_returns_partial(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """htmx request returns partial without full page wrapper."""
        mock_user_client.list_documents = AsyncMock(
            return_value=_make_paginated_response(),
        )
        response = client.get(
            "/documents",
            headers={"HX-Request": "true"},
        )
        assert response.status_code == 200
        assert "<html" not in response.text

    def test_empty_list(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Shows message when no documents found."""
        mock_user_client.list_documents = AsyncMock(
            return_value=_make_paginated_response([]),
        )
        response = client.get("/documents")
        assert "No documents found" in response.text

    def test_forwards_query_param(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Search query is forwarded to PaperlessClient."""
        mock_user_client.list_documents = AsyncMock(
            return_value=_make_paginated_response([]),
        )
        client.get("/documents?query=invoice")
        call_kwargs = mock_user_client.list_documents.call_args.kwargs
        assert call_kwargs["query"] == "invoice"

    def test_pagination_params(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Pagination parameters are forwarded."""
        mock_user_client.list_documents = AsyncMock(
            return_value=_make_paginated_response([]),
        )
        client.get("/documents?page=3&page_size=10")
        call_kwargs = mock_user_client.list_documents.call_args.kwargs
        assert call_kwargs["page"] == 3
        assert call_kwargs["page_size"] == 10

    def test_has_row_checkboxes(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Document rows include selection checkboxes."""
        mock_user_client.list_documents = AsyncMock(
            return_value=_make_paginated_response(),
        )
        response = client.get("/documents")
        assert 'class="doc-select' in response.text
        assert 'data-document-id="1"' in response.text

    def test_has_header_checkbox(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Document table includes header select-all checkbox."""
        mock_user_client.list_documents = AsyncMock(
            return_value=_make_paginated_response(),
        )
        response = client.get("/documents")
        assert 'id="select-page"' in response.text

    def test_has_bulk_action_bar(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Document list page includes the bulk action bar."""
        mock_user_client.list_documents = AsyncMock(
            return_value=_make_paginated_response(),
        )
        response = client.get("/documents")
        assert 'id="bulk-action-bar"' in response.text
        assert 'id="bulk-process-btn"' in response.text


# ---------------------------------------------------------------------------
# TestDocumentDetailView
# ---------------------------------------------------------------------------


class TestDocumentDetailView:
    """Tests for GET /documents/{document_id}."""

    def test_returns_html(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Returns 200 with HTML content."""
        doc = _make_document(document_id=42)
        mock_user_client.get_document = AsyncMock(
            return_value=doc,
        )
        response = client.get("/documents/42")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_contains_document_title(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Page contains the document title."""
        doc = _make_document(document_id=42)
        mock_user_client.get_document = AsyncMock(
            return_value=doc,
        )
        response = client.get("/documents/42")
        assert "Test Document 42" in response.text

    def test_contains_process_buttons(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Detail page contains process and dry-run buttons."""
        doc = _make_document(document_id=42)
        mock_user_client.get_document = AsyncMock(
            return_value=doc,
        )
        response = client.get("/documents/42")
        assert "Process" in response.text
        assert "Dry Run" in response.text

    def test_not_found_returns_404(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Missing document returns 404."""
        mock_user_client.get_document = AsyncMock(
            side_effect=PaperlessNotFoundError(
                resource_type="document",
                resource_id=999,
            ),
        )
        response = client.get("/documents/999")
        assert response.status_code == 404

    def test_shows_stage_status(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Detail page shows stage enabled/disabled status."""
        doc = _make_document(document_id=1)
        mock_user_client.get_document = AsyncMock(
            return_value=doc,
        )
        response = client.get("/documents/1")
        assert "Stage 1 (OCR)" in response.text
        assert "Stage 2 (Markdown)" in response.text

    def test_shows_tag_names(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Tags are displayed as names, not numeric IDs."""
        doc = _make_document(document_id=1, tags=[1, 2])
        mock_user_client.get_document = AsyncMock(
            return_value=doc,
        )
        response = client.get("/documents/1")
        assert "Tag 1" in response.text
        assert "Tag 2" in response.text

    def test_shows_correspondent_name(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Correspondent name is resolved and displayed."""
        doc = _make_document(document_id=1, correspondent=1)
        mock_user_client.get_document = AsyncMock(
            return_value=doc,
        )
        response = client.get("/documents/1")
        assert "ACME Corp" in response.text

    def test_shows_document_type_name(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Document type name is resolved and displayed."""
        doc = _make_document(document_id=1, document_type=1)
        mock_user_client.get_document = AsyncMock(
            return_value=doc,
        )
        response = client.get("/documents/1")
        assert "Invoice" in response.text

    def test_shows_archive_serial_number(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Archive serial number is displayed."""
        doc = _make_document(
            document_id=1,
            archive_serial_number=42,
        )
        mock_user_client.get_document = AsyncMock(
            return_value=doc,
        )
        response = client.get("/documents/1")
        assert "42" in response.text
        assert ">ASN<" in response.text

    def test_shows_pdf_viewer(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Detail page contains PDF viewer iframe."""
        doc = _make_document(document_id=7)
        mock_user_client.get_document = AsyncMock(
            return_value=doc,
        )
        response = client.get("/documents/7")
        assert "/documents/7/pdf" in response.text
        assert "PDF Preview" in response.text

    def test_shows_metadata_section(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Detail page contains collapsible metadata."""
        doc = _make_document(document_id=1)
        mock_user_client.get_document = AsyncMock(
            return_value=doc,
        )
        mock_user_client.get_document_metadata = AsyncMock(
            return_value=DocumentMetadata(
                original_metadata=[
                    {
                        "namespace": "dc",
                        "prefix": "dc",
                        "key": "title",
                        "value": "Test",
                    },
                ],
            ),
        )
        response = client.get("/documents/1")
        assert "Original Metadata" in response.text

    def test_no_correspondent_shows_na(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Shows N/A when correspondent is None."""
        doc = _make_document(
            document_id=1,
            correspondent=None,
        )
        mock_user_client.get_document = AsyncMock(
            return_value=doc,
        )
        response = client.get("/documents/1")
        assert "Correspondent" in response.text


# ---------------------------------------------------------------------------
# TestDocumentPdfProxy
# ---------------------------------------------------------------------------


class TestDocumentPdfProxy:
    """Tests for GET /documents/{document_id}/pdf."""

    def test_returns_pdf_content_type(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Returns response with PDF content type."""
        mock_user_client.download_document = _mock_pdf_download(
            b"%PDF-1.4 test",
        )
        response = client.get("/documents/1/pdf")
        assert response.status_code == 200
        assert "application/pdf" in response.headers["content-type"]

    def test_streams_pdf_bytes(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Response body contains the streamed PDF bytes."""
        mock_user_client.download_document = _mock_pdf_download(
            b"%PDF-1.4 test content",
        )
        response = client.get("/documents/1/pdf")
        assert response.content == b"%PDF-1.4 test content"


# ---------------------------------------------------------------------------
# TestProcessDocumentView
# ---------------------------------------------------------------------------


class TestProcessDocumentView:
    """Tests for POST /documents/{document_id}/process."""

    def test_returns_html(self, test_app: FastAPI, client: TestClient) -> None:
        """Returns HTML partial with job card."""
        mock_job = _make_mock_job()
        test_app.state.job_queue.submit = AsyncMock(
            return_value=mock_job,
        )

        with patch(
            "paperless_ngx_smart_ocr.web.routes.views.make_job_coroutine",
        ):
            response = client.post("/documents/1/process")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_returns_job_card(self, test_app: FastAPI, client: TestClient) -> None:
        """Response contains job status information."""
        mock_job = _make_mock_job(
            job_id="test123",
            document_id=5,
        )
        test_app.state.job_queue.submit = AsyncMock(
            return_value=mock_job,
        )

        with patch(
            "paperless_ngx_smart_ocr.web.routes.views.make_job_coroutine",
        ):
            response = client.post("/documents/5/process")

        assert "pending" in response.text
        assert "Process document 5" in response.text

    def test_submits_to_queue(self, test_app: FastAPI, client: TestClient) -> None:
        """Coroutine is submitted to the job queue."""
        mock_job = _make_mock_job()
        test_app.state.job_queue.submit = AsyncMock(
            return_value=mock_job,
        )

        with patch(
            "paperless_ngx_smart_ocr.web.routes.views.make_job_coroutine",
        ):
            client.post("/documents/7/process")

        submit_kwargs = test_app.state.job_queue.submit.call_args.kwargs
        assert submit_kwargs["name"] == "Process document 7"
        assert submit_kwargs["document_id"] == 7


# ---------------------------------------------------------------------------
# TestDryRunView
# ---------------------------------------------------------------------------


class TestDryRunView:
    """Tests for POST /documents/{document_id}/dry-run."""

    def test_returns_html(self, test_app: FastAPI, client: TestClient) -> None:
        """Returns HTML partial with result."""
        mock_result = _make_mock_pipeline_result()

        with patch(
            "paperless_ngx_smart_ocr.pipeline.orchestrator.PipelineOrchestrator",
        ) as mock_cls:
            mock_cls.return_value.process = AsyncMock(
                return_value=mock_result,
            )
            response = client.post("/documents/1/dry-run")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_shows_result(self, test_app: FastAPI, client: TestClient) -> None:
        """Response contains pipeline result information."""
        mock_result = _make_mock_pipeline_result()

        with patch(
            "paperless_ngx_smart_ocr.pipeline.orchestrator.PipelineOrchestrator",
        ) as mock_cls:
            mock_cls.return_value.process = AsyncMock(
                return_value=mock_result,
            )
            response = client.post("/documents/1/dry-run")

        assert "Dry Run Result" in response.text
        assert "Success" in response.text

    def test_passes_dry_run_true(self, test_app: FastAPI, client: TestClient) -> None:
        """dry_run=True is passed to orchestrator.process()."""
        mock_result = _make_mock_pipeline_result()

        with patch(
            "paperless_ngx_smart_ocr.pipeline.orchestrator.PipelineOrchestrator",
        ) as mock_cls:
            mock_cls.return_value.process = AsyncMock(
                return_value=mock_result,
            )
            client.post("/documents/1/dry-run")

        call_kwargs = mock_cls.return_value.process.call_args.kwargs
        assert call_kwargs["dry_run"] is True

    def test_does_not_use_job_queue(
        self, test_app: FastAPI, client: TestClient
    ) -> None:
        """Dry-run runs synchronously, not via job queue."""
        mock_result = _make_mock_pipeline_result()

        with patch(
            "paperless_ngx_smart_ocr.pipeline.orchestrator.PipelineOrchestrator",
        ) as mock_cls:
            mock_cls.return_value.process = AsyncMock(
                return_value=mock_result,
            )
            client.post("/documents/1/dry-run")

        test_app.state.job_queue.submit.assert_not_called()


# ---------------------------------------------------------------------------
# TestBulkProcessView
# ---------------------------------------------------------------------------


class TestBulkProcessView:
    """Tests for POST /documents/bulk-process."""

    def test_returns_html(self, test_app: FastAPI, client: TestClient) -> None:
        """Returns HTML with job cards."""
        mock_job = _make_mock_job()
        test_app.state.job_queue.submit = AsyncMock(
            return_value=mock_job,
        )
        with patch(
            "paperless_ngx_smart_ocr.web.routes.views.make_job_coroutine",
        ):
            response = client.post(
                "/documents/bulk-process",
                data={"document_ids": "1,2,3"},
            )
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_submits_multiple_jobs(self, test_app: FastAPI, client: TestClient) -> None:
        """Each document ID gets its own job."""
        mock_job = _make_mock_job()
        test_app.state.job_queue.submit = AsyncMock(
            return_value=mock_job,
        )
        with patch(
            "paperless_ngx_smart_ocr.web.routes.views.make_job_coroutine",
        ):
            client.post(
                "/documents/bulk-process",
                data={"document_ids": "1,2,3"},
            )
        assert test_app.state.job_queue.submit.call_count == 3

    def test_shows_count(self, test_app: FastAPI, client: TestClient) -> None:
        """Response shows the number of submitted jobs."""
        mock_job = _make_mock_job()
        test_app.state.job_queue.submit = AsyncMock(
            return_value=mock_job,
        )
        with patch(
            "paperless_ngx_smart_ocr.web.routes.views.make_job_coroutine",
        ):
            response = client.post(
                "/documents/bulk-process",
                data={"document_ids": "1,2"},
            )
        assert "2 documents submitted" in response.text

    def test_empty_ids_returns_zero(
        self, test_app: FastAPI, client: TestClient
    ) -> None:
        """Empty document_ids returns 0 jobs."""
        response = client.post(
            "/documents/bulk-process",
            data={"document_ids": ""},
        )
        assert response.status_code == 200
        assert "0 documents submitted" in response.text

    def test_filter_query_mode(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """filter_query triggers server-side re-query."""
        mock_user_client.list_documents = AsyncMock(
            return_value=_make_paginated_response(
                [_make_document(1), _make_document(2)],
            ),
        )
        mock_job = _make_mock_job()
        test_app.state.job_queue.submit = AsyncMock(
            return_value=mock_job,
        )
        with patch(
            "paperless_ngx_smart_ocr.web.routes.views.make_job_coroutine",
        ):
            response = client.post(
                "/documents/bulk-process",
                data={
                    "filter_query": "&query=invoice",
                },
            )
        assert response.status_code == 200
        assert test_app.state.job_queue.submit.call_count == 2

    def test_force_flag_forwarded(self, test_app: FastAPI, client: TestClient) -> None:
        """force=true is forwarded to make_job_coroutine."""
        mock_job = _make_mock_job()
        test_app.state.job_queue.submit = AsyncMock(
            return_value=mock_job,
        )
        with patch(
            "paperless_ngx_smart_ocr.web.routes.views.make_job_coroutine",
        ) as mock_make:
            mock_make.return_value = MagicMock()
            client.post(
                "/documents/bulk-process",
                data={
                    "document_ids": "1",
                    "force": "true",
                },
            )
        mock_make.assert_called_once()
        call_kwargs = mock_make.call_args.kwargs
        assert call_kwargs["force"] is True


# ---------------------------------------------------------------------------
# TestJobListView
# ---------------------------------------------------------------------------


class TestJobListView:
    """Tests for GET /jobs."""

    def test_returns_html(self, test_app: FastAPI, client: TestClient) -> None:
        """Returns 200 with HTML content."""
        response = client.get("/jobs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_contains_title(self, test_app: FastAPI, client: TestClient) -> None:
        """Page contains the Jobs heading."""
        response = client.get("/jobs")
        assert "Jobs" in response.text

    def test_shows_empty_message(self, test_app: FastAPI, client: TestClient) -> None:
        """Shows message when no jobs exist."""
        response = client.get("/jobs")
        assert "No jobs found" in response.text

    def test_htmx_request_returns_partial(
        self, test_app: FastAPI, client: TestClient
    ) -> None:
        """htmx request returns partial without full page wrapper."""
        response = client.get(
            "/jobs",
            headers={"HX-Request": "true"},
        )
        assert response.status_code == 200
        assert "<html" not in response.text

    def test_shows_job_card(self, test_app: FastAPI, client: TestClient) -> None:
        """Shows job card when jobs exist."""
        mock_job = _make_mock_job(status="running")
        test_app.state.job_queue.list_jobs = AsyncMock(
            return_value=[mock_job],
        )
        response = client.get("/jobs")
        assert "running" in response.text
        assert "Process document 1" in response.text


# ---------------------------------------------------------------------------
# TestJobDetailView
# ---------------------------------------------------------------------------


class TestJobDetailView:
    """Tests for GET /jobs/{job_id}."""

    def test_returns_html(self, test_app: FastAPI, client: TestClient) -> None:
        """Returns 200 with HTML content."""
        mock_job = _make_mock_job()
        test_app.state.job_queue.get = AsyncMock(
            return_value=mock_job,
        )
        response = client.get("/jobs/abc123def456")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_contains_job_name(self, test_app: FastAPI, client: TestClient) -> None:
        """Page contains the job name."""
        mock_job = _make_mock_job()
        test_app.state.job_queue.get = AsyncMock(
            return_value=mock_job,
        )
        response = client.get("/jobs/abc123def456")
        assert "Process document 1" in response.text

    def test_not_found_returns_404(self, test_app: FastAPI, client: TestClient) -> None:
        """Missing job returns 404."""
        test_app.state.job_queue.get = AsyncMock(
            side_effect=JobNotFoundError("not-found"),
        )
        response = client.get("/jobs/not-found")
        assert response.status_code == 404

    def test_htmx_request_returns_partial(
        self, test_app: FastAPI, client: TestClient
    ) -> None:
        """htmx request returns job card partial."""
        mock_job = _make_mock_job()
        test_app.state.job_queue.get = AsyncMock(
            return_value=mock_job,
        )
        response = client.get(
            "/jobs/abc123def456",
            headers={"HX-Request": "true"},
        )
        assert response.status_code == 200
        assert "<html" not in response.text


# ---------------------------------------------------------------------------
# TestCancelJobView
# ---------------------------------------------------------------------------


class TestCancelJobView:
    """Tests for POST /jobs/{job_id}/cancel."""

    def test_returns_html(self, test_app: FastAPI, client: TestClient) -> None:
        """Returns updated job card as HTML."""
        mock_job = _make_mock_job(status="cancelled")
        test_app.state.job_queue.cancel = AsyncMock(
            return_value=mock_job,
        )
        response = client.post("/jobs/abc123def456/cancel")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "cancelled" in response.text

    def test_not_found_returns_404(self, test_app: FastAPI, client: TestClient) -> None:
        """Missing job returns 404."""
        test_app.state.job_queue.cancel = AsyncMock(
            side_effect=JobNotFoundError("not-found"),
        )
        response = client.post("/jobs/not-found/cancel")
        assert response.status_code == 404

    def test_already_cancelled_returns_409(
        self, test_app: FastAPI, client: TestClient
    ) -> None:
        """Already-cancelled job returns 409."""
        test_app.state.job_queue.cancel = AsyncMock(
            side_effect=JobAlreadyCancelledError("abc123"),
        )
        response = client.post("/jobs/abc123/cancel")
        assert response.status_code == 409


# ---------------------------------------------------------------------------
# TestDarkModeConfig
# ---------------------------------------------------------------------------


class TestDarkModeConfig:
    """Tests for dark mode configuration in templates."""

    def test_auto_theme(self, test_app: FastAPI, client: TestClient) -> None:
        """Default auto theme is set in HTML."""
        response = client.get("/")
        assert 'data-theme-mode="auto"' in response.text

    def test_dark_theme(self) -> None:
        """Dark theme config is reflected in HTML attribute."""
        settings = _make_settings(web={"theme": "dark"})
        mock_queue = _make_mock_queue()
        mock_service_client = MagicMock()
        mock_service_client.health_check = AsyncMock(return_value=True)
        mock_service_client.close = AsyncMock()

        mock_user_client = _make_mock_client()

        with (
            patch(
                "paperless_ngx_smart_ocr.workers.JobQueue",
                return_value=mock_queue,
            ),
            patch(
                "paperless_ngx_smart_ocr.paperless.PaperlessClient",
                return_value=mock_service_client,
            ),
        ):
            app = create_app(settings=settings)

            async def _override() -> MagicMock:  # type: ignore[misc]
                return mock_user_client

            app.dependency_overrides[get_user_client] = _override

        with TestClient(app) as c:
            c.cookies.set(AUTH_COOKIE_NAME, "test-token")
            response = c.get("/")
        assert 'data-theme-mode="dark"' in response.text
        app.dependency_overrides.clear()

    def test_light_theme(self) -> None:
        """Light theme config is reflected in HTML attribute."""
        settings = _make_settings(web={"theme": "light"})
        mock_queue = _make_mock_queue()
        mock_service_client = MagicMock()
        mock_service_client.health_check = AsyncMock(return_value=True)
        mock_service_client.close = AsyncMock()

        mock_user_client = _make_mock_client()

        with (
            patch(
                "paperless_ngx_smart_ocr.workers.JobQueue",
                return_value=mock_queue,
            ),
            patch(
                "paperless_ngx_smart_ocr.paperless.PaperlessClient",
                return_value=mock_service_client,
            ),
        ):
            app = create_app(settings=settings)

            async def _override() -> MagicMock:  # type: ignore[misc]
                return mock_user_client

            app.dependency_overrides[get_user_client] = _override

        with TestClient(app) as c:
            c.cookies.set(AUTH_COOKIE_NAME, "test-token")
            response = c.get("/")
        assert 'data-theme-mode="light"' in response.text
        app.dependency_overrides.clear()
