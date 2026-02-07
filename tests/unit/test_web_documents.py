"""Unit tests for document API routes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from paperless_ngx_smart_ocr.config import Settings
from paperless_ngx_smart_ocr.paperless.exceptions import (
    PaperlessConnectionError,
    PaperlessNotFoundError,
)
from paperless_ngx_smart_ocr.paperless.models import (
    Document,
    PaginatedResponse,
)
from paperless_ngx_smart_ocr.web import create_app
from paperless_ngx_smart_ocr.web.auth import AUTH_COOKIE_NAME, get_user_client


if TYPE_CHECKING:
    from collections.abc import Generator

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
    return queue


def _make_mock_client() -> MagicMock:
    """Create a mock PaperlessClient with sensible defaults."""
    client = MagicMock()
    client.health_check = AsyncMock(return_value=True)
    client.close = AsyncMock()
    client.list_documents = AsyncMock()
    client.get_document = AsyncMock()
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
    }
    defaults.update(overrides)
    return Document(**defaults)  # type: ignore[arg-type]


def _make_paginated_response(
    documents: list[Document] | None = None,
) -> PaginatedResponse[Document]:
    """Create a PaginatedResponse for testing."""
    docs = documents or [_make_document()]
    return PaginatedResponse[Document](
        count=len(docs),
        results=docs,
    )


def _make_mock_job(
    job_id: str = "abc123def456",
    document_id: int = 1,
) -> MagicMock:
    """Create a mock Job with to_dict() support."""
    job = MagicMock()
    job.id = job_id
    job.to_dict.return_value = {
        "id": job_id,
        "name": f"Process document {document_id}",
        "status": "pending",
        "document_id": document_id,
        "created_at": _NOW.isoformat(),
        "started_at": None,
        "completed_at": None,
        "duration_seconds": None,
        "result": None,
        "metadata": {},
    }
    return job


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
# TestListDocuments
# ---------------------------------------------------------------------------


class TestListDocuments:
    """Tests for GET /api/documents."""

    def test_returns_200(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Returns 200 with document list."""
        response_data = _make_paginated_response()
        mock_user_client.list_documents = AsyncMock(
            return_value=response_data,
        )

        response = client.get("/api/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["id"] == 1

    def test_forwards_query_params(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Query parameters are forwarded to PaperlessClient."""
        mock_user_client.list_documents = AsyncMock(
            return_value=_make_paginated_response([]),
        )

        client.get(
            "/api/documents?page=2&page_size=10&query=invoice&ordering=-created",
        )

        call_kwargs = mock_user_client.list_documents.call_args.kwargs
        assert call_kwargs["page"] == 2
        assert call_kwargs["page_size"] == 10
        assert call_kwargs["query"] == "invoice"
        assert call_kwargs["ordering"] == "-created"

    def test_parses_tags_include(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Comma-separated tags_include is parsed to list[int]."""
        mock_user_client.list_documents = AsyncMock(
            return_value=_make_paginated_response([]),
        )

        client.get("/api/documents?tags_include=1,2,3")

        call_kwargs = mock_user_client.list_documents.call_args.kwargs
        assert call_kwargs["tags_include"] == [1, 2, 3]

    def test_parses_tags_exclude(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Comma-separated tags_exclude is parsed to list[int]."""
        mock_user_client.list_documents = AsyncMock(
            return_value=_make_paginated_response([]),
        )

        client.get("/api/documents?tags_exclude=4,5")

        call_kwargs = mock_user_client.list_documents.call_args.kwargs
        assert call_kwargs["tags_exclude"] == [4, 5]

    def test_tags_none_when_omitted(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Tags are None when not provided."""
        mock_user_client.list_documents = AsyncMock(
            return_value=_make_paginated_response([]),
        )

        client.get("/api/documents")

        call_kwargs = mock_user_client.list_documents.call_args.kwargs
        assert call_kwargs["tags_include"] is None
        assert call_kwargs["tags_exclude"] is None

    def test_paperless_error_returns_502(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """PaperlessConnectionError returns 502."""
        mock_user_client.list_documents = AsyncMock(
            side_effect=PaperlessConnectionError("refused"),
        )

        response = client.get("/api/documents")
        assert response.status_code == 502


# ---------------------------------------------------------------------------
# TestGetDocument
# ---------------------------------------------------------------------------


class TestGetDocument:
    """Tests for GET /api/documents/{document_id}."""

    def test_returns_200(
        self,
        test_app: FastAPI,
        mock_user_client: MagicMock,
        client: TestClient,
    ) -> None:
        """Returns 200 with document detail."""
        doc = _make_document(document_id=42)
        mock_user_client.get_document = AsyncMock(
            return_value=doc,
        )

        response = client.get("/api/documents/42")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == 42
        assert data["title"] == "Test Document 42"

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

        response = client.get("/api/documents/999")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# TestProcessDocument
# ---------------------------------------------------------------------------


class TestProcessDocument:
    """Tests for POST /api/documents/{document_id}/process."""

    def test_returns_202(self, test_app: FastAPI, client: TestClient) -> None:
        """Returns 202 Accepted with job info."""
        mock_job = _make_mock_job()
        test_app.state.job_queue.submit = AsyncMock(
            return_value=mock_job,
        )

        with patch(
            "paperless_ngx_smart_ocr.web.routes.documents.make_job_coroutine",
        ):
            response = client.post("/api/documents/1/process")

        assert response.status_code == 202

    def test_returns_job_dict(self, test_app: FastAPI, client: TestClient) -> None:
        """Response body matches Job.to_dict() shape."""
        mock_job = _make_mock_job(job_id="test123", document_id=5)
        test_app.state.job_queue.submit = AsyncMock(
            return_value=mock_job,
        )

        with patch(
            "paperless_ngx_smart_ocr.web.routes.documents.make_job_coroutine",
        ):
            response = client.post("/api/documents/5/process")

        data = response.json()
        assert data["id"] == "test123"
        assert data["document_id"] == 5
        assert data["status"] == "pending"

    def test_submits_to_queue(self, test_app: FastAPI, client: TestClient) -> None:
        """Coroutine is submitted to the job queue."""
        mock_job = _make_mock_job()
        test_app.state.job_queue.submit = AsyncMock(
            return_value=mock_job,
        )

        with patch(
            "paperless_ngx_smart_ocr.web.routes.documents.make_job_coroutine",
        ):
            client.post("/api/documents/7/process")

        submit_kwargs = test_app.state.job_queue.submit.call_args.kwargs
        assert submit_kwargs["name"] == "Process document 7"
        assert submit_kwargs["document_id"] == 7

    def test_force_param(self, test_app: FastAPI, client: TestClient) -> None:
        """force=true is passed through to make_job_coroutine."""
        mock_job = _make_mock_job()
        test_app.state.job_queue.submit = AsyncMock(
            return_value=mock_job,
        )

        with patch(
            "paperless_ngx_smart_ocr.web.routes.documents.make_job_coroutine",
        ) as mock_make:
            client.post("/api/documents/1/process?force=true")

        mock_make.assert_called_once()
        call_kwargs = mock_make.call_args.kwargs
        assert call_kwargs["force"] is True


# ---------------------------------------------------------------------------
# TestDryRunDocument
# ---------------------------------------------------------------------------


class TestDryRunDocument:
    """Tests for POST /api/documents/{document_id}/dry-run."""

    def test_returns_200(self, test_app: FastAPI, client: TestClient) -> None:
        """Returns 200 with pipeline result."""
        mock_result = _make_mock_pipeline_result()

        with patch(
            "paperless_ngx_smart_ocr.pipeline.orchestrator.PipelineOrchestrator",
        ) as mock_cls:
            mock_cls.return_value.process = AsyncMock(
                return_value=mock_result,
            )
            response = client.post("/api/documents/1/dry-run")

        assert response.status_code == 200

    def test_returns_pipeline_result(
        self, test_app: FastAPI, client: TestClient
    ) -> None:
        """Response body matches PipelineResult.to_dict() shape."""
        mock_result = _make_mock_pipeline_result(document_id=3)

        with patch(
            "paperless_ngx_smart_ocr.pipeline.orchestrator.PipelineOrchestrator",
        ) as mock_cls:
            mock_cls.return_value.process = AsyncMock(
                return_value=mock_result,
            )
            response = client.post("/api/documents/3/dry-run")

        data = response.json()
        assert data["document_id"] == 3
        assert data["success"] is True
        assert data["dry_run"] is True
        assert data["tags_updated"] is False
        assert data["content_updated"] is False

    def test_passes_dry_run_true(self, test_app: FastAPI, client: TestClient) -> None:
        """dry_run=True is passed to orchestrator.process()."""
        mock_result = _make_mock_pipeline_result()

        with patch(
            "paperless_ngx_smart_ocr.pipeline.orchestrator.PipelineOrchestrator",
        ) as mock_cls:
            mock_cls.return_value.process = AsyncMock(
                return_value=mock_result,
            )
            client.post("/api/documents/1/dry-run")

        call_kwargs = mock_cls.return_value.process.call_args.kwargs
        assert call_kwargs["dry_run"] is True

    def test_force_param(self, test_app: FastAPI, client: TestClient) -> None:
        """force=true is passed through."""
        mock_result = _make_mock_pipeline_result()

        with patch(
            "paperless_ngx_smart_ocr.pipeline.orchestrator.PipelineOrchestrator",
        ) as mock_cls:
            mock_cls.return_value.process = AsyncMock(
                return_value=mock_result,
            )
            client.post("/api/documents/1/dry-run?force=true")

        call_kwargs = mock_cls.return_value.process.call_args.kwargs
        assert call_kwargs["force"] is True

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
            client.post("/api/documents/1/dry-run")

        test_app.state.job_queue.submit.assert_not_called()
