"""Unit tests for job API routes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from paperless_ngx_smart_ocr.config import Settings
from paperless_ngx_smart_ocr.web import create_app
from paperless_ngx_smart_ocr.web.auth import AUTH_COOKIE_NAME
from paperless_ngx_smart_ocr.workers.exceptions import (
    JobAlreadyCancelledError,
    JobNotFoundError,
)


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


def _make_mock_queue() -> AsyncMock:
    """Create a mock JobQueue with sensible defaults."""
    queue = AsyncMock()
    queue.is_running = True
    queue.start = AsyncMock()
    queue.stop = AsyncMock()
    return queue


def _make_mock_client() -> AsyncMock:
    """Create a mock PaperlessClient with sensible defaults."""
    client = AsyncMock()
    client.health_check = AsyncMock(return_value=True)
    client.close = AsyncMock()
    return client


def _make_mock_job(
    job_id: str = "abc123def456",
    status: str = "running",
    document_id: int | None = 1,
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
        "started_at": _NOW.isoformat(),
        "completed_at": None,
        "duration_seconds": None,
        "result": None,
        "metadata": {},
    }
    return job


@pytest.fixture
def test_app() -> Generator[FastAPI, None, None]:
    """Create a FastAPI app with mocked dependencies."""
    settings = _make_settings()
    mock_queue = _make_mock_queue()
    mock_client = _make_mock_client()

    with (
        patch(
            "paperless_ngx_smart_ocr.workers.JobQueue",
            return_value=mock_queue,
        ),
        patch(
            "paperless_ngx_smart_ocr.paperless.PaperlessClient",
            return_value=mock_client,
        ),
    ):
        app = create_app(settings=settings)
        yield app


@pytest.fixture
def client(test_app: FastAPI) -> Generator[TestClient, None, None]:
    """Create a test client with lifespan management."""
    with TestClient(test_app) as c:
        c.cookies.set(AUTH_COOKIE_NAME, "test-token")
        yield c


# ---------------------------------------------------------------------------
# TestListJobs
# ---------------------------------------------------------------------------


class TestListJobs:
    """Tests for GET /api/jobs."""

    def test_returns_200(self, test_app: FastAPI, client: TestClient) -> None:
        """Returns 200 with job list."""
        jobs = [_make_mock_job(), _make_mock_job(job_id="xyz789")]
        test_app.state.job_queue.list_jobs = AsyncMock(
            return_value=jobs,
        )

        response = client.get("/api/jobs")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_returns_empty_list(self, test_app: FastAPI, client: TestClient) -> None:
        """Returns empty list when no jobs exist."""
        test_app.state.job_queue.list_jobs = AsyncMock(
            return_value=[],
        )

        response = client.get("/api/jobs")
        assert response.status_code == 200
        assert response.json() == []

    def test_filters_by_status(self, test_app: FastAPI, client: TestClient) -> None:
        """status query param is forwarded to list_jobs."""
        test_app.state.job_queue.list_jobs = AsyncMock(
            return_value=[],
        )

        client.get("/api/jobs?status=completed")

        call_kwargs = test_app.state.job_queue.list_jobs.call_args.kwargs
        assert call_kwargs["status"].value == "completed"

    def test_filters_by_document_id(
        self, test_app: FastAPI, client: TestClient
    ) -> None:
        """document_id query param is forwarded to list_jobs."""
        test_app.state.job_queue.list_jobs = AsyncMock(
            return_value=[],
        )

        client.get("/api/jobs?document_id=42")

        call_kwargs = test_app.state.job_queue.list_jobs.call_args.kwargs
        assert call_kwargs["document_id"] == 42

    def test_filters_by_limit(self, test_app: FastAPI, client: TestClient) -> None:
        """limit query param is forwarded to list_jobs."""
        test_app.state.job_queue.list_jobs = AsyncMock(
            return_value=[],
        )

        client.get("/api/jobs?limit=50")

        call_kwargs = test_app.state.job_queue.list_jobs.call_args.kwargs
        assert call_kwargs["limit"] == 50

    def test_invalid_status_returns_422(
        self, test_app: FastAPI, client: TestClient
    ) -> None:
        """Invalid status value returns 422."""
        response = client.get("/api/jobs?status=invalid_status")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# TestGetJob
# ---------------------------------------------------------------------------


class TestGetJob:
    """Tests for GET /api/jobs/{job_id}."""

    def test_returns_200(self, test_app: FastAPI, client: TestClient) -> None:
        """Returns 200 with job detail."""
        mock_job = _make_mock_job(job_id="test123")
        test_app.state.job_queue.get = AsyncMock(
            return_value=mock_job,
        )

        response = client.get("/api/jobs/test123")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test123"
        assert data["status"] == "running"

    def test_not_found_returns_404(
        self,
        test_app: FastAPI,
        client: TestClient,
    ) -> None:
        """Missing job returns 404."""
        test_app.state.job_queue.get = AsyncMock(
            side_effect=JobNotFoundError("nonexistent"),
        )

        response = client.get("/api/jobs/nonexistent")
        assert response.status_code == 404
        data = response.json()
        assert data["error_type"] == "JobNotFoundError"


# ---------------------------------------------------------------------------
# TestCancelJob
# ---------------------------------------------------------------------------


class TestCancelJob:
    """Tests for POST /api/jobs/{job_id}/cancel."""

    def test_returns_200(self, test_app: FastAPI, client: TestClient) -> None:
        """Returns 200 with cancelled job."""
        mock_job = _make_mock_job(job_id="cancel-me", status="cancelled")
        test_app.state.job_queue.cancel = AsyncMock(
            return_value=mock_job,
        )

        response = client.post("/api/jobs/cancel-me/cancel")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "cancel-me"
        assert data["status"] == "cancelled"

    def test_not_found_returns_404(
        self,
        test_app: FastAPI,
        client: TestClient,
    ) -> None:
        """Missing job returns 404."""
        test_app.state.job_queue.cancel = AsyncMock(
            side_effect=JobNotFoundError("nonexistent"),
        )

        response = client.post("/api/jobs/nonexistent/cancel")
        assert response.status_code == 404

    def test_already_cancelled_returns_409(
        self,
        test_app: FastAPI,
        client: TestClient,
    ) -> None:
        """Already cancelled job returns 409."""
        test_app.state.job_queue.cancel = AsyncMock(
            side_effect=JobAlreadyCancelledError("already-done"),
        )

        response = client.post("/api/jobs/already-done/cancel")
        assert response.status_code == 409
        data = response.json()
        assert data["error_type"] == "JobAlreadyCancelledError"
