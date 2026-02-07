"""Unit tests for the web application module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from paperless_ngx_smart_ocr.config import Settings
from paperless_ngx_smart_ocr.paperless.exceptions import (
    PaperlessConnectionError,
    PaperlessNotFoundError,
)
from paperless_ngx_smart_ocr.web import create_app
from paperless_ngx_smart_ocr.web.auth import AUTH_COOKIE_NAME
from paperless_ngx_smart_ocr.workers.exceptions import JobNotFoundError


if TYPE_CHECKING:
    from collections.abc import Generator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_mock_client(*, healthy: bool = True) -> AsyncMock:
    """Create a mock PaperlessClient with sensible defaults."""
    client = AsyncMock()
    client.health_check = AsyncMock(return_value=healthy)
    client.close = AsyncMock()
    return client


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
        yield create_app(settings=settings)


@pytest.fixture
def client(test_app: FastAPI) -> Generator[TestClient, None, None]:
    """Create a test client with lifespan management."""
    with TestClient(test_app) as c:
        yield c


# ---------------------------------------------------------------------------
# TestCreateApp
# ---------------------------------------------------------------------------


class TestCreateApp:
    """Tests for create_app factory."""

    def test_creates_app(self) -> None:
        """Factory returns a FastAPI instance with settings on state."""
        settings = _make_settings()
        app = create_app(settings=settings)
        assert isinstance(app, FastAPI)
        assert app.state.settings is settings

    def test_creates_app_with_default_settings(self) -> None:
        """Factory loads default settings when none provided."""
        with patch(
            "paperless_ngx_smart_ocr.config.get_settings",
            return_value=_make_settings(),
        ):
            app = create_app()
            assert isinstance(app, FastAPI)

    def test_app_has_title(self) -> None:
        """App has the expected title."""
        settings = _make_settings()
        app = create_app(settings=settings)
        assert app.title == "paperless-ngx-smart-ocr"

    def test_app_has_version(self) -> None:
        """App version matches package version."""
        from paperless_ngx_smart_ocr import __version__

        settings = _make_settings()
        app = create_app(settings=settings)
        assert app.version == __version__


# ---------------------------------------------------------------------------
# TestHealthEndpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for GET /api/health."""

    def test_returns_200(self, client: TestClient) -> None:
        """Health endpoint returns 200 OK."""
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_returns_ok_status(self, client: TestClient) -> None:
        """Health endpoint returns status ok."""
        data = client.get("/api/health").json()
        assert data["status"] == "ok"

    def test_returns_version(self, client: TestClient) -> None:
        """Health endpoint includes version."""
        from paperless_ngx_smart_ocr import __version__

        data = client.get("/api/health").json()
        assert data["version"] == __version__


# ---------------------------------------------------------------------------
# TestReadinessEndpoint
# ---------------------------------------------------------------------------


class TestReadinessEndpoint:
    """Tests for GET /api/ready."""

    def test_returns_200_when_ready(self, client: TestClient) -> None:
        """Readiness returns 200 when all checks pass."""
        response = client.get("/api/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["checks"]["job_queue"] is True
        assert data["checks"]["paperless"] is True

    def test_returns_503_when_paperless_down(self) -> None:
        """Readiness returns 503 when paperless is unreachable."""
        settings = _make_settings()
        mock_queue = _make_mock_queue()
        mock_client = _make_mock_client(healthy=False)

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
            with TestClient(app) as c:
                response = c.get("/api/ready")
                assert response.status_code == 503
                data = response.json()
                assert data["status"] == "not_ready"
                assert data["checks"]["paperless"] is False

    def test_returns_503_when_queue_stopped(self) -> None:
        """Readiness returns 503 when job queue is not running."""
        settings = _make_settings()
        mock_queue = _make_mock_queue()
        mock_queue.is_running = False
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
            with TestClient(app) as c:
                response = c.get("/api/ready")
                assert response.status_code == 503
                data = response.json()
                assert data["checks"]["job_queue"] is False

    def test_returns_503_when_health_check_raises(self) -> None:
        """Readiness returns 503 when health check raises."""
        settings = _make_settings()
        mock_queue = _make_mock_queue()
        mock_client = _make_mock_client()
        mock_client.health_check = AsyncMock(
            side_effect=ConnectionError("refused"),
        )

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
            with TestClient(app) as c:
                response = c.get("/api/ready")
                assert response.status_code == 503
                assert response.json()["checks"]["paperless"] is False


# ---------------------------------------------------------------------------
# TestRequestIDMiddleware
# ---------------------------------------------------------------------------


class TestRequestIDMiddleware:
    """Tests for request ID middleware."""

    def test_generates_request_id(self, client: TestClient) -> None:
        """Responses include a generated X-Request-ID header."""
        response = client.get("/api/health")
        assert "x-request-id" in response.headers
        assert len(response.headers["x-request-id"]) == 8

    def test_uses_provided_request_id(
        self,
        client: TestClient,
    ) -> None:
        """Provided X-Request-ID header is propagated."""
        response = client.get(
            "/api/health",
            headers={"X-Request-ID": "custom-id"},
        )
        assert response.headers["x-request-id"] == "custom-id"


# ---------------------------------------------------------------------------
# TestExceptionHandlers
# ---------------------------------------------------------------------------


class TestExceptionHandlers:
    """Tests for global exception handlers."""

    def test_paperless_connection_error_returns_502(
        self,
        test_app: FastAPI,
    ) -> None:
        """PaperlessConnectionError returns 502."""

        @test_app.get("/test-paperless-error")
        async def _raise() -> None:
            msg = "connection refused"
            raise PaperlessConnectionError(msg)

        with TestClient(
            test_app,
            raise_server_exceptions=False,
        ) as c:
            c.cookies.set(AUTH_COOKIE_NAME, "test-token")
            response = c.get("/test-paperless-error")
            assert response.status_code == 502
            data = response.json()
            assert data["error_type"] == "PaperlessConnectionError"

    def test_paperless_not_found_returns_404(
        self,
        test_app: FastAPI,
    ) -> None:
        """PaperlessNotFoundError returns 404."""

        @test_app.get("/test-not-found")
        async def _raise() -> None:
            raise PaperlessNotFoundError(
                resource_type="document",
                resource_id=999,
            )

        with TestClient(
            test_app,
            raise_server_exceptions=False,
        ) as c:
            c.cookies.set(AUTH_COOKIE_NAME, "test-token")
            response = c.get("/test-not-found")
            assert response.status_code == 404

    def test_job_not_found_returns_404(
        self,
        test_app: FastAPI,
    ) -> None:
        """JobNotFoundError returns 404."""

        @test_app.get("/test-job-not-found")
        async def _raise() -> None:
            job_id = "abc123"
            raise JobNotFoundError(job_id)

        with TestClient(
            test_app,
            raise_server_exceptions=False,
        ) as c:
            c.cookies.set(AUTH_COOKIE_NAME, "test-token")
            response = c.get("/test-job-not-found")
            assert response.status_code == 404
            data = response.json()
            assert data["error_type"] == "JobNotFoundError"

    def test_generic_error_returns_500(
        self,
        test_app: FastAPI,
    ) -> None:
        """Unhandled exception returns 500."""

        @test_app.get("/test-generic-error")
        async def _raise() -> None:
            msg = "unexpected"
            raise RuntimeError(msg)

        with TestClient(
            test_app,
            raise_server_exceptions=False,
        ) as c:
            c.cookies.set(AUTH_COOKIE_NAME, "test-token")
            response = c.get("/test-generic-error")
            assert response.status_code == 500
            data = response.json()
            assert data["detail"] == "Internal server error"
            assert data["error_type"] == "RuntimeError"
