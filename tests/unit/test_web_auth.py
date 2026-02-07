"""Unit tests for authentication middleware, login/logout, and helpers."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from paperless_ngx_smart_ocr.config import Settings
from paperless_ngx_smart_ocr.web import create_app
from paperless_ngx_smart_ocr.web.auth import (
    AUTH_COOKIE_NAME,
    get_user_client,
)


if TYPE_CHECKING:
    from collections.abc import Generator

    from fastapi import FastAPI


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


def _make_mock_queue() -> MagicMock:
    """Create a mock JobQueue with sensible defaults."""
    queue = MagicMock()
    queue.is_running = True
    queue.start = AsyncMock()
    queue.stop = AsyncMock()
    return queue


def _make_mock_client(*, healthy: bool = True) -> MagicMock:
    """Create a mock PaperlessClient with sensible defaults."""
    client = MagicMock()
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


def _set_auth_cookie(client: TestClient, token: str = "test-token") -> None:
    """Set the auth cookie on the test client."""
    client.cookies.set(AUTH_COOKIE_NAME, token)


# ---------------------------------------------------------------------------
# TestAuthMiddleware
# ---------------------------------------------------------------------------


class TestAuthMiddleware:
    """Tests for AuthMiddleware cookie enforcement."""

    def test_unauthenticated_html_redirects_to_login(self, client: TestClient) -> None:
        """Missing cookie on HTML request redirects to /login."""
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 302
        assert response.headers["location"] == "/login"

    def test_unauthenticated_api_returns_401(self, client: TestClient) -> None:
        """Missing cookie on API request returns 401 JSON."""
        response = client.get("/api/documents")
        assert response.status_code == 401
        data = response.json()
        assert data["error_type"] == "AuthenticationRequired"

    def test_health_is_public(self, client: TestClient) -> None:
        """GET /api/health works without cookie."""
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_ready_is_public(self, client: TestClient) -> None:
        """GET /api/ready works without cookie."""
        response = client.get("/api/ready")
        assert response.status_code == 200

    def test_login_page_is_public(self, client: TestClient) -> None:
        """GET /login works without cookie."""
        response = client.get("/login")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_authenticated_request_passes(
        self,
        test_app: FastAPI,
        client: TestClient,
    ) -> None:
        """Request with valid cookie reaches the route handler."""
        _set_auth_cookie(client)
        mock_user_client = _make_mock_client()

        async def _override() -> MagicMock:  # type: ignore[misc]
            return mock_user_client

        test_app.dependency_overrides[get_user_client] = _override
        try:
            response = client.get("/")
            assert response.status_code == 200
        finally:
            test_app.dependency_overrides.clear()

    def test_htmx_unauthenticated_gets_hx_redirect(self, client: TestClient) -> None:
        """htmx request without cookie gets HX-Redirect header."""
        response = client.get(
            "/documents",
            headers={"HX-Request": "true"},
        )
        assert response.headers.get("HX-Redirect") == "/login"

    def test_static_is_public(self, client: TestClient) -> None:
        """Static file paths are exempt from auth."""
        # Static mount may or may not serve files in test,
        # but the middleware should not redirect
        response = client.get("/static/css/app.css")
        assert response.status_code != 302


# ---------------------------------------------------------------------------
# TestLoginPage
# ---------------------------------------------------------------------------


class TestLoginPage:
    """Tests for GET /login."""

    def test_renders_login_form(self, client: TestClient) -> None:
        """Login page contains token input and submit button."""
        response = client.get("/login")
        assert response.status_code == 200
        assert 'name="token"' in response.text
        assert "Sign In" in response.text

    def test_redirects_if_already_authenticated(self, client: TestClient) -> None:
        """Redirects to / if cookie already present."""
        _set_auth_cookie(client)
        response = client.get("/login", follow_redirects=False)
        assert response.status_code == 302
        assert response.headers["location"] == "/"


# ---------------------------------------------------------------------------
# TestLoginSubmit
# ---------------------------------------------------------------------------


class TestLoginSubmit:
    """Tests for POST /login."""

    def test_valid_token_sets_cookie_and_redirects(self, client: TestClient) -> None:
        """Valid token sets HttpOnly cookie and redirects to /."""
        with patch(
            "paperless_ngx_smart_ocr.web.routes.auth.validate_token",
            new_callable=AsyncMock,
            return_value=True,
        ):
            response = client.post(
                "/login",
                data={"token": "valid-token"},
                follow_redirects=False,
            )
        assert response.status_code == 302
        assert response.headers["location"] == "/"
        # Cookie should be set
        cookie = response.cookies.get(AUTH_COOKIE_NAME)
        assert cookie == "valid-token"

    def test_invalid_token_re_renders_with_error(self, client: TestClient) -> None:
        """Invalid token re-renders login page with error message."""
        with patch(
            "paperless_ngx_smart_ocr.web.routes.auth.validate_token",
            new_callable=AsyncMock,
            return_value=False,
        ):
            response = client.post(
                "/login",
                data={"token": "bad-token"},
            )
        assert response.status_code == 200
        assert "Invalid token" in response.text


# ---------------------------------------------------------------------------
# TestLogout
# ---------------------------------------------------------------------------


class TestLogout:
    """Tests for POST /logout."""

    def test_clears_cookie_and_redirects(self, client: TestClient) -> None:
        """Logout clears cookie and redirects to /login."""
        _set_auth_cookie(client)
        response = client.post("/logout", follow_redirects=False)
        assert response.status_code == 302
        assert response.headers["location"] == "/login"


# ---------------------------------------------------------------------------
# TestGetUserClient
# ---------------------------------------------------------------------------


class TestGetUserClient:
    """Tests for the get_user_client dependency."""

    async def test_creates_and_closes_client(self, test_app: FastAPI) -> None:
        """Dependency creates a client, yields it, then closes."""
        mock_client = _make_mock_client()

        with patch(
            "paperless_ngx_smart_ocr.paperless.PaperlessClient",
            return_value=mock_client,
        ):
            request = MagicMock()
            request.app.state.settings.paperless.url = "http://localhost:8000"
            request.state.paperless_token = "user-token"

            from paperless_ngx_smart_ocr.web.auth import (
                get_user_client,
            )

            gen = get_user_client(request)
            client = await gen.__anext__()
            assert client is mock_client

            # Cleanup
            with contextlib.suppress(StopAsyncIteration):
                await gen.__anext__()

            mock_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# TestMakeJobCoroutine
# ---------------------------------------------------------------------------


class TestMakeJobCoroutine:
    """Tests for make_job_coroutine."""

    async def test_creates_client_and_processes(self) -> None:
        """Job coroutine creates its own client and processes."""
        mock_result = MagicMock()
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        settings = _make_settings()

        with (
            patch(
                "paperless_ngx_smart_ocr.paperless.PaperlessClient",
                return_value=mock_client,
            ),
            patch(
                "paperless_ngx_smart_ocr.pipeline.orchestrator.process_document",
                new_callable=AsyncMock,
                return_value=mock_result,
            ) as mock_process,
        ):
            from paperless_ngx_smart_ocr.web.auth import (
                make_job_coroutine,
            )

            coro = make_job_coroutine(
                42,
                settings=settings,
                base_url="http://localhost:8000",
                token="user-token",
            )
            result = await coro

        assert result is mock_result
        mock_process.assert_called_once_with(
            42,
            settings=settings,
            client=mock_client,
            force=True,
        )
        mock_client.__aexit__.assert_called_once()
