"""Unit tests for the Paperless-ngx API client."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator  # noqa: TC003
from pathlib import Path  # noqa: TC003
from typing import Any

import httpx
import pytest
import respx  # noqa: TC002

from paperless_ngx_smart_ocr.paperless import (
    Correspondent,
    Document,
    DocumentMetadata,
    DocumentType,
    DocumentUpdate,
    PaginatedResponse,
    PaperlessAuthenticationError,
    PaperlessClient,
    PaperlessConnectionError,
    PaperlessNotFoundError,
    PaperlessRateLimitError,
    PaperlessServerError,
    PaperlessValidationError,
    StoragePath,
    Tag,
    TagCreate,
    TaskState,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_url() -> str:
    """Base URL for test client."""
    return "http://paperless.test:8000"


@pytest.fixture
def api_token() -> str:
    """API token for test client."""
    return "test-token-12345"


@pytest.fixture
async def client(
    base_url: str,
    api_token: str,
) -> AsyncGenerator[PaperlessClient, None]:
    """Create a test client with reduced retries."""
    async with PaperlessClient(base_url, api_token, max_retries=1) as c:
        yield c


@pytest.fixture
def sample_document_json() -> dict[str, Any]:
    """Sample document JSON response."""
    return {
        "id": 123,
        "title": "Test Document",
        "content": "This is test content",
        "correspondent": 1,
        "document_type": 2,
        "storage_path": None,
        "tags": [1, 2, 3],
        "created": "2024-01-15T10:30:00Z",
        "created_date": "2024-01-15",
        "modified": "2024-01-16T14:00:00Z",
        "added": "2024-01-15T10:30:00Z",
        "archive_serial_number": None,
        "original_file_name": "test.pdf",
        "archived_file_name": None,
        "owner": 1,
        "user_can_change": True,
        "is_shared_by_requester": False,
        "notes": [],
        "custom_fields": [],
    }


@pytest.fixture
def sample_tag_json() -> dict[str, Any]:
    """Sample tag JSON response."""
    return {
        "id": 1,
        "slug": "smart-ocr-pending",
        "name": "smart-ocr:pending",
        "color": "#a6cee3",
        "text_color": "#000000",
        "is_inbox_tag": False,
        "document_count": 5,
        "owner": 1,
        "user_can_change": True,
    }


# ---------------------------------------------------------------------------
# Document Tests
# ---------------------------------------------------------------------------


class TestListDocuments:
    """Tests for list_documents method."""

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_list_documents(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
        sample_document_json: dict[str, Any],
    ) -> None:
        """Test listing documents with pagination."""
        respx_mock.get("/api/documents/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "count": 1,
                    "next": None,
                    "previous": None,
                    "results": [sample_document_json],
                    "all": [123],
                },
            )
        )

        result = await client.list_documents()

        assert isinstance(result, PaginatedResponse)
        assert result.count == 1
        assert len(result.results) == 1
        assert result.results[0].id == 123
        assert result.results[0].title == "Test Document"

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_list_documents_with_tag_filters(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
        sample_document_json: dict[str, Any],
    ) -> None:
        """Test listing documents with tag filters."""
        route = respx_mock.get("/api/documents/").mock(
            return_value=httpx.Response(
                200,
                json={"count": 1, "results": [sample_document_json]},
            )
        )

        await client.list_documents(tags_include=[1, 2], tags_exclude=[3])

        assert route.called
        request = route.calls.last.request
        url_str = str(request.url)
        # URL-encoded comma is %2C
        assert "tags__id__all=1%2C2" in url_str or "tags__id__all=1,2" in url_str
        assert "tags__id__none=3" in url_str

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_list_documents_with_query(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
        sample_document_json: dict[str, Any],
    ) -> None:
        """Test listing documents with search query."""
        route = respx_mock.get("/api/documents/").mock(
            return_value=httpx.Response(
                200,
                json={"count": 1, "results": [sample_document_json]},
            )
        )

        await client.list_documents(query="invoice")

        assert route.called
        request = route.calls.last.request
        assert "query=invoice" in str(request.url)


class TestGetDocument:
    """Tests for get_document method."""

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_get_document(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
        sample_document_json: dict[str, Any],
    ) -> None:
        """Test getting a single document."""
        respx_mock.get("/api/documents/123/").mock(
            return_value=httpx.Response(200, json=sample_document_json)
        )

        doc = await client.get_document(123)

        assert isinstance(doc, Document)
        assert doc.id == 123
        assert doc.title == "Test Document"
        assert doc.tags == [1, 2, 3]

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_get_document_not_found(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test getting a non-existent document."""
        respx_mock.get("/api/documents/999/").mock(
            return_value=httpx.Response(404, json={"detail": "Not found."})
        )

        with pytest.raises(PaperlessNotFoundError) as exc_info:
            await client.get_document(999)

        assert exc_info.value.resource_type == "Document"
        assert exc_info.value.resource_id == 999


class TestUpdateDocument:
    """Tests for update_document method."""

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_update_document_content(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
        sample_document_json: dict[str, Any],
    ) -> None:
        """Test updating document content field."""
        updated = {**sample_document_json, "content": "# New Markdown Content"}
        respx_mock.patch("/api/documents/123/").mock(
            return_value=httpx.Response(200, json=updated)
        )

        result = await client.update_document(
            123,
            DocumentUpdate(content="# New Markdown Content"),
        )

        assert result.content == "# New Markdown Content"

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_update_document_tags(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
        sample_document_json: dict[str, Any],
    ) -> None:
        """Test updating document tags."""
        updated = {**sample_document_json, "tags": [1, 2, 3, 4]}
        route = respx_mock.patch("/api/documents/123/").mock(
            return_value=httpx.Response(200, json=updated)
        )

        result = await client.update_document(
            123,
            DocumentUpdate(tags=[1, 2, 3, 4]),
        )

        assert result.tags == [1, 2, 3, 4]
        # Verify request body
        request = route.calls.last.request
        body = json.loads(request.content)
        assert body["tags"] == [1, 2, 3, 4]


class TestDownloadDocument:
    """Tests for download_document method."""

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_download_document(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test streaming document download."""
        pdf_content = b"%PDF-1.4 test content"
        respx_mock.get("/api/documents/123/download/").mock(
            return_value=httpx.Response(
                200,
                content=pdf_content,
                headers={"Content-Type": "application/pdf"},
            )
        )

        chunks = []
        async with client.download_document(123) as response:
            async for chunk in response.aiter_bytes():
                chunks.append(chunk)  # noqa: PERF401

        assert b"".join(chunks) == pdf_content

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_download_document_preview(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test downloading document preview (archive version)."""
        pdf_content = b"%PDF-1.4 archive content"
        respx_mock.get("/api/documents/123/preview/").mock(
            return_value=httpx.Response(
                200,
                content=pdf_content,
                headers={"Content-Type": "application/pdf"},
            )
        )

        chunks = []
        async with client.download_document(123, original=False) as response:
            async for chunk in response.aiter_bytes():
                chunks.append(chunk)  # noqa: PERF401

        assert b"".join(chunks) == pdf_content


# ---------------------------------------------------------------------------
# Tag Tests
# ---------------------------------------------------------------------------


class TestListTags:
    """Tests for list_tags method."""

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_list_tags(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
        sample_tag_json: dict[str, Any],
    ) -> None:
        """Test listing tags."""
        respx_mock.get("/api/tags/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "count": 1,
                    "next": None,
                    "previous": None,
                    "results": [sample_tag_json],
                },
            )
        )

        result = await client.list_tags()

        assert result.count == 1
        assert len(result.results) == 1
        assert result.results[0].name == "smart-ocr:pending"


class TestCreateTag:
    """Tests for create_tag method."""

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_create_tag(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
        sample_tag_json: dict[str, Any],
    ) -> None:
        """Test creating a new tag."""
        respx_mock.post("/api/tags/").mock(
            return_value=httpx.Response(201, json=sample_tag_json)
        )

        tag = await client.create_tag(TagCreate(name="smart-ocr:pending"))

        assert isinstance(tag, Tag)
        assert tag.name == "smart-ocr:pending"


class TestEnsureTag:
    """Tests for ensure_tag method."""

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_ensure_tag_exists(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
        sample_tag_json: dict[str, Any],
    ) -> None:
        """Test ensure_tag when tag already exists."""
        respx_mock.get("/api/tags/").mock(
            return_value=httpx.Response(
                200,
                json={"count": 1, "results": [sample_tag_json]},
            )
        )

        tag = await client.ensure_tag("smart-ocr:pending")

        assert tag.name == "smart-ocr:pending"
        assert tag.id == 1

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_ensure_tag_creates(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
        sample_tag_json: dict[str, Any],
    ) -> None:
        """Test ensure_tag creates tag when it doesn't exist."""
        # First call returns empty results
        respx_mock.get("/api/tags/").mock(
            return_value=httpx.Response(
                200,
                json={"count": 0, "results": []},
            )
        )
        # Second call creates the tag
        respx_mock.post("/api/tags/").mock(
            return_value=httpx.Response(201, json=sample_tag_json)
        )

        tag = await client.ensure_tag("smart-ocr:pending")

        assert tag.name == "smart-ocr:pending"


class TestAddRemoveTags:
    """Tests for add_tags_to_document and remove_tags_from_document."""

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_add_tags_to_document(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
        sample_document_json: dict[str, Any],
    ) -> None:
        """Test adding tags to a document."""
        # First call to get document
        respx_mock.get("/api/documents/123/").mock(
            return_value=httpx.Response(200, json=sample_document_json)
        )

        # Second call to patch document
        updated = {**sample_document_json, "tags": [1, 2, 3, 4]}
        respx_mock.patch("/api/documents/123/").mock(
            return_value=httpx.Response(200, json=updated)
        )

        result = await client.add_tags_to_document(123, [4])

        assert 4 in result.tags

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_remove_tags_from_document(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
        sample_document_json: dict[str, Any],
    ) -> None:
        """Test removing tags from a document."""
        # First call to get document
        respx_mock.get("/api/documents/123/").mock(
            return_value=httpx.Response(200, json=sample_document_json)
        )

        # Second call to patch document (remove tag 3)
        updated = {**sample_document_json, "tags": [1, 2]}
        respx_mock.patch("/api/documents/123/").mock(
            return_value=httpx.Response(200, json=updated)
        )

        result = await client.remove_tags_from_document(123, [3])

        assert 3 not in result.tags
        assert result.tags == [1, 2]


# ---------------------------------------------------------------------------
# Task Status Tests
# ---------------------------------------------------------------------------


class TestTaskStatus:
    """Tests for task status methods."""

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_get_task_status(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test getting task status."""
        respx_mock.get("/api/tasks/").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "task_id": "abc-123",
                        "task_file_name": "test.pdf",
                        "status": "SUCCESS",
                        "result": None,
                        "acknowledged": False,
                    }
                ],
            )
        )

        status = await client.get_task_status("abc-123")

        assert isinstance(status, TaskStatus)
        assert status.id == "abc-123"
        assert status.status == TaskState.SUCCESS

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_get_task_status_not_found(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test getting status of unknown task returns pending."""
        respx_mock.get("/api/tasks/").mock(return_value=httpx.Response(200, json=[]))

        status = await client.get_task_status("unknown-task")

        assert status.status == TaskState.PENDING


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_authentication_error_401(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test handling of 401 authentication errors."""
        respx_mock.get("/api/documents/").mock(
            return_value=httpx.Response(401, json={"detail": "Invalid token."})
        )

        with pytest.raises(PaperlessAuthenticationError):
            await client.list_documents()

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_authentication_error_403(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test handling of 403 forbidden errors."""
        respx_mock.get("/api/documents/").mock(
            return_value=httpx.Response(403, json={"detail": "Permission denied."})
        )

        with pytest.raises(PaperlessAuthenticationError):
            await client.list_documents()

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_validation_error(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test handling of 400 validation errors."""
        respx_mock.patch("/api/documents/123/").mock(
            return_value=httpx.Response(
                400,
                json={"title": ["This field is required."]},
            )
        )

        with pytest.raises(PaperlessValidationError) as exc_info:
            await client.update_document(123, DocumentUpdate(title=""))

        assert exc_info.value.errors is not None

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_rate_limit_error(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test handling of 429 rate limit errors."""
        respx_mock.get("/api/documents/").mock(
            return_value=httpx.Response(
                429,
                headers={"Retry-After": "30"},
            )
        )

        with pytest.raises(PaperlessRateLimitError) as exc_info:
            await client.list_documents()

        assert exc_info.value.retry_after is not None

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_server_error(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test handling of 5xx server errors after retries."""
        respx_mock.get("/api/documents/").mock(return_value=httpx.Response(503))

        with pytest.raises(PaperlessServerError):
            await client.list_documents()

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_connection_error(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test handling of connection errors."""
        respx_mock.get("/api/documents/").mock(side_effect=httpx.ConnectError("Failed"))

        with pytest.raises(PaperlessConnectionError):
            await client.list_documents()


# ---------------------------------------------------------------------------
# Retry Logic Tests
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """Tests for retry behavior."""

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_retry_on_server_error(
        self,
        base_url: str,
        api_token: str,
        respx_mock: respx.MockRouter,
        sample_document_json: dict[str, Any],
    ) -> None:
        """Test retry logic on server errors."""
        # First two calls fail, third succeeds
        route = respx_mock.get("/api/documents/")
        route.side_effect = [
            httpx.Response(503),
            httpx.Response(503),
            httpx.Response(200, json={"count": 1, "results": [sample_document_json]}),
        ]

        async with PaperlessClient(base_url, api_token, max_retries=3) as client:
            result = await client.list_documents()

        assert result.count == 1
        assert route.call_count == 3

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_retry_exhausted(
        self,
        base_url: str,
        api_token: str,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test that retries eventually give up."""
        respx_mock.get("/api/documents/").mock(return_value=httpx.Response(503))

        async with PaperlessClient(base_url, api_token, max_retries=2) as client:
            with pytest.raises(PaperlessServerError):
                await client.list_documents()


# ---------------------------------------------------------------------------
# Health Check Tests
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """Tests for health_check method."""

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_health_check_success(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test successful health check."""
        respx_mock.get("/api/tags/").mock(
            return_value=httpx.Response(
                200,
                json={"count": 0, "results": []},
            )
        )

        result = await client.health_check()

        assert result is True

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_health_check_failure_auth(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test failed health check due to auth error."""
        respx_mock.get("/api/tags/").mock(return_value=httpx.Response(401))

        result = await client.health_check()

        assert result is False

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_health_check_failure_connection(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test failed health check due to connection error."""
        respx_mock.get("/api/tags/").mock(side_effect=httpx.ConnectError("Failed"))

        result = await client.health_check()

        assert result is False


# ---------------------------------------------------------------------------
# Model Tests
# ---------------------------------------------------------------------------


class TestModels:
    """Tests for Pydantic models."""

    def test_document_model_validation(
        self,
        sample_document_json: dict[str, Any],
    ) -> None:
        """Test Document model validates correctly."""
        doc = Document.model_validate(sample_document_json)

        assert doc.id == 123
        assert doc.title == "Test Document"
        assert doc.tags == [1, 2, 3]
        assert doc.content == "This is test content"

    def test_document_model_ignores_extra_fields(
        self,
        sample_document_json: dict[str, Any],
    ) -> None:
        """Test Document model ignores unknown fields from API."""
        data = {**sample_document_json, "unknown_field": "should be ignored"}
        doc = Document.model_validate(data)

        assert doc.id == 123
        assert not hasattr(doc, "unknown_field")

    def test_tag_model_validation(
        self,
        sample_tag_json: dict[str, Any],
    ) -> None:
        """Test Tag model validates correctly."""
        tag = Tag.model_validate(sample_tag_json)

        assert tag.id == 1
        assert tag.name == "smart-ocr:pending"
        assert tag.color == "#a6cee3"

    def test_document_update_excludes_none(self) -> None:
        """Test DocumentUpdate only includes non-None fields."""
        update = DocumentUpdate(content="New content")
        data = update.model_dump(exclude_none=True)

        assert data == {"content": "New content"}
        assert "title" not in data
        assert "tags" not in data

    def test_task_status_alias(self) -> None:
        """Test TaskStatus handles task_id alias."""
        status = TaskStatus.model_validate(
            {
                "task_id": "abc-123",
                "status": "SUCCESS",
            }
        )

        assert status.id == "abc-123"
        assert status.status == TaskState.SUCCESS


# ---------------------------------------------------------------------------
# Additional Coverage Tests
# ---------------------------------------------------------------------------


class TestTimeoutHandling:
    """Tests for timeout error handling."""

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_timeout_error(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test handling of timeout errors."""
        respx_mock.get("/api/documents/").mock(
            side_effect=httpx.TimeoutException("Request timed out")
        )

        with pytest.raises(PaperlessConnectionError) as exc_info:
            await client.list_documents()

        assert "timed out" in str(exc_info.value).lower()


class TestWaitForTask:
    """Tests for wait_for_task method."""

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_wait_for_task_success(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test waiting for a task that completes successfully."""
        # First call: pending, second call: success
        route = respx_mock.get("/api/tasks/")
        route.side_effect = [
            httpx.Response(
                200,
                json=[{"task_id": "abc-123", "status": "PENDING"}],
            ),
            httpx.Response(
                200,
                json=[{"task_id": "abc-123", "status": "SUCCESS", "result": "Done"}],
            ),
        ]

        status = await client.wait_for_task("abc-123", poll_interval=0.01)

        assert status.status == TaskState.SUCCESS
        assert route.call_count == 2

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_wait_for_task_failure(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test waiting for a task that fails."""
        respx_mock.get("/api/tasks/").mock(
            return_value=httpx.Response(
                200,
                json=[{"task_id": "abc-123", "status": "FAILURE", "result": "Error"}],
            )
        )

        status = await client.wait_for_task("abc-123")

        assert status.status == TaskState.FAILURE

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_wait_for_task_timeout(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test wait_for_task times out."""
        respx_mock.get("/api/tasks/").mock(
            return_value=httpx.Response(
                200,
                json=[{"task_id": "abc-123", "status": "PENDING"}],
            )
        )

        with pytest.raises(TimeoutError):
            await client.wait_for_task("abc-123", timeout=0.05, poll_interval=0.01)


class TestDocumentMetadata:
    """Tests for get_document_metadata method."""

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_get_document_metadata(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test getting document metadata."""
        respx_mock.get("/api/documents/123/metadata/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "original_checksum": "abc123",
                    "original_size": 12345,
                    "original_mime_type": "application/pdf",
                    "has_archive_version": True,
                },
            )
        )

        metadata = await client.get_document_metadata(123)

        assert isinstance(metadata, DocumentMetadata)
        assert metadata.original_checksum == "abc123"
        assert metadata.original_size == 12345
        assert metadata.original_mime_type == "application/pdf"


class TestUploadDocument:
    """Tests for upload_document method."""

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_upload_document(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
        tmp_path: Path,
    ) -> None:
        """Test uploading a document."""
        # Create a temporary file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 test content")

        respx_mock.post("/api/documents/post_document/").mock(
            return_value=httpx.Response(
                200,
                json={"task_id": "upload-task-123"},
            )
        )

        task_id = await client.upload_document(
            test_file,
            title="Test Document",
            tags=[1, 2],
        )

        assert task_id == "upload-task-123"


class TestGetTag:
    """Tests for get_tag method."""

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_get_tag(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
        sample_tag_json: dict[str, Any],
    ) -> None:
        """Test getting a single tag by ID."""
        respx_mock.get("/api/tags/1/").mock(
            return_value=httpx.Response(200, json=sample_tag_json)
        )

        tag = await client.get_tag(1)

        assert tag.id == 1
        assert tag.name == "smart-ocr:pending"


# ---------------------------------------------------------------------------
# TestGetCorrespondent
# ---------------------------------------------------------------------------


class TestGetCorrespondent:
    """Tests for get_correspondent method."""

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_get_correspondent(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test getting a single correspondent by ID."""
        respx_mock.get("/api/correspondents/1/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": 1,
                    "slug": "acme-corp",
                    "name": "ACME Corp",
                    "match": "",
                    "matching_algorithm": 0,
                    "is_insensitive": True,
                    "document_count": 10,
                    "owner": 1,
                    "user_can_change": True,
                },
            ),
        )

        result = await client.get_correspondent(1)

        assert isinstance(result, Correspondent)
        assert result.id == 1
        assert result.name == "ACME Corp"

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_not_found(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Missing correspondent raises PaperlessNotFoundError."""
        respx_mock.get("/api/correspondents/999/").mock(
            return_value=httpx.Response(404),
        )

        with pytest.raises(PaperlessNotFoundError):
            await client.get_correspondent(999)


# ---------------------------------------------------------------------------
# TestGetDocumentType
# ---------------------------------------------------------------------------


class TestGetDocumentType:
    """Tests for get_document_type method."""

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_get_document_type(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test getting a single document type by ID."""
        respx_mock.get("/api/document_types/2/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": 2,
                    "slug": "invoice",
                    "name": "Invoice",
                    "match": "",
                    "matching_algorithm": 0,
                    "is_insensitive": True,
                    "document_count": 5,
                    "owner": 1,
                    "user_can_change": True,
                },
            ),
        )

        result = await client.get_document_type(2)

        assert isinstance(result, DocumentType)
        assert result.id == 2
        assert result.name == "Invoice"

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_not_found(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Missing document type raises PaperlessNotFoundError."""
        respx_mock.get("/api/document_types/999/").mock(
            return_value=httpx.Response(404),
        )

        with pytest.raises(PaperlessNotFoundError):
            await client.get_document_type(999)


# ---------------------------------------------------------------------------
# TestGetStoragePath
# ---------------------------------------------------------------------------


class TestGetStoragePath:
    """Tests for get_storage_path method."""

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_get_storage_path(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Test getting a single storage path by ID."""
        respx_mock.get("/api/storage_paths/3/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": 3,
                    "slug": "archive",
                    "name": "Archive",
                    "path": "archive/{created_year}/",
                    "match": "",
                    "matching_algorithm": 0,
                    "is_insensitive": True,
                    "document_count": 100,
                    "owner": 1,
                    "user_can_change": True,
                },
            ),
        )

        result = await client.get_storage_path(3)

        assert isinstance(result, StoragePath)
        assert result.id == 3
        assert result.name == "Archive"
        assert result.path == "archive/{created_year}/"

    @pytest.mark.respx(base_url="http://paperless.test:8000")
    async def test_not_found(
        self,
        client: PaperlessClient,
        respx_mock: respx.MockRouter,
    ) -> None:
        """Missing storage path raises PaperlessNotFoundError."""
        respx_mock.get("/api/storage_paths/999/").mock(
            return_value=httpx.Response(404),
        )

        with pytest.raises(PaperlessNotFoundError):
            await client.get_storage_path(999)
