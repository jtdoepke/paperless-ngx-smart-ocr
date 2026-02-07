"""Async HTTP client for the Paperless-ngx API."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator  # noqa: TC003
from contextlib import asynccontextmanager
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Any, Self

import httpx
import structlog

from paperless_ngx_smart_ocr.paperless.exceptions import (
    PaperlessAuthenticationError,
    PaperlessConnectionError,
    PaperlessError,
    PaperlessNotFoundError,
    PaperlessRateLimitError,
    PaperlessServerError,
    PaperlessValidationError,
)
from paperless_ngx_smart_ocr.paperless.models import (
    Correspondent,
    Document,
    DocumentMetadata,
    DocumentType,
    DocumentUpdate,
    PaginatedResponse,
    StoragePath,
    Tag,
    TagCreate,
    TaskState,
    TaskStatus,
)


if TYPE_CHECKING:
    from collections.abc import Mapping


__all__ = ["PaperlessClient"]


class PaperlessClient:
    """Async client for interacting with the Paperless-ngx REST API.

    This client provides methods for document CRUD operations, tag management,
    and file upload/download with retry logic and rate limiting support.

    Example:
        ```python
        async with PaperlessClient(
            base_url="http://paperless:8000",
            token="your-api-token",
        ) as client:
            docs = await client.list_documents(tags_include=[1, 2])
            for doc in docs.results:
                print(doc.title)
        ```

    Attributes:
        base_url: The base URL of the Paperless-ngx instance.
        timeout: Default timeout for requests.
        max_retries: Maximum number of retry attempts for transient errors.
    """

    DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
    DEFAULT_MAX_RETRIES = 3
    RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
    API_VERSION = "9"

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout: httpx.Timeout | None = None,
        max_retries: int | None = None,
        transport: httpx.AsyncHTTPTransport | None = None,
    ) -> None:
        """Initialize the Paperless-ngx client.

        Args:
            base_url: Base URL of the Paperless-ngx instance
                (e.g., "http://localhost:8000").
            token: API authentication token.
            timeout: Optional custom timeout configuration.
            max_retries: Maximum retry attempts for transient errors (default: 3).
            transport: Optional custom transport for testing or advanced config.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.max_retries = (
            max_retries if max_retries is not None else self.DEFAULT_MAX_RETRIES
        )
        self._token = token
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._logger = structlog.get_logger(__name__)

    @property
    def _headers(self) -> dict[str, str]:
        """Default headers for API requests."""
        return {
            "Authorization": f"Token {self._token}",
            "Accept": f"application/json; version={self.API_VERSION}",
            "Content-Type": "application/json",
        }

    async def __aenter__(self) -> Self:
        """Enter async context and create HTTP client."""
        await self._ensure_client()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit async context and close HTTP client."""
        await self.close()

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure the HTTP client is initialized."""
        if self._client is None or self._client.is_closed:
            transport = self._transport or httpx.AsyncHTTPTransport(retries=1)
            self._client = httpx.AsyncClient(
                base_url=f"{self.base_url}/api",
                headers=self._headers,
                timeout=self.timeout,
                transport=transport,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # -------------------------------------------------------------------------
    # Core Request Methods with Retry Logic
    # -------------------------------------------------------------------------

    async def _request(  # noqa: C901, PLR0913
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any | None = None,  # noqa: ANN401
        data: Mapping[str, Any] | None = None,
        files: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: httpx.Timeout | None = None,  # noqa: ASYNC109
    ) -> httpx.Response:
        """Execute an HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, PATCH, DELETE).
            path: API endpoint path (relative to /api/).
            params: Query parameters.
            json: JSON body data.
            data: Form data.
            files: Multipart file uploads.
            headers: Additional headers (merged with defaults).
            timeout: Override default timeout.

        Returns:
            The HTTP response.

        Raises:
            PaperlessAuthenticationError: For 401/403 responses.
            PaperlessNotFoundError: For 404 responses.
            PaperlessRateLimitError: For 429 responses (after retries exhausted).
            PaperlessServerError: For 5xx responses (after retries exhausted).
            PaperlessValidationError: For 400 responses.
            PaperlessConnectionError: For connection failures.
        """
        client = await self._ensure_client()
        log = self._logger.bind(method=method, path=path)

        last_exception: Exception | None = None
        retry_after: float = 0.5  # Initial backoff

        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    log.debug(
                        "retrying_request",
                        attempt=attempt,
                        max_retries=self.max_retries,
                    )
                    await asyncio.sleep(retry_after)

                request_headers = dict(self._headers)
                if headers:
                    request_headers.update(headers)

                # Remove Content-Type for multipart uploads
                if files:
                    request_headers.pop("Content-Type", None)

                response = await client.request(
                    method,
                    path,
                    params=dict(params) if params else None,
                    json=json,
                    data=dict(data) if data else None,
                    files=files,
                    headers=request_headers,
                    timeout=timeout or self.timeout,
                )

                log.debug(
                    "api_response",
                    status_code=response.status_code,
                    elapsed_ms=response.elapsed.total_seconds() * 1000,
                )

                # Handle rate limiting
                if response.status_code == 429:  # noqa: PLR2004
                    retry_after = self._parse_retry_after(response, retry_after)
                    if attempt < self.max_retries:
                        continue
                    raise PaperlessRateLimitError(
                        retry_after=retry_after,
                        response=response,
                    )

                # Handle retryable server errors
                if response.status_code in self.RETRY_STATUS_CODES:
                    retry_after = min(retry_after * 2, 30.0)
                    if attempt < self.max_retries:
                        continue
                    raise PaperlessServerError(  # noqa: TRY003
                        f"Server error: {response.status_code}",  # noqa: EM102
                        response=response,
                    )

                # Handle non-retryable errors
                self._raise_for_status(response)
                return response  # noqa: TRY300

            except httpx.ConnectError as exc:
                last_exception = exc
                retry_after = min(retry_after * 2, 30.0)
                if attempt < self.max_retries:
                    log.warning("connection_error", error=str(exc), attempt=attempt)
                    continue
                raise PaperlessConnectionError(cause=exc) from exc

            except httpx.TimeoutException as exc:
                last_exception = exc
                retry_after = min(retry_after * 2, 30.0)
                if attempt < self.max_retries:
                    log.warning("timeout_error", error=str(exc), attempt=attempt)
                    continue
                raise PaperlessConnectionError(
                    message="Request timed out",
                    cause=exc,
                ) from exc

        # Should not reach here, but handle edge case
        msg = "Max retries exceeded"
        raise PaperlessError(msg) from last_exception

    def _parse_retry_after(
        self,
        response: httpx.Response,
        default: float,
    ) -> float:
        """Parse Retry-After header value."""
        retry_after_header = response.headers.get("Retry-After")
        if retry_after_header:
            try:
                return float(retry_after_header)
            except ValueError:
                pass
        return min(default * 2, 60.0)  # Double with cap at 60s

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Raise appropriate exception for error status codes."""
        if response.is_success:
            return

        status = response.status_code

        if status in {401, 403}:
            raise PaperlessAuthenticationError(  # noqa: TRY003
                "Authentication failed",  # noqa: EM101
                response=response,
            )

        if status == 404:  # noqa: PLR2004
            raise PaperlessNotFoundError(
                "Resource",  # noqa: EM101
                "unknown",
                response=response,
            )

        if status == 400:  # noqa: PLR2004
            errors = None
            with contextlib.suppress(Exception):
                errors = response.json()
            raise PaperlessValidationError(  # noqa: TRY003
                "Validation error",  # noqa: EM101
                errors=errors if isinstance(errors, dict) else None,
                response=response,
            )

        if status >= 500:  # noqa: PLR2004
            raise PaperlessServerError(  # noqa: TRY003
                f"Server error: {status}",  # noqa: EM102
                response=response,
            )

        raise PaperlessError(  # noqa: TRY003
            f"Unexpected error: {status}",  # noqa: EM102
            response=response,
        )

    # -------------------------------------------------------------------------
    # Document Operations
    # -------------------------------------------------------------------------

    async def list_documents(  # noqa: PLR0913
        self,
        *,
        page: int = 1,
        page_size: int = 25,
        ordering: str | None = None,
        tags_include: list[int] | None = None,
        tags_exclude: list[int] | None = None,
        correspondent: int | None = None,
        document_type: int | None = None,
        query: str | None = None,
        truncate_content: bool = True,
        created_from: str | None = None,
        created_to: str | None = None,
        added_from: str | None = None,
        added_to: str | None = None,
    ) -> PaginatedResponse[Document]:
        """List documents with optional filtering.

        Args:
            page: Page number (1-indexed).
            page_size: Number of results per page.
            ordering: Field to order by (prefix with '-' for descending).
            tags_include: Only include documents with ALL of these tag IDs.
            tags_exclude: Exclude documents with ANY of these tag IDs.
            correspondent: Filter by correspondent ID.
            document_type: Filter by document type ID.
            query: Full-text search query.
            truncate_content: If True, truncate content in response.
            created_from: Filter by created date (YYYY-MM-DD), from.
            created_to: Filter by created date (YYYY-MM-DD), to.
            added_from: Filter by added date (YYYY-MM-DD), from.
            added_to: Filter by added date (YYYY-MM-DD), to.

        Returns:
            Paginated response containing documents.
        """
        params: dict[str, Any] = {
            "page": page,
            "page_size": page_size,
            "truncate_content": str(truncate_content).lower(),
        }

        if ordering:
            params["ordering"] = ordering
        if tags_include:
            params["tags__id__all"] = ",".join(str(t) for t in tags_include)
        if tags_exclude:
            params["tags__id__none"] = ",".join(str(t) for t in tags_exclude)
        if correspondent is not None:
            params["correspondent__id"] = correspondent
        if document_type is not None:
            params["document_type__id"] = document_type
        if query:
            params["query"] = query

        # Date range filters
        params.update(
            {
                k: v
                for k, v in {
                    "created__date__gt": created_from,
                    "created__date__lt": created_to,
                    "added__date__gt": added_from,
                    "added__date__lt": added_to,
                }.items()
                if v
            }
        )

        response = await self._request("GET", "/documents/", params=params)
        data = response.json()

        return PaginatedResponse[Document](
            count=data["count"],
            next=data.get("next"),
            previous=data.get("previous"),
            results=[Document.model_validate(d) for d in data["results"]],
            all=data.get("all", []),
        )

    async def get_document(self, document_id: int) -> Document:
        """Get a single document by ID.

        Args:
            document_id: The document ID.

        Returns:
            The document.

        Raises:
            PaperlessNotFoundError: If document not found.
        """
        try:
            response = await self._request("GET", f"/documents/{document_id}/")
            return Document.model_validate(response.json())
        except PaperlessNotFoundError:
            raise PaperlessNotFoundError("Document", document_id) from None  # noqa: EM101

    async def update_document(
        self,
        document_id: int,
        update: DocumentUpdate,
    ) -> Document:
        """Update a document (PATCH).

        Args:
            document_id: The document ID.
            update: Fields to update.

        Returns:
            The updated document.
        """
        # Only include non-None fields
        data = update.model_dump(exclude_none=True, by_alias=True)

        response = await self._request(
            "PATCH",
            f"/documents/{document_id}/",
            json=data,
        )
        return Document.model_validate(response.json())

    async def get_document_metadata(self, document_id: int) -> DocumentMetadata:
        """Get document metadata.

        Args:
            document_id: The document ID.

        Returns:
            Document metadata including checksums, sizes, and file info.
        """
        response = await self._request("GET", f"/documents/{document_id}/metadata/")
        return DocumentMetadata.model_validate(response.json())

    async def get_archive_filename(
        self,
        document_id: int,
    ) -> str | None:
        """Get the archive media filename for a document.

        Args:
            document_id: The document ID.

        Returns:
            The relative archive filename within ARCHIVE_DIR,
            or ``None`` if the document has no archive version.
        """
        metadata = await self.get_document_metadata(document_id)
        return metadata.archive_media_filename

    @asynccontextmanager
    async def download_document(
        self,
        document_id: int,
        *,
        original: bool = True,
    ) -> AsyncIterator[httpx.Response]:
        """Download document file as a streaming response.

        Args:
            document_id: The document ID.
            original: If True, download original file;
                if False, download archive version.

        Yields:
            Streaming HTTP response for reading file content.

        Example:
            ```python
            async with client.download_document(123) as response:
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    file.write(chunk)
            ```
        """
        endpoint = "download" if original else "preview"
        client = await self._ensure_client()

        async with client.stream(
            "GET",
            f"/documents/{document_id}/{endpoint}/",
        ) as response:
            self._raise_for_status(response)
            yield response

    async def download_document_to_path(
        self,
        document_id: int,
        dest_path: Path,
        *,
        original: bool = True,
        chunk_size: int = 65536,
    ) -> Path:
        """Download document to a file path.

        Args:
            document_id: The document ID.
            dest_path: Destination file path.
            original: If True, download original; if False, download archive.
            chunk_size: Size of chunks for streaming download.

        Returns:
            The destination path.
        """
        async with self.download_document(document_id, original=original) as response:
            with dest_path.open("wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                    f.write(chunk)
        return dest_path

    async def upload_document(  # noqa: PLR0913
        self,
        file_path: Path,
        *,
        title: str | None = None,
        correspondent: int | None = None,
        document_type: int | None = None,
        tags: list[int] | None = None,
        created: str | None = None,
        archive_serial_number: int | None = None,
    ) -> str:
        """Upload a new document.

        Args:
            file_path: Path to the document file.
            title: Document title (defaults to filename).
            correspondent: Correspondent ID.
            document_type: Document type ID.
            tags: List of tag IDs.
            created: Created date (YYYY-MM-DD format).
            archive_serial_number: ASN for the document.

        Returns:
            Task ID for tracking the upload.
        """
        data: dict[str, Any] = {}
        if title:
            data["title"] = title
        if correspondent is not None:
            data["correspondent"] = correspondent
        if document_type is not None:
            data["document_type"] = document_type
        if tags:
            data["tags"] = tags
        if created:
            data["created"] = created
        if archive_serial_number is not None:
            data["archive_serial_number"] = archive_serial_number

        with file_path.open("rb") as f:
            files = {"document": (file_path.name, f, "application/pdf")}
            response = await self._request(
                "POST",
                "/documents/post_document/",
                data=data,
                files=files,
                timeout=httpx.Timeout(300.0),  # 5 minute timeout for uploads
            )

        # Response is task ID as plain text or JSON
        try:
            return str(response.json().get("task_id", response.text))
        except Exception:  # noqa: BLE001
            text: str = response.text.strip()
            return text

    # -------------------------------------------------------------------------
    # Tag Operations
    # -------------------------------------------------------------------------

    async def list_tags(
        self,
        *,
        page: int = 1,
        page_size: int = 100,
    ) -> PaginatedResponse[Tag]:
        """List all tags.

        Args:
            page: Page number.
            page_size: Results per page.

        Returns:
            Paginated response of tags.
        """
        response = await self._request(
            "GET",
            "/tags/",
            params={"page": page, "page_size": page_size},
        )
        data = response.json()
        return PaginatedResponse[Tag](
            count=data["count"],
            next=data.get("next"),
            previous=data.get("previous"),
            results=[Tag.model_validate(t) for t in data["results"]],
        )

    async def get_tag(self, tag_id: int) -> Tag:
        """Get a single tag by ID.

        Args:
            tag_id: The tag ID.

        Returns:
            The tag.
        """
        response = await self._request("GET", f"/tags/{tag_id}/")
        return Tag.model_validate(response.json())

    async def get_tag_by_name(self, name: str) -> Tag | None:
        """Find a tag by its name.

        Args:
            name: The tag name to search for.

        Returns:
            The tag if found, None otherwise.
        """
        response = await self._request(
            "GET",
            "/tags/",
            params={"name__iexact": name, "page_size": 1},
        )
        data = response.json()
        if data["results"]:
            return Tag.model_validate(data["results"][0])
        return None

    async def create_tag(self, tag: TagCreate) -> Tag:
        """Create a new tag.

        Args:
            tag: Tag creation data.

        Returns:
            The created tag.
        """
        response = await self._request(
            "POST",
            "/tags/",
            json=tag.model_dump(),
        )
        return Tag.model_validate(response.json())

    async def ensure_tag(self, name: str, **kwargs: Any) -> Tag:  # noqa: ANN401
        """Get or create a tag by name.

        Args:
            name: Tag name.
            **kwargs: Additional tag creation parameters (color, text_color, etc.).

        Returns:
            The existing or newly created tag.
        """
        existing = await self.get_tag_by_name(name)
        if existing:
            return existing
        return await self.create_tag(TagCreate(name=name, **kwargs))

    async def add_tags_to_document(
        self,
        document_id: int,
        tag_ids: list[int],
    ) -> Document:
        """Add tags to a document.

        Args:
            document_id: The document ID.
            tag_ids: Tag IDs to add.

        Returns:
            The updated document.
        """
        doc = await self.get_document(document_id)
        current_tags = set(doc.tags)
        new_tags = list(current_tags | set(tag_ids))

        return await self.update_document(
            document_id,
            DocumentUpdate(tags=new_tags),
        )

    async def remove_tags_from_document(
        self,
        document_id: int,
        tag_ids: list[int],
    ) -> Document:
        """Remove tags from a document.

        Args:
            document_id: The document ID.
            tag_ids: Tag IDs to remove.

        Returns:
            The updated document.
        """
        doc = await self.get_document(document_id)
        current_tags = set(doc.tags)
        new_tags = list(current_tags - set(tag_ids))

        return await self.update_document(
            document_id,
            DocumentUpdate(tags=new_tags),
        )

    # -------------------------------------------------------------------------
    # Correspondent Operations
    # -------------------------------------------------------------------------

    async def list_correspondents(
        self,
        *,
        page: int = 1,
        page_size: int = 100,
    ) -> PaginatedResponse[Correspondent]:
        """List all correspondents.

        Args:
            page: Page number.
            page_size: Results per page.

        Returns:
            Paginated response of correspondents.
        """
        response = await self._request(
            "GET",
            "/correspondents/",
            params={"page": page, "page_size": page_size},
        )
        data = response.json()
        return PaginatedResponse[Correspondent](
            count=data["count"],
            next=data.get("next"),
            previous=data.get("previous"),
            results=[Correspondent.model_validate(c) for c in data["results"]],
        )

    async def get_correspondent(
        self,
        correspondent_id: int,
    ) -> Correspondent:
        """Get a single correspondent by ID.

        Args:
            correspondent_id: The correspondent ID.

        Returns:
            The correspondent.
        """
        response = await self._request(
            "GET",
            f"/correspondents/{correspondent_id}/",
        )
        return Correspondent.model_validate(response.json())

    # -------------------------------------------------------------------------
    # Document Type Operations
    # -------------------------------------------------------------------------

    async def list_document_types(
        self,
        *,
        page: int = 1,
        page_size: int = 100,
    ) -> PaginatedResponse[DocumentType]:
        """List all document types.

        Args:
            page: Page number.
            page_size: Results per page.

        Returns:
            Paginated response of document types.
        """
        response = await self._request(
            "GET",
            "/document_types/",
            params={"page": page, "page_size": page_size},
        )
        data = response.json()
        return PaginatedResponse[DocumentType](
            count=data["count"],
            next=data.get("next"),
            previous=data.get("previous"),
            results=[DocumentType.model_validate(d) for d in data["results"]],
        )

    async def get_document_type(
        self,
        document_type_id: int,
    ) -> DocumentType:
        """Get a single document type by ID.

        Args:
            document_type_id: The document type ID.

        Returns:
            The document type.
        """
        response = await self._request(
            "GET",
            f"/document_types/{document_type_id}/",
        )
        return DocumentType.model_validate(response.json())

    # -------------------------------------------------------------------------
    # Storage Path Operations
    # -------------------------------------------------------------------------

    async def get_storage_path(
        self,
        storage_path_id: int,
    ) -> StoragePath:
        """Get a single storage path by ID.

        Args:
            storage_path_id: The storage path ID.

        Returns:
            The storage path.
        """
        response = await self._request(
            "GET",
            f"/storage_paths/{storage_path_id}/",
        )
        return StoragePath.model_validate(response.json())

    # -------------------------------------------------------------------------
    # Task Status
    # -------------------------------------------------------------------------

    async def get_task_status(self, task_id: str) -> TaskStatus:
        """Get the status of an async task.

        Args:
            task_id: The task ID (from upload, etc.).

        Returns:
            Task status information.
        """
        response = await self._request("GET", f"/tasks/?task_id={task_id}")
        data = response.json()
        # Tasks endpoint returns a list
        if data and isinstance(data, list) and len(data) > 0:
            return TaskStatus.model_validate(data[0])
        # If no task found, return a pending status
        return TaskStatus(id=task_id, status=TaskState.PENDING)

    async def wait_for_task(
        self,
        task_id: str,
        *,
        timeout: float = 300.0,  # noqa: ASYNC109
        poll_interval: float = 2.0,
    ) -> TaskStatus:
        """Wait for an async task to complete.

        Args:
            task_id: The task ID.
            timeout: Maximum time to wait in seconds.
            poll_interval: Time between status checks.

        Returns:
            Final task status.

        Raises:
            TimeoutError: If task doesn't complete within timeout.
        """
        start = asyncio.get_event_loop().time()

        while True:
            status = await self.get_task_status(task_id)

            if status.status in {
                TaskState.SUCCESS,
                TaskState.FAILURE,
                TaskState.REVOKED,
            }:
                return status

            elapsed = asyncio.get_event_loop().time() - start
            if elapsed >= timeout:
                msg = f"Task {task_id} did not complete within {timeout}s"
                raise TimeoutError(msg)

            await asyncio.sleep(poll_interval)

    # -------------------------------------------------------------------------
    # Health Check
    # -------------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Check if Paperless-ngx is reachable and authenticated.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            # Use a lightweight endpoint
            await self._request("GET", "/tags/", params={"page_size": 1})
            return True  # noqa: TRY300
        except PaperlessError:
            return False
