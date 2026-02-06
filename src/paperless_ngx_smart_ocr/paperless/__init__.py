"""Paperless-ngx API client module.

This module provides an async HTTP client for interacting with the
Paperless-ngx REST API, including document and tag management,
file download/upload, and task status tracking.

Example:
    ```python
    from paperless_ngx_smart_ocr.paperless import PaperlessClient, DocumentUpdate

    async with PaperlessClient(
        base_url="http://paperless:8000",
        token="your-api-token",
    ) as client:
        # List documents with a specific tag
        pending_tag = await client.ensure_tag("smart-ocr:pending")
        docs = await client.list_documents(tags_include=[pending_tag.id])

        for doc in docs.results:
            # Download and process document
            async with client.download_document(doc.id) as response:
                pdf_bytes = await response.aread()

            # Update content with processed markdown
            await client.update_document(
                doc.id,
                DocumentUpdate(content="# Processed content"),
            )
    ```
"""

from __future__ import annotations

from paperless_ngx_smart_ocr.paperless.client import PaperlessClient
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


__all__ = [
    "Correspondent",
    "Document",
    "DocumentMetadata",
    "DocumentType",
    "DocumentUpdate",
    "PaginatedResponse",
    "PaperlessAuthenticationError",
    "PaperlessClient",
    "PaperlessConnectionError",
    "PaperlessError",
    "PaperlessNotFoundError",
    "PaperlessRateLimitError",
    "PaperlessServerError",
    "PaperlessValidationError",
    "StoragePath",
    "Tag",
    "TagCreate",
    "TaskState",
    "TaskStatus",
]
