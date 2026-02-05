"""Pydantic models for Paperless-ngx API responses."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


__all__ = [
    "Correspondent",
    "Document",
    "DocumentMetadata",
    "DocumentType",
    "DocumentUpdate",
    "PaginatedResponse",
    "Tag",
    "TagCreate",
    "TaskState",
    "TaskStatus",
]


class TaskState(StrEnum):
    """Celery task states used by Paperless-ngx."""

    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"
    REVOKED = "REVOKED"


class PaperlessBaseModel(BaseModel):
    """Base model with common configuration for all Paperless-ngx models."""

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        frozen=False,
        extra="ignore",  # Ignore unknown fields from API
    )


class Tag(PaperlessBaseModel):
    """Represents a tag in Paperless-ngx.

    Tags are used to categorize documents. This service uses tags like
    "smart-ocr:pending", "smart-ocr:completed" to track processing state.
    """

    id: int
    slug: str
    name: str
    color: str = Field(default="#a6cee3")
    text_color: str = Field(default="#000000")
    is_inbox_tag: bool = Field(default=False)
    document_count: int = Field(default=0)
    owner: int | None = None
    user_can_change: bool = Field(default=True)


class TagCreate(PaperlessBaseModel):
    """Model for creating a new tag."""

    name: str
    color: str = "#a6cee3"
    text_color: str = "#000000"
    is_inbox_tag: bool = False


class Correspondent(PaperlessBaseModel):
    """Represents a correspondent (sender/recipient) in Paperless-ngx."""

    id: int
    slug: str
    name: str
    match: str = ""
    matching_algorithm: int = Field(default=0)
    is_insensitive: bool = Field(default=True)
    document_count: int = Field(default=0)
    owner: int | None = None
    user_can_change: bool = Field(default=True)


class DocumentType(PaperlessBaseModel):
    """Represents a document type in Paperless-ngx."""

    id: int
    slug: str
    name: str
    match: str = ""
    matching_algorithm: int = Field(default=0)
    is_insensitive: bool = Field(default=True)
    document_count: int = Field(default=0)
    owner: int | None = None
    user_can_change: bool = Field(default=True)


class DocumentMetadata(PaperlessBaseModel):
    """Document metadata returned from the metadata endpoint.

    Contains file checksums, sizes, and MIME type information.
    """

    original_checksum: str | None = None
    original_size: int | None = None
    original_mime_type: str | None = None
    media_filename: str | None = None
    has_archive_version: bool = False
    original_metadata: list[dict[str, str]] = Field(default_factory=list)
    archive_checksum: str | None = None
    archive_size: int | None = None
    archive_metadata: list[dict[str, str]] = Field(default_factory=list)


class Document(PaperlessBaseModel):
    """Represents a document in Paperless-ngx.

    This is the primary model for interacting with documents. The content
    field contains the extracted text (or Markdown after Stage 2 processing).
    """

    id: int
    correspondent: int | None = None
    document_type: int | None = None
    storage_path: int | None = None
    title: str
    content: str = ""
    tags: list[int] = Field(default_factory=list)
    created: datetime
    created_date: str  # Date string YYYY-MM-DD
    modified: datetime
    added: datetime
    archive_serial_number: int | None = None
    original_file_name: str = Field(default="")
    archived_file_name: str | None = None
    owner: int | None = None
    user_can_change: bool = Field(default=True)
    is_shared_by_requester: bool = Field(default=False)
    notes: list[dict[str, str | int | datetime]] = Field(default_factory=list)
    custom_fields: list[dict[str, int | str | None]] = Field(default_factory=list)


class DocumentUpdate(PaperlessBaseModel):
    """Model for updating a document via PATCH request.

    Only non-None fields are included in the request body.
    Use this to update the content field with Markdown output from Stage 2.
    """

    title: str | None = None
    content: str | None = None
    correspondent: int | None = None
    document_type: int | None = None
    storage_path: int | None = None
    tags: list[int] | None = None
    archive_serial_number: int | None = None
    owner: int | None = None


class PaginatedResponse[T](PaperlessBaseModel):
    """Paginated response wrapper for list endpoints.

    Paperless-ngx returns paginated results for list operations.
    Use the `next` URL to fetch additional pages, or iterate using
    the client's pagination helpers.

    Attributes:
        count: Total number of items matching the query.
        next: URL for the next page, or None if this is the last page.
        previous: URL for the previous page, or None if this is the first page.
        results: List of items on this page.
        all: List of all matching IDs (if requested with return_all=true).
    """

    count: int
    next: HttpUrl | None = None
    previous: HttpUrl | None = None
    results: list[T]
    all: list[int] = Field(default_factory=list)


class TaskStatus(PaperlessBaseModel):
    """Status of an asynchronous task in Paperless-ngx.

    When uploading documents, Paperless-ngx returns a task ID that can be
    polled to check processing status.
    """

    id: str = Field(alias="task_id")
    task_file_name: str | None = None
    date_created: datetime | None = None
    date_done: datetime | None = None
    type: str | None = None
    status: TaskState = TaskState.PENDING
    result: str | None = None
    acknowledged: bool = False
    related_document: str | None = None
