"""Models for background job processing."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


__all__ = [
    "Job",
    "JobResult",
    "JobStatus",
]


class JobStatus(StrEnum):
    """Status of a background job.

    Attributes:
        PENDING: Job is queued but not yet started.
        RUNNING: Job is currently executing.
        COMPLETED: Job finished successfully.
        FAILED: Job finished with an error.
        CANCELLED: Job was cancelled before completion.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobResult:
    """Result of a completed job.

    Attributes:
        value: The return value if job completed successfully.
        error: The exception if job failed.
        error_message: Human-readable error message if job failed.
    """

    value: Any = None
    error: BaseException | None = None
    error_message: str | None = None

    @property
    def success(self) -> bool:
        """Return True if job completed successfully."""
        return self.error is None and self.error_message is None


@dataclass
class Job:
    """Represents a background job with enhanced status tracking.

    Attributes:
        id: Unique job identifier (12-char hex string).
        name: Human-readable job name/description.
        status: Current job status.
        document_id: Associated document ID, if applicable.
        created_at: When the job was created.
        started_at: When the job started executing.
        completed_at: When the job finished (success, failure, or cancelled).
        result: Job result after completion.
        metadata: Additional job metadata.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    status: JobStatus = JobStatus.PENDING
    document_id: int | None = None
    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC),
    )
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: JobResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        """Return True if job is in a terminal state."""
        return self.status in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }

    @property
    def duration_seconds(self) -> float | None:
        """Return job duration in seconds, or None if not completed."""
        if self.started_at is None or self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary for API responses.

        Returns:
            Dictionary representation of the job.
        """
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "document_id": self.document_id,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "duration_seconds": self.duration_seconds,
            "result": (
                {
                    "success": self.result.success,
                    "error_message": self.result.error_message,
                }
                if self.result
                else None
            ),
            "metadata": self.metadata,
        }
