"""Custom exceptions for the background job queue."""

from __future__ import annotations


__all__ = [
    "JobAlreadyCancelledError",
    "JobError",
    "JobNotFoundError",
    "JobQueueFullError",
    "JobTimeoutError",
]


class JobError(Exception):
    """Base exception for all job queue errors.

    Attributes:
        message: Human-readable error description.
        job_id: The job ID related to this error, if applicable.
    """

    def __init__(
        self,
        message: str,
        *,
        job_id: str | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            job_id: The job ID related to this error.
        """
        super().__init__(message)
        self.message = message
        self.job_id = job_id

    def __str__(self) -> str:
        """Return string representation with job ID if available."""
        if self.job_id is not None:
            return f"{self.message} (job_id={self.job_id})"
        return self.message


class JobNotFoundError(JobError):
    """Raised when a job with the specified ID is not found.

    Attributes:
        job_id: The job ID that was not found.
    """

    def __init__(self, job_id: str) -> None:
        """Initialize the exception.

        Args:
            job_id: The job ID that was not found.
        """
        super().__init__(f"Job not found: {job_id}", job_id=job_id)


class JobAlreadyCancelledError(JobError):
    """Raised when attempting to cancel an already cancelled job.

    Attributes:
        job_id: The job ID that was already cancelled.
    """

    def __init__(self, job_id: str) -> None:
        """Initialize the exception.

        Args:
            job_id: The job ID that was already cancelled.
        """
        super().__init__(f"Job already cancelled: {job_id}", job_id=job_id)


class JobQueueFullError(JobError):
    """Raised when the job queue has reached its pending limit.

    Attributes:
        pending_limit: The maximum pending job limit that was exceeded.
    """

    def __init__(self, pending_limit: int) -> None:
        """Initialize the exception.

        Args:
            pending_limit: The maximum pending job limit that was exceeded.
        """
        super().__init__(f"Job queue full: pending_limit={pending_limit}")
        self.pending_limit = pending_limit


class JobTimeoutError(JobError):
    """Raised when a job exceeds its timeout.

    Attributes:
        timeout: The timeout value in seconds that was exceeded.
    """

    def __init__(
        self,
        job_id: str,
        timeout: float,
    ) -> None:
        """Initialize the exception.

        Args:
            job_id: The job ID that timed out.
            timeout: The timeout value in seconds.
        """
        super().__init__(
            f"Job timed out after {timeout}s",
            job_id=job_id,
        )
        self.timeout = timeout
