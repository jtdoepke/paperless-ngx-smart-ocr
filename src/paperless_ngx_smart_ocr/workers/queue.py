"""In-memory async job queue with status tracking."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Self, TypeVar

import aiojobs
import structlog

from paperless_ngx_smart_ocr.workers.exceptions import (
    JobAlreadyCancelledError,
    JobError,
    JobNotFoundError,
    JobQueueFullError,
    JobTimeoutError,
)
from paperless_ngx_smart_ocr.workers.models import Job, JobResult, JobStatus


if TYPE_CHECKING:
    from collections.abc import Coroutine

    from paperless_ngx_smart_ocr.config.schema import JobsConfig


__all__ = ["JobQueue"]


_T = TypeVar("_T")


class JobQueue:
    """In-memory async job queue with enhanced status tracking.

    Wraps aiojobs.Scheduler to provide:
    - Extended job status (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED)
    - Job result storage (success value or error)
    - Job lookup by ID
    - Configurable concurrent workers and timeout

    Example:
        ```python
        from paperless_ngx_smart_ocr.workers import JobQueue

        async with JobQueue(workers=4, timeout=600) as queue:
            job = await queue.submit(
                process_document(doc_id),
                name="Process document 123",
                document_id=123,
            )
            print(f"Submitted job: {job.id}")

            # Wait for completion
            await queue.wait(job.id)
            completed = await queue.get(job.id)
            print(f"Status: {completed.status}")
        ```

    Attributes:
        workers: Maximum number of concurrent jobs.
        timeout: Default timeout per job in seconds.
    """

    DEFAULT_WORKERS = 2
    DEFAULT_TIMEOUT = 600.0  # 10 minutes
    DEFAULT_PENDING_LIMIT = 10000

    def __init__(
        self,
        *,
        workers: int | None = None,
        timeout: float | None = None,
        pending_limit: int | None = None,
        config: JobsConfig | None = None,
    ) -> None:
        """Initialize the job queue.

        Args:
            workers: Maximum concurrent jobs. Overrides config if provided.
            timeout: Default job timeout in seconds. Overrides config if provided.
            pending_limit: Maximum pending jobs in queue.
            config: JobsConfig instance for default values.
        """
        # Apply config values as defaults, then override with explicit params
        if config is not None:
            self._workers = workers if workers is not None else config.workers
            self._timeout = (
                float(timeout) if timeout is not None else float(config.timeout)
            )
        else:
            self._workers = workers if workers is not None else self.DEFAULT_WORKERS
            self._timeout = (
                float(timeout) if timeout is not None else self.DEFAULT_TIMEOUT
            )

        self._pending_limit = pending_limit or self.DEFAULT_PENDING_LIMIT
        self._scheduler: aiojobs.Scheduler | None = None
        self._jobs: dict[str, Job] = {}
        self._aiojobs_map: dict[str, aiojobs.Job[Any]] = {}
        self._coro_map: dict[str, Coroutine[Any, Any, Any]] = {}  # Track coroutines
        self._lock = asyncio.Lock()
        self._logger = structlog.get_logger(__name__)

    @property
    def workers(self) -> int:
        """Return maximum concurrent workers."""
        return self._workers

    @property
    def timeout(self) -> float:
        """Return default job timeout in seconds."""
        return self._timeout

    @property
    def is_running(self) -> bool:
        """Return True if queue is running."""
        return self._scheduler is not None and not self._scheduler.closed

    @property
    def active_count(self) -> int:
        """Return number of currently running jobs."""
        if self._scheduler is None:
            return 0
        return int(self._scheduler.active_count)

    @property
    def pending_count(self) -> int:
        """Return number of pending jobs."""
        if self._scheduler is None:
            return 0
        return int(self._scheduler.pending_count)

    async def __aenter__(self) -> Self:
        """Enter async context and start the scheduler."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit async context and stop the scheduler."""
        await self.stop()

    async def start(self) -> None:
        """Start the job queue scheduler.

        Creates the underlying aiojobs.Scheduler with configured limits.
        """
        if self._scheduler is not None and not self._scheduler.closed:
            return

        self._scheduler = aiojobs.Scheduler(
            limit=self._workers,
            pending_limit=self._pending_limit,
            close_timeout=10.0,  # Grace period for shutdown
        )
        self._logger.info(
            "job_queue_started",
            workers=self._workers,
            timeout=self._timeout,
            pending_limit=self._pending_limit,
        )

    async def stop(
        self,
        *,
        timeout: float | None = None,  # noqa: ASYNC109
    ) -> None:
        """Stop the job queue gracefully.

        Waits for running jobs to complete within the timeout, then
        cancels any remaining jobs.

        Args:
            timeout: Maximum time to wait for jobs to complete.
        """
        if self._scheduler is None or self._scheduler.closed:
            return

        self._logger.info(
            "job_queue_stopping",
            active_count=self.active_count,
            pending_count=self.pending_count,
        )

        # Wait for jobs then close
        await self._scheduler.wait_and_close(timeout=timeout)

        # Mark any remaining non-terminal jobs as cancelled and clean up coroutines
        async with self._lock:
            for job_id, job in self._jobs.items():
                if not job.is_terminal:
                    job.status = JobStatus.CANCELLED
                    job.completed_at = datetime.now(UTC)
                # Close any pending coroutines
                pending_coro = self._coro_map.pop(job_id, None)
                if pending_coro is not None:
                    pending_coro.close()

        self._scheduler = None
        self._logger.info("job_queue_stopped")

    async def submit(
        self,
        coro: Coroutine[Any, Any, _T],
        *,
        name: str = "",
        document_id: int | None = None,
        timeout: float | None = None,  # noqa: ASYNC109
        metadata: dict[str, Any] | None = None,
    ) -> Job:
        """Submit a coroutine for background execution.

        Args:
            coro: The coroutine to execute.
            name: Human-readable job name.
            document_id: Associated document ID for tracking.
            timeout: Job-specific timeout (overrides default).
            metadata: Additional metadata to attach to the job.

        Returns:
            The created Job with status PENDING.

        Raises:
            JobQueueFullError: If pending queue is full.
            JobError: If queue is not running.
        """
        if self._scheduler is None or self._scheduler.closed:
            msg = "Job queue is not running"
            raise JobError(msg)

        # Check pending limit
        if self._scheduler.pending_count >= self._pending_limit:
            raise JobQueueFullError(self._pending_limit)

        # Create our job tracking object
        job = Job(
            name=name,
            document_id=document_id,
            metadata=metadata or {},
        )
        job_timeout = timeout if timeout is not None else self._timeout

        # Wrap the coroutine to track status
        wrapped = self._wrap_coro(job.id, coro, job_timeout)

        async with self._lock:
            self._jobs[job.id] = job
            # Store coroutine reference for proper cleanup if cancelled while pending
            self._coro_map[job.id] = coro

            # Spawn in aiojobs scheduler
            aiojob = await self._scheduler.spawn(wrapped)
            self._aiojobs_map[job.id] = aiojob

        self._logger.debug(
            "job_submitted",
            job_id=job.id,
            name=name,
            document_id=document_id,
            timeout=job_timeout,
        )

        return job

    async def _wrap_coro(
        self,
        job_id: str,
        coro: Coroutine[Any, Any, _T],
        timeout: float,  # noqa: ASYNC109
    ) -> _T | None:
        """Wrap a coroutine with status tracking and timeout.

        Args:
            job_id: The job ID for status updates.
            coro: The coroutine to execute.
            timeout: Timeout in seconds.

        Returns:
            The coroutine result, or None on failure.
        """
        job = self._jobs.get(job_id)
        if job is None:
            coro.close()  # Clean up the unused coroutine
            return None

        # Mark as running and remove from pending coroutine map
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(UTC)
        self._coro_map.pop(job_id, None)  # No longer pending
        self._logger.debug("job_started", job_id=job_id)

        try:
            # Execute with timeout
            result = await asyncio.wait_for(coro, timeout=timeout)
        except TimeoutError:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now(UTC)
            job.result = JobResult(
                error=JobTimeoutError(job_id, timeout),
                error_message=f"Job timed out after {timeout}s",
            )
            self._logger.warning(
                "job_timeout",
                job_id=job_id,
                timeout=timeout,
            )
            return None

        except asyncio.CancelledError:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now(UTC)
            job.result = JobResult(error_message="Job was cancelled")
            self._logger.info("job_cancelled", job_id=job_id)
            raise  # Re-raise to let aiojobs handle it

        except Exception as exc:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now(UTC)
            job.result = JobResult(
                error=exc,
                error_message=str(exc),
            )
            self._logger.exception(
                "job_failed",
                job_id=job_id,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return None
        else:
            # Mark as completed (only reached if no exception)
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now(UTC)
            job.result = JobResult(value=result)
            self._logger.info(
                "job_completed",
                job_id=job_id,
                duration_seconds=job.duration_seconds,
            )
            return result

    async def get(self, job_id: str) -> Job:
        """Get a job by ID.

        Args:
            job_id: The job ID to look up.

        Returns:
            The Job instance.

        Raises:
            JobNotFoundError: If no job with this ID exists.
        """
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            return job

    async def get_status(self, job_id: str) -> JobStatus:
        """Get the status of a job.

        Args:
            job_id: The job ID to check.

        Returns:
            The current JobStatus.

        Raises:
            JobNotFoundError: If no job with this ID exists.
        """
        job = await self.get(job_id)
        return job.status

    async def cancel(
        self,
        job_id: str,
        *,
        timeout: float = 5.0,  # noqa: ASYNC109
    ) -> Job:
        """Cancel a running or pending job.

        Args:
            job_id: The job ID to cancel.
            timeout: Time to wait for graceful cancellation.

        Returns:
            The cancelled Job.

        Raises:
            JobNotFoundError: If no job with this ID exists.
            JobAlreadyCancelledError: If job is already cancelled.
        """
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)

            if job.status == JobStatus.CANCELLED:
                raise JobAlreadyCancelledError(job_id)

            if job.is_terminal:
                # Already completed, nothing to cancel
                return job

            # Close the pending coroutine if it never started
            pending_coro = self._coro_map.pop(job_id, None)
            if pending_coro is not None:
                pending_coro.close()

            # Close the aiojobs job
            aiojob = self._aiojobs_map.get(job_id)
            if aiojob is not None and not aiojob.closed:
                with suppress(Exception):
                    await aiojob.close(timeout=timeout)

            # Update status (might already be set by _wrap_coro)
            if not job.is_terminal:
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now(UTC)
                job.result = JobResult(error_message="Job was cancelled")

            self._logger.info("job_cancelled_by_request", job_id=job_id)
            return job

    async def wait(
        self,
        job_id: str,
        *,
        timeout: float | None = None,  # noqa: ASYNC109
        poll_interval: float = 0.1,
    ) -> Job:
        """Wait for a job to complete.

        Args:
            job_id: The job ID to wait for.
            timeout: Maximum time to wait (None = wait indefinitely).
            poll_interval: Time between status checks.

        Returns:
            The completed Job.

        Raises:
            JobNotFoundError: If no job with this ID exists.
            TimeoutError: If timeout is exceeded.
        """
        async with self._lock:
            if job_id not in self._jobs:
                raise JobNotFoundError(job_id)

        async def _wait_loop() -> Job:
            while True:
                job = await self.get(job_id)
                if job.is_terminal:
                    return job
                await asyncio.sleep(poll_interval)

        if timeout is not None:
            return await asyncio.wait_for(_wait_loop(), timeout=timeout)
        return await _wait_loop()

    async def list_jobs(
        self,
        *,
        status: JobStatus | None = None,
        document_id: int | None = None,
        limit: int = 100,
    ) -> list[Job]:
        """List jobs with optional filtering.

        Args:
            status: Filter by job status.
            document_id: Filter by document ID.
            limit: Maximum number of jobs to return.

        Returns:
            List of matching jobs, newest first.
        """
        async with self._lock:
            jobs = list(self._jobs.values())

        # Apply filters
        if status is not None:
            jobs = [j for j in jobs if j.status == status]
        if document_id is not None:
            jobs = [j for j in jobs if j.document_id == document_id]

        # Sort by created_at descending (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return jobs[:limit]

    async def list_active(self) -> list[Job]:
        """List all currently running jobs.

        Returns:
            List of jobs with RUNNING status.
        """
        return await self.list_jobs(status=JobStatus.RUNNING)

    async def list_pending(self) -> list[Job]:
        """List all pending jobs.

        Returns:
            List of jobs with PENDING status.
        """
        return await self.list_jobs(status=JobStatus.PENDING)

    async def cleanup_completed(
        self,
        *,
        max_age_seconds: float = 3600,
    ) -> int:
        """Remove old completed jobs from memory.

        Args:
            max_age_seconds: Maximum age of completed jobs to retain.

        Returns:
            Number of jobs removed.
        """
        now = datetime.now(UTC)
        removed = 0

        async with self._lock:
            to_remove = []
            for job_id, job in self._jobs.items():
                if job.is_terminal and job.completed_at is not None:
                    age = (now - job.completed_at).total_seconds()
                    if age > max_age_seconds:
                        to_remove.append(job_id)

            for job_id in to_remove:
                del self._jobs[job_id]
                self._aiojobs_map.pop(job_id, None)
                self._coro_map.pop(job_id, None)  # Clean up any stale references
                removed += 1

        if removed > 0:
            self._logger.debug("jobs_cleaned_up", count=removed)

        return removed
