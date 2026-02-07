"""Unit tests for the job queue module."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from paperless_ngx_smart_ocr.workers import (
    Job,
    JobAlreadyCancelledError,
    JobError,
    JobNotFoundError,
    JobQueue,
    JobQueueFullError,
    JobResult,
    JobStatus,
    JobTimeoutError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def queue() -> AsyncGenerator[JobQueue, None]:
    """Create a test job queue."""
    async with JobQueue(workers=2, timeout=0.5) as q:
        yield q


# ---------------------------------------------------------------------------
# TestJobStatus
# ---------------------------------------------------------------------------


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_status_values(self) -> None:
        """Test JobStatus enum values."""
        assert JobStatus.PENDING == "pending"
        assert JobStatus.RUNNING == "running"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"
        assert JobStatus.CANCELLED == "cancelled"

    def test_status_from_string(self) -> None:
        """Test JobStatus can be created from string."""
        assert JobStatus("pending") == JobStatus.PENDING
        assert JobStatus("completed") == JobStatus.COMPLETED


# ---------------------------------------------------------------------------
# TestJob
# ---------------------------------------------------------------------------


class TestJob:
    """Tests for Job dataclass."""

    def test_job_default_values(self) -> None:
        """Test Job default initialization."""
        job = Job()
        assert len(job.id) == 12
        assert job.name == ""
        assert job.status == JobStatus.PENDING
        assert job.document_id is None
        assert job.created_at is not None
        assert job.started_at is None
        assert job.completed_at is None
        assert job.result is None

    def test_job_is_terminal(self) -> None:
        """Test is_terminal property."""
        job = Job()
        assert not job.is_terminal

        job.status = JobStatus.RUNNING
        assert not job.is_terminal

        job.status = JobStatus.COMPLETED
        assert job.is_terminal

        job.status = JobStatus.FAILED
        assert job.is_terminal

        job.status = JobStatus.CANCELLED
        assert job.is_terminal

    def test_job_to_dict(self) -> None:
        """Test Job.to_dict() method."""
        job = Job(name="Test job", document_id=123)
        data = job.to_dict()

        assert data["id"] == job.id
        assert data["name"] == "Test job"
        assert data["status"] == "pending"
        assert data["document_id"] == 123


# ---------------------------------------------------------------------------
# TestJobResult
# ---------------------------------------------------------------------------


class TestJobResult:
    """Tests for JobResult dataclass."""

    def test_success_result(self) -> None:
        """Test successful result."""
        result = JobResult(value={"markdown": "# Hello"})
        assert result.success
        assert result.value == {"markdown": "# Hello"}
        assert result.error is None

    def test_failed_result_with_error(self) -> None:
        """Test failed result with exception."""
        exc = ValueError("Test error")
        result = JobResult(error=exc, error_message="Test error")
        assert not result.success
        assert result.error is exc
        assert result.error_message == "Test error"

    def test_failed_result_with_message_only(self) -> None:
        """Test failed result with only error message."""
        result = JobResult(error_message="Something went wrong")
        assert not result.success
        assert result.error is None
        assert result.error_message == "Something went wrong"


# ---------------------------------------------------------------------------
# TestJobQueue - Lifecycle
# ---------------------------------------------------------------------------


class TestJobQueueLifecycle:
    """Tests for JobQueue lifecycle."""

    async def test_start_stop(self) -> None:
        """Test starting and stopping the queue."""
        queue = JobQueue(workers=2)
        assert not queue.is_running

        await queue.start()
        assert queue.is_running

        await queue.stop()
        assert not queue.is_running

    async def test_context_manager(self) -> None:
        """Test async context manager."""
        async with JobQueue(workers=2) as queue:
            assert queue.is_running
        assert not queue.is_running

    async def test_properties(self, queue: JobQueue) -> None:
        """Test queue properties."""
        assert queue.workers == 2
        assert queue.timeout == 0.5
        assert queue.active_count == 0
        assert queue.pending_count == 0


# ---------------------------------------------------------------------------
# TestJobQueue - Submission
# ---------------------------------------------------------------------------


class TestJobSubmission:
    """Tests for job submission."""

    async def test_submit_simple_job(self, queue: JobQueue) -> None:
        """Test submitting a simple job."""

        async def simple_task() -> str:
            return "done"

        job = await queue.submit(simple_task(), name="Simple task")

        assert job.id is not None
        assert job.name == "Simple task"
        # Job may be PENDING or already RUNNING
        assert job.status in {JobStatus.PENDING, JobStatus.RUNNING}

    async def test_submit_with_document_id(self, queue: JobQueue) -> None:
        """Test submitting a job with document_id."""

        async def task() -> None:
            pass

        job = await queue.submit(
            task(),
            name="Process doc",
            document_id=123,
        )

        assert job.document_id == 123

    async def test_submit_with_metadata(self, queue: JobQueue) -> None:
        """Test submitting a job with metadata."""

        async def task() -> None:
            pass

        job = await queue.submit(
            task(),
            metadata={"stage": 1, "priority": "high"},
        )

        assert job.metadata["stage"] == 1
        assert job.metadata["priority"] == "high"

    async def test_submit_to_stopped_queue_raises(self) -> None:
        """Test submitting to stopped queue raises error."""
        queue = JobQueue()
        # Not started

        async def task() -> None:
            pass

        coro = task()
        try:
            with pytest.raises(JobError, match="not running"):
                await queue.submit(coro)
        finally:
            # Close the coroutine to avoid warning
            coro.close()


# ---------------------------------------------------------------------------
# TestJobQueue - Execution
# ---------------------------------------------------------------------------


class TestJobExecution:
    """Tests for job execution."""

    async def test_job_completes_successfully(self, queue: JobQueue) -> None:
        """Test job runs to completion."""

        async def task() -> int:
            await asyncio.sleep(0.01)
            return 42

        job = await queue.submit(task(), name="Compute")
        completed = await queue.wait(job.id)

        assert completed.status == JobStatus.COMPLETED
        assert completed.result is not None
        assert completed.result.success
        assert completed.result.value == 42
        assert completed.duration_seconds is not None

    async def test_job_fails_with_exception(self, queue: JobQueue) -> None:
        """Test job failure is captured."""

        async def failing_task() -> None:
            msg = "Something went wrong"
            raise ValueError(msg)

        job = await queue.submit(failing_task(), name="Failing")
        completed = await queue.wait(job.id)

        assert completed.status == JobStatus.FAILED
        assert completed.result is not None
        assert not completed.result.success
        assert "Something went wrong" in str(completed.result.error_message)

    async def test_job_times_out(self) -> None:
        """Test job timeout handling."""
        async with JobQueue(workers=1, timeout=0.1) as queue:

            async def slow_task() -> None:
                await asyncio.sleep(10)

            job = await queue.submit(slow_task(), name="Slow")
            completed = await queue.wait(job.id, timeout=1.0)

            assert completed.status == JobStatus.FAILED
            assert "timed out" in str(completed.result.error_message).lower()


# ---------------------------------------------------------------------------
# TestJobQueue - Cancellation
# ---------------------------------------------------------------------------


class TestJobCancellation:
    """Tests for job cancellation."""

    async def test_cancel_pending_job(self) -> None:
        """Test cancelling a pending job."""
        # Use 1 worker so jobs queue up
        async with JobQueue(workers=1, timeout=0.5) as queue:

            async def blocking() -> None:
                await asyncio.sleep(0.2)

            # Submit a blocking job to occupy the worker
            await queue.submit(blocking(), name="Blocker")

            # This job should be pending
            async def pending_task() -> None:
                pass

            job = await queue.submit(pending_task(), name="Will cancel")
            await asyncio.sleep(0.01)

            # Cancel it
            cancelled = await queue.cancel(job.id)
            assert cancelled.status == JobStatus.CANCELLED

    async def test_cancel_running_job(self, queue: JobQueue) -> None:
        """Test cancelling a running job."""

        async def long_task() -> None:
            await asyncio.sleep(10)

        job = await queue.submit(long_task(), name="Long")
        await asyncio.sleep(0.05)  # Let it start

        cancelled = await queue.cancel(job.id)
        assert cancelled.status == JobStatus.CANCELLED

    async def test_cancel_completed_job_noop(self, queue: JobQueue) -> None:
        """Test cancelling completed job returns it unchanged."""

        async def quick() -> str:
            return "done"

        job = await queue.submit(quick())
        await queue.wait(job.id)

        result = await queue.cancel(job.id)
        assert result.status == JobStatus.COMPLETED

    async def test_cancel_already_cancelled_raises(self, queue: JobQueue) -> None:
        """Test cancelling twice raises error."""

        async def task() -> None:
            await asyncio.sleep(10)

        job = await queue.submit(task())
        await asyncio.sleep(0.01)  # Let it register
        await queue.cancel(job.id)

        with pytest.raises(JobAlreadyCancelledError):
            await queue.cancel(job.id)


# ---------------------------------------------------------------------------
# TestJobQueue - Lookup
# ---------------------------------------------------------------------------


class TestJobLookup:
    """Tests for job lookup methods."""

    async def test_get_job(self, queue: JobQueue) -> None:
        """Test getting a job by ID."""

        async def task() -> None:
            pass

        job = await queue.submit(task())
        found = await queue.get(job.id)

        assert found.id == job.id

    async def test_get_nonexistent_raises(self, queue: JobQueue) -> None:
        """Test getting nonexistent job raises error."""
        with pytest.raises(JobNotFoundError):
            await queue.get("nonexistent-id")

    async def test_get_status(self, queue: JobQueue) -> None:
        """Test getting job status."""

        async def task() -> None:
            await asyncio.sleep(0.1)

        job = await queue.submit(task())
        status = await queue.get_status(job.id)

        assert status in {JobStatus.PENDING, JobStatus.RUNNING}


# ---------------------------------------------------------------------------
# TestJobQueue - Listing
# ---------------------------------------------------------------------------


class TestJobListing:
    """Tests for job listing methods."""

    async def test_list_jobs(self, queue: JobQueue) -> None:
        """Test listing all jobs."""

        async def task() -> None:
            pass

        await queue.submit(task(), name="Job 1")
        await queue.submit(task(), name="Job 2")
        await queue.submit(task(), name="Job 3")

        jobs = await queue.list_jobs()
        assert len(jobs) == 3

    async def test_list_jobs_with_status_filter(self, queue: JobQueue) -> None:
        """Test listing jobs filtered by status."""

        async def quick() -> None:
            pass

        async def slow() -> None:
            await asyncio.sleep(10)

        job1 = await queue.submit(quick())
        await queue.submit(slow())

        await queue.wait(job1.id)

        completed = await queue.list_jobs(status=JobStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0].id == job1.id

    async def test_list_jobs_with_document_filter(self, queue: JobQueue) -> None:
        """Test listing jobs filtered by document_id."""

        async def task() -> None:
            pass

        await queue.submit(task(), document_id=100)
        await queue.submit(task(), document_id=200)
        await queue.submit(task(), document_id=100)

        jobs = await queue.list_jobs(document_id=100)
        assert len(jobs) == 2

    async def test_list_active(self, queue: JobQueue) -> None:
        """Test listing active jobs."""

        async def slow() -> None:
            await asyncio.sleep(10)

        await queue.submit(slow())
        await asyncio.sleep(0.05)  # Let it start

        active = await queue.list_active()
        assert len(active) == 1
        assert active[0].status == JobStatus.RUNNING


# ---------------------------------------------------------------------------
# TestJobQueue - Wait
# ---------------------------------------------------------------------------


class TestJobWait:
    """Tests for job wait functionality."""

    async def test_wait_for_job(self, queue: JobQueue) -> None:
        """Test waiting for job completion."""

        async def task() -> str:
            await asyncio.sleep(0.05)
            return "result"

        job = await queue.submit(task())
        completed = await queue.wait(job.id)

        assert completed.status == JobStatus.COMPLETED

    async def test_wait_with_timeout(self, queue: JobQueue) -> None:
        """Test wait timeout."""

        async def slow() -> None:
            await asyncio.sleep(10)

        job = await queue.submit(slow())

        with pytest.raises(TimeoutError):
            await queue.wait(job.id, timeout=0.1)

    async def test_wait_nonexistent_raises(self, queue: JobQueue) -> None:
        """Test waiting for nonexistent job raises."""
        with pytest.raises(JobNotFoundError):
            await queue.wait("nonexistent")


# ---------------------------------------------------------------------------
# TestJobQueue - Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    """Tests for concurrency limits."""

    async def test_respects_worker_limit(self) -> None:
        """Test that worker limit is respected."""
        async with JobQueue(workers=2) as queue:
            running_count = 0
            max_concurrent = 0
            lock = asyncio.Lock()

            async def tracked_task() -> None:
                nonlocal running_count, max_concurrent
                async with lock:
                    running_count += 1
                    max_concurrent = max(max_concurrent, running_count)
                await asyncio.sleep(0.05)
                async with lock:
                    running_count -= 1

            # Submit 5 jobs
            jobs = [await queue.submit(tracked_task()) for _ in range(5)]

            # Wait for all
            for job in jobs:
                await queue.wait(job.id, timeout=2.0)

            # Should never exceed 2 concurrent
            assert max_concurrent <= 2


# ---------------------------------------------------------------------------
# TestJobQueue - Cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    """Tests for job cleanup."""

    async def test_cleanup_completed(self, queue: JobQueue) -> None:
        """Test cleanup of old completed jobs."""

        async def task() -> None:
            pass

        job = await queue.submit(task())
        await queue.wait(job.id)

        # Cleanup with 0 max age removes immediately
        removed = await queue.cleanup_completed(max_age_seconds=0)
        assert removed == 1

        # Job should no longer exist
        with pytest.raises(JobNotFoundError):
            await queue.get(job.id)


# ---------------------------------------------------------------------------
# TestExceptions
# ---------------------------------------------------------------------------


class TestExceptions:
    """Tests for exception classes."""

    def test_job_error_str(self) -> None:
        """Test JobError string representation."""
        err = JobError("Something failed", job_id="abc123")
        assert "Something failed" in str(err)
        assert "abc123" in str(err)

    def test_job_error_str_without_job_id(self) -> None:
        """Test JobError string without job_id."""
        err = JobError("Something failed")
        assert str(err) == "Something failed"

    def test_job_not_found_error(self) -> None:
        """Test JobNotFoundError."""
        err = JobNotFoundError("xyz789")
        assert err.job_id == "xyz789"
        assert "xyz789" in str(err)

    def test_job_queue_full_error(self) -> None:
        """Test JobQueueFullError."""
        err = JobQueueFullError(1000)
        assert err.pending_limit == 1000
        assert "1000" in str(err)

    def test_job_timeout_error(self) -> None:
        """Test JobTimeoutError."""
        err = JobTimeoutError("abc", 60.0)
        assert err.job_id == "abc"
        assert err.timeout == 60.0
        assert "60" in str(err)

    def test_job_already_cancelled_error(self) -> None:
        """Test JobAlreadyCancelledError."""
        err = JobAlreadyCancelledError("def456")
        assert err.job_id == "def456"
        assert "def456" in str(err)
