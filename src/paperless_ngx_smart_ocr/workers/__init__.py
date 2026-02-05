"""Background workers and integration handlers.

This module provides an in-memory async job queue for background document
processing, along with integration handlers for polling, webhooks, and
post-consume script modes.

Example:
    ```python
    from paperless_ngx_smart_ocr.workers import JobQueue, JobStatus

    async with JobQueue(workers=4, timeout=600) as queue:
        # Submit a job
        job = await queue.submit(
            process_document(doc_id),
            name="Process document",
            document_id=doc_id,
        )

        # Check status
        status = await queue.get_status(job.id)
        if status == JobStatus.COMPLETED:
            print("Done!")
    ```
"""

from __future__ import annotations

from paperless_ngx_smart_ocr.workers.exceptions import (
    JobAlreadyCancelledError,
    JobError,
    JobNotFoundError,
    JobQueueFullError,
    JobTimeoutError,
)
from paperless_ngx_smart_ocr.workers.models import Job, JobResult, JobStatus
from paperless_ngx_smart_ocr.workers.queue import JobQueue


__all__ = [
    # Models
    "Job",
    # Exceptions
    "JobAlreadyCancelledError",
    "JobError",
    "JobNotFoundError",
    # Queue
    "JobQueue",
    "JobQueueFullError",
    "JobResult",
    "JobStatus",
    "JobTimeoutError",
]
