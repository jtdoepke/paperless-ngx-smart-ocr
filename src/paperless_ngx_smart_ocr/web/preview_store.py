"""Server-side cache for pipeline preview results.

Stores dry-run results between the preview and apply steps so that
temporary files (OCR'd PDF bytes, markdown) survive beyond the
pipeline's ``TemporaryDirectory`` lifetime.
"""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from paperless_ngx_smart_ocr.pipeline.models import PipelineResult


__all__ = [
    "BulkPreviewBatch",
    "PreviewEntry",
    "PreviewStore",
]


# -------------------------------------------------------------------
# Data models
# -------------------------------------------------------------------


@dataclass(slots=True)
class PreviewEntry:
    """A single cached preview result.

    Attributes:
        preview_id: Unique 12-char hex identifier.
        document_id: Paperless-ngx document ID.
        pipeline_result: Full pipeline result from dry-run.
        markdown: Stage 2 markdown content (empty string if Stage 2
            was not run or failed).
        ocr_pdf_bytes: OCR'd PDF bytes from Stage 1, or ``None`` if
            Stage 1 was not run or failed.
        created_at: Timestamp when the entry was stored.
    """

    preview_id: str
    document_id: int
    pipeline_result: PipelineResult
    markdown: str
    ocr_pdf_bytes: bytes | None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class BulkPreviewBatch:
    """Tracks a batch of bulk preview dry-runs.

    Attributes:
        batch_id: Unique 12-char hex identifier for the batch.
        document_ids: Ordered list of document IDs in the batch.
        job_ids: Mapping of document_id to job_id in the queue.
        preview_ids: Mapping of document_id to preview_id, populated
            as individual dry-runs complete.
        excluded_ids: Document IDs the user unchecked before apply.
        created_at: Timestamp when the batch was created.
    """

    batch_id: str
    document_ids: list[int]
    job_ids: dict[int, str]
    preview_ids: dict[int, str] = field(default_factory=dict)
    excluded_ids: set[int] = field(default_factory=set)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


# -------------------------------------------------------------------
# Store
# -------------------------------------------------------------------


class PreviewStore:
    """In-memory store for preview entries with TTL-based expiration.

    Thread-safe via an asyncio lock. Limits total entries to prevent
    unbounded memory growth.

    Attributes:
        TTL_SECONDS: Time-to-live for entries (default 600 = 10 min).
        MAX_ENTRIES: Maximum number of stored entries (default 100).
    """

    TTL_SECONDS: int = 600
    MAX_ENTRIES: int = 100

    def __init__(
        self,
        *,
        ttl_seconds: int | None = None,
        max_entries: int | None = None,
    ) -> None:
        self._entries: dict[str, PreviewEntry] = {}
        self._batches: dict[str, BulkPreviewBatch] = {}
        self._lock = asyncio.Lock()
        if ttl_seconds is not None:
            self.TTL_SECONDS = ttl_seconds
        if max_entries is not None:
            self.MAX_ENTRIES = max_entries

    # -- Preview entries ------------------------------------------

    async def store(self, entry: PreviewEntry) -> str:
        """Store a preview entry.

        If the store is at capacity, the oldest entry is evicted.

        Args:
            entry: The preview entry to store.

        Returns:
            The ``preview_id`` of the stored entry.
        """
        async with self._lock:
            # Evict oldest if at capacity
            while len(self._entries) >= self.MAX_ENTRIES:
                oldest_id = next(iter(self._entries))
                del self._entries[oldest_id]
            self._entries[entry.preview_id] = entry
        return entry.preview_id

    async def get(self, preview_id: str) -> PreviewEntry | None:
        """Retrieve a preview entry by ID.

        Returns ``None`` if the entry does not exist or has expired.

        Args:
            preview_id: The unique preview identifier.

        Returns:
            The preview entry, or ``None``.
        """
        async with self._lock:
            entry = self._entries.get(preview_id)
            if entry is None:
                return None
            age = (datetime.now(UTC) - entry.created_at).total_seconds()
            if age > self.TTL_SECONDS:
                del self._entries[preview_id]
                return None
            return entry

    async def remove(self, preview_id: str) -> None:
        """Remove a preview entry.

        No-op if the entry does not exist.

        Args:
            preview_id: The unique preview identifier.
        """
        async with self._lock:
            self._entries.pop(preview_id, None)

    # -- Bulk batches ---------------------------------------------

    async def store_batch(self, batch: BulkPreviewBatch) -> str:
        """Store a bulk preview batch.

        Args:
            batch: The batch to store.

        Returns:
            The ``batch_id``.
        """
        async with self._lock:
            self._batches[batch.batch_id] = batch
        return batch.batch_id

    async def get_batch(self, batch_id: str) -> BulkPreviewBatch | None:
        """Retrieve a bulk batch by ID.

        Returns ``None`` if expired or not found.

        Args:
            batch_id: The unique batch identifier.

        Returns:
            The batch, or ``None``.
        """
        async with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                return None
            age = (datetime.now(UTC) - batch.created_at).total_seconds()
            if age > self.TTL_SECONDS:
                del self._batches[batch_id]
                return None
            return batch

    async def remove_batch(self, batch_id: str) -> None:
        """Remove a bulk batch and all associated preview entries.

        Args:
            batch_id: The unique batch identifier.
        """
        async with self._lock:
            batch = self._batches.pop(batch_id, None)
            if batch is not None:
                for pid in batch.preview_ids.values():
                    self._entries.pop(pid, None)

    # -- Maintenance ----------------------------------------------

    async def cleanup_expired(self) -> int:
        """Remove all expired entries and batches.

        Returns:
            Number of entries removed.
        """
        now = datetime.now(UTC)
        removed = 0
        async with self._lock:
            expired_entries = [
                pid
                for pid, entry in self._entries.items()
                if (now - entry.created_at).total_seconds() > self.TTL_SECONDS
            ]
            for pid in expired_entries:
                del self._entries[pid]
                removed += 1

            expired_batches = [
                bid
                for bid, batch in self._batches.items()
                if (now - batch.created_at).total_seconds() > self.TTL_SECONDS
            ]
            for bid in expired_batches:
                batch = self._batches.pop(bid)
                for pid in batch.preview_ids.values():
                    if pid in self._entries:
                        del self._entries[pid]
                        removed += 1
        return removed

    @staticmethod
    def generate_id() -> str:
        """Generate a unique 12-character hex identifier."""
        return secrets.token_hex(6)
