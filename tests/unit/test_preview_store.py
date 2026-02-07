"""Unit tests for the preview store."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from paperless_ngx_smart_ocr.web.preview_store import (
    BulkPreviewBatch,
    PreviewEntry,
    PreviewStore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    preview_id: str = "abc123",
    document_id: int = 1,
    *,
    markdown: str = "# Hello",
    ocr_pdf_bytes: bytes | None = b"%PDF-1.4",
    created_at: datetime | None = None,
) -> PreviewEntry:
    return PreviewEntry(
        preview_id=preview_id,
        document_id=document_id,
        pipeline_result=MagicMock(),
        markdown=markdown,
        ocr_pdf_bytes=ocr_pdf_bytes,
        created_at=created_at or datetime.now(UTC),
    )


def _make_batch(
    batch_id: str = "batch1",
    document_ids: list[int] | None = None,
    job_ids: dict[int, str] | None = None,
    preview_ids: dict[int, str] | None = None,
    *,
    created_at: datetime | None = None,
) -> BulkPreviewBatch:
    return BulkPreviewBatch(
        batch_id=batch_id,
        document_ids=document_ids or [1, 2, 3],
        job_ids=job_ids or {1: "j1", 2: "j2", 3: "j3"},
        preview_ids=preview_ids or {},
        created_at=created_at or datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# TestPreviewStore - Entry operations
# ---------------------------------------------------------------------------


class TestPreviewStoreEntries:
    """Tests for store/get/remove of preview entries."""

    async def test_store_and_get(self) -> None:
        """Can store and retrieve an entry."""
        store = PreviewStore()
        entry = _make_entry()
        await store.store(entry)

        result = await store.get("abc123")
        assert result is not None
        assert result.preview_id == "abc123"
        assert result.document_id == 1

    async def test_get_missing_returns_none(self) -> None:
        """Getting a nonexistent entry returns None."""
        store = PreviewStore()
        assert await store.get("nonexistent") is None

    async def test_get_expired_returns_none(self) -> None:
        """Expired entries return None on get."""
        store = PreviewStore(ttl_seconds=10)
        entry = _make_entry(created_at=datetime.now(UTC) - timedelta(seconds=20))
        await store.store(entry)

        assert await store.get("abc123") is None

    async def test_remove_entry(self) -> None:
        """Removing an entry makes it unavailable."""
        store = PreviewStore()
        entry = _make_entry()
        await store.store(entry)
        await store.remove("abc123")

        assert await store.get("abc123") is None

    async def test_remove_nonexistent_is_noop(self) -> None:
        """Removing a nonexistent entry does not raise."""
        store = PreviewStore()
        await store.remove("nonexistent")  # Should not raise

    async def test_evicts_oldest_at_capacity(self) -> None:
        """Oldest entry is evicted when store hits max_entries."""
        store = PreviewStore(max_entries=2)
        await store.store(_make_entry(preview_id="a", document_id=1))
        await store.store(_make_entry(preview_id="b", document_id=2))
        await store.store(_make_entry(preview_id="c", document_id=3))

        # "a" should be evicted
        assert await store.get("a") is None
        assert await store.get("b") is not None
        assert await store.get("c") is not None


# ---------------------------------------------------------------------------
# TestPreviewStore - Batch operations
# ---------------------------------------------------------------------------


class TestPreviewStoreBatches:
    """Tests for store/get/remove of bulk preview batches."""

    async def test_store_and_get_batch(self) -> None:
        """Can store and retrieve a batch."""
        store = PreviewStore()
        batch = _make_batch()
        await store.store_batch(batch)

        result = await store.get_batch("batch1")
        assert result is not None
        assert result.batch_id == "batch1"
        assert result.document_ids == [1, 2, 3]

    async def test_get_missing_batch_returns_none(self) -> None:
        """Getting a nonexistent batch returns None."""
        store = PreviewStore()
        assert await store.get_batch("nonexistent") is None

    async def test_get_expired_batch_returns_none(self) -> None:
        """Expired batches return None on get."""
        store = PreviewStore(ttl_seconds=10)
        batch = _make_batch(created_at=datetime.now(UTC) - timedelta(seconds=20))
        await store.store_batch(batch)

        assert await store.get_batch("batch1") is None

    async def test_remove_batch(self) -> None:
        """Removing a batch removes it and associated entries."""
        store = PreviewStore()
        entry = _make_entry(preview_id="p1", document_id=1)
        await store.store(entry)

        batch = _make_batch(preview_ids={1: "p1"})
        await store.store_batch(batch)
        await store.remove_batch("batch1")

        assert await store.get_batch("batch1") is None
        assert await store.get("p1") is None


# ---------------------------------------------------------------------------
# TestPreviewStore - Cleanup
# ---------------------------------------------------------------------------


class TestPreviewStoreCleanup:
    """Tests for cleanup_expired."""

    async def test_cleanup_removes_expired_entries(self) -> None:
        """Expired entries are cleaned up."""
        store = PreviewStore(ttl_seconds=10)
        old_entry = _make_entry(
            preview_id="old",
            created_at=datetime.now(UTC) - timedelta(seconds=20),
        )
        fresh_entry = _make_entry(preview_id="fresh")
        await store.store(old_entry)
        await store.store(fresh_entry)

        removed = await store.cleanup_expired()
        assert removed == 1
        assert await store.get("old") is None
        assert await store.get("fresh") is not None

    async def test_cleanup_removes_expired_batches(self) -> None:
        """Expired batches and their entries are cleaned up."""
        store = PreviewStore(ttl_seconds=10)
        entry = _make_entry(
            preview_id="bp1",
            created_at=datetime.now(UTC) - timedelta(seconds=20),
        )
        await store.store(entry)

        batch = _make_batch(
            created_at=datetime.now(UTC) - timedelta(seconds=20),
            preview_ids={1: "bp1"},
        )
        await store.store_batch(batch)

        removed = await store.cleanup_expired()
        # 1 entry + 1 batch-associated entry (same entry "bp1")
        assert removed >= 1
        assert await store.get_batch("batch1") is None

    async def test_cleanup_returns_zero_when_nothing_expired(
        self,
    ) -> None:
        """Returns 0 when no entries are expired."""
        store = PreviewStore()
        await store.store(_make_entry())
        removed = await store.cleanup_expired()
        assert removed == 0


# ---------------------------------------------------------------------------
# TestPreviewStore - ID generation
# ---------------------------------------------------------------------------


class TestPreviewStoreGenerateId:
    """Tests for generate_id."""

    def test_generates_12_char_hex(self) -> None:
        """Generated ID is 12 hex characters."""
        pid = PreviewStore.generate_id()
        assert len(pid) == 12
        int(pid, 16)  # Should not raise if valid hex

    def test_generates_unique_ids(self) -> None:
        """Each call produces a different ID."""
        ids = {PreviewStore.generate_id() for _ in range(100)}
        assert len(ids) == 100
