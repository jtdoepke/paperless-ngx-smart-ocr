"""Unit tests for archive PDF replacement."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from paperless_ngx_smart_ocr.web.archive import (
    replace_archive_pdf,
    update_archive_checksum,
)


if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# TestReplaceArchivePdf
# ---------------------------------------------------------------------------


class TestReplaceArchivePdf:
    """Tests for replace_archive_pdf."""

    async def test_writes_pdf_bytes(self, tmp_path: Path) -> None:
        """PDF bytes are written to the correct path."""
        pdf_bytes = b"%PDF-1.4 test content"

        with patch(
            "paperless_ngx_smart_ocr.web.archive.update_archive_checksum",
            new_callable=AsyncMock,
        ):
            await replace_archive_pdf(
                archive_dir=tmp_path,
                archive_media_filename="test.pdf",
                pdf_bytes=pdf_bytes,
                database_url="postgresql://test",
                document_id=1,
            )

        assert (tmp_path / "test.pdf").read_bytes() == pdf_bytes

    async def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Parent directories are created if they don't exist."""
        pdf_bytes = b"%PDF-1.4"

        with patch(
            "paperless_ngx_smart_ocr.web.archive.update_archive_checksum",
            new_callable=AsyncMock,
        ):
            await replace_archive_pdf(
                archive_dir=tmp_path,
                archive_media_filename="subdir/nested/test.pdf",
                pdf_bytes=pdf_bytes,
                database_url="postgresql://test",
                document_id=1,
            )

        assert (tmp_path / "subdir" / "nested" / "test.pdf").exists()

    async def test_returns_md5_checksum(self, tmp_path: Path) -> None:
        """Returns the MD5 checksum of written bytes."""
        pdf_bytes = b"%PDF-1.4 checksum test"
        expected = hashlib.md5(pdf_bytes).hexdigest()  # noqa: S324

        with patch(
            "paperless_ngx_smart_ocr.web.archive.update_archive_checksum",
            new_callable=AsyncMock,
        ):
            result = await replace_archive_pdf(
                archive_dir=tmp_path,
                archive_media_filename="test.pdf",
                pdf_bytes=pdf_bytes,
                database_url="postgresql://test",
                document_id=1,
            )

        assert result == expected

    async def test_calls_update_checksum(self, tmp_path: Path) -> None:
        """Calls update_archive_checksum with correct args."""
        pdf_bytes = b"%PDF-1.4"
        expected_checksum = hashlib.md5(pdf_bytes).hexdigest()  # noqa: S324

        with patch(
            "paperless_ngx_smart_ocr.web.archive.update_archive_checksum",
            new_callable=AsyncMock,
        ) as mock_update:
            await replace_archive_pdf(
                archive_dir=tmp_path,
                archive_media_filename="test.pdf",
                pdf_bytes=pdf_bytes,
                database_url="postgresql://test:5432/db",
                document_id=42,
            )

        mock_update.assert_awaited_once_with(
            database_url="postgresql://test:5432/db",
            document_id=42,
            checksum=expected_checksum,
        )


# ---------------------------------------------------------------------------
# TestUpdateArchiveChecksum
# ---------------------------------------------------------------------------


class TestUpdateArchiveChecksum:
    """Tests for update_archive_checksum."""

    async def test_executes_sql_update(self) -> None:
        """Executes the correct SQL update statement."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")

        with patch(
            "asyncpg.connect",
            new_callable=AsyncMock,
            return_value=mock_conn,
        ):
            await update_archive_checksum(
                database_url="postgresql://test",
                document_id=7,
                checksum="abc123",
            )

        mock_conn.execute.assert_awaited_once()
        sql_arg = mock_conn.execute.call_args[0][0]
        assert "archive_checksum" in sql_arg
        assert "$1" in sql_arg
        assert "$2" in sql_arg

        # Positional params
        assert mock_conn.execute.call_args[0][1] == "abc123"
        assert mock_conn.execute.call_args[0][2] == 7

    async def test_closes_connection(self) -> None:
        """Connection is closed even on success."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")

        with patch(
            "asyncpg.connect",
            new_callable=AsyncMock,
            return_value=mock_conn,
        ):
            await update_archive_checksum(
                database_url="postgresql://test",
                document_id=1,
                checksum="abc",
            )

        mock_conn.close.assert_awaited_once()

    async def test_closes_connection_on_error(self) -> None:
        """Connection is closed even when execute raises."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=RuntimeError("db error"))

        with (
            patch(
                "asyncpg.connect",
                new_callable=AsyncMock,
                return_value=mock_conn,
            ),
            pytest.raises(RuntimeError, match="db error"),
        ):
            await update_archive_checksum(
                database_url="postgresql://test",
                document_id=1,
                checksum="abc",
            )

        mock_conn.close.assert_awaited_once()
