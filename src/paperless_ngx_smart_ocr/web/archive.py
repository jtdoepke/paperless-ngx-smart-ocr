"""Archive PDF replacement via shared filesystem.

Writes OCR'd PDF bytes to the paperless-ngx archive directory and
updates the ``archive_checksum`` in the database so HTTP ETag
caching works correctly.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from paperless_ngx_smart_ocr.observability import get_logger


if TYPE_CHECKING:
    from pathlib import Path


__all__ = [
    "replace_archive_pdf",
    "update_archive_checksum",
]


async def replace_archive_pdf(
    *,
    archive_dir: Path,
    archive_media_filename: str,
    pdf_bytes: bytes,
    database_url: str,
    document_id: int,
) -> str:
    """Replace the archive PDF and update the database checksum.

    Args:
        archive_dir: Mount point for the paperless-ngx archive dir.
        archive_media_filename: Relative path within archive_dir.
        pdf_bytes: The OCR'd PDF bytes to write.
        database_url: PostgreSQL connection URL.
        document_id: The paperless-ngx document ID.

    Returns:
        The MD5 checksum of the written file.

    Raises:
        OSError: If the file write fails.
        Exception: If the database update fails.
    """
    logger = get_logger(__name__)

    target_path = archive_dir / archive_media_filename
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(pdf_bytes)

    checksum = hashlib.md5(pdf_bytes).hexdigest()  # noqa: S324

    logger.info(
        "archive_pdf_replaced",
        document_id=document_id,
        path=str(target_path),
        checksum=checksum,
        size_bytes=len(pdf_bytes),
    )

    await update_archive_checksum(
        database_url=database_url,
        document_id=document_id,
        checksum=checksum,
    )

    return checksum


async def update_archive_checksum(
    *,
    database_url: str,
    document_id: int,
    checksum: str,
) -> None:
    """Update archive_checksum in the paperless-ngx database.

    Uses a direct SQL UPDATE via asyncpg. This ensures the HTTP
    ETag on the paperless preview endpoint matches the new file.

    Args:
        database_url: PostgreSQL connection URL.
        document_id: The paperless-ngx document ID.
        checksum: The MD5 checksum of the new archive file.
    """
    import asyncpg

    logger = get_logger(__name__)

    conn = await asyncpg.connect(database_url)
    try:
        result = await conn.execute(
            "UPDATE documents_document SET archive_checksum = $1 WHERE id = $2",
            checksum,
            document_id,
        )
        logger.info(
            "archive_checksum_updated",
            document_id=document_id,
            checksum=checksum,
            result=result,
        )
    finally:
        await conn.close()
