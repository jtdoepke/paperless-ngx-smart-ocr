"""Pipeline orchestrator for coordinating OCR and Markdown stages.

This module provides the PipelineOrchestrator class that coordinates the
full document processing pipeline: downloading from paperless-ngx,
running Stage 1 (OCR) and Stage 2 (Markdown), updating content and tags,
and cleaning up temporary files.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from paperless_ngx_smart_ocr.config import ContentMode
from paperless_ngx_smart_ocr.observability import get_logger
from paperless_ngx_smart_ocr.paperless.exceptions import PaperlessError
from paperless_ngx_smart_ocr.paperless.models import DocumentUpdate
from paperless_ngx_smart_ocr.pipeline.models import (
    PipelineResult,
    Stage1Result,
    Stage2Result,
)
from paperless_ngx_smart_ocr.pipeline.stage1_ocr import Stage1Processor
from paperless_ngx_smart_ocr.pipeline.stage2_markdown import Stage2Processor


if TYPE_CHECKING:
    from paperless_ngx_smart_ocr.config import Settings
    from paperless_ngx_smart_ocr.paperless import PaperlessClient


__all__ = [
    "PipelineOrchestrator",
    "process_document",
]


class PipelineOrchestrator:
    """Orchestrates the full document processing pipeline.

    Coordinates Stage 1 (OCR) and Stage 2 (Markdown) processing, handling
    document downloads, stage sequencing, content patching, tag management,
    and temp file cleanup.

    The orchestrator does NOT depend on the job queue - it is a coroutine
    that the queue can submit.

    Example:
        ```python
        from paperless_ngx_smart_ocr.config import get_settings
        from paperless_ngx_smart_ocr.paperless import PaperlessClient

        settings = get_settings()
        async with PaperlessClient(
            base_url=settings.paperless.url,
            token=settings.paperless.token,
        ) as client:
            orchestrator = PipelineOrchestrator(
                settings=settings,
                client=client,
            )
            result = await orchestrator.process(document_id=123)
        ```
    """

    def __init__(
        self,
        *,
        settings: Settings,
        client: PaperlessClient,
    ) -> None:
        """Initialize the pipeline orchestrator.

        Args:
            settings: Application settings.
            client: Paperless-ngx API client.
        """
        self._settings = settings
        self._client = client
        self._logger = get_logger(__name__)

    async def process(
        self,
        document_id: int,
        *,
        force: bool = False,
    ) -> PipelineResult:
        """Process a document through the full pipeline.

        Downloads the document from paperless-ngx, runs enabled stages,
        updates content and tags, and cleans up temporary files.

        This method never raises exceptions - it always returns a
        PipelineResult capturing the outcome.

        Args:
            document_id: The paperless-ngx document ID.
            force: If True, force processing regardless of born-digital status.

        Returns:
            PipelineResult with the processing outcome.
        """
        start_time = time.monotonic()
        stage1_config = self._settings.pipeline.stage1
        stage2_config = self._settings.pipeline.stage2

        self._logger.info(
            "pipeline_starting",
            document_id=document_id,
            stage1_enabled=stage1_config.enabled,
            stage2_enabled=stage2_config.enabled,
            force=force,
        )

        stage1_result: Stage1Result | None = None
        stage2_result: Stage2Result | None = None
        content_updated = False
        tags_updated = False

        try:
            with tempfile.TemporaryDirectory(
                prefix=f"smart-ocr-{document_id}-"
            ) as tmp_dir:
                tmp_path = Path(tmp_dir)
                original_path = tmp_path / "original.pdf"
                ocr_output_path = tmp_path / "ocr_output.pdf"

                # Step 1: Download original PDF
                try:
                    await self._client.download_document_to_path(
                        document_id, original_path, original=True
                    )
                    self._logger.info(
                        "pipeline_downloaded",
                        document_id=document_id,
                        path=str(original_path),
                    )
                except PaperlessError as exc:
                    self._logger.exception(
                        "pipeline_download_failed",
                        document_id=document_id,
                        error=str(exc),
                    )
                    processing_time = time.monotonic() - start_time
                    tags_updated = await self._update_tags(document_id, success=False)
                    return PipelineResult(
                        document_id=document_id,
                        success=False,
                        stage1_result=None,
                        stage2_result=None,
                        stage1_skipped_by_config=not stage1_config.enabled,
                        stage2_skipped_by_config=not stage2_config.enabled,
                        tags_updated=tags_updated,
                        content_updated=False,
                        document_uploaded=False,
                        error=f"Download failed: {exc}",
                        processing_time_seconds=processing_time,
                    )

                # Step 2: Fetch document metadata (for existing content)
                try:
                    document = await self._client.get_document(document_id)
                    existing_content = document.content
                except PaperlessError:
                    existing_content = ""

                # Step 3: Stage 1 (OCR)
                if stage1_config.enabled:
                    stage1_result = await self._run_stage1(
                        original_path, ocr_output_path, force=force
                    )
                    self._logger.info(
                        "pipeline_stage1_completed",
                        document_id=document_id,
                        success=stage1_result.success,
                        skipped=stage1_result.skipped,
                        processing_time_seconds=stage1_result.processing_time_seconds,
                    )
                    # Log known limitation: OCR PDF upload not supported
                    if stage1_result.success and not stage1_result.skipped:
                        self._logger.info(
                            "pipeline_ocr_upload_skipped",
                            document_id=document_id,
                            reason="Paperless-ngx API does not support "
                            "replacing existing document files",
                        )

                # Step 4: Stage 2 (Markdown)
                if stage2_config.enabled:
                    stage2_input = self._determine_stage2_input(
                        original_path, stage1_result
                    )
                    stage2_result = await self._run_stage2(stage2_input, force=force)
                    self._logger.info(
                        "pipeline_stage2_completed",
                        document_id=document_id,
                        success=stage2_result.success,
                        skipped=stage2_result.skipped,
                        processing_time_seconds=stage2_result.processing_time_seconds,
                    )

                    # Update content if Stage 2 succeeded
                    if stage2_result.success and not stage2_result.skipped:
                        content_updated = await self._update_content(
                            document_id,
                            stage2_result.markdown,
                            existing_content,
                        )

            # Step 5: Determine overall success
            success = self._determine_success(
                stage1_result=stage1_result,
                stage2_result=stage2_result,
                stage1_enabled=stage1_config.enabled,
                stage2_enabled=stage2_config.enabled,
            )

            # Step 6: Update tags
            tags_updated = await self._update_tags(document_id, success=success)

            processing_time = time.monotonic() - start_time
            self._logger.info(
                "pipeline_completed",
                document_id=document_id,
                success=success,
                processing_time_seconds=processing_time,
            )

            return PipelineResult(
                document_id=document_id,
                success=success,
                stage1_result=stage1_result,
                stage2_result=stage2_result,
                stage1_skipped_by_config=not stage1_config.enabled,
                stage2_skipped_by_config=not stage2_config.enabled,
                tags_updated=tags_updated,
                content_updated=content_updated,
                document_uploaded=False,
                processing_time_seconds=processing_time,
            )

        except Exception as exc:
            processing_time = time.monotonic() - start_time
            self._logger.exception(
                "pipeline_error",
                document_id=document_id,
                error=str(exc),
            )
            tags_updated = await self._update_tags(document_id, success=False)
            return PipelineResult(
                document_id=document_id,
                success=False,
                stage1_result=stage1_result,
                stage2_result=stage2_result,
                stage1_skipped_by_config=not stage1_config.enabled,
                stage2_skipped_by_config=not stage2_config.enabled,
                tags_updated=tags_updated,
                content_updated=content_updated,
                document_uploaded=False,
                error=str(exc),
                processing_time_seconds=processing_time,
            )

    async def _run_stage1(
        self,
        input_path: Path,
        output_path: Path,
        *,
        force: bool,
    ) -> Stage1Result:
        """Run Stage 1 OCR processing.

        Args:
            input_path: Path to input PDF.
            output_path: Path for OCR'd output PDF.
            force: If True, force OCR regardless of born-digital status.

        Returns:
            Stage1Result with processing outcome.
        """
        processor = Stage1Processor(
            config=self._settings.pipeline.stage1,
            gpu_mode=self._settings.gpu.enabled,
        )
        return await processor.process(input_path, output_path, force=force)

    async def _run_stage2(
        self,
        input_path: Path,
        *,
        force: bool,
    ) -> Stage2Result:
        """Run Stage 2 Markdown conversion.

        Args:
            input_path: Path to input PDF.
            force: If True, force processing even if disabled.

        Returns:
            Stage2Result with conversion outcome.
        """
        processor = Stage2Processor(
            config=self._settings.pipeline.stage2,
            gpu_mode=self._settings.gpu.enabled,
        )
        return await processor.process(input_path, force=force)

    async def _update_content(
        self,
        document_id: int,
        markdown: str,
        existing_content: str,
    ) -> bool:
        """Update document content in paperless-ngx.

        Respects the configured content mode (REPLACE or APPEND).

        Args:
            document_id: The document ID.
            markdown: The new Markdown content.
            existing_content: The document's existing content.

        Returns:
            True if the content was successfully updated.
        """
        content_mode = self._settings.pipeline.stage2.content_mode

        if content_mode == ContentMode.APPEND and existing_content:
            content = f"{existing_content}\n\n{markdown}"
        else:
            content = markdown

        try:
            await self._client.update_document(
                document_id,
                DocumentUpdate(content=content),
            )
        except PaperlessError as exc:
            self._logger.exception(
                "content_update_failed",
                document_id=document_id,
                error=str(exc),
            )
            return False
        else:
            self._logger.info(
                "content_updated",
                document_id=document_id,
                content_mode=content_mode.value,
                content_length=len(content),
            )
            return True

    async def _update_tags(
        self,
        document_id: int,
        *,
        success: bool,
    ) -> bool:
        """Update workflow tags on the document.

        Removes the pending tag and adds either the completed or failed tag.

        Args:
            document_id: The document ID.
            success: Whether the pipeline succeeded.

        Returns:
            True if tags were successfully updated.
        """
        prefix = self._settings.tags.prefix
        pending_name = f"{prefix}:pending"
        result_name = f"{prefix}:completed" if success else f"{prefix}:failed"

        try:
            # Ensure tags exist
            pending_tag = await self._client.ensure_tag(pending_name)
            result_tag = await self._client.ensure_tag(result_name)

            # Remove pending, add result
            await self._client.remove_tags_from_document(document_id, [pending_tag.id])
            await self._client.add_tags_to_document(document_id, [result_tag.id])
        except PaperlessError as exc:
            self._logger.exception(
                "tag_update_failed",
                document_id=document_id,
                error=str(exc),
            )
            return False
        else:
            self._logger.info(
                "tags_updated",
                document_id=document_id,
                success=success,
            )
            return True

    def _determine_stage2_input(
        self,
        original_path: Path,
        stage1_result: Stage1Result | None,
    ) -> Path:
        """Determine the input file for Stage 2.

        Uses the OCR'd output from Stage 1 if it succeeded, otherwise
        falls back to the original PDF.

        Args:
            original_path: Path to the original PDF.
            stage1_result: Stage 1 result, or None if Stage 1 was not run.

        Returns:
            Path to the PDF file to use for Stage 2.
        """
        if (
            stage1_result is not None
            and stage1_result.success
            and not stage1_result.skipped
            and stage1_result.output_path is not None
            and stage1_result.output_path.exists()
        ):
            return stage1_result.output_path
        return original_path

    def _determine_success(
        self,
        *,
        stage1_result: Stage1Result | None,
        stage2_result: Stage2Result | None,
        stage1_enabled: bool,
        stage2_enabled: bool,
    ) -> bool:
        """Determine overall pipeline success.

        Both enabled stages must succeed (or be skipped within the stage)
        for the pipeline to be considered successful.

        Args:
            stage1_result: Stage 1 result, or None if not run.
            stage2_result: Stage 2 result, or None if not run.
            stage1_enabled: Whether Stage 1 was enabled in config.
            stage2_enabled: Whether Stage 2 was enabled in config.

        Returns:
            True if the pipeline succeeded overall.
        """
        if stage1_enabled and stage1_result is not None and not stage1_result.success:
            return False

        return not (
            stage2_enabled and stage2_result is not None and not stage2_result.success
        )


async def process_document(
    document_id: int,
    *,
    settings: Settings,
    client: PaperlessClient,
    force: bool = False,
) -> PipelineResult:
    """Convenience function to process a document through the full pipeline.

    Creates a PipelineOrchestrator and processes the document in a single call.

    Args:
        document_id: The paperless-ngx document ID.
        settings: Application settings.
        client: Paperless-ngx API client.
        force: If True, force processing regardless of born-digital status.

    Returns:
        PipelineResult with the processing outcome.

    Example:
        ```python
        from paperless_ngx_smart_ocr.config import get_settings
        from paperless_ngx_smart_ocr.paperless import PaperlessClient

        settings = get_settings()
        async with PaperlessClient(
            base_url=settings.paperless.url,
            token=settings.paperless.token,
        ) as client:
            result = await process_document(
                123,
                settings=settings,
                client=client,
            )
        ```
    """
    orchestrator = PipelineOrchestrator(settings=settings, client=client)
    return await orchestrator.process(document_id, force=force)
