"""Unit tests for the pipeline orchestrator module."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from paperless_ngx_smart_ocr.config import Settings
from paperless_ngx_smart_ocr.paperless.exceptions import (
    PaperlessConnectionError,
    PaperlessError,
)
from paperless_ngx_smart_ocr.paperless.models import (
    Document,
    DocumentUpdate,
    Tag,
)
from paperless_ngx_smart_ocr.pipeline import (
    PipelineOrchestrator,
    PipelineResult,
    process_document,
)
from paperless_ngx_smart_ocr.pipeline.models import (
    DocumentAnalysis,
    PageAnalysis,
    Stage1Result,
    Stage2Result,
)


if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides: object) -> Settings:
    """Create a Settings instance with sensible defaults for testing."""
    return Settings(
        paperless={"url": "http://localhost:8000", "token": "test-token"},
        **overrides,  # type: ignore[arg-type]
    )


def _make_mock_client() -> AsyncMock:
    """Create a mock PaperlessClient with sensible defaults."""
    client = AsyncMock()

    # download_document_to_path returns the path and creates a file
    async def _download(
        doc_id: int,
        dest: Path,
        *,
        original: bool = True,
    ) -> Path:
        dest.write_bytes(b"%PDF-1.4 fake")
        return dest

    client.download_document_to_path = AsyncMock(side_effect=_download)

    # get_document returns a stub Document
    client.get_document = AsyncMock(
        return_value=Document(
            id=1,
            title="Test Document",
            content="Existing content",
            tags=[10],
            created=datetime(2024, 1, 1, tzinfo=UTC),
            created_date="2024-01-01",
            modified=datetime(2024, 1, 1, tzinfo=UTC),
            added=datetime(2024, 1, 1, tzinfo=UTC),
        )
    )

    # update_document returns the same document
    client.update_document = AsyncMock(
        return_value=Document(
            id=1,
            title="Test Document",
            content="Updated",
            tags=[10],
            created=datetime(2024, 1, 1, tzinfo=UTC),
            created_date="2024-01-01",
            modified=datetime(2024, 1, 1, tzinfo=UTC),
            added=datetime(2024, 1, 1, tzinfo=UTC),
        )
    )

    # Tag operations
    pending_tag = Tag(id=100, slug="smart-ocr-pending", name="smart-ocr:pending")
    completed_tag = Tag(id=101, slug="smart-ocr-completed", name="smart-ocr:completed")
    failed_tag = Tag(id=102, slug="smart-ocr-failed", name="smart-ocr:failed")

    def _ensure_tag(name: str, **kwargs: object) -> Tag:
        if name == "smart-ocr:pending":
            return pending_tag
        if name == "smart-ocr:completed":
            return completed_tag
        if name == "smart-ocr:failed":
            return failed_tag
        return Tag(id=999, slug=name, name=name)

    client.ensure_tag = AsyncMock(side_effect=_ensure_tag)
    client.add_tags_to_document = AsyncMock(
        return_value=Document(
            id=1,
            title="Test Document",
            content="",
            tags=[],
            created=datetime(2024, 1, 1, tzinfo=UTC),
            created_date="2024-01-01",
            modified=datetime(2024, 1, 1, tzinfo=UTC),
            added=datetime(2024, 1, 1, tzinfo=UTC),
        )
    )
    client.remove_tags_from_document = AsyncMock(
        return_value=Document(
            id=1,
            title="Test Document",
            content="",
            tags=[],
            created=datetime(2024, 1, 1, tzinfo=UTC),
            created_date="2024-01-01",
            modified=datetime(2024, 1, 1, tzinfo=UTC),
            added=datetime(2024, 1, 1, tzinfo=UTC),
        )
    )

    return client


def _make_stage1_success(tmp_path: Path) -> Stage1Result:
    """Create a successful Stage1Result."""
    output = tmp_path / "ocr_output.pdf"
    output.write_bytes(b"%PDF-1.4 ocr")
    return Stage1Result(
        success=True,
        input_path=tmp_path / "original.pdf",
        output_path=output,
        document_analysis=DocumentAnalysis(
            total_pages=2,
            pages=[
                PageAnalysis(
                    0,
                    has_text=True,
                    text_char_count=500,
                    is_image_only=False,
                ),
                PageAnalysis(
                    1,
                    has_text=True,
                    text_char_count=300,
                    is_image_only=False,
                ),
            ],
            has_any_text=True,
            is_born_digital=False,
        ),
        layout_results=None,
        pages_processed=2,
        skipped=False,
        skip_reason=None,
        processing_time_seconds=3.0,
    )


def _make_stage1_skipped(tmp_path: Path) -> Stage1Result:
    """Create a skipped (born-digital) Stage1Result."""
    return Stage1Result(
        success=True,
        input_path=tmp_path / "original.pdf",
        output_path=None,
        document_analysis=DocumentAnalysis(
            total_pages=2,
            pages=[
                PageAnalysis(
                    0,
                    has_text=True,
                    text_char_count=500,
                    is_image_only=False,
                ),
                PageAnalysis(
                    1,
                    has_text=True,
                    text_char_count=300,
                    is_image_only=False,
                ),
            ],
            has_any_text=True,
            is_born_digital=True,
        ),
        layout_results=None,
        pages_processed=0,
        skipped=True,
        skip_reason="Document has existing text layer (born-digital)",
        processing_time_seconds=0.1,
    )


def _make_stage1_failed(tmp_path: Path) -> Stage1Result:
    """Create a failed Stage1Result."""
    return Stage1Result(
        success=False,
        input_path=tmp_path / "original.pdf",
        output_path=None,
        document_analysis=DocumentAnalysis(
            total_pages=0,
            pages=[],
            has_any_text=False,
            is_born_digital=False,
        ),
        layout_results=None,
        pages_processed=0,
        skipped=False,
        skip_reason=None,
        error="OCR failed: missing dependency",
        processing_time_seconds=1.0,
    )


def _make_stage2_success(tmp_path: Path) -> Stage2Result:
    """Create a successful Stage2Result."""
    return Stage2Result(
        success=True,
        input_path=tmp_path / "ocr_output.pdf",
        markdown="# Test Document\n\nHello world.\n",
        page_count=2,
        images={},
        metadata={},
        llm_used=False,
        skipped=False,
        skip_reason=None,
        processing_time_seconds=2.0,
    )


def _make_stage2_failed(tmp_path: Path) -> Stage2Result:
    """Create a failed Stage2Result."""
    return Stage2Result(
        success=False,
        input_path=tmp_path / "ocr_output.pdf",
        markdown="",
        page_count=0,
        images={},
        metadata={},
        llm_used=False,
        skipped=False,
        skip_reason=None,
        error="Marker conversion failed",
        processing_time_seconds=1.0,
    )


# ---------------------------------------------------------------------------
# PipelineResult Tests
# ---------------------------------------------------------------------------


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_to_dict_success(self, tmp_path: Path) -> None:
        """Test to_dict serialization for a successful result."""
        stage1 = _make_stage1_success(tmp_path)
        stage2 = _make_stage2_success(tmp_path)
        result = PipelineResult(
            document_id=42,
            success=True,
            stage1_result=stage1,
            stage2_result=stage2,
            stage1_skipped_by_config=False,
            stage2_skipped_by_config=False,
            tags_updated=True,
            content_updated=True,
            document_uploaded=False,
            processing_time_seconds=5.0,
        )

        d = result.to_dict()
        assert d["document_id"] == 42
        assert d["success"] is True
        assert d["stage1_result"] is not None
        assert d["stage2_result"] is not None
        assert d["tags_updated"] is True
        assert d["content_updated"] is True
        assert d["document_uploaded"] is False
        assert d["error"] is None
        assert d["processing_time_seconds"] == 5.0
        assert "created_at" in d

    def test_to_dict_failure(self) -> None:
        """Test to_dict serialization for a failed result."""
        result = PipelineResult(
            document_id=42,
            success=False,
            stage1_result=None,
            stage2_result=None,
            stage1_skipped_by_config=False,
            stage2_skipped_by_config=False,
            tags_updated=False,
            content_updated=False,
            document_uploaded=False,
            error="Download failed",
        )

        d = result.to_dict()
        assert d["success"] is False
        assert d["stage1_result"] is None
        assert d["stage2_result"] is None
        assert d["error"] == "Download failed"

    def test_is_terminal_success(self) -> None:
        """Test is_terminal for successful result."""
        result = PipelineResult(
            document_id=1,
            success=True,
            stage1_result=None,
            stage2_result=None,
            stage1_skipped_by_config=True,
            stage2_skipped_by_config=True,
            tags_updated=True,
            content_updated=False,
            document_uploaded=False,
        )
        assert result.is_terminal is True

    def test_is_terminal_with_error(self) -> None:
        """Test is_terminal for result with error."""
        result = PipelineResult(
            document_id=1,
            success=False,
            stage1_result=None,
            stage2_result=None,
            stage1_skipped_by_config=False,
            stage2_skipped_by_config=False,
            tags_updated=False,
            content_updated=False,
            document_uploaded=False,
            error="Something went wrong",
        )
        assert result.is_terminal is True

    def test_is_terminal_pending(self) -> None:
        """Test is_terminal for a non-terminal (in-progress) result."""
        result = PipelineResult(
            document_id=1,
            success=False,
            stage1_result=None,
            stage2_result=None,
            stage1_skipped_by_config=False,
            stage2_skipped_by_config=False,
            tags_updated=False,
            content_updated=False,
            document_uploaded=False,
        )
        assert result.is_terminal is False

    def test_created_at_default(self) -> None:
        """Test that created_at defaults to now."""
        before = datetime.now(UTC)
        result = PipelineResult(
            document_id=1,
            success=True,
            stage1_result=None,
            stage2_result=None,
            stage1_skipped_by_config=True,
            stage2_skipped_by_config=True,
            tags_updated=True,
            content_updated=False,
            document_uploaded=False,
        )
        after = datetime.now(UTC)
        assert before <= result.created_at <= after

    def test_both_stages_skipped_by_config(self) -> None:
        """Test result when both stages are disabled by config."""
        result = PipelineResult(
            document_id=1,
            success=True,
            stage1_result=None,
            stage2_result=None,
            stage1_skipped_by_config=True,
            stage2_skipped_by_config=True,
            tags_updated=True,
            content_updated=False,
            document_uploaded=False,
        )
        assert result.success is True
        assert result.stage1_skipped_by_config is True
        assert result.stage2_skipped_by_config is True

    def test_to_dict_includes_nested_results(self, tmp_path: Path) -> None:
        """Test that to_dict includes stage result dicts."""
        stage1 = _make_stage1_success(tmp_path)
        result = PipelineResult(
            document_id=42,
            success=True,
            stage1_result=stage1,
            stage2_result=None,
            stage1_skipped_by_config=False,
            stage2_skipped_by_config=True,
            tags_updated=True,
            content_updated=False,
            document_uploaded=False,
        )
        d = result.to_dict()
        assert d["stage1_result"]["success"] is True
        assert d["stage1_result"]["pages_processed"] == 2
        assert d["stage2_result"] is None


# ---------------------------------------------------------------------------
# PipelineOrchestrator Tests
# ---------------------------------------------------------------------------


class TestPipelineOrchestrator:
    """Tests for PipelineOrchestrator."""

    # ----- Happy path -----

    @pytest.mark.asyncio
    async def test_both_stages_succeed(self, tmp_path: Path) -> None:
        """Test pipeline with both stages enabled and succeeding."""
        settings = _make_settings()
        client = _make_mock_client()

        stage1_result = _make_stage1_success(tmp_path)
        stage2_result = _make_stage2_success(tmp_path)

        with (
            patch.object(
                PipelineOrchestrator,
                "_run_stage1",
                return_value=stage1_result,
            ) as mock_s1,
            patch.object(
                PipelineOrchestrator,
                "_run_stage2",
                return_value=stage2_result,
            ) as mock_s2,
        ):
            orchestrator = PipelineOrchestrator(settings=settings, client=client)
            result = await orchestrator.process(42)

        assert result.success is True
        assert result.document_id == 42
        assert result.stage1_result is stage1_result
        assert result.stage2_result is stage2_result
        assert result.stage1_skipped_by_config is False
        assert result.stage2_skipped_by_config is False
        assert result.tags_updated is True
        assert result.content_updated is True
        assert result.document_uploaded is False
        assert result.error is None
        assert result.processing_time_seconds > 0
        mock_s1.assert_called_once()
        mock_s2.assert_called_once()

    @pytest.mark.asyncio
    async def test_stage1_only_enabled(self, tmp_path: Path) -> None:
        """Test pipeline with only Stage 1 enabled."""
        settings = _make_settings(
            pipeline={
                "stage1": {"enabled": True},
                "stage2": {"enabled": False},
            }
        )
        client = _make_mock_client()

        stage1_result = _make_stage1_success(tmp_path)

        with patch.object(
            PipelineOrchestrator,
            "_run_stage1",
            return_value=stage1_result,
        ):
            orchestrator = PipelineOrchestrator(settings=settings, client=client)
            result = await orchestrator.process(42)

        assert result.success is True
        assert result.stage1_result is stage1_result
        assert result.stage2_result is None
        assert result.stage2_skipped_by_config is True
        assert result.content_updated is False

    @pytest.mark.asyncio
    async def test_stage2_only_enabled(self, tmp_path: Path) -> None:
        """Test pipeline with only Stage 2 enabled."""
        settings = _make_settings(
            pipeline={
                "stage1": {"enabled": False},
                "stage2": {"enabled": True},
            }
        )
        client = _make_mock_client()

        stage2_result = _make_stage2_success(tmp_path)

        with patch.object(
            PipelineOrchestrator,
            "_run_stage2",
            return_value=stage2_result,
        ):
            orchestrator = PipelineOrchestrator(settings=settings, client=client)
            result = await orchestrator.process(42)

        assert result.success is True
        assert result.stage1_result is None
        assert result.stage1_skipped_by_config is True
        assert result.stage2_result is stage2_result
        assert result.content_updated is True

    @pytest.mark.asyncio
    async def test_both_stages_disabled(self) -> None:
        """Test both stages disabled succeeds with no processing."""
        settings = _make_settings(
            pipeline={
                "stage1": {"enabled": False},
                "stage2": {"enabled": False},
            }
        )
        client = _make_mock_client()

        orchestrator = PipelineOrchestrator(settings=settings, client=client)
        result = await orchestrator.process(42)

        assert result.success is True
        assert result.stage1_result is None
        assert result.stage2_result is None
        assert result.stage1_skipped_by_config is True
        assert result.stage2_skipped_by_config is True
        assert result.content_updated is False
        assert result.tags_updated is True

    # ----- Stage 1 edge cases -----

    @pytest.mark.asyncio
    async def test_stage1_skipped_born_digital(self, tmp_path: Path) -> None:
        """Test that Stage 2 uses original PDF when Stage 1 skips (born-digital)."""
        settings = _make_settings()
        client = _make_mock_client()

        stage1_result = _make_stage1_skipped(tmp_path)
        stage2_result = _make_stage2_success(tmp_path)

        with (
            patch.object(
                PipelineOrchestrator,
                "_run_stage1",
                return_value=stage1_result,
            ),
            patch.object(
                PipelineOrchestrator,
                "_run_stage2",
                return_value=stage2_result,
            ) as mock_s2,
        ):
            orchestrator = PipelineOrchestrator(settings=settings, client=client)
            result = await orchestrator.process(42)

        assert result.success is True
        # Stage 2 should have been called with the original PDF (not OCR output)
        call_args = mock_s2.call_args
        input_path = call_args[0][0]
        assert input_path.name == "original.pdf"

    @pytest.mark.asyncio
    async def test_stage1_fails_stage2_uses_original(self, tmp_path: Path) -> None:
        """Test that Stage 2 uses original PDF when Stage 1 fails, overall fails."""
        settings = _make_settings()
        client = _make_mock_client()

        stage1_result = _make_stage1_failed(tmp_path)
        stage2_result = _make_stage2_success(tmp_path)

        with (
            patch.object(
                PipelineOrchestrator,
                "_run_stage1",
                return_value=stage1_result,
            ),
            patch.object(
                PipelineOrchestrator,
                "_run_stage2",
                return_value=stage2_result,
            ) as mock_s2,
        ):
            orchestrator = PipelineOrchestrator(settings=settings, client=client)
            result = await orchestrator.process(42)

        assert result.success is False  # Stage 1 failed -> overall fails
        assert result.stage1_result is stage1_result
        assert result.stage2_result is stage2_result
        # Stage 2 should have been called with the original PDF
        call_args = mock_s2.call_args
        input_path = call_args[0][0]
        assert input_path.name == "original.pdf"

    # ----- Stage 2 edge cases -----

    @pytest.mark.asyncio
    async def test_stage2_fails_tags_updated_to_failed(self, tmp_path: Path) -> None:
        """Test that tags are updated to failed when Stage 2 fails."""
        settings = _make_settings()
        client = _make_mock_client()

        stage1_result = _make_stage1_success(tmp_path)
        stage2_result = _make_stage2_failed(tmp_path)

        with (
            patch.object(
                PipelineOrchestrator,
                "_run_stage1",
                return_value=stage1_result,
            ),
            patch.object(
                PipelineOrchestrator,
                "_run_stage2",
                return_value=stage2_result,
            ),
        ):
            orchestrator = PipelineOrchestrator(settings=settings, client=client)
            result = await orchestrator.process(42)

        assert result.success is False
        assert result.tags_updated is True
        # Should have called ensure_tag with "smart-ocr:failed"
        ensure_calls = [call.args[0] for call in client.ensure_tag.call_args_list]
        assert "smart-ocr:failed" in ensure_calls

    @pytest.mark.asyncio
    async def test_content_mode_replace(self, tmp_path: Path) -> None:
        """Test content update with REPLACE mode."""
        settings = _make_settings(pipeline={"stage2": {"content_mode": "replace"}})
        client = _make_mock_client()

        stage1_result = _make_stage1_success(tmp_path)
        stage2_result = _make_stage2_success(tmp_path)

        with (
            patch.object(
                PipelineOrchestrator,
                "_run_stage1",
                return_value=stage1_result,
            ),
            patch.object(
                PipelineOrchestrator,
                "_run_stage2",
                return_value=stage2_result,
            ),
        ):
            orchestrator = PipelineOrchestrator(settings=settings, client=client)
            result = await orchestrator.process(42)

        assert result.content_updated is True
        # Check that update_document was called with just the markdown
        update_call = client.update_document.call_args
        update: DocumentUpdate = update_call[0][1]
        # Pydantic strips whitespace, so compare stripped versions
        assert update.content == stage2_result.markdown.strip()

    @pytest.mark.asyncio
    async def test_content_mode_append(self, tmp_path: Path) -> None:
        """Test content update with APPEND mode."""
        settings = _make_settings(pipeline={"stage2": {"content_mode": "append"}})
        client = _make_mock_client()

        stage1_result = _make_stage1_success(tmp_path)
        stage2_result = _make_stage2_success(tmp_path)

        with (
            patch.object(
                PipelineOrchestrator,
                "_run_stage1",
                return_value=stage1_result,
            ),
            patch.object(
                PipelineOrchestrator,
                "_run_stage2",
                return_value=stage2_result,
            ),
        ):
            orchestrator = PipelineOrchestrator(settings=settings, client=client)
            result = await orchestrator.process(42)

        assert result.content_updated is True
        update_call = client.update_document.call_args
        update: DocumentUpdate = update_call[0][1]
        assert update.content is not None
        assert update.content.startswith("Existing content\n\n")
        assert update.content.endswith(stage2_result.markdown.strip())

    @pytest.mark.asyncio
    async def test_content_mode_append_empty_existing(self, tmp_path: Path) -> None:
        """Test APPEND mode with empty existing content uses markdown directly."""
        settings = _make_settings(pipeline={"stage2": {"content_mode": "append"}})
        client = _make_mock_client()
        # Override get_document to return empty content
        client.get_document = AsyncMock(
            return_value=Document(
                id=1,
                title="Test",
                content="",
                tags=[],
                created=datetime(2024, 1, 1, tzinfo=UTC),
                created_date="2024-01-01",
                modified=datetime(2024, 1, 1, tzinfo=UTC),
                added=datetime(2024, 1, 1, tzinfo=UTC),
            )
        )

        stage1_result = _make_stage1_success(tmp_path)
        stage2_result = _make_stage2_success(tmp_path)

        with (
            patch.object(
                PipelineOrchestrator,
                "_run_stage1",
                return_value=stage1_result,
            ),
            patch.object(
                PipelineOrchestrator,
                "_run_stage2",
                return_value=stage2_result,
            ),
        ):
            orchestrator = PipelineOrchestrator(settings=settings, client=client)
            result = await orchestrator.process(42)

        assert result.content_updated is True
        update_call = client.update_document.call_args
        update: DocumentUpdate = update_call[0][1]
        # With empty existing content, should just use the markdown directly
        # Pydantic strips whitespace, so compare stripped versions
        assert update.content == stage2_result.markdown.strip()

    # ----- Tag management -----

    @pytest.mark.asyncio
    async def test_success_tags_updated(self, tmp_path: Path) -> None:
        """Test that on success, pending tag is removed and completed tag is added."""
        settings = _make_settings()
        client = _make_mock_client()

        with (
            patch.object(
                PipelineOrchestrator,
                "_run_stage1",
                return_value=_make_stage1_success(tmp_path),
            ),
            patch.object(
                PipelineOrchestrator,
                "_run_stage2",
                return_value=_make_stage2_success(tmp_path),
            ),
        ):
            orchestrator = PipelineOrchestrator(settings=settings, client=client)
            result = await orchestrator.process(42)

        assert result.tags_updated is True
        # Verify pending tag was removed
        client.remove_tags_from_document.assert_called_once_with(42, [100])
        # Verify completed tag was added
        client.add_tags_to_document.assert_called_once_with(42, [101])

    @pytest.mark.asyncio
    async def test_failure_tags_updated(self, tmp_path: Path) -> None:
        """Test that on failure, pending tag is removed and failed tag is added."""
        settings = _make_settings()
        client = _make_mock_client()

        with (
            patch.object(
                PipelineOrchestrator,
                "_run_stage1",
                return_value=_make_stage1_failed(tmp_path),
            ),
            patch.object(
                PipelineOrchestrator,
                "_run_stage2",
                return_value=_make_stage2_success(tmp_path),
            ),
        ):
            orchestrator = PipelineOrchestrator(settings=settings, client=client)
            result = await orchestrator.process(42)

        assert result.tags_updated is True
        client.remove_tags_from_document.assert_called_once_with(42, [100])
        client.add_tags_to_document.assert_called_once_with(42, [102])

    @pytest.mark.asyncio
    async def test_tag_update_failure_logged_not_crash(self, tmp_path: Path) -> None:
        """Test that tag update failure doesn't crash the pipeline."""
        settings = _make_settings(
            pipeline={"stage1": {"enabled": False}, "stage2": {"enabled": False}}
        )
        client = _make_mock_client()
        client.ensure_tag = AsyncMock(
            side_effect=PaperlessConnectionError("Connection refused")
        )

        orchestrator = PipelineOrchestrator(settings=settings, client=client)
        result = await orchestrator.process(42)

        assert result.success is True  # Processing itself succeeded
        assert result.tags_updated is False  # But tags failed

    # ----- Error paths -----

    @pytest.mark.asyncio
    async def test_download_failure(self) -> None:
        """Test pipeline failure when document download fails."""
        settings = _make_settings()
        client = _make_mock_client()
        client.download_document_to_path = AsyncMock(
            side_effect=PaperlessError("Connection refused")
        )

        orchestrator = PipelineOrchestrator(settings=settings, client=client)
        result = await orchestrator.process(42)

        assert result.success is False
        assert result.error is not None
        assert "Download failed" in result.error
        assert result.stage1_result is None
        assert result.stage2_result is None

    @pytest.mark.asyncio
    async def test_content_patch_failure(self, tmp_path: Path) -> None:
        """Test that content PATCH failure is recorded but pipeline still tagged."""
        settings = _make_settings()
        client = _make_mock_client()
        client.update_document = AsyncMock(side_effect=PaperlessError("Server error"))

        stage1_result = _make_stage1_success(tmp_path)
        stage2_result = _make_stage2_success(tmp_path)

        with (
            patch.object(
                PipelineOrchestrator,
                "_run_stage1",
                return_value=stage1_result,
            ),
            patch.object(
                PipelineOrchestrator,
                "_run_stage2",
                return_value=stage2_result,
            ),
        ):
            orchestrator = PipelineOrchestrator(settings=settings, client=client)
            result = await orchestrator.process(42)

        # Pipeline succeeds, but content update failed
        assert result.success is True
        assert result.content_updated is False
        assert result.tags_updated is True

    @pytest.mark.asyncio
    async def test_force_flag_propagated(self, tmp_path: Path) -> None:
        """Test that the force flag is propagated to both stages."""
        settings = _make_settings()
        client = _make_mock_client()

        stage1_result = _make_stage1_success(tmp_path)
        stage2_result = _make_stage2_success(tmp_path)

        with (
            patch.object(
                PipelineOrchestrator,
                "_run_stage1",
                return_value=stage1_result,
            ) as mock_s1,
            patch.object(
                PipelineOrchestrator,
                "_run_stage2",
                return_value=stage2_result,
            ) as mock_s2,
        ):
            orchestrator = PipelineOrchestrator(settings=settings, client=client)
            await orchestrator.process(42, force=True)

        # Verify force was passed
        _, s1_kwargs = mock_s1.call_args
        assert s1_kwargs["force"] is True
        _, s2_kwargs = mock_s2.call_args
        assert s2_kwargs["force"] is True

    @pytest.mark.asyncio
    async def test_unexpected_exception_caught(self) -> None:
        """Test that unexpected exceptions are caught and returned as failures."""
        settings = _make_settings()
        client = _make_mock_client()
        client.download_document_to_path = AsyncMock(
            side_effect=RuntimeError("Unexpected!")
        )

        orchestrator = PipelineOrchestrator(settings=settings, client=client)
        result = await orchestrator.process(42)

        assert result.success is False
        assert result.error is not None
        assert "Unexpected!" in result.error

    # ----- Cleanup -----

    @pytest.mark.asyncio
    async def test_temp_files_cleaned_up_on_success(self, tmp_path: Path) -> None:
        """Test that temporary files are cleaned up after success."""
        settings = _make_settings(
            pipeline={"stage1": {"enabled": False}, "stage2": {"enabled": False}}
        )
        client = _make_mock_client()

        orchestrator = PipelineOrchestrator(settings=settings, client=client)
        result = await orchestrator.process(42)

        assert result.success is True
        # TemporaryDirectory context manager handles cleanup

    @pytest.mark.asyncio
    async def test_temp_files_cleaned_up_on_failure(self) -> None:
        """Test that temporary files are cleaned up after failure."""
        settings = _make_settings()
        client = _make_mock_client()
        client.download_document_to_path = AsyncMock(side_effect=PaperlessError("fail"))

        orchestrator = PipelineOrchestrator(settings=settings, client=client)
        result = await orchestrator.process(42)

        assert result.success is False
        # TemporaryDirectory context manager handles cleanup even on failure

    @pytest.mark.asyncio
    async def test_processing_time_tracked(self, tmp_path: Path) -> None:
        """Test that processing time is tracked in the result."""
        settings = _make_settings(
            pipeline={"stage1": {"enabled": False}, "stage2": {"enabled": False}}
        )
        client = _make_mock_client()

        orchestrator = PipelineOrchestrator(settings=settings, client=client)
        result = await orchestrator.process(42)

        assert result.processing_time_seconds > 0

    # ----- _determine_stage2_input -----

    def test_determine_stage2_input_uses_ocr_output(self, tmp_path: Path) -> None:
        """Test that Stage 2 uses OCR output when available."""
        settings = _make_settings()
        client = _make_mock_client()
        orchestrator = PipelineOrchestrator(settings=settings, client=client)

        stage1_result = _make_stage1_success(tmp_path)
        original_path = tmp_path / "original.pdf"

        result = orchestrator._determine_stage2_input(original_path, stage1_result)
        assert result == stage1_result.output_path

    def test_stage2_input_uses_original_when_skipped(self, tmp_path: Path) -> None:
        """Test that Stage 2 uses original when Stage 1 was skipped."""
        settings = _make_settings()
        client = _make_mock_client()
        orchestrator = PipelineOrchestrator(settings=settings, client=client)

        stage1_result = _make_stage1_skipped(tmp_path)
        original_path = tmp_path / "original.pdf"

        result = orchestrator._determine_stage2_input(original_path, stage1_result)
        assert result == original_path

    def test_stage2_input_uses_original_when_failed(self, tmp_path: Path) -> None:
        """Test that Stage 2 uses original when Stage 1 failed."""
        settings = _make_settings()
        client = _make_mock_client()
        orchestrator = PipelineOrchestrator(settings=settings, client=client)

        stage1_result = _make_stage1_failed(tmp_path)
        original_path = tmp_path / "original.pdf"

        result = orchestrator._determine_stage2_input(original_path, stage1_result)
        assert result == original_path

    def test_stage2_input_uses_original_when_no_stage1(self, tmp_path: Path) -> None:
        """Test that Stage 2 uses original when Stage 1 was not run."""
        settings = _make_settings()
        client = _make_mock_client()
        orchestrator = PipelineOrchestrator(settings=settings, client=client)

        original_path = tmp_path / "original.pdf"

        result = orchestrator._determine_stage2_input(original_path, None)
        assert result == original_path

    # ----- _determine_success -----

    def test_determine_success_both_succeed(self, tmp_path: Path) -> None:
        """Test overall success when both stages succeed."""
        settings = _make_settings()
        client = _make_mock_client()
        orchestrator = PipelineOrchestrator(settings=settings, client=client)

        assert (
            orchestrator._determine_success(
                stage1_result=_make_stage1_success(tmp_path),
                stage2_result=_make_stage2_success(tmp_path),
                stage1_enabled=True,
                stage2_enabled=True,
            )
            is True
        )

    def test_determine_success_stage1_fails(self, tmp_path: Path) -> None:
        """Test overall failure when Stage 1 fails."""
        settings = _make_settings()
        client = _make_mock_client()
        orchestrator = PipelineOrchestrator(settings=settings, client=client)

        assert (
            orchestrator._determine_success(
                stage1_result=_make_stage1_failed(tmp_path),
                stage2_result=_make_stage2_success(tmp_path),
                stage1_enabled=True,
                stage2_enabled=True,
            )
            is False
        )

    def test_determine_success_stage2_fails(self, tmp_path: Path) -> None:
        """Test overall failure when Stage 2 fails."""
        settings = _make_settings()
        client = _make_mock_client()
        orchestrator = PipelineOrchestrator(settings=settings, client=client)

        assert (
            orchestrator._determine_success(
                stage1_result=_make_stage1_success(tmp_path),
                stage2_result=_make_stage2_failed(tmp_path),
                stage1_enabled=True,
                stage2_enabled=True,
            )
            is False
        )

    def test_determine_success_disabled_stages(self) -> None:
        """Test success when stages are disabled (no results)."""
        settings = _make_settings()
        client = _make_mock_client()
        orchestrator = PipelineOrchestrator(settings=settings, client=client)

        assert (
            orchestrator._determine_success(
                stage1_result=None,
                stage2_result=None,
                stage1_enabled=False,
                stage2_enabled=False,
            )
            is True
        )


# ---------------------------------------------------------------------------
# process_document Function Tests
# ---------------------------------------------------------------------------


class TestProcessDocumentFunction:
    """Tests for the process_document convenience function."""

    @pytest.mark.asyncio
    async def test_creates_orchestrator_and_delegates(self) -> None:
        """Test that process_document creates an orchestrator and calls process."""
        settings = _make_settings(
            pipeline={"stage1": {"enabled": False}, "stage2": {"enabled": False}}
        )
        client = _make_mock_client()

        result = await process_document(
            42,
            settings=settings,
            client=client,
            force=True,
        )

        assert result.document_id == 42
        assert result.success is True
