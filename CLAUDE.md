# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

paperless-ngx-smart-ocr is a Python service that enhances paperless-ngx with intelligent, layout-aware OCR and Markdown conversion. It processes PDFs through a two-stage pipeline:
- **Stage 1**: Add searchable text layers using OCRmyPDF + Surya layout detection
- **Stage 2**: Convert to structured Markdown using Marker, store in paperless-ngx content field

See README.md for the full implementation checklist and current status.

## Development Commands

```bash
# Setup
mise install                   # Install Python 3.12 + uv + jq
uv sync --all-extras           # Install all dependencies including dev

# Running the application
uv run smart-ocr serve         # Start web server with workers
uv run smart-ocr process <id>  # Process single document
uv run smart-ocr --verbose     # Enable debug logging
uv run smart-ocr --quiet       # Only warnings and errors

# Testing
uv run pytest                              # Run all tests
uv run pytest tests/unit/test_config.py   # Run single test file
uv run pytest -k "test_name"              # Run tests matching pattern
uv run pytest --cov=paperless_ngx_smart_ocr --cov-report=html  # With coverage

# Code quality (run before committing)
uv run ruff check .            # Lint code
uv run ruff check --fix .      # Auto-fix lint issues
uv run ruff format .           # Format code
uv run mypy src/               # Type check (strict mode)
uv run pre-commit run --all-files  # Run all hooks
```

## Architecture

### Processing Pipeline

```
Document → Pre-Processing Analysis → Stage 1 (OCR) → Stage 2 (Markdown) → Post-Processing
                                         ↓                ↓
                                   OCRmyPDF +        Marker →
                                   Surya Layout      PATCH content
```

Each stage can be independently enabled/disabled via configuration. The `PipelineOrchestrator` coordinates the full workflow: download, Stage 1, Stage 2, content PATCH, tag update, and temp file cleanup.

### Integration Patterns

Three ways to trigger processing:
1. **Polling** (recommended): Service polls paperless-ngx API at intervals
2. **Webhook**: paperless-ngx Workflow sends POST on document events
3. **Post-consume**: CLI runs synchronously after document consumption

### Implemented Modules

- **`config/`** - Pydantic settings with YAML + env vars, `${VAR}` interpolation, secret file support
- **`paperless/`** - Async API client for paperless-ngx (PaperlessClient, Document/Tag models)
- **`observability/`** - Structured logging with structlog, request ID tracking
- **`workers/`** - Background job queue (JobQueue wrapping aiojobs), job status tracking
- **`pipeline/`** - Two-stage processing pipeline with orchestrator
  - `stage1_ocr.py` - Stage1Processor for OCR via OCRmyPDF + Surya layout
  - `stage2_markdown.py` - Stage2Processor for Markdown via Marker
  - `orchestrator.py` - PipelineOrchestrator coordinating both stages end-to-end
  - `models.py` - Result dataclasses (Stage1Result, Stage2Result, PipelineResult)
- **`web/`** - FastAPI application with htmx UI
  - `app.py` - Application factory (`create_app`), lifespan, middleware, exception handlers
  - `routes/health.py` - Health (`/api/health`) and readiness (`/api/ready`) endpoints
  - Dependency helpers: `get_app_settings`, `get_job_queue`, `get_paperless_client`
- **`cli/`** - Typer CLI (entry point: `smart-ocr`)

### Key Architectural Patterns

- **Processors never raise**: `Stage1Processor`, `Stage2Processor`, and `PipelineOrchestrator.process()` always return result dataclasses (never raise exceptions). Errors are captured in the result's `error` field.
- **Stage chaining**: If Stage 1 produces OCR output, Stage 2 uses it; if Stage 1 skips/fails, Stage 2 falls back to original PDF.
- **Tag-based workflow**: Documents tracked via tags (smart-ocr:pending, :completed, :failed, :skip). The orchestrator manages tag transitions.
- **Content modes**: Stage 2 supports REPLACE (overwrite) or APPEND (preserve existing content) modes when patching the paperless-ngx content field.
- **Known limitation**: The paperless-ngx API has no endpoint for replacing a document's file in-place, so OCR'd PDFs are only used within the pipeline run (not re-uploaded).

### Module Usage Examples

#### Configuration

```python
from paperless_ngx_smart_ocr.config import load_settings, get_settings

# Load from YAML file or environment
settings = load_settings()  # Searches ./config.yaml, ~/.config/smart-ocr/config.yaml
settings = load_settings("/path/to/config.yaml")

# Access nested settings
settings.paperless.url           # http://localhost:8000
settings.pipeline.stage1.enabled # True
settings.web.port                # 8080

# Use cached singleton
settings = get_settings()
```

Environment variables use `SMARTOCR_` prefix with `__` delimiter (e.g., `SMARTOCR_WEB__PORT=9000`).

#### Paperless-ngx Client

```python
from paperless_ngx_smart_ocr.paperless import PaperlessClient, DocumentUpdate

async with PaperlessClient(base_url, token) as client:
    docs = await client.list_documents(tags_include=[1], tags_exclude=[2])
    await client.update_document(doc_id, DocumentUpdate(content=markdown))
    tag = await client.ensure_tag("smart-ocr:pending")
    await client.add_tags_to_document(doc_id, [tag.id])
```

#### Logging

```python
from paperless_ngx_smart_ocr.observability import configure_logging, get_logger, set_request_id

configure_logging(level="debug")  # At startup
logger = get_logger(__name__)
set_request_id()  # For request correlation
logger.info("processing_document", document_id=123, stage="ocr")
```

#### Pipeline Orchestrator

```python
from paperless_ngx_smart_ocr.pipeline import PipelineOrchestrator, process_document

# Full pipeline: download → Stage 1 → Stage 2 → content PATCH → tag update
async with PaperlessClient(base_url, token) as client:
    result = await process_document(
        document_id,
        settings=settings,
        client=client,
    )
    # result.success, result.stage1_result, result.stage2_result,
    # result.tags_updated, result.content_updated, result.error
```

#### Pipeline (Stage 1 OCR)

```python
from paperless_ngx_smart_ocr.pipeline import analyze_document, process_stage1

# Analyze document for born-digital detection
analysis = analyze_document(pdf_path)

# Process through Stage 1 OCR
result = await process_stage1(
    input_path, output_path,
    config=settings.pipeline.stage1,
    gpu_mode=settings.gpu.enabled,
)
```

#### Pipeline (Stage 2 Markdown)

```python
from paperless_ngx_smart_ocr.pipeline import process_stage2, get_marker_models

# Pre-load Marker models (cached singleton, expensive first call)
await get_marker_models()

# Process through Stage 2 Markdown conversion
result = await process_stage2(
    pdf_path,
    config=settings.pipeline.stage2,
    gpu_mode=settings.gpu.enabled,
)
```

#### Job Queue

```python
from paperless_ngx_smart_ocr.workers import JobQueue, JobStatus

async with JobQueue(workers=4, timeout=600) as queue:
    job = await queue.submit(
        process_document(doc_id),
        name="Process document 123",
        document_id=doc_id,
    )
    completed = await queue.wait(job.id)
```

#### Web Application

```python
from paperless_ngx_smart_ocr.web import create_app
from paperless_ngx_smart_ocr.config import load_settings

# Create app with custom settings
settings = load_settings("/path/to/config.yaml")
app = create_app(settings=settings)

# Or use defaults (loads from standard config locations)
app = create_app()

# Lifespan auto-manages JobQueue and PaperlessClient on app.state
# Access in routes via request.app.state.job_queue, request.app.state.client
# Or use dependency helpers: get_app_settings, get_job_queue, get_paperless_client
```

## Code Style

- All modules use `from __future__ import annotations`
- All modules define `__all__` exports
- PEP 561 type checking enabled (py.typed marker)
- Google-style docstrings
- Python 3.12+ features (e.g., `class Foo[T]` for generics)
- Result dataclasses use `@dataclass(slots=True)` (not `frozen=True`)

### Lint/Format Constraints

- **ruff line length**: 88 characters (Black-compatible)
- **ruff enforces**: SIM102 (combine nested ifs), SIM103 (inline return conditions), TRY300 (move return to `else` block), TRY400 (use `logger.exception` not `logger.error` in `except` blocks)
- **Path imports** go in `TYPE_CHECKING` block per TC003 rule
- **mypy**: strict mode with pydantic plugin
- **Pydantic models** have `str_strip_whitespace=True` - be aware `DocumentUpdate` strips trailing whitespace from content strings

### Test Patterns

- Tests use `pytest-asyncio` with `mode=Mode.AUTO` (no need for explicit `@pytest.mark.asyncio` in practice, though it's used for clarity)
- Test files ignore: S101 (assert), ARG001/ARG002 (fixtures), PLR2004 (magic values), SLF001 (private access)
- Fixtures in `tests/conftest.py`: `fixtures_dir`, `pdfs_dir` for test data paths

## Design Principles

1. **No hallucination**: Text extraction is deterministic; LLM mode off by default
2. **Complex layouts supported**: Surya layout detection before OCR handles multi-column, tables, figures
3. **No noise-as-text**: Layout detection filters artifacts, watermarks, headers/footers
4. **Tag-based workflow**: Documents tracked via tags (smart-ocr:pending, :completed, :failed, :skip)
