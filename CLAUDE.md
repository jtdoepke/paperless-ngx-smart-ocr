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

Each stage can be independently enabled/disabled via configuration.

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
- **`cli/`** - Typer CLI (entry point: `smart-ocr`)

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

Output (logfmt, auto-detects TTY for colors):
```
timestamp=2024-01-15T10:30:45Z level=info event=processing_document document_id=123 request_id=abc12345
```

#### Job Queue

```python
from paperless_ngx_smart_ocr.workers import JobQueue, JobStatus

async with JobQueue(workers=4, timeout=600) as queue:
    # Submit a background job
    job = await queue.submit(
        process_document(doc_id),
        name="Process document 123",
        document_id=doc_id,
    )

    # Wait for completion
    completed = await queue.wait(job.id)
    if completed.status == JobStatus.COMPLETED:
        print(f"Result: {completed.result.value}")

    # Or cancel a running job
    await queue.cancel(job.id)

    # List jobs by status
    active_jobs = await queue.list_active()
    pending_jobs = await queue.list_pending()
```

## Code Style

- All modules use `from __future__ import annotations`
- Empty modules define `__all__: list[str] = []`
- PEP 561 type checking enabled (py.typed marker)
- Google-style docstrings
- Python 3.12+ features (e.g., `class Foo[T]` for generics)

## Design Principles

1. **No hallucination**: Text extraction is deterministic; LLM mode off by default
2. **Complex layouts supported**: Surya layout detection before OCR handles multi-column, tables, figures
3. **No noise-as-text**: Layout detection filters artifacts, watermarks, headers/footers
4. **Tag-based workflow**: Documents tracked via tags (smart-ocr:pending, :completed, :failed, :skip)
