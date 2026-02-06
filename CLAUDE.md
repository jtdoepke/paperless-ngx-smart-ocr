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
mise run serve                 # Start web server with workers (via mise task)
uv run smart-ocr serve         # Start web server with workers (directly)

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

Each stage can be independently enabled/disabled via configuration. The `PipelineOrchestrator` coordinates the full workflow: download from paperless-ngx, Stage 1, Stage 2, content PATCH, tag update, and temp file cleanup. The orchestrator supports `dry_run=True` mode which runs stages but skips content updates and tag changes.

### Key Architectural Patterns

- **Processors never raise**: `Stage1Processor`, `Stage2Processor`, and `PipelineOrchestrator.process()` always return result dataclasses (never raise exceptions). Errors are captured in the result's `error` field.
- **Stage chaining**: If Stage 1 produces OCR output, Stage 2 uses it; if Stage 1 skips/fails, Stage 2 falls back to original PDF.
- **Tag-based workflow**: Documents tracked via tags (smart-ocr:pending, :completed, :failed, :skip). The orchestrator manages tag transitions.
- **Content modes**: Stage 2 supports REPLACE (overwrite) or APPEND (preserve existing content) modes when patching the paperless-ngx content field.
- **Lazy imports in route handlers**: Pipeline modules (`process_document`, `PipelineOrchestrator`) are imported inside endpoint functions to avoid loading heavy ML models at import time. These trigger PLC0415 lint warnings which are suppressed with `# noqa: PLC0415`.
- **Known limitation**: The paperless-ngx API has no endpoint for replacing a document's file in-place, so OCR'd PDFs are only used within the pipeline run (not re-uploaded).

### Web Application

The FastAPI app (`web/app.py`) uses an application factory pattern (`create_app`). The lifespan context manager auto-creates and manages `JobQueue` and `PaperlessClient` on `app.state`. Routes access dependencies via `request.app.state.*`.

**API Endpoints (9 total across 3 routers):**

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Liveness probe (status + version) |
| `GET /api/ready` | Readiness probe (job queue + paperless connectivity) |
| `GET /api/documents` | List documents with filtering (pagination, tags, query) |
| `GET /api/documents/{id}` | Get single document |
| `POST /api/documents/{id}/process` | Submit for background processing (returns 202) |
| `POST /api/documents/{id}/dry-run` | Synchronous preview (no side effects) |
| `GET /api/jobs` | List jobs with filtering (status, document_id, limit) |
| `GET /api/jobs/{id}` | Get job status |
| `POST /api/jobs/{id}/cancel` | Cancel pending/running job |

**Exception handlers** map domain errors to HTTP status codes:
- `PaperlessNotFoundError` → 404, `PaperlessValidationError` → 400
- `PaperlessAuthenticationError`/`PaperlessConnectionError`/`PaperlessServerError` → 502
- `PaperlessRateLimitError` → 503
- `JobNotFoundError` → 404, `JobAlreadyCancelledError` → 409

### Exception Hierarchy

Four exception families, each with a base class:
- `ConfigurationError` → `ConfigurationFileNotFoundError`, `ConfigurationValidationError`
- `PaperlessError` → `PaperlessConnectionError`, `PaperlessAuthenticationError`, `PaperlessNotFoundError`, `PaperlessRateLimitError`, `PaperlessServerError`, `PaperlessValidationError`
- `PipelineError` → `PreprocessingError`, `LayoutDetectionError`, `OCRError`, `MarkerConversionError`
- `JobError` → `JobNotFoundError`, `JobAlreadyCancelledError`, `JobQueueFullError`, `JobTimeoutError`

### Web UI (htmx + Tailwind CSS)

The UI uses Jinja2 templates with htmx for dynamic interactions and Tailwind CSS (CDN) for styling. Dark mode uses `darkMode: 'class'` with system preference detection and `localStorage` persistence.

**View routes** (`web/routes/views.py`) serve full pages or htmx partials based on the `HX-Request` header. Templates extend `base.html` and use `{% include %}` for partials. Jinja2 globals (`version`, `theme_mode`) are injected in `_configure_templates()`.

**htmx patterns**:
- Document list pagination: `hx-get` targeting `#document-table` with `hx-push-url="true"`
- Job auto-poll: `hx-get="/jobs/{id}" hx-trigger="every 2s"` on non-terminal job cards
- Process/dry-run forms: `hx-post` returning partial templates into `#result-area`

### Not Yet Implemented

- **CLI commands**: Only `serve` works; `process`, `config`, `post-consume` are stubs
- **Integration patterns**: Polling, webhook, and post-consume handlers not yet built
- **Observability**: Prometheus metrics and OpenTelemetry tracing modules not yet implemented
- **Integration tests**: Directory exists but no tests yet; no test PDF fixtures

## Code Style

- All modules use `from __future__ import annotations`
- All modules define `__all__` exports
- Google-style docstrings
- Python 3.12+ features (e.g., `class Foo[T]` for generics)
- Result dataclasses use `@dataclass(slots=True)` (not `frozen=True`)

### Lint/Format Constraints

- **ruff line length**: 88 characters (Black-compatible)
- **ruff enforces**: SIM102 (combine nested ifs), SIM103 (inline return conditions), TRY300 (move return to `else` block), TRY400 (use `logger.exception` not `logger.error` in `except` blocks)
- **Third-party imports** go in `TYPE_CHECKING` block per TC001/TC002/TC003 rules. Exception: imports FastAPI needs at runtime for query param validation (e.g., `JobStatus` enum) stay at module level with `# noqa: TC001`.
- **mypy**: strict mode with pydantic plugin
- **Pydantic models** have `str_strip_whitespace=True` - be aware `DocumentUpdate` strips trailing whitespace from content strings

### Test Patterns

- Tests use `pytest-asyncio` with `mode=Mode.AUTO` (no explicit `@pytest.mark.asyncio` needed)
- Test files ignore: S101 (assert), ARG001/ARG002 (fixtures), PLR2004 (magic values), SLF001 (private access)
- Fixtures in `tests/conftest.py`: `fixtures_dir`, `pdfs_dir` for test data paths
- Web route tests use `TestClient` with mocked `JobQueue` and `PaperlessClient` via `unittest.mock.patch`. Use `MagicMock()` as base for mock objects (not `AsyncMock()`) with explicit `AsyncMock()` for specific async methods to avoid unawaited coroutine warnings on garbage collection.
- Pipeline orchestrator patches target the source module (`paperless_ngx_smart_ocr.pipeline.orchestrator.process_document`) not the route module, since lazy imports don't create module-level attributes.
- `pyproject.toml` has a `filterwarnings` entry to ignore `AsyncMockMixin` `PytestUnraisableExceptionWarning` from TestClient lifecycle.

## Module Usage Examples

### Configuration

```python
from paperless_ngx_smart_ocr.config import load_settings, get_settings

settings = load_settings()  # Searches ./config.yaml, ~/.config/smart-ocr/config.yaml
settings.paperless.url           # http://localhost:8000
settings.pipeline.stage1.enabled # True
```

Environment variables use `SMARTOCR_` prefix with `__` delimiter (e.g., `SMARTOCR_WEB__PORT=9000`).

### Paperless-ngx Client

```python
from paperless_ngx_smart_ocr.paperless import PaperlessClient, DocumentUpdate

async with PaperlessClient(base_url, token) as client:
    docs = await client.list_documents(tags_include=[1], tags_exclude=[2])
    await client.update_document(doc_id, DocumentUpdate(content=markdown))
    tag = await client.ensure_tag("smart-ocr:pending")
    await client.add_tags_to_document(doc_id, [tag.id])
```

### Pipeline Orchestrator

```python
from paperless_ngx_smart_ocr.pipeline import process_document

async with PaperlessClient(base_url, token) as client:
    result = await process_document(document_id, settings=settings, client=client)
    # result.success, result.stage1_result, result.stage2_result,
    # result.tags_updated, result.content_updated, result.dry_run, result.error
```

### Job Queue

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

### Web Application

```python
from paperless_ngx_smart_ocr.web import create_app

app = create_app(settings=settings)
# Lifespan auto-manages JobQueue and PaperlessClient on app.state
# Access in routes via request.app.state.job_queue, request.app.state.client
```

## Design Principles

1. **No hallucination**: Text extraction is deterministic; LLM mode off by default
2. **Complex layouts supported**: Surya layout detection before OCR handles multi-column, tables, figures
3. **No noise-as-text**: Layout detection filters artifacts, watermarks, headers/footers
4. **Tag-based workflow**: Documents tracked via tags (smart-ocr:pending, :completed, :failed, :skip)
