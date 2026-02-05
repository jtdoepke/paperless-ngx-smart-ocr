# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

paperless-ngx-smart-ocr is a Python service that enhances paperless-ngx with intelligent, layout-aware OCR and Markdown conversion. It processes PDFs through a two-stage pipeline:
- **Stage 1**: Add searchable text layers using OCRmyPDF + Surya layout detection
- **Stage 2**: Convert to structured Markdown using Marker, store in paperless-ngx content field

**Current Status**: Phase 1 complete (project foundation with stub modules). Implementation phases 2-13 remain. The directory structure exists but modules contain only `__init__.py` stubs.

## Development Commands

```bash
# Tool management (mise)
mise install                    # Install Python 3.12 + uv + jq

# Package management (uv)
uv sync --all-extras           # Install all dependencies including dev
uv add <package>               # Add dependency
uv add --dev <package>         # Add dev dependency

# Running the application
uv run smart-ocr serve         # Start web server with workers
uv run smart-ocr process <id>  # Process single document
uv run smart-ocr config        # Validate configuration
uv run smart-ocr post-consume  # Post-consume script mode

# Testing
uv run pytest                                              # Run all tests
uv run pytest --cov=paperless_ngx_smart_ocr --cov-report=html  # With coverage
uv run pytest tests/unit/                                  # Unit tests only
uv run pytest tests/integration/                           # Integration tests only
uv run pytest -k "test_name"                              # Run specific test

# Code quality
uv run ruff check .            # Lint code
uv run ruff format .           # Format code
uv run mypy src/               # Type check (strict mode)

# Pre-commit
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

### Key Components (Planned Architecture)

```
src/paperless_ngx_smart_ocr/
├── config/          # Pydantic settings (YAML + env vars)
├── paperless/       # Async API client for paperless-ngx (httpx)
├── pipeline/        # Processing stages
│   ├── orchestrator.py     # Stage coordination
│   ├── stage1_ocr.py       # OCRmyPDF + Surya
│   ├── stage2_markdown.py  # Marker conversion
│   └── layout.py           # Surya layout detection
├── workers/         # Integration implementations
│   ├── queue.py     # Background job queue (asyncio)
│   ├── polling.py   # Polling loop
│   └── webhook.py   # Webhook handler
├── web/             # FastAPI + htmx + Tailwind
└── cli/             # Typer CLI (entry point: smart-ocr)
```

Note: Currently only stub `__init__.py` files exist. See README.md for the full implementation checklist.

## Technology Stack

- **Python 3.12+** with strict mypy type checking
- **mise** for tool/runtime management (.mise.toml)
- **uv** for package management
- **ruff** for linting/formatting (strict mode, comprehensive rules)
- **FastAPI** + uvicorn for web framework
- **htmx** + Tailwind CSS for frontend (minimal JS)
- **httpx** for async HTTP client
- **Pydantic v2** for settings and validation
- **pytest** + pytest-asyncio + respx for testing (80% coverage target)
- **structlog** for JSON structured logging
- **OCRmyPDF** + Surya + Marker for PDF processing

## Code Style

- All modules use `from __future__ import annotations`
- Empty modules define `__all__: list[str] = []`
- PEP 561 type checking enabled (py.typed marker)
- Google-style docstrings

## Design Principles

1. **No hallucination**: Text extraction is deterministic; LLM mode off by default
2. **Complex layouts supported**: Surya layout detection before OCR handles multi-column, tables, figures
3. **No noise-as-text**: Layout detection filters artifacts, watermarks, headers/footers
4. **Tag-based workflow**: Documents tracked via tags (smart-ocr:pending, :completed, :failed, :skip)
