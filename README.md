# paperless-ngx-smart-ocr

A service that enhances [paperless-ngx](https://github.com/paperless-ngx/paperless-ngx) with intelligent, layout-aware OCR and Markdown conversion. It processes PDF documents through a two-stage pipeline, producing searchable PDFs with accurate text layers and high-quality Markdown output.

## Guiding Philosophy

1. **Support complex document layouts** - Multi-column documents, tables, figures, and mixed layouts are handled correctly through vision-based layout detection before OCR
2. **No noise-as-text** - Scanning artifacts, watermarks, and visual noise are filtered out using layout detection models, not OCR'd as garbage text
3. **No hallucination** - Text extraction is deterministic; no generative LLMs that could invent content. Marker's optional LLM mode is configurable but off by default
4. **Excellent Markdown** - Output preserves document structure: headings, paragraphs, lists, tables, code blocks, with proper reading order

## Core Features

- **Two-stage processing pipeline**:
  - Stage 1: OCR scanned PDFs using OCRmyPDF + Surya layout detection
  - Stage 2: Convert to Markdown using Marker, store in paperless-ngx content field
- **Flexible execution**: Either stage can be skipped independently
- **Multiple integration patterns**: Polling, webhooks, post-consume scripts
- **Web UI**: Manual triggers, dry runs, processing status
- **Tag-based workflow**: Target documents via include/exclude tag filters

## Technical Stack

| Category | Decision |
|----------|----------|
| Language | Python 3.12+ |
| Type Checking | mypy strict mode, fully typed |
| Package Manager | uv |
| Linting/Formatting | ruff |
| Testing | pytest, pytest-cov, 80%+ coverage |
| Web Framework | FastAPI + htmx + Tailwind CSS |
| Stage 1 OCR | OCRmyPDF + Surya (layout detection) |
| Stage 2 Markdown | Marker (LLM configurable, off by default) |
| Storage | YAML configs, structured logs (no database) |
| Job Queue | Background queue + sync mode for dry runs |
| Auth | Paperless-ngx API tokens |
| Deployment | Docker (ghcr.io) + pip install + Proxmox guide |
| License | MIT |

---

## Architecture

### System Context

```
┌─────────────────────────────────────────────────────────────────┐
│                        User                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  paperless-ngx-smart-ocr                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Web UI    │  │  REST API   │  │   Background Workers    │  │
│  │  (htmx)     │  │  (FastAPI)  │  │   (async processing)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────┴───────────────────────────────┐  │
│  │                    Processing Pipeline                     │  │
│  │  ┌─────────────────┐         ┌─────────────────────────┐  │  │
│  │  │     Stage 1     │         │        Stage 2          │  │  │
│  │  │  OCRmyPDF +     │────────▶│   Marker → Markdown     │  │  │
│  │  │  Surya Layout   │         │   → PATCH content       │  │  │
│  │  └─────────────────┘         └─────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      paperless-ngx                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  REST API   │  │  Documents  │  │        Tags             │  │
│  │  /api/...   │  │  Storage    │  │  (processing state)     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Processing Pipeline

```
Document Input
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Pre-Processing Analysis                       │
│  • Check if document has existing text layer                     │
│  • Determine if Stage 1 needed (based on config + doc state)     │
│  • Determine if Stage 2 needed (based on config)                 │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Stage 1: OCR                             │
│  1. Download original PDF from paperless-ngx                     │
│  2. Run Surya layout detection → identify text regions + order   │
│  3. For each text region, run Tesseract (PSM 6) via OCRmyPDF     │
│  4. Composite results with correct reading order                 │
│  5. Write invisible text layer to PDF (fpdf2 renderer)           │
│  6. Upload OCR'd PDF to paperless-ngx (replaces original)        │
│                                                                  │
│  Configurable: skip_if_text_exists, force_ocr                    │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Stage 2: Markdown                            │
│  1. Download PDF from paperless-ngx (may be OCR'd from Stage 1)  │
│  2. Run Marker extraction (configurable: --use_llm flag)         │
│  3. Post-process Markdown (clean up, validate structure)         │
│  4. PATCH document content field with Markdown                   │
│                                                                  │
│  Configurable: use_llm, replace_content vs append_content        │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Post-Processing                             │
│  • Update tags (remove pending, add completed/failed)            │
│  • Log results (structured JSON logs)                            │
│  • Emit OpenTelemetry traces                                     │
│  • Update Prometheus metrics                                     │
└─────────────────────────────────────────────────────────────────┘
```

### Integration Patterns

The service supports three integration patterns with paperless-ngx:

#### 1. Polling-Based (Recommended)

```
┌──────────────┐     periodic poll      ┌─────────────────┐
│   smart-ocr  │ ───────────────────▶   │  paperless-ngx  │
│   service    │ ◀───────────────────   │  GET /api/docs  │
└──────────────┘   docs with tags       └─────────────────┘
```

- Service polls paperless-ngx API at configurable interval
- Queries documents matching include tags, excluding exclude tags
- Most flexible, works with any paperless-ngx setup

#### 2. Webhook-Triggered

```
┌─────────────────┐    webhook POST     ┌──────────────┐
│  paperless-ngx  │ ─────────────────▶  │   smart-ocr  │
│   Workflow      │                     │   /webhook   │
└─────────────────┘                     └──────────────┘
```

- Paperless-ngx Workflow sends webhook on document added/updated
- Near real-time processing
- Requires paperless-ngx Workflow configuration

#### 3. Post-Consume Script

```
┌─────────────────┐   env vars + exec   ┌──────────────┐
│  paperless-ngx  │ ─────────────────▶  │  smart-ocr   │
│  consumption    │                     │    CLI       │
└─────────────────┘                     └──────────────┘
```

- Script runs synchronously after document consumption
- Immediate processing, tightly coupled
- Requires mounting smart-ocr into paperless container

---

## Project Structure

```
paperless-ngx-smart-ocr/
├── src/
│   └── paperless_ngx_smart_ocr/
│       ├── __init__.py
│       ├── __main__.py              # CLI entry point
│       ├── py.typed                 # PEP 561 marker
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   ├── settings.py          # Pydantic settings (env + YAML)
│       │   └── schema.py            # Config validation schemas
│       │
│       ├── paperless/
│       │   ├── __init__.py
│       │   ├── client.py            # Paperless-ngx API client
│       │   ├── models.py            # Document, Tag, etc. models
│       │   └── exceptions.py
│       │
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── orchestrator.py      # Pipeline coordination
│       │   ├── stage1_ocr.py        # OCRmyPDF + Surya
│       │   ├── stage2_markdown.py   # Marker conversion
│       │   ├── layout.py            # Surya layout detection
│       │   └── preprocessing.py     # Document analysis
│       │
│       ├── workers/
│       │   ├── __init__.py
│       │   ├── queue.py             # Background job queue
│       │   ├── polling.py           # Polling integration
│       │   ├── webhook.py           # Webhook handler
│       │   └── post_consume.py      # Post-consume script mode
│       │
│       ├── web/
│       │   ├── __init__.py
│       │   ├── app.py               # FastAPI application
│       │   ├── routes/
│       │   │   ├── __init__.py
│       │   │   ├── documents.py     # Document processing endpoints
│       │   │   ├── jobs.py          # Job status endpoints
│       │   │   ├── settings.py      # Settings endpoints
│       │   │   └── health.py        # Health check endpoints
│       │   ├── templates/           # Jinja2 templates for htmx
│       │   │   ├── base.html
│       │   │   ├── index.html
│       │   │   ├── documents/
│       │   │   │   ├── list.html
│       │   │   │   ├── detail.html
│       │   │   │   └── process.html
│       │   │   ├── jobs/
│       │   │   │   └── status.html
│       │   │   └── partials/        # htmx partial templates
│       │   └── static/
│       │       ├── css/
│       │       │   └── app.css      # Tailwind output
│       │       └── js/
│       │           └── app.js       # Minimal JS (htmx extensions)
│       │
│       ├── observability/
│       │   ├── __init__.py
│       │   ├── logging.py           # Structured JSON logging
│       │   ├── metrics.py           # Prometheus metrics
│       │   └── tracing.py           # OpenTelemetry setup
│       │
│       └── cli/
│           ├── __init__.py
│           └── commands.py          # CLI commands (typer)
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Fixtures, test config
│   ├── fixtures/
│   │   └── pdfs/                    # Sample PDF test documents
│   │       ├── scanned_single_column.pdf
│   │       ├── scanned_multi_column.pdf
│   │       ├── born_digital.pdf
│   │       ├── tables_and_figures.pdf
│   │       └── noisy_scan.pdf
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_paperless_client.py
│   │   ├── test_pipeline.py
│   │   └── test_layout.py
│   └── integration/
│       ├── test_stage1_ocr.py
│       ├── test_stage2_markdown.py
│       └── test_full_pipeline.py
│
├── docs/                            # MkDocs documentation source
│   ├── index.md
│   ├── getting-started/
│   │   ├── installation.md
│   │   ├── configuration.md
│   │   └── quick-start.md
│   ├── user-guide/
│   │   ├── web-ui.md
│   │   ├── processing-documents.md
│   │   ├── tag-workflow.md
│   │   └── troubleshooting.md
│   ├── integration/
│   │   ├── polling.md
│   │   ├── webhooks.md
│   │   └── post-consume.md
│   ├── deployment/
│   │   ├── docker.md
│   │   ├── proxmox.md
│   │   └── pip-install.md
│   ├── development/
│   │   ├── contributing.md
│   │   ├── architecture.md
│   │   └── testing.md
│   └── reference/
│       ├── configuration.md
│       ├── api.md
│       └── cli.md
│
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.dev
│   └── docker-compose.yml           # For local development
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                   # Lint, test, coverage
│   │   ├── release.yml              # Build & publish on tag
│   │   └── docs.yml                 # Deploy docs to GitHub Pages
│   ├── dependabot.yml
│   └── ISSUE_TEMPLATE/
│
├── pyproject.toml                   # Project config (uv, ruff, mypy, pytest)
├── uv.lock                          # Lockfile
├── .pre-commit-config.yaml
├── .python-version                  # 3.12
├── mkdocs.yml                       # MkDocs configuration
├── tailwind.config.js               # Tailwind configuration
├── LICENSE                          # MIT
├── README.md
├── CHANGELOG.md
└── config.example.yaml              # Example configuration file
```

---

## Configuration

### Configuration File (config.yaml)

```yaml
# paperless-ngx-smart-ocr configuration

# Paperless-ngx connection
paperless:
  url: "http://paperless-ngx:8000"
  token: "${PAPERLESS_TOKEN}"  # Environment variable interpolation
  # Or use token_file for secrets management
  # token_file: "/run/secrets/paperless_token"

# Tag configuration
tags:
  prefix: "smart-ocr"  # Creates tags like smart-ocr:pending, smart-ocr:completed
  # Include documents with ANY of these tags for processing
  include:
    - "smart-ocr:pending"
  # Exclude documents with ANY of these tags
  exclude:
    - "smart-ocr:completed"
    - "smart-ocr:failed"
    - "smart-ocr:skip"

# Processing pipeline configuration
pipeline:
  # Stage 1: OCR
  stage1:
    enabled: true
    # How to handle documents that already have text
    # "skip" - skip OCR if text detected (default for auto)
    # "force" - always OCR, replacing existing text
    born_digital_handling: "skip"

    # OCRmyPDF options
    ocrmypdf:
      deskew: true
      clean: true
      rotate_pages: true
      language: "eng"  # Tesseract language(s)
      # Additional OCRmyPDF arguments
      extra_args: []

    # Surya layout detection
    layout_detection:
      enabled: true
      confidence_threshold: 0.5
      # Filter out these region types
      exclude_regions:
        - "advertisement"
        - "page_number"
        - "header"
        - "footer"

  # Stage 2: Markdown conversion
  stage2:
    enabled: true
    # How to handle existing content
    # "replace" - replace content entirely (default for auto)
    # "append" - append below existing content
    content_mode: "replace"

    # Marker options
    marker:
      use_llm: false  # Enable LLM assistance (may introduce hallucinations)
      # If use_llm is true, configure the LLM
      llm:
        provider: "openai"  # or "anthropic", "ollama"
        model: "gpt-4o-mini"
        api_key: "${OPENAI_API_KEY}"

# Integration mode
integration:
  # Polling configuration
  polling:
    enabled: true
    interval_seconds: 300  # 5 minutes
    batch_size: 10  # Max documents per poll cycle

  # Webhook configuration
  webhook:
    enabled: false
    # Webhook endpoint is always at /api/webhook
    # Configure secret for validation
    secret: "${WEBHOOK_SECRET}"

  # Post-consume script mode
  post_consume:
    enabled: false
    # When enabled, runs in CLI mode triggered by paperless

# Auto-processing
auto_processing:
  enabled: true
  # Only process documents matching tag filters automatically
  # Manual processing via web UI always available

# Background job processing
jobs:
  # Number of concurrent workers
  workers: 2
  # Timeout for individual document processing (seconds)
  timeout: 600  # 10 minutes

# GPU configuration
gpu:
  enabled: "auto"  # "auto", "cuda", "cpu"
  # For Surya and Marker

# Web UI configuration
web:
  host: "0.0.0.0"
  port: 8080
  # Dark mode: "auto", "dark", "light"
  theme: "auto"

# Observability
observability:
  logging:
    level: "INFO"
    format: "json"  # "json" or "console"

  metrics:
    enabled: true
    port: 9090  # Prometheus metrics endpoint

  tracing:
    enabled: false
    # OpenTelemetry configuration
    otlp_endpoint: "http://jaeger:4317"
```

### Environment Variables

All configuration can also be set via environment variables:

```bash
# Paperless connection
PAPERLESS_URL=http://paperless-ngx:8000
PAPERLESS_TOKEN=your_api_token

# Tag prefix
SMARTOCR_TAGS_PREFIX=smart-ocr

# Pipeline
SMARTOCR_PIPELINE_STAGE1_ENABLED=true
SMARTOCR_PIPELINE_STAGE2_ENABLED=true
SMARTOCR_PIPELINE_STAGE2_MARKER_USE_LLM=false

# Integration
SMARTOCR_INTEGRATION_POLLING_ENABLED=true
SMARTOCR_INTEGRATION_POLLING_INTERVAL_SECONDS=300

# GPU
SMARTOCR_GPU_ENABLED=auto

# Web
SMARTOCR_WEB_HOST=0.0.0.0
SMARTOCR_WEB_PORT=8080

# Observability
SMARTOCR_OBSERVABILITY_LOGGING_LEVEL=INFO
SMARTOCR_OBSERVABILITY_METRICS_ENABLED=true
```

---

## API Endpoints

```
GET  /                           # Web UI home
GET  /documents                  # Document list view
GET  /documents/{id}             # Document detail view
POST /documents/{id}/process     # Trigger processing
POST /documents/{id}/dry-run     # Dry run (preview)

GET  /jobs                       # Active jobs list
GET  /jobs/{id}                  # Job status

GET  /api/health                 # Health check
GET  /api/ready                  # Readiness check
GET  /metrics                    # Prometheus metrics

POST /api/webhook                # Paperless webhook receiver
```

---

## Key Dependencies

### Core
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `httpx` - Async HTTP client
- `pydantic` - Data validation
- `pydantic-settings` - Configuration management

### OCR Pipeline
- `ocrmypdf` - PDF OCR tool
- `surya-ocr` - Layout detection
- `marker-pdf` - Markdown conversion
- `pymupdf` (fitz) - PDF manipulation

### Web UI
- `jinja2` - Template engine
- `python-multipart` - Form handling

### Observability
- `structlog` - Structured logging
- `prometheus-client` - Metrics
- `opentelemetry-api` - Tracing
- `opentelemetry-sdk`
- `opentelemetry-exporter-otlp`

### CLI
- `typer` - CLI framework
- `rich` - Terminal formatting

### Development
- `pytest` - Testing
- `pytest-cov` - Coverage
- `pytest-asyncio` - Async test support
- `respx` - HTTP mocking
- `ruff` - Linting/formatting
- `mypy` - Type checking
- `pre-commit` - Git hooks

---

## Implementation Checklist

### Phase 1: Project Foundation ✓

Completed: Project initialized with uv, pyproject.toml configured with all dependencies and strict tooling (ruff, mypy, pytest), pre-commit hooks, .gitignore, MIT LICENSE, and full directory structure.

### Phase 2: Core Infrastructure

- [x] **Configuration module** (`src/paperless_ngx_smart_ocr/config/`)
  - [x] Pydantic settings with YAML + env support
  - [x] Configuration validation schemas
  - [x] Environment variable interpolation (`${VAR}` syntax)
  - [x] Secret file support (`token_file`)
- [x] **Logging setup** (`src/paperless_ngx_smart_ocr/observability/logging.py`)
  - [x] Structured logfmt-style logging with structlog (human + machine readable)
  - [x] Configurable log levels and formatting (--verbose/-V, --quiet/-q CLI flags)
  - [x] Request ID tracking for correlation via contextvars
- [x] **Paperless-ngx client** (`src/paperless_ngx_smart_ocr/paperless/`)
  - [x] Async API client with httpx
  - [x] Document CRUD operations (list, get, update, download, upload)
  - [x] Tag management (list, create, add/remove from documents)
  - [x] File download/upload with streaming
  - [x] Retry logic with exponential backoff
  - [x] Rate limiting support

### Phase 3: Processing Pipeline ✓

- [x] **Stage 1: OCR Pipeline** (`src/paperless_ngx_smart_ocr/pipeline/stage1_ocr.py`)
  - [x] Surya layout detection integration
  - [x] Region filtering (exclude advertisements, noise, headers, footers)
  - [x] Reading order determination from layout
  - [x] OCRmyPDF integration with configurable options
  - [x] Per-region OCR with appropriate PSM modes
  - [x] Text layer composition with correct reading order
  - [x] Born-digital detection (check for existing text)
- [x] **Stage 2: Markdown Pipeline** (`src/paperless_ngx_smart_ocr/pipeline/stage2_markdown.py`)
  - [x] Marker integration
  - [x] LLM toggle support (configurable on/off)
  - [x] Markdown post-processing and cleanup
  - [x] Content field patching via paperless API
- [x] **Pipeline orchestrator** (`src/paperless_ngx_smart_ocr/pipeline/orchestrator.py`)
  - [x] Stage coordination and sequencing
  - [x] Skip logic based on configuration
  - [x] Error handling and recovery
  - [x] Progress tracking and reporting
- [x] **Layout detection** (`src/paperless_ngx_smart_ocr/pipeline/layout.py`)
  - [x] Surya model loading and caching
  - [x] Confidence thresholding
  - [x] Region type classification
- [x] **Preprocessing** (`src/paperless_ngx_smart_ocr/pipeline/preprocessing.py`)
  - [x] PDF text layer detection
  - [x] Document analysis for stage determination

### Phase 4: Integration Patterns

- [ ] **Polling integration** (`src/paperless_ngx_smart_ocr/workers/polling.py`)
  - [ ] Periodic document query loop
  - [ ] Tag-based filtering (include/exclude)
  - [ ] Batch processing with configurable size
  - [ ] Graceful shutdown handling
- [ ] **Webhook integration** (`src/paperless_ngx_smart_ocr/workers/webhook.py`)
  - [ ] Webhook endpoint handler
  - [ ] Signature/secret validation
  - [ ] Event parsing and routing
- [ ] **Post-consume script** (`src/paperless_ngx_smart_ocr/workers/post_consume.py`)
  - [ ] CLI command for post-consume mode
  - [ ] Environment variable parsing (DOCUMENT_ID, etc.)
  - [ ] Synchronous processing mode

### Phase 5: Background Processing ✓

- [x] **Job queue** (`src/paperless_ngx_smart_ocr/workers/queue.py`)
  - [x] In-memory async queue with asyncio (using aiojobs)
  - [x] Job status tracking (pending, running, completed, failed, cancelled)
  - [x] Configurable concurrent workers
  - [x] Timeout handling per job
- [x] **Job management**
  - [x] Queue new jobs
  - [x] Cancel running jobs
  - [x] Get job status by ID
  - [x] List active/completed jobs

### Phase 6: Web UI

- [x] **FastAPI application** (`src/paperless_ngx_smart_ocr/web/app.py`)
  - [x] Application factory pattern
  - [x] Middleware setup (CORS, request ID)
  - [x] Exception handlers
  - [x] Static file serving
  - [x] Lifespan management (startup/shutdown)
- [ ] **API routes** (`src/paperless_ngx_smart_ocr/web/routes/`)
  - [ ] Document listing with filtering (`documents.py`)
  - [ ] Document processing trigger
  - [ ] Dry run endpoint
  - [ ] Job status endpoints (`jobs.py`)
  - [x] Health check endpoints (`health.py`)
- [ ] **htmx templates** (`src/paperless_ngx_smart_ocr/web/templates/`)
  - [ ] Base layout with Tailwind CSS
  - [ ] Dark mode support with system preference detection
  - [ ] Document list view
  - [ ] Document detail view
  - [ ] Processing form (stage selection, options override)
  - [ ] Job progress display
  - [ ] Partial templates for htmx updates
- [ ] **Tailwind CSS setup**
  - [ ] tailwind.config.js configuration
  - [ ] Build pipeline (npm scripts or standalone CLI)
  - [ ] Dark mode class configuration

### Phase 7: Observability

- [ ] **Prometheus metrics** (`src/paperless_ngx_smart_ocr/observability/metrics.py`)
  - [ ] Documents processed counter (by stage, status)
  - [ ] Processing duration histogram
  - [ ] Stage-specific timing metrics
  - [ ] Error counters by type
  - [ ] Queue depth gauge
  - [ ] Metrics endpoint (`/metrics`)
- [ ] **OpenTelemetry tracing** (`src/paperless_ngx_smart_ocr/observability/tracing.py`)
  - [ ] Trace provider setup
  - [ ] Span creation for pipeline stages
  - [ ] Attribute enrichment (document ID, stage, etc.)
  - [ ] OTLP exporter configuration

### Phase 8: CLI

- [ ] **Typer CLI** (`src/paperless_ngx_smart_ocr/cli/commands.py`)
  - [ ] `serve` command - Start web server with workers
  - [ ] `process` command - Process single document by ID
  - [ ] `config` command - Validate configuration file
  - [ ] `post-consume` command - Post-consume script mode
  - [ ] Rich output formatting

### Phase 9: Docker

- [ ] **Dockerfile** (`docker/Dockerfile`)
  - [ ] Multi-stage build for smaller image
  - [ ] GPU support variant (NVIDIA base image)
  - [ ] Non-root user for security
  - [ ] Health check configuration
  - [ ] Proper layer caching
- [ ] **docker-compose.yml** (`docker/docker-compose.yml`)
  - [ ] Service definition with environment variables
  - [ ] Volume mounts for config and logs
  - [ ] Network configuration
  - [ ] GPU passthrough example (commented)
  - [ ] Optional paperless-ngx service for development

### Phase 10: Documentation

- [ ] **MkDocs setup**
  - [ ] mkdocs.yml configuration
  - [ ] Material theme configuration
  - [ ] Navigation structure
  - [ ] Search configuration
- [ ] **Documentation pages** (`docs/`)
  - [ ] Getting started guide (installation, configuration, quick-start)
  - [ ] User guide (web UI, processing documents, tag workflow, troubleshooting)
  - [ ] Integration guides (polling, webhooks, post-consume)
  - [ ] Deployment guides (Docker, Proxmox, pip install)
  - [ ] Development guide (contributing, architecture, testing)
  - [ ] Reference (configuration options, API, CLI)

### Phase 11: Testing

- [ ] **Test fixtures** (`tests/fixtures/`)
  - [ ] Sample PDF files (scanned single-column, multi-column, born-digital, tables, noisy)
  - [ ] Mock paperless-ngx API responses
  - [ ] Configuration fixtures
- [ ] **Unit tests** (`tests/unit/`)
  - [ ] Configuration parsing and validation
  - [ ] Paperless client methods
  - [ ] Pipeline components (layout, preprocessing)
  - [ ] Job queue operations
- [ ] **Integration tests** (`tests/integration/`)
  - [ ] Full Stage 1 pipeline with sample PDFs
  - [ ] Full Stage 2 pipeline with sample PDFs
  - [ ] End-to-end processing flow
- [ ] **Coverage configuration**
  - [ ] 80% coverage target
  - [ ] Coverage reporting in CI
  - [ ] Coverage badge in README

### Phase 12: CI/CD

- [ ] **GitHub Actions: CI** (`.github/workflows/ci.yml`)
  - [ ] Trigger on push and PR
  - [ ] Lint with ruff
  - [ ] Type check with mypy
  - [ ] Test with pytest
  - [ ] Coverage report upload
  - [ ] Matrix testing (Python versions)
- [ ] **GitHub Actions: Release** (`.github/workflows/release.yml`)
  - [ ] Trigger on version tag
  - [ ] Build Docker image
  - [ ] Push to ghcr.io
  - [ ] Build Python package
  - [ ] Publish to PyPI
  - [ ] Create GitHub release
- [ ] **GitHub Actions: Docs** (`.github/workflows/docs.yml`)
  - [ ] Build MkDocs site
  - [ ] Deploy to GitHub Pages
- [ ] **Dependabot** (`.github/dependabot.yml`)
  - [ ] Python dependency updates
  - [ ] GitHub Actions updates

### Phase 13: Proxmox Quick Setup

- [ ] **LXC/VM template guide** (`docs/deployment/proxmox.md`)
  - [ ] Container/VM creation steps
  - [ ] GPU passthrough instructions (if applicable)
  - [ ] Network configuration
  - [ ] Storage configuration
- [ ] **Helper script** (`scripts/proxmox-setup.sh`)
  - [ ] Automated setup script
  - [ ] Systemd service file generation
  - [ ] Configuration template creation

---

## Verification Plan

### Manual Testing

1. Start service with `uv run smart-ocr serve`
2. Open web UI at http://localhost:8080
3. Verify paperless-ngx connection (health indicator)
4. Select a scanned PDF document
5. Run dry run - verify preview shows expected output
6. Run actual processing
7. Verify in paperless-ngx:
   - Document is now searchable (Stage 1)
   - Content field contains Markdown (Stage 2)
   - Tags updated correctly

### Automated Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=paperless_ngx_smart_ocr --cov-report=html

# Lint
uv run ruff check .

# Type check
uv run mypy .
```

### Docker Testing

```bash
# Build and run
docker compose -f docker/docker-compose.yml up -d

# Access web UI
open http://localhost:8080

# Check logs
docker compose -f docker/docker-compose.yml logs -f

# Test with sample documents
# Verify GPU detection (if applicable)
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.
