"""Web application module (FastAPI + htmx)."""

from __future__ import annotations

from paperless_ngx_smart_ocr.web.app import (
    create_app,
    get_app_settings,
    get_job_queue,
    get_preview_store,
    get_service_client,
)


__all__ = [
    "create_app",
    "get_app_settings",
    "get_job_queue",
    "get_preview_store",
    "get_service_client",
]
