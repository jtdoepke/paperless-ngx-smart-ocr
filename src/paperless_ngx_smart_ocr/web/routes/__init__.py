"""API and web routes."""

from __future__ import annotations

from paperless_ngx_smart_ocr.web.routes.health import router as health_router


__all__ = ["health_router"]
