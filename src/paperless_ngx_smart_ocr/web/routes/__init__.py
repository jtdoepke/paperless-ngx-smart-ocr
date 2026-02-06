"""API and web routes."""

from __future__ import annotations

from paperless_ngx_smart_ocr.web.routes.documents import (
    router as documents_router,
)
from paperless_ngx_smart_ocr.web.routes.health import (
    router as health_router,
)
from paperless_ngx_smart_ocr.web.routes.jobs import (
    router as jobs_router,
)


__all__ = ["documents_router", "health_router", "jobs_router"]
