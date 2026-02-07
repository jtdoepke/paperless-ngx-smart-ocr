"""Strict LLM service subclasses for Marker.

Marker's built-in service classes swallow all exceptions and return
``{}`` on failure, making errors invisible.  The subclasses here
re-raise so that failures propagate up to the pipeline.

This module is **not** imported at package level — it is only loaded
when Marker's ``PdfConverter`` resolves the dotted class-path string
returned by :func:`~..stage2_markdown.get_llm_service_info`.  This
keeps heavy Marker/Surya imports out of the critical startup path.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import requests  # type: ignore[import-untyped]
from marker.services.ollama import OllamaService

from paperless_ngx_smart_ocr.observability import get_logger


if TYPE_CHECKING:
    import PIL.Image


__all__ = ["StrictOllamaService"]

logger = get_logger(__name__)


def _resolve_refs(
    obj: Any,  # noqa: ANN401
    defs: dict[str, Any],
) -> Any:  # noqa: ANN401
    """Recursively resolve ``$ref`` pointers using ``$defs``.

    Pydantic's ``model_json_schema()`` emits ``$defs`` + ``$ref`` for
    nested models, but Ollama's ``format`` parameter requires a
    self-contained JSON Schema without references.  This function
    inlines every ``$ref`` so the schema stands alone.
    """
    if isinstance(obj, dict):
        if "$ref" in obj:
            ref_name = obj["$ref"].rsplit("/", 1)[-1]
            if ref_name in defs:
                return _resolve_refs(defs[ref_name], defs)
            return obj  # unresolvable ref, leave as-is
        return {
            k: _resolve_refs(v, defs)
            for k, v in obj.items()
            if k not in ("$defs", "title")
        }
    if isinstance(obj, list):
        return [_resolve_refs(item, defs) for item in obj]
    return obj


class StrictOllamaService(OllamaService):  # type: ignore[misc]
    """OllamaService that propagates errors instead of swallowing them.

    Marker's ``OllamaService.__call__`` wraps its HTTP request in a
    bare ``except Exception`` that logs a warning and returns ``{}``.
    This subclass re-implements the call to let errors propagate.
    """

    def process_images(
        self,
        images: list[PIL.Image.Image],
    ) -> list[str]:
        """Encode images as PNG base64.

        The default ``OllamaService`` uses WEBP, which causes some
        Ollama vision models (e.g. ``granite3.2-vision``) to crash
        with a 500 error.  PNG is universally supported.
        """
        return [self.img_to_base64(img, format="PNG") for img in images]

    def __call__(
        self,
        prompt: str,
        image: Any,  # noqa: ANN401
        block: Any,  # noqa: ANN401
        response_schema: Any,  # noqa: ANN401
        max_retries: int | None = None,  # noqa: ARG002
        timeout: int | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Call the Ollama API, raising on failure."""
        url = f"{self.ollama_base_url}/api/generate"
        headers = {"Content-Type": "application/json"}

        schema = response_schema.model_json_schema()
        defs = schema.get("$defs", {})
        format_schema = {
            "type": "object",
            "properties": _resolve_refs(
                schema["properties"],
                defs,
            ),
            "required": schema.get("required", []),
        }

        image_bytes = self.format_image_for_llm(image)

        payload: dict[str, Any] = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "format": format_schema,
            "images": image_bytes,
        }

        resp = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        if not resp.ok:
            body = resp.text[:500]
            logger.error(
                "ollama_request_failed",
                status=resp.status_code,
                body=body,
                model=self.ollama_model,
            )
        resp.raise_for_status()

        data: dict[str, Any] = resp.json()
        total_tokens = data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
        if block:
            block.update_metadata(
                llm_request_count=1,
                llm_tokens_used=total_tokens,
            )
        return json.loads(data["response"])  # type: ignore[no-any-return]
