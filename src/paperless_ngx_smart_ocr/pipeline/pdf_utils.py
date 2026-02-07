"""PDF property detection for archive format matching.

Detects PDF/A conformance level and dominant color space from existing
archive PDFs so that Smart OCR can produce replacement archives with
identical format settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from paperless_ngx_smart_ocr.observability import get_logger


if TYPE_CHECKING:
    from pathlib import Path


__all__ = [
    "PdfProperties",
    "detect_pdf_properties",
]


_PDFA_PART_MAP: dict[str, str] = {
    "1": "pdfa-1",
    "2": "pdfa-2",
    "3": "pdfa-3",
}

_COLORSPACE_TO_STRATEGY: dict[str, str] = {
    "/DeviceRGB": "RGB",
    "/CalRGB": "RGB",
    "/DeviceCMYK": "CMYK",
    "/CalCMYK": "CMYK",
    "/DeviceGray": "Gray",
    "/CalGray": "Gray",
}


@dataclass(slots=True)
class PdfProperties:
    """Detected PDF properties for matching ocrmypdf settings.

    Attributes:
        pdfa_status: Raw PDF/A conformance string (e.g. ``"2B"``),
            or empty string if not PDF/A.
        output_type: Value suitable for ``ocrmypdf.ocr(output_type=)``.
        color_conversion_strategy: Value suitable for
            ``ocrmypdf.ocr(color_conversion_strategy=)``.
    """

    pdfa_status: str
    output_type: str
    color_conversion_strategy: str


def detect_pdf_properties(pdf_path: Path) -> PdfProperties:
    """Detect PDF/A conformance and color space from a PDF.

    Opens the PDF with pikepdf and inspects XMP metadata for PDF/A
    status, then scans image color spaces (weighted by pixel area)
    to determine the dominant color conversion strategy.

    Args:
        pdf_path: Path to the PDF file to analyse.

    Returns:
        ``PdfProperties`` with ``output_type`` and
        ``color_conversion_strategy`` suitable for ``ocrmypdf.ocr()``.
    """
    import pikepdf  # type: ignore[import-not-found]

    logger = get_logger(__name__)

    with pikepdf.open(pdf_path) as pdf:
        # --- PDF/A detection via XMP metadata ---
        meta = pdf.open_metadata()
        pdfa_status: str = meta.pdfa_status
        output_type = (
            _PDFA_PART_MAP.get(pdfa_status[0], "pdfa") if pdfa_status else "pdfa"
        )

        # --- Dominant color space detection ---
        strategy_weights = _scan_color_spaces(pdf, pdf_path)

    color_strategy = (
        max(strategy_weights, key=strategy_weights.__getitem__)
        if strategy_weights
        else "LeaveColorUnchanged"
    )

    logger.debug(
        "pdf_properties_detected",
        path=str(pdf_path),
        pdfa_status=pdfa_status,
        output_type=output_type,
        color_conversion_strategy=color_strategy,
    )

    return PdfProperties(
        pdfa_status=pdfa_status,
        output_type=output_type,
        color_conversion_strategy=color_strategy,
    )


def _scan_color_spaces(pdf: object, pdf_path: Path) -> dict[str, int]:
    """Scan all images in a PDF and tally color space weights.

    Each image's color space is mapped to a strategy string and
    weighted by pixel area (width * height).

    Args:
        pdf: An open pikepdf.Pdf instance.
        pdf_path: Path for logging purposes.

    Returns:
        Mapping of strategy string to total pixel weight.
    """
    from pikepdf import PdfImage

    logger = get_logger(__name__)
    strategy_weights: dict[str, int] = {}
    seen_objects: set[int] = set()

    for page in pdf.pages:  # type: ignore[attr-defined]
        try:
            images = page.images
        except AttributeError:
            continue

        for raw_image in images.values():
            obj_id = id(raw_image)
            if obj_id in seen_objects:
                continue
            seen_objects.add(obj_id)

            try:
                pdfimage = PdfImage(raw_image)
                strategy = _image_strategy(pdfimage)
                if strategy is None:
                    continue
                weight = pdfimage.width * pdfimage.height
                strategy_weights[strategy] = strategy_weights.get(strategy, 0) + weight
            except Exception:  # noqa: BLE001
                logger.debug(
                    "pdf_image_analysis_skipped",
                    path=str(pdf_path),
                )

    return strategy_weights


def _image_strategy(pdfimage: object) -> str | None:
    """Map a PdfImage's color space to a strategy string.

    Returns:
        Strategy string, or ``None`` if the color space is
        unrecognised.
    """
    cs = pdfimage.colorspace  # type: ignore[attr-defined]
    if cs is None:
        return None
    if cs == "/ICCBased":
        return _resolve_icc_strategy(pdfimage)
    return _COLORSPACE_TO_STRATEGY.get(cs)


def _resolve_icc_strategy(pdfimage: object) -> str:
    """Resolve ICCBased color space to a strategy string.

    ICCBased profiles encode channel count in the ``/N`` key:
    1 = Gray, 3 = RGB, 4 = CMYK.
    """
    try:
        colorspaces = pdfimage._colorspaces  # type: ignore[attr-defined]  # noqa: SLF001
        icc_stream = (
            colorspaces[1][1]
            if pdfimage.indexed  # type: ignore[attr-defined]
            else colorspaces[1]
        )
        n_channels = int(icc_stream["/N"])
    except (IndexError, KeyError, TypeError):
        return "RGB"

    if n_channels == 1:
        return "Gray"
    if n_channels == 4:  # noqa: PLR2004
        return "CMYK"
    return "RGB"
