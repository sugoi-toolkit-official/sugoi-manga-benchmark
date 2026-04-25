"""Recognizer registry — lazy imports to avoid loading all dependencies at once."""

from utils import build_registry

__all__ = ["get_all_recognizers"]


_SPECS = [
    ("manga_ocr", "recognizers.manga_ocr", "MangaOcrRecognizer", "manga-ocr"),
    ("manga_ocr_2025", "recognizers.manga_ocr_2025", "MangaOcr2025Recognizer", "manga-ocr-2025"),
    ("paddleocr", "recognizers.paddleocr_rec", "PaddleOcrRecognizer", "PaddleOCR"),
    ("paddleocr_vl_manga", "recognizers.paddleocr_vl_manga", "PaddleOcrVlMangaRecognizer", "PaddleOCR-VL-For-Manga"),
]


def get_all_recognizers() -> dict:
    """Return dict of {name: recognizer_class}. Import errors are caught per-model."""
    return build_registry(_SPECS)
