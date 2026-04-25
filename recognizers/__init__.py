"""Recognizer registry — lazy imports to avoid loading all dependencies at once."""

import functools
import json
import os
from pathlib import Path

from utils import build_registry

__all__ = ["get_all_recognizers"]


_LOCAL_SPECS = [
    ("manga_ocr", "recognizers.manga_ocr", "MangaOcrRecognizer", "manga-ocr"),
    ("manga_ocr_2025", "recognizers.manga_ocr_2025", "MangaOcr2025Recognizer", "manga-ocr-2025"),
    ("paddleocr", "recognizers.paddleocr_rec", "PaddleOcrRecognizer", "PaddleOCR"),
    ("paddleocr_vl_manga", "recognizers.paddleocr_vl_manga", "PaddleOcrVlMangaRecognizer", "PaddleOCR-VL-For-Manga"),
]


def get_all_recognizers() -> dict:
    """Return dict of {name: recognizer_class_or_factory}.

    Local models use direct class references; OpenRouter VLMs use
    functools.partial factories bound to a model slug.
    Import errors are caught per-model.
    """
    registry = build_registry(_LOCAL_SPECS)

    if not os.environ.get("OPENROUTER_API_KEY"):
        return registry

    try:
        from recognizers.openrouter import OpenRouterRecognizer
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[WARN] OpenRouter recognizer unavailable: {e}")
        return registry

    cfg_path = Path(__file__).parent / "models.json"
    slugs = json.loads(cfg_path.read_text(encoding="utf-8"))["models"]
    for slug in slugs:
        registry[slug] = functools.partial(OpenRouterRecognizer, slug)

    return registry
