"""Recognizer registry — lazy imports to avoid loading all dependencies at once."""

__all__ = ["get_all_recognizers"]


def get_all_recognizers() -> dict:
    """Return dict of {name: recognizer_class}. Import errors are caught per-model."""
    recognizers = {}

    try:
        from recognizers.manga_ocr import MangaOcrRecognizer
        recognizers["manga_ocr"] = MangaOcrRecognizer
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[WARN] manga-ocr unavailable: {e}")

    try:
        from recognizers.manga_ocr_2025 import MangaOcr2025Recognizer
        recognizers["manga_ocr_2025"] = MangaOcr2025Recognizer
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[WARN] manga-ocr-2025 unavailable: {e}")

    try:
        from recognizers.paddleocr_rec import PaddleOcrRecognizer
        recognizers["paddleocr"] = PaddleOcrRecognizer
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[WARN] PaddleOCR unavailable: {e}")

    try:
        from recognizers.paddleocr_vl_manga import PaddleOcrVlMangaRecognizer
        recognizers["paddleocr_vl_manga"] = PaddleOcrVlMangaRecognizer
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[WARN] PaddleOCR-VL-For-Manga unavailable: {e}")

    try:
        from recognizers.surya_rec import SuryaRecognizer
        recognizers["surya"] = SuryaRecognizer
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[WARN] Surya unavailable: {e}")

    return recognizers
