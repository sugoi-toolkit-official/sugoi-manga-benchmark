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
        from recognizers.paddleocr_rec import PaddleOcrRecognizer
        recognizers["paddleocr"] = PaddleOcrRecognizer
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[WARN] PaddleOCR unavailable: {e}")

    try:
        from recognizers.surya_rec import SuryaRecognizer
        recognizers["surya"] = SuryaRecognizer
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[WARN] Surya unavailable: {e}")

    try:
        from recognizers.easyocr_rec import EasyOcrRecognizer
        recognizers["easyocr"] = EasyOcrRecognizer
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[WARN] EasyOCR unavailable: {e}")

    return recognizers
