"""manga-ocr recognizer (kha-white/manga-ocr-base)."""

from PIL import Image
from recognizers.benchmark import TextRecognizer


class MangaOcrRecognizer(TextRecognizer):
    """TrOCR-based model specialized for Japanese manga text.

    Install: pip install manga-ocr
    HuggingFace: https://huggingface.co/kha-white/manga-ocr-base
    """

    def __init__(self):
        from manga_ocr import MangaOcr
        self.mocr = MangaOcr()

    def recognize(self, image: Image.Image) -> str:
        return self.mocr(image)
