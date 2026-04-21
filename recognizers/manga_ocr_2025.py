"""manga-ocr 2025 recognizer (jzhang533/manga-ocr-base-2025).

Same VisionEncoderDecoder architecture as kha-white/manga-ocr-base, retrained
on Manga109-s + synthetic data by jzhang533. Drop-in via the `manga_ocr`
package by pointing it at the 2025 checkpoint.
"""

from PIL import Image
from recognizers.benchmark import TextRecognizer


class MangaOcr2025Recognizer(TextRecognizer):
    """manga-ocr with the 2025 retrained checkpoint.

    HuggingFace: https://huggingface.co/jzhang533/manga-ocr-base-2025
    """

    MODEL_ID = "jzhang533/manga-ocr-base-2025"

    def __init__(self):
        from manga_ocr import MangaOcr
        self.mocr = MangaOcr(pretrained_model_name_or_path=self.MODEL_ID)

    def recognize(self, image: Image.Image) -> str:
        return self.mocr(image)
