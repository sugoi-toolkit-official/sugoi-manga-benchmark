"""PaddleOCR Japanese recognizer (PaddleOCR 3.x API)."""

import numpy as np
from PIL import Image
from recognizers.benchmark import TextRecognizer


class PaddleOcrRecognizer(TextRecognizer):
    """PaddleOCR with Japanese language support.

    Install: pip install paddleocr
    Uses full pipeline (detection + recognition) on cropped regions
    to handle multi-line text in manga speech bubbles.
    PaddleOCR 3.x API: uses predict() instead of ocr().
    """

    def __init__(self):
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(
            lang="japan",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def recognize(self, image: Image.Image) -> str:
        img_array = np.array(image)
        result = self.ocr.predict(input=img_array)

        if not result:
            return ""

        # PaddleOCR 3.x result: list of result objects with
        # rec_texts (list[str]), rec_scores (list[float]), dt_polys (list[ndarray])
        lines = []
        for res in result:
            texts = res.get("rec_texts", [])
            polys = res.get("dt_polys", [])

            for text, poly in zip(texts, polys):
                if not text:
                    continue
                # poly is ndarray shape (4, 2): [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                poly = np.array(poly)
                x_center = poly[:, 0].mean()
                y_center = poly[:, 1].mean()
                lines.append((x_center, y_center, text))

        if not lines:
            return ""

        # Sort: right-to-left (x descending), then top-to-bottom (y ascending)
        # This matches Japanese vertical reading order
        lines.sort(key=lambda t: (-t[0], t[1]))
        return "".join(line[2] for line in lines)
