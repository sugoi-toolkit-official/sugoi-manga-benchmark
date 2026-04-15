"""EasyOCR Japanese recognizer."""

import numpy as np
from PIL import Image
from recognizers.benchmark import TextRecognizer


class EasyOcrRecognizer(TextRecognizer):
    """EasyOCR with Japanese language support.

    Install: pip install easyocr
    Uses readtext() which performs detection + recognition on cropped regions.
    """

    def __init__(self, gpu: bool = True):
        import easyocr
        self.reader = easyocr.Reader(["ja"], gpu=gpu)

    def recognize(self, image: Image.Image) -> str:
        img_array = np.array(image)
        results = self.reader.readtext(img_array)

        if not results:
            return ""

        # results is [(bbox, text, confidence), ...]
        # bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        lines = []
        for bbox, text, _conf in results:
            x_center = (bbox[0][0] + bbox[1][0]) / 2
            y_center = (bbox[0][1] + bbox[2][1]) / 2
            lines.append((x_center, y_center, text))

        # Sort: right-to-left, then top-to-bottom (vertical reading order)
        lines.sort(key=lambda t: (-t[0], t[1]))
        return "".join(line[2] for line in lines)
