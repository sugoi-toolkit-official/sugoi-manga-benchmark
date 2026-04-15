"""Surya OCR recognizer (surya-ocr 0.17.x API)."""

from PIL import Image
from recognizers.benchmark import TextRecognizer


class SuryaRecognizer(TextRecognizer):
    """Surya OCR with Japanese support.

    Install: pip install surya-ocr
    Uses FoundationPredictor + DetectionPredictor + RecognitionPredictor.
    Surya 0.17.x requires FoundationPredictor as a shared backbone.
    """

    def __init__(self):
        from surya.foundation import FoundationPredictor
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor

        self.foundation = FoundationPredictor()
        self.det_predictor = DetectionPredictor()
        self.rec_predictor = RecognitionPredictor(self.foundation)

    def recognize(self, image: Image.Image) -> str:
        results = self.rec_predictor(
            [image],
            det_predictor=self.det_predictor,
            sort_lines=True,
        )

        if not results or not results[0].text_lines:
            return ""

        # Extract text lines with positions
        lines = []
        for text_line in results[0].text_lines:
            bbox = text_line.bbox  # [x1, y1, x2, y2]
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            lines.append((x_center, y_center, text_line.text))

        # Sort: right-to-left (x descending), then top-to-bottom (y ascending)
        # This matches Japanese vertical reading order
        lines.sort(key=lambda t: (-t[0], t[1]))
        return "".join(line[2] for line in lines)
