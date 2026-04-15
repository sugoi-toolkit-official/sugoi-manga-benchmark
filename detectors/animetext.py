"""AnimeText YOLO12 detector (deepghs/AnimeText_yolo)."""

import os
from PIL import Image
from detectors.benchmark import TextDetector, xyxy_to_xywh


class AnimeTextDetector(TextDetector):
    """AnimeText YOLO12 text detector trained on 735K anime/manga images.

    Install: pip install ultralytics huggingface_hub
    HuggingFace: https://huggingface.co/deepghs/AnimeText_yolo

    Set HF_TOKEN env var or pass token to constructor for gated repo access.
    """

    def __init__(self, variant: str = "yolo12x", token: str = None):
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download

        hf_token = token or os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "HuggingFace token required for gated repo. "
                "Set HF_TOKEN env var or pass token= to constructor."
            )
        model_path = hf_hub_download("deepghs/AnimeText_yolo",
                                     f"{variant}_animetext/model.pt",
                                     token=hf_token)
        self.model = YOLO(model_path)

    def detect(self, image: Image.Image) -> list[dict]:
        results = self.model(image, verbose=False)
        boxes = results[0].boxes

        return [xyxy_to_xywh(*box.xyxy[0].tolist()) for box in boxes]
