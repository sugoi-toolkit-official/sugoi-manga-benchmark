"""Grounding DINO — zero-shot text detection via text prompt."""

import torch
from PIL import Image
from detectors.benchmark import TextDetector, xyxy_to_xywh
from utils import get_device


class GroundingDinoDetector(TextDetector):
    """Grounding DINO zero-shot object detector prompted for text regions.

    Install: pip install transformers torch
    HuggingFace: https://huggingface.co/IDEA-Research/grounding-dino-base
    """

    def __init__(self, threshold: float = 0.3, prompt: str = "text."):
        from transformers import AutoProcessor, GroundingDinoForObjectDetection

        self.device = get_device()
        self.processor = AutoProcessor.from_pretrained(
            "IDEA-Research/grounding-dino-base"
        )
        self.model = GroundingDinoForObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-base"
        ).to(self.device).eval()
        self.threshold = threshold
        self.prompt = prompt

    def detect(self, image: Image.Image) -> list[dict]:
        inputs = self.processor(
            images=image, text=self.prompt, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            target_sizes=[(image.height, image.width)],
            threshold=self.threshold,
            text_threshold=self.threshold,
        )[0]

        return [xyxy_to_xywh(*box.tolist()) for box in results["boxes"]]
