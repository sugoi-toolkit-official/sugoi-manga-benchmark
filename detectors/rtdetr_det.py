"""ogkalu/comic-text-and-bubble-detector (RT-DETR-v2)."""

import torch
from PIL import Image
from detectors.benchmark import TextDetector, get_device, xyxy_to_xywh


class RTDetrDetector(TextDetector):
    """RT-DETR-v2 text and bubble detector for comics.

    Install: pip install transformers torch
    HuggingFace: https://huggingface.co/ogkalu/comic-text-and-bubble-detector
    Detects 3 classes: bubble(0), text_bubble(1), text_free(2).
    We only use text_bubble and text_free for text detection.
    """

    def __init__(self, threshold: float = 0.5):
        from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

        self.device = get_device()
        self.processor = RTDetrImageProcessor.from_pretrained(
            "ogkalu/comic-text-and-bubble-detector"
        )
        self.model = RTDetrV2ForObjectDetection.from_pretrained(
            "ogkalu/comic-text-and-bubble-detector"
        ).to(self.device).eval()
        self.threshold = threshold
        # Label 1 = text_bubble, Label 2 = text_free
        self.text_labels = {1, 2}

    def detect(self, image: Image.Image) -> list[dict]:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([(image.height, image.width)], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.threshold
        )[0]

        return [
            xyxy_to_xywh(*box.tolist())
            for label, box in zip(results["labels"], results["boxes"])
            if label.item() in self.text_labels
        ]
