"""OWLv2 — zero-shot text detection via open-vocabulary object detection."""

import torch
from PIL import Image
from detectors.benchmark import TextDetector, xyxy_to_xywh
from utils import get_device, load_hf_model

MODEL_ID = "google/owlv2-large-patch14-ensemble"


class Owlv2Detector(TextDetector):
    """OWLv2 zero-shot open-vocabulary detector prompted for text regions.

    Install: pip install transformers torch
    HuggingFace: https://huggingface.co/google/owlv2-large-patch14-ensemble
    """

    def __init__(self, threshold: float = 0.1):
        from transformers import Owlv2Processor, Owlv2ForObjectDetection

        self.device = get_device()
        self.processor, self.model = load_hf_model(
            Owlv2Processor, Owlv2ForObjectDetection, MODEL_ID, self.device,
        )
        self.threshold = threshold

    def detect(self, image: Image.Image) -> list[dict]:
        inputs = self.processor(
            text=[["text"]], images=image, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([(image.height, image.width)], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.threshold
        )[0]

        return [xyxy_to_xywh(*box.tolist()) for box in results["boxes"]]
