"""Comic Text Detector (CTD) — manga-specific text detection.

Requires cloning the repo first:
    git clone https://github.com/dmMaze/comic-text-detector.git
    Download weights from: https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.2.1
    Place comictextdetector.pt in comic-text-detector/data/
"""

import sys
import os
import numpy as np
import cv2
from PIL import Image
from detectors.benchmark import TextDetector, xyxy_to_xywh
from utils import get_device

CTD_REPO_PATH = os.environ.get("CTD_REPO_PATH", "comic-text-detector")


class CTDDetector(TextDetector):
    """Comic Text Detector (DBNet + YOLOv5 hybrid) trained on manga data.

    Setup:
        git clone https://github.com/dmMaze/comic-text-detector.git
        Download comictextdetector.pt from manga-image-translator releases
        Place weights in comic-text-detector/data/comictextdetector.pt

    Or set CTD_REPO_PATH env var to the cloned repo location.
    """

    def __init__(self, model_path: str = None):
        # Patch numpy 2.0+ for older CTD code that uses removed aliases
        for alias, replacement in [
            ("bool8", "bool_"), ("int0", "intp"), ("float_", "float64"),
            ("complex_", "complex128"), ("object_", "object_"), ("str_", "str_"),
            ("long", "int64"), ("unicode_", "str_"),
        ]:
            if not hasattr(np, alias):
                setattr(np, alias, getattr(np, replacement))

        if CTD_REPO_PATH not in sys.path:
            sys.path.insert(0, CTD_REPO_PATH)

        from inference import TextDetector as CTDModel

        if model_path is None:
            model_path = os.path.join(CTD_REPO_PATH, "data", "comictextdetector.pt")

        device = str(get_device())
        self.model = CTDModel(
            model_path=model_path,
            input_size=1024,
            device=device,
            half=False,
            nms_thresh=0.35,
            conf_thresh=0.4,
            mask_thresh=0.3,
        )

    def detect(self, image: Image.Image) -> list[dict]:
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        _mask, _mask_refined, blk_list = self.model(img_bgr)

        return [xyxy_to_xywh(*blk.xyxy) for blk in blk_list]
