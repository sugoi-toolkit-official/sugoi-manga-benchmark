"""ONNX YOLOv8 — manga text detection via ONNX model."""

import os
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from PIL import Image
from detectors.benchmark import TextDetector

MODEL_INPUT_SIZE = 640


class OnnxDetector(TextDetector):
    """ONNX-based manga text detector (YOLOv8 single-class).

    Uses model-v8-1l.onnx in the project root.
    """

    def __init__(self, threshold: float = 0.25, iou_threshold: float = 0.40):
        models_dir = Path(os.environ.get(
            "MANGA_BENCH_MODELS_DIR",
            Path(__file__).resolve().parent.parent / "models",
        ))
        model_path = models_dir / "model-v8-1l.onnx"
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {model_path}\n"
                "Place model-v8-1l.onnx in the models/ directory, "
                "or set MANGA_BENCH_MODELS_DIR env var."
            )

        providers = []
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.threshold = threshold
        self.iou_threshold = iou_threshold

    def detect(self, image: Image.Image) -> list[dict]:
        img = np.array(image)[:, :, ::-1]  # RGB -> BGR
        orig_h, orig_w = img.shape[:2]

        # Preprocess
        blob = cv2.dnn.blobFromImage(
            img, scalefactor=1.0 / 255.0,
            size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            mean=(0, 0, 0), swapRB=True, crop=False,
        )
        scale_x = orig_w / MODEL_INPUT_SIZE
        scale_y = orig_h / MODEL_INPUT_SIZE

        # Inference
        output = self.session.run([self.output_name], {self.input_name: blob})[0]

        # Postprocess — output shape (1, 5, 8400): [cx, cy, w, h, conf]
        preds = output[0].T  # (8400, 5)
        conf = preds[:, 4]
        mask = conf >= self.threshold
        preds = preds[mask]
        conf = conf[mask]

        if len(conf) == 0:
            return []

        cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2

        # NMS
        boxes_for_nms = np.column_stack((x1, y1, w, h)).tolist()
        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms, conf.tolist(), self.threshold, self.iou_threshold,
        )
        if len(indices) == 0:
            return []
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()

        x2 = cx + w / 2
        y2 = cy + h / 2

        detections = []
        for i in indices:
            bx1 = int(round(x1[i] * scale_x))
            by1 = int(round(y1[i] * scale_y))
            bx2 = int(round(x2[i] * scale_x))
            by2 = int(round(y2[i] * scale_y))
            bw, bh = bx2 - bx1, by2 - by1
            if bw >= 5 and bh >= 5:
                detections.append({"x": bx1, "y": by1, "w": bw, "h": bh})
        return detections
