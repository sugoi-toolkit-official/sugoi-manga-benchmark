"""Detector registry — lazy imports to avoid loading all dependencies at once."""

from utils import build_registry

__all__ = ["get_all_detectors"]


_SPECS = [
    ("ctd", "detectors.ctd", "CTDDetector", "CTD"),
    ("animetext", "detectors.animetext", "AnimeTextDetector", "AnimeText"),
    ("rtdetr", "detectors.rtdetr_det", "RTDetrDetector", "RT-DETR"),
    ("grounding_dino", "detectors.grounding_dino", "GroundingDinoDetector", "Grounding DINO"),
    ("onnx", "detectors.onnx_det", "OnnxDetector", "ONNX detector"),
    ("owlv2", "detectors.owlv2", "Owlv2Detector", "OWLv2"),
]


def get_all_detectors() -> dict:
    """Return dict of {name: detector_class}. Import errors are caught per-model."""
    return build_registry(_SPECS)
