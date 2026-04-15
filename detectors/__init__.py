"""Detector registry — lazy imports to avoid loading all dependencies at once."""

__all__ = ["get_all_detectors"]


def get_all_detectors() -> dict:
    """Return dict of {name: detector_class}. Import errors are caught per-model."""
    detectors = {}

    try:
        from detectors.ctd import CTDDetector
        detectors["ctd"] = CTDDetector
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[WARN] CTD unavailable: {e}")

    try:
        from detectors.animetext import AnimeTextDetector
        detectors["animetext"] = AnimeTextDetector
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[WARN] AnimeText unavailable: {e}")

    try:
        from detectors.rtdetr_det import RTDetrDetector
        detectors["rtdetr"] = RTDetrDetector
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[WARN] RT-DETR unavailable: {e}")

    try:
        from detectors.grounding_dino import GroundingDinoDetector
        detectors["grounding_dino"] = GroundingDinoDetector
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[WARN] Grounding DINO unavailable: {e}")

    try:
        from detectors.onnx_det import OnnxDetector
        detectors["onnx"] = OnnxDetector
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[WARN] ONNX detector unavailable: {e}")

    try:
        from detectors.owlv2 import Owlv2Detector
        detectors["owlv2"] = Owlv2Detector
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[WARN] OWLv2 unavailable: {e}")

    return detectors
