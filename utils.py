"""Cross-task helpers shared by detect/ocr/translate benchmarks."""

import importlib
import json
import unicodedata
from pathlib import Path
from typing import Iterable

from PIL import Image


__all__ = [
    "get_device",
    "load_annotations",
    "iter_pages",
    "normalize_text",
    "build_registry",
]


# ---------------------------------------------------------------------------
# Torch device
# ---------------------------------------------------------------------------

def get_device():
    """Auto-detect best available torch device: CUDA > MPS > CPU."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Annotation I/O
# ---------------------------------------------------------------------------

def load_annotations(annotation_path: str) -> list[dict]:
    """Load annotation.json and return the list of books."""
    with open(annotation_path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_pages(annotations: list[dict], dataset_root: str):
    """Yield (book_title, image_rel_path, image, gt_entries) for each page.

    Skips pages whose image file is missing on disk.
    """
    for book in annotations:
        book_title = book["book_title"]
        for page in book["pages"]:
            image_rel_path = page["image_paths"]["ja"]
            image_path = Path(dataset_root) / image_rel_path

            if not image_path.exists():
                print(f"[WARN] Image not found: {image_path}, skipping.")
                continue

            image = Image.open(image_path).convert("RGB")
            gt_entries = page.get("text", [])
            yield book_title, image_rel_path, image, gt_entries


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """NFKC-normalize and strip whitespace for fair comparison."""
    return unicodedata.normalize("NFKC", text).strip()


# ---------------------------------------------------------------------------
# Lazy-import registry factory
# ---------------------------------------------------------------------------

def build_registry(specs: Iterable[tuple[str, str, str, str]]) -> dict:
    """Return {key: imported_attr} for every spec that imported successfully.

    Each spec is (key, module_path, attr_name, display_name). Failed imports
    are warned about per-spec so a missing optional dependency doesn't take
    down the whole registry.
    """
    out: dict = {}
    for key, module_path, attr_name, display_name in specs:
        try:
            module = importlib.import_module(module_path)
            out[key] = getattr(module, attr_name)
        except (ImportError, ModuleNotFoundError, AttributeError) as e:
            print(f"[WARN] {display_name} unavailable: {e}")
    return out
