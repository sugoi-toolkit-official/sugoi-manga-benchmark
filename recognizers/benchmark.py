"""OCR Recognition Benchmark for OpenMantra Dataset.

Evaluates OCR models on Japanese text recognition using ground-truth
bounding boxes from the dataset. Metrics: CER, Accuracy, 1-NED.
"""

import json
import unicodedata
from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image

from rapidfuzz.distance import Levenshtein


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_annotations(annotation_path: str) -> list[dict]:
    """Load annotation.json and return the list of books."""
    with open(annotation_path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_pages(annotations: list[dict], dataset_root: str):
    """Yield (book_title, image_rel_path, image, gt_entries) for each page."""
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
# Abstract base class
# ---------------------------------------------------------------------------

class TextRecognizer(ABC):
    """Base class for OCR recognition models.
    Subclass this and implement the `recognize` method to benchmark your model."""

    @abstractmethod
    def recognize(self, image: Image.Image) -> str:
        """Recognize text from a cropped image of a single text region.

        Args:
            image: PIL Image of a cropped text region.

        Returns:
            Recognized text string.
        """
        pass


# ---------------------------------------------------------------------------
# Text normalization and metrics
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize text for fair comparison."""
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()
    return text


def compute_cer(pred: str, gt: str) -> float:
    """Character Error Rate: edit_distance / max(len(gt), 1)."""
    if not gt:
        return 0.0 if not pred else 1.0
    return Levenshtein.distance(pred, gt) / max(len(gt), 1)


def compute_ned(pred: str, gt: str) -> float:
    """Normalized Edit Distance similarity: 1 - edit_distance / max(len(pred), len(gt), 1)."""
    max_len = max(len(pred), len(gt), 1)
    return 1.0 - Levenshtein.distance(pred, gt) / max_len


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------

def run_ocr_benchmark(
    recognizer: TextRecognizer,
    annotation_path: str,
    dataset_root: str,
    output_dir: Path | None = None,
) -> dict:
    """Run OCR benchmark on all text regions in the dataset."""
    annotations = load_annotations(annotation_path)

    all_cer = []
    all_ned = []
    all_acc = []
    per_book: dict[str, dict] = {}
    per_page = []

    for book_title, image_rel_path, image, gt_entries in iter_pages(annotations, dataset_root):
        if book_title not in per_book:
            per_book[book_title] = {"cer": [], "ned": [], "acc": []}

        page_cer = []
        page_ned = []
        page_acc = []
        page_details = []

        for i, entry in enumerate(gt_entries):
            gt_text = entry.get("text_ja", "")
            if not gt_text:
                continue

            x, y, w, h = entry["x"], entry["y"], entry["w"], entry["h"]
            img_w, img_h = image.size
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_w, x + w)
            y2 = min(img_h, y + h)
            if x2 <= x1 or y2 <= y1:
                continue

            cropped = image.crop((x1, y1, x2, y2))
            pred_text = recognizer.recognize(cropped)

            gt_norm = normalize_text(gt_text)
            pred_norm = normalize_text(pred_text)

            cer = compute_cer(pred_norm, gt_norm)
            ned = compute_ned(pred_norm, gt_norm)
            acc = 1.0 if pred_norm == gt_norm else 0.0

            page_cer.append(cer)
            page_ned.append(ned)
            page_acc.append(acc)
            all_cer.append(cer)
            all_ned.append(ned)
            all_acc.append(acc)
            per_book[book_title]["cer"].append(cer)
            per_book[book_title]["ned"].append(ned)
            per_book[book_title]["acc"].append(acc)

            page_details.append({
                "gt": gt_norm, "pred": pred_norm,
                "cer": cer, "ned": ned, "acc": acc,
            })

            if output_dir is not None:
                page_stem = Path(image_rel_path).stem
                save_dir = output_dir / Path(image_rel_path).parent
                save_dir.mkdir(parents=True, exist_ok=True)
                cropped.save(save_dir / f"{page_stem}_{i}.png")

        if page_cer:
            per_page.append({
                "book": book_title,
                "page": image_rel_path,
                "cer": _avg(page_cer),
                "ned": _avg(page_ned),
                "acc": _avg(page_acc),
                "samples": len(page_cer),
                "details": page_details,
            })

    per_book_agg = {}
    for book, data in per_book.items():
        if data["cer"]:
            per_book_agg[book] = {
                "cer": _avg(data["cer"]),
                "ned": _avg(data["ned"]),
                "acc": _avg(data["acc"]),
                "samples": len(data["cer"]),
            }

    overall = {
        "cer": _avg(all_cer),
        "ned": _avg(all_ned),
        "acc": _avg(all_acc),
        "samples": len(all_cer),
    }

    return {"overall": overall, "per_book": per_book_agg, "per_page": per_page}


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_ocr_results(results: dict, model_name: str = "") -> None:
    """Print OCR benchmark results."""
    title = "OCR BENCHMARK RESULTS"
    if model_name:
        title += f" - {model_name}"

    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)

    o = results["overall"]
    print(f"\n--- Overall ---")
    print(f"{'CER':<12} {'Accuracy':<12} {'1-NED':<12} {'Samples':<10}")
    print("-" * 50)
    print(f"{o['cer']:<12.4f} {o['acc']:<12.4f} {o['ned']:<12.4f} {o['samples']:<10}")

    print(f"\n--- Per Book ---")
    print(f"{'Book':<25} {'CER':<12} {'Accuracy':<12} {'1-NED':<12} {'Samples':<10}")
    print("-" * 70)
    for book, m in results["per_book"].items():
        print(f"{book:<25} {m['cer']:<12.4f} {m['acc']:<12.4f} "
              f"{m['ned']:<12.4f} {m['samples']:<10}")
    print()


def write_ocr_per_page(results: dict, filepath: str, model_name: str) -> None:
    """Write per-page OCR results to a text file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Per-Page OCR Results — model: {model_name}\n")
        f.write(f"{'Page':<45} {'CER':<10} {'Acc':<10} {'1-NED':<10} {'N':<6}\n")
        f.write("-" * 85 + "\n")
        for entry in results.get("per_page", []):
            f.write(
                f"{entry['page']:<45} "
                f"{entry['cer']:<10.4f} "
                f"{entry['acc']:<10.4f} "
                f"{entry['ned']:<10.4f} "
                f"{entry['samples']:<6}\n"
            )
    print(f"[INFO] Per-page OCR results saved to {filepath}")


def print_ocr_comparison(all_results: dict[str, dict]) -> None:
    """Print side-by-side comparison of all OCR models."""
    print(f"\n{'=' * 75}")
    print("OCR MODEL COMPARISON")
    print("=" * 75)
    print(f"{'Model':<20} {'CER':<12} {'Accuracy':<12} {'1-NED':<12} {'Time(s)':<10}")
    print("-" * 75)
    for name, data in all_results.items():
        o = data["results"]["overall"]
        t = data["time"]
        print(f"{name:<20} {o['cer']:<12.4f} {o['acc']:<12.4f} "
              f"{o['ned']:<12.4f} {t:<10.1f}")
    print()
