"""OCR Recognition Benchmark for OpenMantra Dataset.

Evaluates OCR models on Japanese text recognition using ground-truth
bounding boxes from the dataset. Metrics: CER, Accuracy, 1-NED.
"""

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image

from rapidfuzz.distance import Levenshtein
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from utils import iter_pages, load_annotations, normalize_text


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
# Metrics
# ---------------------------------------------------------------------------

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

async def run_ocr_benchmark(
    recognizer: TextRecognizer,
    annotation_path: str,
    dataset_root: str,
    output_dir: Path | None = None,
    concurrency: int = 8,
    progress_desc: str | None = None,
    progress_position: int = 0,
) -> dict:
    """Run OCR benchmark on all text regions in the dataset.

    Phase 1: load all valid crops from every page into a flat list.
    Phase 2: recognize — async recognizers use atqdm.gather with a semaphore;
             sync recognizers use a plain tqdm loop.
    Phase 3: group by page/book and compute metrics.
    """
    is_async = asyncio.iscoroutinefunction(recognizer.recognize)
    sem = asyncio.Semaphore(concurrency)
    desc = progress_desc or "recognizing"

    annotations = load_annotations(annotation_path)

    # Phase 1: collect all valid crops (full-page images are freed after cropping).
    entries: list[tuple[str, str, int, str, object]] = []
    for book_title, image_rel_path, image, gt_entries in iter_pages(annotations, dataset_root):
        for i, entry in enumerate(gt_entries):
            gt_text = entry.get("text_ja", "")
            if not gt_text:
                continue
            x, y, w, h = entry["x"], entry["y"], entry["w"], entry["h"]
            img_w, img_h = image.size
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img_w, x + w), min(img_h, y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            entries.append((book_title, image_rel_path, i, gt_text, image.crop((x1, y1, x2, y2))))

    # Phase 2: recognize.
    if is_async:
        async def _bounded(crop):
            async with sem:
                return await recognizer.recognize(crop)

        pred_texts = await atqdm.gather(
            *[_bounded(crop) for *_, crop in entries],
            desc=desc,
            position=progress_position,
            leave=True,
            total=len(entries),
        )
    else:
        pred_texts = []
        for *_, crop in tqdm(entries, desc=desc, position=progress_position, leave=True):
            pred_texts.append(recognizer.recognize(crop))

    # Phase 3: score and group by page / book.
    all_cer, all_ned, all_acc = [], [], []
    per_book: dict[str, dict] = {}
    per_page_map: dict[str, dict] = {}

    for (book_title, image_rel_path, i, gt_text, cropped), pred_text in zip(entries, pred_texts):
        gt_norm = normalize_text(gt_text)
        pred_norm = normalize_text(pred_text)

        cer = compute_cer(pred_norm, gt_norm)
        ned = compute_ned(pred_norm, gt_norm)
        acc = 1.0 if pred_norm == gt_norm else 0.0

        all_cer.append(cer)
        all_ned.append(ned)
        all_acc.append(acc)

        if book_title not in per_book:
            per_book[book_title] = {"cer": [], "ned": [], "acc": []}
        per_book[book_title]["cer"].append(cer)
        per_book[book_title]["ned"].append(ned)
        per_book[book_title]["acc"].append(acc)

        if image_rel_path not in per_page_map:
            per_page_map[image_rel_path] = {"book": book_title, "cer": [], "ned": [], "acc": [], "details": []}
        per_page_map[image_rel_path]["cer"].append(cer)
        per_page_map[image_rel_path]["ned"].append(ned)
        per_page_map[image_rel_path]["acc"].append(acc)
        per_page_map[image_rel_path]["details"].append({"gt": gt_norm, "pred": pred_norm, "cer": cer, "ned": ned, "acc": acc})

        if output_dir is not None:
            page_stem = Path(image_rel_path).stem
            save_dir = output_dir / Path(image_rel_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            cropped.save(save_dir / f"{page_stem}_{i}.png")

    per_book_agg = {
        book: {"cer": _avg(d["cer"]), "ned": _avg(d["ned"]), "acc": _avg(d["acc"]), "samples": len(d["cer"])}
        for book, d in per_book.items() if d["cer"]
    }
    per_page = [
        {"book": d["book"], "page": page, "cer": _avg(d["cer"]), "ned": _avg(d["ned"]),
         "acc": _avg(d["acc"]), "samples": len(d["cer"]), "details": d["details"]}
        for page, d in per_page_map.items()
    ]

    return {
        "overall": {"cer": _avg(all_cer), "ned": _avg(all_ned), "acc": _avg(all_acc), "samples": len(all_cer)},
        "per_book": per_book_agg,
        "per_page": per_page,
    }


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
