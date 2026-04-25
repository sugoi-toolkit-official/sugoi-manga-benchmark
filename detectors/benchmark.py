"""Text Detection Benchmark for OpenMantra Dataset.

Evaluates text detection models against ground truth bounding boxes using
standard metrics: Precision, Recall, F1-score at multiple IoU thresholds.
"""

import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

from utils import iter_pages, load_annotations


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def xyxy_to_xywh(x1, y1, x2, y2) -> dict:
    """Convert (x1, y1, x2, y2) to {"x", "y", "w", "h"} dict."""
    return {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)}


def _filter_valid_boxes(boxes: list[dict], min_size: int = 2) -> list[dict]:
    """Remove degenerate boxes (zero/negative dimensions)."""
    return [b for b in boxes if b["w"] >= min_size and b["h"] >= min_size]


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class TextDetector(ABC):
    """Base class for text detection models.
    Subclass this and implement the `detect` method to benchmark your model."""

    @abstractmethod
    def detect(self, image: Image.Image) -> list[dict]:
        """Detect text regions in an image.

        Args:
            image: PIL Image object.

        Returns:
            List of dicts, each with keys "x", "y", "w", "h" (int/float).
            (x, y) is top-left corner, (w, h) is width and height.
        """
        pass


# ---------------------------------------------------------------------------
# IoU and matching
# ---------------------------------------------------------------------------

def compute_iou(box_a: dict, box_b: dict) -> float:
    """Compute IoU between two boxes. Each box is {"x", "y", "w", "h"} (top-left + size)."""
    ax1, ay1 = box_a["x"], box_a["y"]
    ax2, ay2 = ax1 + box_a["w"], ay1 + box_a["h"]
    bx1, by1 = box_b["x"], box_b["y"]
    bx2, by2 = bx1 + box_b["w"], by1 + box_b["h"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = box_a["w"] * box_a["h"]
    area_b = box_b["w"] * box_b["h"]
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def match_boxes(
    pred_boxes: list[dict],
    gt_boxes: list[dict],
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """Match predicted boxes to GT boxes using Hungarian algorithm.

    Returns:
        (true_positives, false_positives, false_negatives)
    """
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 0, 0, 0
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou(pb, gb)

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    tp = 0
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            tp += 1

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def draw_boxes(
    image: Image.Image,
    gt_boxes: list[dict],
    pred_boxes: list[dict],
) -> Image.Image:
    """Draw GT (green) and prediction (red) bounding boxes on a copy of the image."""
    vis = image.copy()
    draw = ImageDraw.Draw(vis)

    for b in gt_boxes:
        x1, y1 = b["x"], b["y"]
        x2, y2 = x1 + b["w"], y1 + b["h"]
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)

    for b in pred_boxes:
        x1, y1 = b["x"], b["y"]
        x2, y2 = x1 + b["w"], y1 + b["h"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    return vis


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(
    detector: TextDetector,
    annotation_path: str,
    dataset_root: str,
    iou_thresholds: list[float] | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Run benchmark on all pages in the annotation file."""
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    annotations = load_annotations(annotation_path)

    totals = {t: {"tp": 0, "fp": 0, "fn": 0} for t in iou_thresholds}
    per_book = {}
    per_page = []

    for book_title, image_rel_path, image, gt_boxes in iter_pages(annotations, dataset_root):
        if book_title not in per_book:
            per_book[book_title] = {t: {"tp": 0, "fp": 0, "fn": 0} for t in iou_thresholds}

        pred_boxes = _filter_valid_boxes(detector.detect(image))

        if output_dir is not None:
            vis = draw_boxes(image, gt_boxes, pred_boxes)
            save_path = output_dir / image_rel_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            vis.save(save_path, quality=85)

        page_counts = {}
        for t in iou_thresholds:
            tp, fp, fn = match_boxes(pred_boxes, gt_boxes, iou_threshold=t)
            totals[t]["tp"] += tp
            totals[t]["fp"] += fp
            totals[t]["fn"] += fn
            per_book[book_title][t]["tp"] += tp
            per_book[book_title][t]["fp"] += fp
            per_book[book_title][t]["fn"] += fn
            page_counts[t] = {"tp": tp, "fp": fp, "fn": fn}

        per_page.append({
            "book": book_title,
            "page": image_rel_path,
            "metrics": _calc_metrics(page_counts),
        })

    per_book = {k: _calc_metrics(v) for k, v in per_book.items()}
    overall = _calc_metrics(totals)
    return {"overall": overall, "per_book": per_book, "per_page": per_page}


def _calc_metrics(totals: dict) -> dict:
    """Convert tp/fp/fn counts into precision/recall/f1 per threshold."""
    results = {}
    for t, counts in totals.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        results[t] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(results: dict) -> None:
    """Print benchmark results in a readable table format."""
    print("\n" + "=" * 70)
    print("TEXT DETECTION BENCHMARK RESULTS")
    print("=" * 70)

    print("\n--- Overall ---")
    print(f"{'IoU Thresh':<12} {'Precision':<12} {'Recall':<12} "
          f"{'F1':<12} {'TP':<8} {'FP':<8} {'FN':<8}")
    print("-" * 70)
    for t in sorted(results["overall"].keys()):
        m = results["overall"][t]
        print(f"{t:<12.1f} {m['precision']:<12.4f} {m['recall']:<12.4f} "
              f"{m['f1']:<12.4f} {m['tp']:<8} {m['fp']:<8} {m['fn']:<8}")

    print("\n--- Per Book (IoU=0.5) ---")
    print(f"{'Book':<25} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 60)
    for book, metrics in results["per_book"].items():
        if 0.5 in metrics:
            m = metrics[0.5]
            print(f"{book:<25} {m['precision']:<12.4f} "
                  f"{m['recall']:<12.4f} {m['f1']:<12.4f}")
    print()


def write_per_page(results: dict, filepath: str, model_name: str, iou: float = 0.5) -> None:
    """Write per-page accuracy to a text file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Per-Page Results — model: {model_name}, IoU: {iou}\n")
        f.write(f"{'Page':<45} {'Prec':<8} {'Rec':<8} {'F1':<8} {'TP':<6} {'FP':<6} {'FN':<6}\n")
        f.write("-" * 90 + "\n")
        for entry in results.get("per_page", []):
            m = entry["metrics"].get(iou, {})
            f.write(
                f"{entry['page']:<45} "
                f"{m.get('precision',0):<8.4f} "
                f"{m.get('recall',0):<8.4f} "
                f"{m.get('f1',0):<8.4f} "
                f"{m.get('tp',0):<6} "
                f"{m.get('fp',0):<6} "
                f"{m.get('fn',0):<6}\n"
            )
    print(f"[INFO] Per-page results saved to {filepath}")


def print_comparison(all_results: dict[str, dict], iou: float = 0.5) -> None:
    """Print side-by-side comparison of all models at a given IoU threshold."""
    print("\n" + "=" * 75)
    print(f"COMPARISON (IoU={iou})")
    print("=" * 75)
    print(f"{'Model':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Time(s)':<10}")
    print("-" * 75)
    for name, data in all_results.items():
        m = data["results"]["overall"].get(iou, {})
        t = data["time"]
        print(f"{name:<20} {m.get('precision',0):<12.4f} {m.get('recall',0):<12.4f} "
              f"{m.get('f1',0):<12.4f} {t:<10.1f}")
    print()
