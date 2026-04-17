"""manga-bench — Benchmark SOTA models for manga text detection, OCR, and translation.

Usage:
    python main.py detect                              # Run all detection models
    python main.py detect --model rtdetr               # Run specific model
    python main.py detect --model rtdetr --model ctd   # Run multiple models
    python main.py ocr                                 # Run all OCR models
    python main.py ocr --model manga_ocr               # Run specific model
"""

import asyncio
import os
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add shared CLI arguments to a subparser."""
    parser.add_argument(
        "--annotation", type=str, default="dataset/open-mantra-dataset/annotation.json",
        help="Path to annotation.json",
    )
    parser.add_argument(
        "--dataset-root", type=str, default="dataset/open-mantra-dataset",
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--model", type=str, action="append", default=None,
        help="Model(s) to run. Omit to run all available.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Directory to save output (visualizations, etc.)",
    )
    parser.add_argument(
        "--per-page", type=str, default=None, metavar="FILE",
        help="Save per-page results to a text file",
    )


def select_models(available: dict, requested: list[str] | None) -> dict:
    """Filter available models by user request."""
    if not requested:
        return available
    selected = {}
    for name in requested:
        if name in available:
            selected[name] = available[name]
        else:
            print(f"[WARN] Model '{name}' not available. Options: {list(available.keys())}")
    return selected


# ---------------------------------------------------------------------------
# detect subcommand
# ---------------------------------------------------------------------------

def run_detect(args: argparse.Namespace) -> None:
    from detectors import get_all_detectors
    from detectors.benchmark import (
        run_benchmark, print_results, write_per_page, print_comparison,
    )

    available = get_all_detectors()
    if not available:
        print("[ERROR] No detectors available. Install dependencies first.")
        return

    selected = select_models(available, args.model)
    if not selected:
        return

    print(f"[INFO] Running {len(selected)} model(s): {list(selected.keys())}")

    all_results = {}
    for name, detector_cls in selected.items():
        print(f"\n{'=' * 70}")
        print(f"[INFO] Loading {name}...")
        try:
            detector = detector_cls()
        except Exception as e:
            print(f"[ERROR] Failed to load {name}: {e}")
            continue

        output_dir = Path(args.output) / name if args.output else None
        print(f"[INFO] Running benchmark with {name}...")
        start = time.time()
        results = run_benchmark(
            detector, args.annotation, args.dataset_root,
            output_dir=output_dir,
        )
        elapsed = time.time() - start

        all_results[name] = {"results": results, "time": elapsed}
        print_results(results)
        print(f"[INFO] {name} completed in {elapsed:.1f}s")

        if args.per_page:
            per_page_file = args.per_page
            if len(selected) > 1:
                base, ext = Path(per_page_file).stem, Path(per_page_file).suffix or ".txt"
                per_page_file = f"{base}_{name}{ext}"
            write_per_page(results, per_page_file, name)

    if len(all_results) > 1:
        print_comparison(all_results)


# ---------------------------------------------------------------------------
# ocr subcommand
# ---------------------------------------------------------------------------

def run_ocr(args: argparse.Namespace) -> None:
    from recognizers import get_all_recognizers
    from recognizers.benchmark import (
        run_ocr_benchmark, print_ocr_results, write_ocr_per_page, print_ocr_comparison,
    )

    available = get_all_recognizers()
    if not available:
        print("[ERROR] No recognizers available. Install dependencies first.")
        return

    selected = select_models(available, args.model)
    if not selected:
        return

    print(f"[INFO] Running {len(selected)} OCR model(s): {list(selected.keys())}")

    all_results = {}
    for name, recognizer_cls in selected.items():
        print(f"\n{'=' * 70}")
        print(f"[INFO] Loading {name}...")
        try:
            recognizer = recognizer_cls()
        except Exception as e:
            print(f"[ERROR] Failed to load {name}: {e}")
            continue

        output_dir = Path(args.output) / name if args.output else None
        print(f"[INFO] Running OCR benchmark with {name}...")
        start = time.time()
        results = run_ocr_benchmark(
            recognizer, args.annotation, args.dataset_root,
            output_dir=output_dir,
        )
        elapsed = time.time() - start

        all_results[name] = {"results": results, "time": elapsed}
        print_ocr_results(results, model_name=name)
        print(f"[INFO] {name} completed in {elapsed:.1f}s")

        if args.per_page:
            per_page_file = args.per_page
            if len(selected) > 1:
                base, ext = Path(per_page_file).stem, Path(per_page_file).suffix or ".txt"
                per_page_file = f"{base}_{name}{ext}"
            write_ocr_per_page(results, per_page_file, name)

    if len(all_results) > 1:
        print_ocr_comparison(all_results)


# ---------------------------------------------------------------------------
# translate subcommand
# ---------------------------------------------------------------------------

def run_translate(args: argparse.Namespace) -> None:
    from translators import get_all_translators
    from translators.benchmark import (
        run_translate_benchmark, print_translation_results,
        write_translation_per_page, print_translation_comparison,
    )

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("[ERROR] OPENROUTER_API_KEY not set. Copy .env.example → .env "
              "and add your OpenRouter key (https://openrouter.ai/keys).")
        return

    available = get_all_translators()
    if not available:
        print("[ERROR] No translators available.")
        return

    selected = select_models(available, args.model)
    if not selected:
        return

    print(f"[INFO] Running {len(selected)} translator(s) in parallel: "
          f"{list(selected.keys())}")
    print(f"[INFO] Per-model concurrency: {args.concurrency}"
          + (f", max_pages: {args.max_pages}" if args.max_pages else ""))

    async def benchmark_one(name: str, factory, position: int):
        try:
            translator = factory()
        except Exception as e:
            print(f"[ERROR] [{name}] load failed: {e}")
            return name, None, 0.0
        start = time.time()
        try:
            results = await run_translate_benchmark(
                translator, args.annotation, args.dataset_root,
                concurrency=args.concurrency, max_pages=args.max_pages,
                progress_desc=name, progress_position=position,
            )
        except Exception as e:
            print(f"[ERROR] [{name}] benchmark failed: {e}")
            return name, None, time.time() - start
        elapsed = time.time() - start
        return name, results, elapsed

    async def run_all():
        return await asyncio.gather(
            *(benchmark_one(name, factory, idx)
              for idx, (name, factory) in enumerate(selected.items()))
        )

    outputs = asyncio.run(run_all())

    all_results: dict[str, dict] = {}
    for name, results, elapsed in outputs:
        if results is None:
            continue
        all_results[name] = {"results": results, "time": elapsed}
        print_translation_results(results, model_name=name)

        if args.per_page:
            per_page_file = args.per_page
            if len(selected) > 1:
                base, ext = Path(per_page_file).stem, Path(per_page_file).suffix or ".tsv"
                slug_safe = name.replace("/", "_")
                per_page_file = f"{base}_{slug_safe}{ext}"
            write_translation_per_page(results, per_page_file, name)

    if len(all_results) > 1:
        print_translation_comparison(all_results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="manga-bench: Benchmark SOTA models for manga text detection, OCR, and translation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # detect
    detect_parser = subparsers.add_parser("detect", help="Run text detection benchmark")
    add_common_args(detect_parser)

    # ocr
    ocr_parser = subparsers.add_parser("ocr", help="Run OCR recognition benchmark")
    add_common_args(ocr_parser)

    # translate
    translate_parser = subparsers.add_parser("translate", help="Run translation benchmark")
    add_common_args(translate_parser)
    translate_parser.add_argument(
        "--concurrency", type=int, default=4,
        help="Max concurrent API calls (default: 4)",
    )
    translate_parser.add_argument(
        "--max-pages", type=int, default=None,
        help="Debug: limit total pages processed across the whole dataset",
    )

    args = parser.parse_args()

    if args.command == "detect":
        run_detect(args)
    elif args.command == "ocr":
        run_ocr(args)
    elif args.command == "translate":
        run_translate(args)


if __name__ == "__main__":
    main()
