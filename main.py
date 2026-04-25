"""manga-bench — Benchmark SOTA models for manga text detection, OCR, and translation.

Usage:
    python main.py detect                              # Run all detection models
    python main.py detect --model rtdetr               # Run specific model
    python main.py detect --model rtdetr --model ctd   # Run multiple models
    python main.py ocr                                 # Run all OCR models
    python main.py ocr --model manga_ocr               # Run specific model
"""

import argparse
import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Shared CLI helpers
# ---------------------------------------------------------------------------

def add_common_args(parser: argparse.ArgumentParser) -> None:
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
    if not requested:
        return available
    selected = {}
    for name in requested:
        if name in available:
            selected[name] = available[name]
        else:
            print(f"[WARN] Model '{name}' not available. Options: {list(available.keys())}")
    return selected


def _per_page_path(template: str, model_name: str, multi: bool,
                   slug_safe: bool, default_ext: str) -> str:
    if not multi:
        return template
    base = Path(template).stem
    ext = Path(template).suffix or default_ext
    safe_name = model_name.replace("/", "_") if slug_safe else model_name
    return f"{base}_{safe_name}{ext}"


# ---------------------------------------------------------------------------
# Generic task dispatcher
# ---------------------------------------------------------------------------

@dataclass
class TaskSpec:
    label: str  # human-readable: "detection" | "OCR" | "translation"
    registry: Callable[[], dict]
    invoke: Callable[..., Awaitable[Any]]  # (model, args, name, position) -> results
    printer: Callable[[Any, str], None]    # (results, model_name)
    writer: Callable[[Any, str, str], None]  # (results, path, model_name)
    compare: Callable[[dict], None]
    per_page_ext: str
    parallel: bool
    slug_safe: bool = False


async def _benchmark_one(spec: TaskSpec, factory, args, name: str, position: int):
    """Load model, run benchmark, time it. Returns (name, results|None, elapsed)."""
    print(f"\n{'=' * 70}")
    print(f"[INFO] Loading {name}...")
    try:
        model = factory()
    except Exception as e:
        print(f"[ERROR] Failed to load {name}: {e}")
        return name, None, 0.0

    print(f"[INFO] Running {spec.label} benchmark with {name}...")
    start = time.time()
    try:
        results = await spec.invoke(model, args, name, position)
    except Exception as e:
        print(f"[ERROR] [{name}] benchmark failed: {e}")
        return name, None, time.time() - start
    return name, results, time.time() - start


def _record_result(spec: TaskSpec, args, all_results: dict, multi: bool,
                   name: str, results: Any, elapsed: float) -> None:
    if results is None:
        return
    all_results[name] = {"results": results, "time": elapsed}
    spec.printer(results, name)
    print(f"[INFO] {name} completed in {elapsed:.1f}s")
    if args.per_page:
        path = _per_page_path(args.per_page, name, multi, spec.slug_safe, spec.per_page_ext)
        spec.writer(results, path, name)


async def _run_task(spec: TaskSpec, args) -> None:
    available = spec.registry()
    if not available:
        print(f"[ERROR] No {spec.label} models available. Install dependencies first.")
        return

    selected = select_models(available, args.model)
    if not selected:
        return

    print(f"[INFO] Running {len(selected)} {spec.label} model(s): {list(selected.keys())}")
    multi = len(selected) > 1
    all_results: dict = {}

    if spec.parallel:
        # Build all tasks up front so tqdm bars stack via unique positions.
        outputs = await asyncio.gather(*(
            _benchmark_one(spec, factory, args, name, idx)
            for idx, (name, factory) in enumerate(selected.items())
        ))
        for name, results, elapsed in outputs:
            _record_result(spec, args, all_results, multi, name, results, elapsed)
    else:
        # Serial: print each model's results before the next loads, so long
        # benchmarks surface progress incrementally.
        for idx, (name, factory) in enumerate(selected.items()):
            name, results, elapsed = await _benchmark_one(spec, factory, args, name, idx)
            _record_result(spec, args, all_results, multi, name, results, elapsed)

    if len(all_results) > 1:
        spec.compare(all_results)


def run_task(spec: TaskSpec, args) -> None:
    asyncio.run(_run_task(spec, args))


# ---------------------------------------------------------------------------
# Subcommands — each just builds a TaskSpec and dispatches.
# Imports are lazy so e.g. `translate` doesn't pull in torch.
# ---------------------------------------------------------------------------

def run_detect(args: argparse.Namespace) -> None:
    from detectors import get_all_detectors
    from detectors.benchmark import (
        run_benchmark, print_results, write_per_page, print_comparison,
    )

    async def invoke(model, args, name, position):
        output_dir = Path(args.output) / name if args.output else None
        return run_benchmark(
            model, args.annotation, args.dataset_root, output_dir=output_dir,
        )

    run_task(TaskSpec(
        label="detection",
        registry=get_all_detectors,
        invoke=invoke,
        printer=print_results,
        writer=write_per_page,
        compare=print_comparison,
        per_page_ext=".txt",
        parallel=False,
    ), args)


def run_ocr(args: argparse.Namespace) -> None:
    from recognizers import get_all_recognizers
    from recognizers.benchmark import (
        run_ocr_benchmark, print_ocr_results, write_ocr_per_page, print_ocr_comparison,
    )

    async def invoke(model, args, name, position):
        output_dir = Path(args.output) / name if args.output else None
        return run_ocr_benchmark(
            model, args.annotation, args.dataset_root, output_dir=output_dir,
        )

    run_task(TaskSpec(
        label="OCR",
        registry=get_all_recognizers,
        invoke=invoke,
        printer=print_ocr_results,
        writer=write_ocr_per_page,
        compare=print_ocr_comparison,
        per_page_ext=".txt",
        parallel=False,
    ), args)


def run_translate(args: argparse.Namespace) -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("[ERROR] OPENROUTER_API_KEY not set. Copy .env.example -> .env "
              "and add your OpenRouter key (https://openrouter.ai/keys).")
        return

    from translators import get_all_translators
    from translators.benchmark import (
        run_translate_benchmark, print_translation_results,
        write_translation_per_page, print_translation_comparison,
    )

    async def invoke(model, args, name, position):
        return await run_translate_benchmark(
            model, args.annotation, args.dataset_root,
            concurrency=args.concurrency, max_pages=args.max_pages,
            progress_desc=name, progress_position=position,
        )

    print(f"[INFO] Per-model concurrency: {args.concurrency}"
          + (f", max_pages: {args.max_pages}" if args.max_pages else ""))

    run_task(TaskSpec(
        label="translation",
        registry=get_all_translators,
        invoke=invoke,
        printer=print_translation_results,
        writer=write_translation_per_page,
        compare=print_translation_comparison,
        per_page_ext=".tsv",
        parallel=True,
        slug_safe=True,
    ), args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="manga-bench: Benchmark SOTA models for manga text detection, OCR, and translation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect_parser = subparsers.add_parser("detect", help="Run text detection benchmark")
    add_common_args(detect_parser)

    ocr_parser = subparsers.add_parser("ocr", help="Run OCR recognition benchmark")
    add_common_args(ocr_parser)

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

    dispatch = {"detect": run_detect, "ocr": run_ocr, "translate": run_translate}
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
