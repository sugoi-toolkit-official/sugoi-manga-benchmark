"""Translation Benchmark for OpenMantra Dataset.

Evaluates LLM translators on Japanese-to-English manga translation using
ground-truth text from annotation.json. Translation is line-by-line per
page with prior (ja, en) pairs stacked into the prompt as inline context;
history resets on each new page. Metric: corpus-level BLEU via sacrebleu.
"""

import asyncio
from abc import ABC, abstractmethod

import sacrebleu
from tqdm import tqdm as _tqdm
from tqdm.asyncio import tqdm as atqdm

from utils import load_annotations, normalize_text


def tqdm_safe_print(*args, **kwargs) -> None:
    """print() that plays nicely with active tqdm progress bars."""
    _tqdm.write(" ".join(str(a) for a in args), **kwargs)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def iter_translation_entries(
    annotations: list[dict],
    max_pages: int | None = None,
):
    """Yield (book_title, page_rel_path, entry_idx, ja_norm, en_norm) tuples
    in JSON order. Entries missing either ja or en text are skipped. Text is
    NFKC-normalized once here so the same string drives both the prompt and
    the BLEU reference.

    max_pages: global cap across flattened (book, page) pairs (not per-book).
    """
    pages_seen = 0
    for book in annotations:
        book_title = book["book_title"]
        for page in book["pages"]:
            if max_pages is not None and pages_seen >= max_pages:
                return
            pages_seen += 1
            page_rel_path = page["image_paths"]["ja"]
            for entry_idx, entry in enumerate(page.get("text", [])):
                ja = entry.get("text_ja", "")
                en = entry.get("text_en", "")
                if not ja or not en:
                    continue
                yield (
                    book_title,
                    page_rel_path,
                    entry_idx,
                    normalize_text(ja),
                    normalize_text(en),
                )


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class Translator(ABC):
    """Base class for translation models.
    Subclass this and implement the async `translate` method."""

    @abstractmethod
    async def translate(
        self,
        text_ja: str,
        history: tuple[tuple[str, str], ...],
    ) -> str:
        """Translate one JA line given prior (ja, en_gt) pairs from the same
        page. History uses ground-truth EN (teacher-forcing)."""
        pass


# ---------------------------------------------------------------------------
# BLEU metrics
# ---------------------------------------------------------------------------

def compute_corpus_bleu(hyps: list[str], refs: list[str]) -> float:
    """Corpus-level BLEU (0..100) via sacrebleu."""
    if not hyps:
        return 0.0
    return sacrebleu.corpus_bleu(hyps, [refs]).score


def compute_sentence_bleu(hyp: str, ref: str) -> float:
    """Sentence-level BLEU (0..100) with exp smoothing. Empty hyp short-circuits to 0."""
    if not hyp:
        return 0.0
    return sacrebleu.sentence_bleu(hyp, [ref], smooth_method="exp").score


# ---------------------------------------------------------------------------
# Async benchmark loop
# ---------------------------------------------------------------------------

async def _bounded_translate(
    semaphore: asyncio.Semaphore,
    translator: Translator,
    text_ja: str,
    history: tuple[tuple[str, str], ...],
) -> str:
    """Run translator.translate under a semaphore. Swallow any uncaught
    exception and return "" so asyncio.gather doesn't abort the batch."""
    async with semaphore:
        try:
            return await translator.translate(text_ja, history)
        except Exception as e:
            tqdm_safe_print(f"[WARN] translate() raised unexpectedly: {type(e).__name__}: {e}")
            return ""


async def run_translate_benchmark(
    translator: Translator,
    annotation_path: str,
    dataset_root: str,
    concurrency: int = 4,
    max_pages: int | None = None,
    progress_desc: str | None = None,
    progress_position: int = 0,
) -> dict:
    """Run translation benchmark across all pages.

    dataset_root is accepted for CLI uniformity with detect/ocr but is unused:
    translation works off the JSON text directly, no images are opened.

    progress_desc / progress_position: shown in the tqdm bar. When multiple
    models run in parallel, each one should get a unique `progress_position`
    so the bars stack instead of overwriting each other.
    """
    del dataset_root  # intentionally unused

    annotations = load_annotations(annotation_path)

    # Phase 1 (sync): build flat task list + parallel metadata list.
    # Teacher-forcing: history known up front from GT, so all coroutines can
    # be constructed before any API call fires.
    semaphore = asyncio.Semaphore(concurrency)
    tasks: list = []
    task_meta: list[tuple[str, str, int, str, str]] = []  # (book, page, idx, ja, en)

    prev_page: str | None = None
    history: list[tuple[str, str]] = []
    for book, page, entry_idx, ja_norm, en_norm in iter_translation_entries(
        annotations, max_pages=max_pages
    ):
        if page != prev_page:
            history = []
            prev_page = page
        snapshot = tuple(history)
        tasks.append(_bounded_translate(semaphore, translator, ja_norm, snapshot))
        task_meta.append((book, page, entry_idx, ja_norm, en_norm))
        history.append((ja_norm, en_norm))

    # Phase 2 (async): fire all tasks; tqdm.gather preserves input order
    # and shows a progress bar as coroutines complete.
    preds_raw = await atqdm.gather(
        *tasks,
        desc=progress_desc or "translating",
        position=progress_position,
        leave=True,
        total=len(tasks),
    )
    preds = [normalize_text(p) for p in preds_raw]

    # Phase 3 (sync): group + score.
    all_hyps: list[str] = []
    all_refs: list[str] = []
    per_book: dict[str, dict[str, list[str]]] = {}
    pages_order: list[tuple[str, str]] = []  # (book, page) in first-seen order
    per_page_buckets: dict[tuple[str, str], list[dict]] = {}

    for (book, page, entry_idx, ja_norm, en_norm), pred in zip(task_meta, preds):
        all_hyps.append(pred)
        all_refs.append(en_norm)

        if book not in per_book:
            per_book[book] = {"hyps": [], "refs": []}
        per_book[book]["hyps"].append(pred)
        per_book[book]["refs"].append(en_norm)

        key = (book, page)
        if key not in per_page_buckets:
            per_page_buckets[key] = []
            pages_order.append(key)
        per_page_buckets[key].append({
            "entry_idx": entry_idx,
            "gt": en_norm,
            "pred": pred,
            "bleu": compute_sentence_bleu(pred, en_norm),
        })

    overall = {
        "bleu": compute_corpus_bleu(all_hyps, all_refs),
        "samples": len(all_hyps),
    }
    per_book_agg = {
        book: {
            "bleu": compute_corpus_bleu(data["hyps"], data["refs"]),
            "samples": len(data["hyps"]),
        }
        for book, data in per_book.items()
    }
    per_page = []
    for book, page in pages_order:
        details = per_page_buckets[(book, page)]
        page_bleu = (
            sum(d["bleu"] for d in details) / len(details) if details else 0.0
        )
        per_page.append({
            "book": book,
            "page": page,
            "bleu": page_bleu,
            "samples": len(details),
            "details": details,
        })

    return {"overall": overall, "per_book": per_book_agg, "per_page": per_page}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_translation_results(results: dict, model_name: str = "") -> None:
    """Print translation benchmark results."""
    title = "TRANSLATION BENCHMARK RESULTS"
    if model_name:
        title += f" - {model_name}"

    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)

    o = results["overall"]
    print(f"\n--- Overall ---")
    print(f"{'BLEU':<12} {'Samples':<10}")
    print("-" * 25)
    print(f"{o['bleu']:<12.2f} {o['samples']:<10}")

    print(f"\n--- Per Book ---")
    print(f"{'Book':<25} {'BLEU':<12} {'Samples':<10}")
    print("-" * 50)
    for book, m in results["per_book"].items():
        print(f"{book:<25} {m['bleu']:<12.2f} {m['samples']:<10}")
    print()


def _tsv_safe(text: str) -> str:
    """Strip tabs and newlines so the field stays on one TSV row."""
    return text.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def write_translation_per_page(results: dict, filepath: str, model_name: str) -> None:
    """Write per-entry TSV with sentence-BLEU + predictions."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Per-entry translation results — model: {model_name}\n")
        f.write("book\tpage\tentry_idx\tsentence_bleu\tgt_en\tpred_en\n")
        for page_entry in results.get("per_page", []):
            book = page_entry["book"]
            page = page_entry["page"]
            for d in page_entry["details"]:
                f.write(
                    f"{book}\t{page}\t{d['entry_idx']}\t"
                    f"{d['bleu']:.2f}\t{_tsv_safe(d['gt'])}\t{_tsv_safe(d['pred'])}\n"
                )
    print(f"[INFO] Per-entry translation results saved to {filepath}")


def print_translation_comparison(all_results: dict[str, dict]) -> None:
    """Print side-by-side comparison of all translation models."""
    print(f"\n{'=' * 70}")
    print("TRANSLATION MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Model':<40} {'BLEU':<10} {'Time(s)':<10}")
    print("-" * 70)
    for name, data in all_results.items():
        o = data["results"]["overall"]
        t = data["time"]
        print(f"{name:<40} {o['bleu']:<10.2f} {t:<10.1f}")
    print()
