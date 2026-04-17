# LLM Translator Benchmark Design

**Date:** 2026-04-17
**Status:** Approved (brainstorm)
**Scope:** Add `translators/` subsystem + `translate` subcommand that benchmarks 5 OpenRouter-hosted LLMs on Japanese-to-English manga translation using the OpenMantra dataset. Metric: BLEU.

## 1. Goals

- Benchmark 5 LLMs on manga translation: ja → en.
- Line-by-line translation per page with in-prompt translation history; history resets on page boundary.
- Evaluate with corpus-level BLEU (sacrebleu), with per-book and per-page breakdowns.
- Follow the existing `detectors/` / `recognizers/` pattern so it plugs into [main.py](../../../main.py) via a new `translate` subcommand.

## 2. Non-goals

- Not benchmarking local LLMs (vLLM, llama.cpp, HF Transformers).
- Not doing OCR → translate end-to-end. Ground-truth `text_ja` is used as input.
- Not implementing response caching (explicitly declined).
- Not doing adaptive prompt engineering per model — same prompt for all 5 models.

## 3. Models

Loaded from `translators/models.json`:

```json
{
  "models": [
    "openai/gpt-5.4",
    "anthropic/claude-sonnet-4.6",
    "google/gemini-3-flash-preview",
    "x-ai/grok-4.20",
    "deepseek/deepseek-v3.2"
  ]
}
```

All 5 are OpenRouter slugs. The same `OpenRouterTranslator` class is instantiated 5× with different `model_id`. Display name = slug.

## 4. File layout

```
translators/
├── __init__.py        # registry: load models.json → {slug: lambda: OpenRouterTranslator(slug)}
├── benchmark.py       # abstract Translator, run_translate_benchmark, BLEU, console + per-page output
├── openrouter.py      # OpenRouterTranslator(model_id) — uses openai.AsyncOpenAI w/ OpenRouter base_url
└── models.json        # model slug list
```

New in repo root:
- [main.py](../../../main.py): add `translate` subparser + `run_translate` function.
- [pyproject.toml](../../../pyproject.toml): add `openai`, `sacrebleu` to `dependencies`.
- [.env.example](../../../.env.example): add `OPENROUTER_API_KEY=`.
- [README.md](../../../README.md): replace "Translation Benchmark — Coming soon" with real section.

## 5. Abstract interface

```python
# translators/benchmark.py
class Translator(ABC):
    @abstractmethod
    async def translate(self, text_ja: str, history: list[tuple[str, str]]) -> str:
        """Translate one JA line given prior (ja, en_gt) pairs from the same page.
        history uses GROUND-TRUTH en (teacher-forcing)."""
```

Why async: 214 pages × ~7 entries × 5 models ≈ 7.5k calls per model. Concurrency needed. Default concurrency = 4 (semaphore) — async-first keeps the code simple.

## 6. Prompt format

**System message (constant):**
```
You are a professional Japanese manga translator. Translate the given Japanese line to natural English. Keep style and terminology consistent with prior lines on the same page. Output ONLY the English translation — no quotes, no romaji, no explanations.
```

**User message — with history (n ≥ 1):**
```
Prior lines on this page:
JA: {h1_ja}
EN: {h1_en}
JA: {h2_ja}
EN: {h2_en}
...

Translate this line:
JA: {current_ja}
EN:
```

**User message — first line of page (no history):**
```
Translate this line:
JA: {current_ja}
EN:
```

History uses **ground-truth `text_en`** (teacher-forcing), not prior predictions. Consequence: every call within a page is independent, so concurrency can span pages *and* lines freely.

Output post-processing:
- `strip()`.
- Strip a leading `EN:` prefix if the model echoes it.
- Strip surrounding matching quotes (`" "`, `「 」`).
- `NFKC` normalize before BLEU.

## 7. Translation flow

```
for book in annotations:
  for page in book.pages:
    history = []
    for entry in page.text:               # order as in JSON
      if not entry.text_ja or not entry.text_en: skip
      task = translate(entry.text_ja, history[:])
      history.append((entry.text_ja, entry.text_en))   # GT EN
      enqueue(task)

gather all tasks with asyncio.Semaphore(4)
```

Concurrency = 4 global across the whole benchmark for one model. Models run sequentially (one model at a time, matching the existing detect/ocr pattern).

## 8. OpenRouter client

```python
# translators/openrouter.py
from openai import AsyncOpenAI

class OpenRouterTranslator(Translator):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    async def translate(self, text_ja, history):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(text_ja, history)},
        ]
        resp = await self._call_with_retry(messages)
        return post_process(resp.choices[0].message.content or "")
```

Retry: 3 attempts, exponential backoff (1s, 2s, 4s) on `openai.RateLimitError`, `openai.APIStatusError` with `status_code >= 500`, and `openai.APIConnectionError`. On final failure → return `""` and log a warning.

Temperature: `0.3` (mild creativity, deterministic-ish). `max_tokens`: `256` (manga lines are short).

## 9. BLEU metric

Using `sacrebleu`:

```python
import sacrebleu
# hypotheses: list[str], references: list[list[str]]
bleu = sacrebleu.corpus_bleu(hyps, [refs])
score = bleu.score  # 0..100
```

- **Overall:** `corpus_bleu(all_hyps, [all_refs]).score`.
- **Per book:** `corpus_bleu(hyps_of_book, [refs_of_book]).score`.
- **Per page (debug only):** average of `sentence_bleu` with `exp` smoothing across entries in that page.

Tokenizer: sacrebleu default (`13a`) — treats EN reference as plain text.

Normalization before BLEU: `unicodedata.normalize("NFKC", t).strip()` on both hypothesis and reference.

Empty hypothesis: keep it as empty string in the hypotheses list. `sacrebleu.corpus_bleu` accepts empty hyps — no ngram matches → contributes a 0 score contribution without raising. For the per-page sentence-BLEU debug path, skip the sentence-BLEU call for empty preds and record `0.0` directly.

## 10. Benchmark result shape

Identical spirit to [recognizers/benchmark.py](../../../recognizers/benchmark.py):

```python
{
  "overall": {"bleu": float, "samples": int},
  "per_book": {book_title: {"bleu": float, "samples": int}, ...},
  "per_page": [
    {"book": str, "page": str, "bleu": float, "samples": int,
     "details": [{"gt": str, "pred": str, "bleu": float}, ...]},
    ...
  ],
}
```

## 11. CLI

```bash
python main.py translate                                 # all models from models.json
python main.py translate --model openai/gpt-5.4          # single
python main.py translate --model openai/gpt-5.4 --model deepseek/deepseek-v3.2
python main.py translate --per-page results.txt          # save per-page
python main.py translate --concurrency 8                 # override default 4
python main.py translate --max-pages 10                  # debug: limit for quick smoke
```

Reuses `add_common_args` from [main.py](../../../main.py) and adds `--concurrency` + `--max-pages` specific to `translate`.

## 12. Error handling

| Failure | Behavior |
|---|---|
| `OPENROUTER_API_KEY` missing | Skip all LLM models with warning; `translate` subcommand exits cleanly if no translators remain. |
| Single API call fails after 3 retries | pred = `""`, log warning, continue. |
| Entry missing `text_en` | Skip entry (no reference = no BLEU). |
| Entry missing `text_ja` | Skip entry. |
| `models.json` missing/malformed | Fail loudly on `translate` startup. |
| Image files not needed (ground-truth text used directly) | N/A — unlike detect/ocr, no image loading. |

## 13. Output (console)

```
======================================================================
TRANSLATION BENCHMARK RESULTS - openai/gpt-5.4
======================================================================

--- Overall ---
BLEU         Samples
--------------------
32.45        1592

--- Per Book ---
Book                       BLEU         Samples
tojime_no_siora            35.21        ...
balloon_dream              31.04        ...
...

[INFO] openai/gpt-5.4 completed in 142.3s
```

Final comparison table across models:
```
TRANSLATION MODEL COMPARISON
Model                            BLEU       Time(s)
openai/gpt-5.4                   32.45      142.3
anthropic/claude-sonnet-4.6      34.12      168.1
...
```

## 14. Dependencies

Add to [pyproject.toml](../../../pyproject.toml):
- `openai>=1.54.0` — OpenAI-compatible client; works with OpenRouter via `base_url`.
- `sacrebleu>=2.4.0` — BLEU.

No new system deps.

## 15. Deferred to plan

- Whether/how to pass OpenRouter analytics headers (`HTTP-Referer`, `X-Title`) through `AsyncOpenAI(default_headers=...)`. Nice-to-have; not blocking.

## 16. Testing

- Smoke test: `python main.py translate --model deepseek/deepseek-v3.2 --max-pages 2` should complete and print a BLEU table.
- Unit-level: mock OpenRouter client; assert prompt format (with/without history), retry logic, post-processing (EN: prefix, quotes), BLEU computation on a fixed (hyp, ref) pair.
- No real API call in CI.
