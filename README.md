# manga-bench

Benchmark SOTA models for manga/anime text detection, OCR recognition, and translation.

Uses the [OpenMantra](https://github.com/mantra-inc/open-mantra-dataset) dataset (5 manga series, 214 pages, 1592 text annotations) from the AAAI 2021 paper *"Towards Fully Automated Manga Translation"*.

- [Text Detection Benchmark](#text-detection-benchmark)
- [OCR Recognition Benchmark](#ocr-recognition-benchmark)
- [Translation Benchmark](#translation-benchmark)

## Quick Start

```bash
# Clone dataset
cd dataset
git clone https://github.com/mantra-inc/open-mantra-dataset.git
cd ..

# Install dependencies
uv sync

# Set up env vars
cp .env.example .env
# Edit .env — add HF_TOKEN (for gated models) and OPENROUTER_API_KEY (for VLM/LLM benchmarks)

# Run benchmarks
python main.py detect                       # text detection
python main.py ocr                          # OCR recognition
python main.py translate                    # translation
```

## Text Detection Benchmark

Evaluates text detection models using IoU-based matching (Hungarian algorithm).
Metrics: Precision, Recall, F1 at IoU thresholds 0.5 - 0.9.

### Available Models

| Model | Type | Source |
|-------|------|--------|
| `animetext` | YOLO12 (manga-specific) | [deepghs/AnimeText_yolo](https://huggingface.co/deepghs/AnimeText_yolo) |
| `rtdetr` | RT-DETR-v2 (comic text+bubble) | [ogkalu/comic-text-and-bubble-detector](https://huggingface.co/ogkalu/comic-text-and-bubble-detector) |
| `grounding_dino` | Zero-shot (prompt: "text.") | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base) |
| `owlv2` | Zero-shot open-vocabulary | [google/owlv2-large-patch14-ensemble](https://huggingface.co/google/owlv2-large-patch14-ensemble) |
| `ctd` | DBNet+YOLOv5 (manga-specific) | [comic-text-detector](https://github.com/dmMaze/comic-text-detector) |

### Usage

```bash
python main.py detect                              # run all available models
python main.py detect --model rtdetr               # run specific model
python main.py detect --model rtdetr --model ctd   # run multiple models
python main.py detect --output output/             # save visualizations
python main.py detect --per-page results.txt       # save per-page results
```

### Results (OpenMantra, IoU=0.5)

| Model | Precision | Recall | F1 | Time |
|-------|-----------|--------|----|------|
| `rtdetr` | **0.9424** | **0.9454** | **0.9439** | 14.4s |
| `owlv2` | 0.5942 | 0.3329 | 0.4267 | 317.4s |
| `animetext` | 0.4075 | 0.5258 | 0.4591 | 13.5s |
| `ctd` | 0.4503 | 0.4523 | 0.4513 | 23.8s |
| `grounding_dino` | 0.3101 | 0.4108 | 0.3534 | 94.7s |

<details>
<summary>Per-book breakdown (IoU=0.5)</summary>

| Book | rtdetr | animetext | ctd | grounding_dino | owlv2 |
|------|--------|-----------|-----|----------------|-------|
| tojime_no_siora | **0.9701** | 0.6025 | 0.6078 | 0.4565 | 0.5448 |
| balloon_dream | **0.9506** | 0.5877 | 0.5746 | 0.5022 | 0.5466 |
| tencho_isoro | **0.9365** | 0.4465 | 0.4236 | 0.3543 | 0.4380 |
| boureisougi | **0.9177** | 0.3062 | 0.3066 | 0.2837 | 0.2729 |
| rasetugari | **0.9400** | 0.3572 | 0.3343 | 0.2114 | 0.3041 |

*Values are F1 scores.*

</details>

## OCR Recognition Benchmark

Evaluates OCR models on Japanese text recognition using ground-truth bounding boxes.
Metrics: CER (Character Error Rate), Accuracy, 1-NED (Normalized Edit Distance).

### Available Models

**Local models** (run on GPU):

| Model | Type | Source |
|-------|------|--------|
| `manga_ocr` | TrOCR (manga-specific) | [kha-white/manga-ocr-base](https://huggingface.co/kha-white/manga-ocr-base) |
| `manga_ocr_2025` | TrOCR retrained 2025 | [jzhang533/manga-ocr-base-2025](https://huggingface.co/jzhang533/manga-ocr-base-2025) |
| `paddleocr` | PaddleOCR (Japanese) | [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) |
| `paddleocr_vl_manga` | PaddleOCR-VL fine-tuned for manga | [jzhang533/PaddleOCR-VL-For-Manga](https://huggingface.co/jzhang533/PaddleOCR-VL-For-Manga) |

**OpenRouter VLMs** (via API, requires `OPENROUTER_API_KEY`):

Slugs are loaded from [`recognizers/models.json`](recognizers/models.json). Add or swap any vision-capable model from [openrouter.ai](https://openrouter.ai/models).

| Slug | Provider |
|------|----------|
| `openai/gpt-5.5` | OpenAI |
| `anthropic/claude-sonnet-4.6` | Anthropic |
| `google/gemini-3-flash-preview` | Google |
| `x-ai/grok-4.20` | xAI |
| `deepseek/deepseek-v4-flash` | DeepSeek |

### Usage

```bash
python main.py ocr                                          # run all available models
python main.py ocr --model manga_ocr                        # run specific model
python main.py ocr --model manga_ocr --model paddleocr      # run multiple
python main.py ocr --model openrouter                       # run all OpenRouter VLMs
python main.py ocr --model openrouter --concurrency 16      # override concurrency (default: 8)
python main.py ocr --per-page results.txt                   # save per-page results
```

### Results (OpenMantra, 1586 samples)

| Model | CER ↓ | Accuracy ↑ | 1-NED ↑ | Time |
|-------|-------|------------|---------|------|
| `manga_ocr` | **0.0349** | **0.8733** | **0.9684** | 460.1s |
| `manga_ocr_2025` | 0.0426 | 0.8506 | 0.9600 | 406.4s |
| `google/gemini-3.1-flash-lite-preview` | 0.2431 | 0.8417 | 0.9459 | 140.7s |
| `anthropic/claude-sonnet-4.6` | 0.2162 | 0.7812 | 0.9311 | 127.1s |
| `paddleocr` | 0.1255 | 0.5303 | 0.8806 | 2216.8s |
| `openai/gpt-5.4` | 0.1294 | 0.6166 | 0.8865 | 65.2s |
| `x-ai/grok-4.20` | 0.2438 | 0.5246 | 0.8404 | 47.7s |
| `z-ai/glm-5v-turbo` | 0.2533 | 0.4224 | 0.7537 | 287.4s |
| `qwen/qwen3.6-plus` | 0.2925 | 0.4142 | 0.7165 | 115.1s |

Sorted by Accuracy. `paddleocr_vl_manga` not benchmarked yet.

<details>
<summary>Per-book breakdown (Accuracy)</summary>

| Book | manga_ocr | manga_ocr_2025 | gemini-3.1-flash-lite | claude-sonnet-4.6 | paddleocr | gpt-5.4 | grok-4.20 | glm-5v-turbo | qwen3.6-plus |
|------|-----------|----------------|-----------------------|-------------------|-----------|---------|-----------|--------------|--------------|
| tojime_no_siora | **0.8889** | 0.8709 | 0.8468 | 0.7898 | 0.5706 | 0.6607 | 0.5676 | 0.3874 | 0.4054 |
| balloon_dream | **0.8758** | 0.8726 | 0.8567 | 0.8280 | 0.5732 | 0.6529 | 0.5064 | 0.4108 | 0.4713 |
| tencho_isoro | 0.8896 | **0.8896** | **0.8669** | 0.7857 | 0.5942 | 0.6461 | 0.5357 | 0.4578 | 0.3766 |
| boureisougi | **0.8864** | 0.8535 | 0.8498 | 0.7839 | 0.3004 | 0.4799 | 0.4579 | 0.3480 | 0.4212 |
| rasetugari | **0.8324** | 0.7961 | 0.7961 | 0.7263 | 0.5754 | 0.6229 | 0.5419 | 0.4916 | 0.3994 |

</details>

## Translation Benchmark

Evaluates LLM translators on Japanese-to-English manga translation using ground-truth text from the dataset.
Translation is line-by-line per page with prior (ja, en) pairs stacked into the prompt as inline context; history
resets on each new page. History uses ground-truth EN (teacher-forcing), so calls within a page are independent
and can run in parallel. Metric: corpus-level BLEU (`sacrebleu`).

### Available Models

All 5 run via [OpenRouter](https://openrouter.ai/). Slugs are loaded from [`translators/models.json`](translators/models.json):

| Slug | Provider |
|------|----------|
| `openai/gpt-5.5` | OpenAI |
| `anthropic/claude-sonnet-4.6` | Anthropic |
| `google/gemini-3-flash-preview` | Google |
| `x-ai/grok-4.20` | xAI |
| `deepseek/deepseek-v4-flash` | DeepSeek |

### Setup

```bash
# Add OpenRouter key to .env (see .env.example)
echo "OPENROUTER_API_KEY=sk-or-v1-..." >> .env
```

### Usage

```bash
python main.py translate                                    # run all models in models.json
python main.py translate --model deepseek/deepseek-v4-flash # run one model
python main.py translate --model openai/gpt-5.5 --model deepseek/deepseek-v4-flash  # multiple
python main.py translate --max-pages 5                      # debug: first 5 pages only
python main.py translate --concurrency 8                    # override default 4
python main.py translate --per-page results.tsv             # save per-entry TSV
```

`--per-page` writes a TSV with one row per text entry (columns: `book`, `page`, `entry_idx`, `sentence_bleu`, `gt_en`, `pred_en`).

### Results (OpenMantra, full dataset, 5 models run in parallel)

| Model | BLEU | Time |
|-------|------|------|
| `deepseek/deepseek-v3.2` | **18.16** | 1273.7s |
| `anthropic/claude-sonnet-4.6` | 17.44 | 967.2s |
| `openai/gpt-5.4` | 16.52 | 461.8s |
| `x-ai/grok-4.20` | 15.65 | 329.7s |
| `google/gemini-3-flash-preview` | 15.32 | 645.0s |

Sorted by BLEU. `Time` is wall-clock for that model (all 5 models run concurrently, so total elapsed ≈ the slowest model, not the sum).

## Adding a New Model

### Detection

1. Create `detectors/your_model.py` subclassing `TextDetector`, implement `detect(image) -> list[dict]`
2. Add a spec tuple to `_SPECS` in `detectors/__init__.py`

### OCR

1. Create `recognizers/your_model.py` subclassing `TextRecognizer`, implement `recognize(image) -> str`
2. Add a spec tuple to `_LOCAL_SPECS` in `recognizers/__init__.py`

For OpenRouter VLMs, just add the model slug to `recognizers/models.json` — no code needed.

### Translation

Add the model slug to `translators/models.json` — no code needed.

## Dataset

This benchmark uses the [OpenMantra dataset](https://github.com/mantra-inc/open-mantra-dataset):

- 5 Japanese manga series across different genres
- 214 pages with 1592 text region annotations
- Professional translations (Japanese, English, Chinese)

### Citation

```bibtex
@inproceedings{hinami2021towards,
  title={Towards Fully Automated Manga Translation},
  author={Hinami, Ryota and Delorme, Shonosuke and Saito, Yusuke and Fadaei, Hossein},
  booktitle={AAAI},
  year={2021}
}
```

## License

MIT
