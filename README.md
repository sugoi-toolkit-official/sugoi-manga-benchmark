# manga-bench

Benchmark SOTA models for manga/anime text detection, OCR recognition, and translation.

Uses the [OpenMantra](https://github.com/mantra-inc/open-mantra-dataset) dataset (5 manga series, 214 pages, 1592 text annotations) from the AAAI 2021 paper *"Towards Fully Automated Manga Translation"*.

## Quick Start

```bash
# Clone dataset
cd dataset
git clone https://github.com/mantra-inc/open-mantra-dataset.git
cd ..

# Install dependencies
uv sync

# Set up env vars (needed for gated models like AnimeText)
cp .env.example .env
# Edit .env and add your HF_TOKEN

# Run benchmarks
python main.py detect                       # text detection
python main.py ocr                          # OCR recognition
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
| `onnx` | YOLOv8 ONNX (manga-specific) | Local model in `models/` |
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
| `onnx` | 0.4482 | 0.4567 | 0.4524 | — |
| `ctd` | 0.4503 | 0.4523 | 0.4513 | 23.8s |
| `grounding_dino` | 0.3101 | 0.4108 | 0.3534 | 94.7s |

<details>
<summary>Per-book breakdown (IoU=0.5)</summary>

| Book | rtdetr | animetext | onnx | ctd | grounding_dino | owlv2 |
|------|--------|-----------|------|-----|----------------|-------|
| tojime_no_siora | **0.9701** | 0.6025 | 0.6190 | 0.6078 | 0.4565 | 0.5448 |
| balloon_dream | **0.9506** | 0.5877 | 0.5920 | 0.5746 | 0.5022 | 0.5466 |
| tencho_isoro | **0.9365** | 0.4465 | 0.4101 | 0.4236 | 0.3543 | 0.4380 |
| boureisougi | **0.9177** | 0.3062 | 0.2801 | 0.3066 | 0.2837 | 0.2729 |
| rasetugari | **0.9400** | 0.3572 | 0.3471 | 0.3343 | 0.2114 | 0.3041 |

*Values are F1 scores.*

</details>

## OCR Recognition Benchmark

Evaluates OCR models on Japanese text recognition using ground-truth bounding boxes.
Metrics: CER (Character Error Rate), Accuracy, 1-NED (Normalized Edit Distance).

### Available Models

| Model | Type | Source |
|-------|------|--------|
| `manga_ocr` | TrOCR (manga-specific) | [kha-white/manga-ocr-base](https://huggingface.co/kha-white/manga-ocr-base) |
| `paddleocr` | PaddleOCR (Japanese) | [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) |
| `surya` | Surya OCR (multi-language) | [surya-ocr](https://github.com/VikParuchuri/surya) |
| `easyocr` | EasyOCR (Japanese) | [JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR) |

### Usage

```bash
python main.py ocr                              # run all available models
python main.py ocr --model manga_ocr            # run specific model
python main.py ocr --model manga_ocr --model paddleocr  # multiple
```

## Translation Benchmark

Evaluates LLM translators on Japanese-to-English manga translation using ground-truth text from the dataset.
Translation is line-by-line per page with prior (ja, en) pairs stacked into the prompt as inline context; history
resets on each new page. History uses ground-truth EN (teacher-forcing), so calls within a page are independent
and can run in parallel. Metric: corpus-level BLEU (`sacrebleu`).

### Available Models

All 5 run via [OpenRouter](https://openrouter.ai/). Slugs are loaded from [`translators/models.json`](translators/models.json):

| Slug | Provider |
|------|----------|
| `openai/gpt-5.4` | OpenAI |
| `anthropic/claude-sonnet-4.6` | Anthropic |
| `google/gemini-3-flash-preview` | Google |
| `x-ai/grok-4.20` | xAI |
| `deepseek/deepseek-v3.2` | DeepSeek |

### Setup

```bash
# Add OpenRouter key to .env (see .env.example)
echo "OPENROUTER_API_KEY=sk-or-v1-..." >> .env
```

### Usage

```bash
python main.py translate                                    # run all models in models.json
python main.py translate --model deepseek/deepseek-v3.2     # run one model
python main.py translate --model openai/gpt-5.4 --model deepseek/deepseek-v3.2   # multiple
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

1. Create `detectors/your_model.py`:

```python
from PIL import Image
from detectors.benchmark import TextDetector

class YourDetector(TextDetector):
    def __init__(self):
        # load your model here
        pass

    def detect(self, image: Image.Image) -> list[dict]:
        # return list of {"x": int, "y": int, "w": int, "h": int}
        ...
```

2. Register in `detectors/__init__.py`:

```python
try:
    from detectors.your_model import YourDetector
    detectors["your_model"] = YourDetector
except Exception as e:
    print(f"[WARN] your_model unavailable: {e}")
```

### OCR

Same pattern: subclass `TextRecognizer` from `recognizers/benchmark.py`, implement `recognize(image) -> str`, register in `recognizers/__init__.py`.

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
