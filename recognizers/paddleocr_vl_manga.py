"""PaddleOCR-VL fine-tuned for manga (jzhang533/PaddleOCR-VL-For-Manga).

0.3B-param ERNIE-4.5-based VLM, fine-tuned on Manga109-s + 1.5M synthetic
samples. Runs via HuggingFace Transformers — no paddlepaddle runtime needed.

HuggingFace: https://huggingface.co/jzhang533/PaddleOCR-VL-For-Manga
"""

import torch
from PIL import Image
from recognizers.benchmark import TextRecognizer


class PaddleOcrVlMangaRecognizer(TextRecognizer):
    """PaddleOCR-VL fine-tuned for Japanese manga text."""

    MODEL_ID = "jzhang533/PaddleOCR-VL-For-Manga"
    PROMPT = "OCR:"

    def __init__(self):
        from transformers import (
            PaddleOCRVLForConditionalGeneration,
            PaddleOCRVLProcessor,
            PaddleOCRVLImageProcessor,
            AutoTokenizer,
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.model = PaddleOCRVLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            dtype=self.dtype,
        ).to(self.device).eval()

        image_processor = PaddleOCRVLImageProcessor.from_pretrained(self.MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.processor = PaddleOCRVLProcessor(
            image_processor=image_processor, tokenizer=tokenizer,
        )

    def recognize(self, image: Image.Image) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self.PROMPT},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=1024)

        prompt_len = inputs["input_ids"].shape[1]
        gen = outputs[:, prompt_len:]
        text = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
        return text.strip()
