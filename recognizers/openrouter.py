"""OpenRouter VLM recognizer — routes vision-capable models through openrouter.ai.

One class serves all models; only `model_id` changes between instances.
Image is base64-encoded and sent as an OpenAI vision message.
"""

import base64
import io
import os

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from recognizers.benchmark import TextRecognizer
from utils import normalize_text


_MAX_ATTEMPTS = 3


def _log_retry(retry_state) -> None:
    exc = retry_state.outcome.exception()
    self_obj = retry_state.args[0] if retry_state.args else None
    model_id = getattr(self_obj, "model_id", "?")
    tqdm.write(
        f"[WARN] {model_id}: retry {retry_state.attempt_number}/{_MAX_ATTEMPTS} "
        f"after {type(exc).__name__}: {exc}"
    )


USER_PROMPT = "Read the Japanese text in this image exactly as written. Output only the text."


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code >= 500:
        return True
    return False


def _encode_image(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class OpenRouterRecognizer(TextRecognizer):
    """Sync VLM recognizer that calls OpenRouter's OpenAI-compatible endpoint.

    The runner is expected to have verified OPENROUTER_API_KEY is set before
    instantiating this class (see main.py run_ocr).
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    async def recognize(self, image) -> str:
        b64 = _encode_image(image)
        try:
            raw = await self._call_api(b64)
        except Exception as e:
            tqdm.write(f"[WARN] {self.model_id}: {type(e).__name__}: {e}")
            return ""
        return normalize_text(raw.strip())

    @retry(
        stop=stop_after_attempt(_MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception(_is_retryable),
        before_sleep=_log_retry,
        reraise=True,
    )
    async def _call_api(self, b64_image: str) -> str:
        resp = await self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                        },
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
            ],
            temperature=0.0,
            max_tokens=256,
        )
        return resp.choices[0].message.content or ""
