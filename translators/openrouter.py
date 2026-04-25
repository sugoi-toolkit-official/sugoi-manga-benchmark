"""OpenRouter LLM translator — routes through https://openrouter.ai.

One class serves all 5 models; only `model_id` changes between instances.
"""

import os
import re

from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from utils import normalize_text
from translators.benchmark import Translator


_MAX_ATTEMPTS = 3


def _log_retry(retry_state) -> None:
    """Surface transient failures so they're not silent (plays nicely with tqdm)."""
    exc = retry_state.outcome.exception()
    self_obj = retry_state.args[0] if retry_state.args else None
    model_id = getattr(self_obj, "model_id", "?")
    tqdm.write(
        f"[WARN] {model_id}: retry {retry_state.attempt_number}/{_MAX_ATTEMPTS} "
        f"after {type(exc).__name__}: {exc}"
    )


SYSTEM_PROMPT = (
    "You are a professional Japanese manga translator. Translate the given "
    "Japanese line to natural English. Keep style and terminology consistent "
    "with prior lines on the same page. Output ONLY the English translation "
    "— no quotes, no romaji, no explanations."
)


def build_user_prompt(
    text_ja: str,
    history: tuple[tuple[str, str], ...],
) -> str:
    """Build the user-message content per spec §6 (inline history, cách B)."""
    if history:
        lines = ["Prior lines on this page:"]
        for ja, en in history:
            lines.append(f"JA: {ja}")
            lines.append(f"EN: {en}")
        lines.append("")
        lines.append("Translate this line:")
        lines.append(f"JA: {text_ja}")
        lines.append("EN:")
        return "\n".join(lines)
    return f"Translate this line:\nJA: {text_ja}\nEN:"


_EN_PREFIX_RE = re.compile(r"^\s*EN\s*:\s*", re.IGNORECASE)
_QUOTE_PAIRS = [('"', '"'), ("'", "'"), ("「", "」"), ("『", "』")]


def post_process(text: str) -> str:
    """Trim, drop leading `EN:`, strip matched quote pairs, normalize."""
    out = text.strip()
    out = _EN_PREFIX_RE.sub("", out).strip()
    for left, right in _QUOTE_PAIRS:
        if len(out) >= 2 and out.startswith(left) and out.endswith(right):
            out = out[len(left):-len(right)].strip()
            break
    return normalize_text(out)


def _is_retryable(exc: BaseException) -> bool:
    """Transient errors worth a retry: rate-limit, connection, 5xx."""
    if isinstance(exc, (RateLimitError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code >= 500:
        return True
    return False


class OpenRouterTranslator(Translator):
    """Async translator that calls OpenRouter's OpenAI-compatible endpoint.

    The runner is expected to have verified OPENROUTER_API_KEY is set before
    instantiating this class (see main.py run_translate).
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    async def translate(
        self,
        text_ja: str,
        history: tuple[tuple[str, str], ...],
    ) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(text_ja, history)},
        ]
        try:
            raw = await self._call_api(messages)
        except Exception as e:  # non-retryable or retries exhausted
            tqdm.write(f"[WARN] {self.model_id}: {type(e).__name__}: {e}")
            return ""
        return post_process(raw)

    @retry(
        stop=stop_after_attempt(_MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception(_is_retryable),
        before_sleep=_log_retry,
        reraise=True,
    )
    async def _call_api(self, messages: list[dict]) -> str:
        resp = await self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=0.0,
            max_tokens=256,
        )
        return resp.choices[0].message.content or ""
