"""Translator registry — loads OpenRouter model slugs from models.json."""

import functools
import json
from pathlib import Path

__all__ = ["get_all_translators"]


def get_all_translators() -> dict:
    """Return dict of {slug: zero-arg factory}. Each factory produces an
    OpenRouterTranslator bound to a specific model slug. Missing openai dep
    is reported once as a warning; missing OPENROUTER_API_KEY surfaces later
    when the factory is called (checked centrally in main.run_translate)."""
    translators: dict = {}
    try:
        from translators.openrouter import OpenRouterTranslator
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[WARN] OpenRouter unavailable: {e}")
        return translators

    cfg_path = Path(__file__).parent / "models.json"
    slugs = json.loads(cfg_path.read_text(encoding="utf-8"))["models"]
    for slug in slugs:
        translators[slug] = functools.partial(OpenRouterTranslator, slug)
    return translators
