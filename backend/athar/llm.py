"""Minimal OpenRouter client (OpenAI-compatible chat completions) with disk cache.

Only this module talks to the network. Every response is cached by a hash of
(model, messages, temperature) so re-runs are free and reproducible.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
from typing import Any

import httpx

from .config import CACHE_DIR, llm_model, openrouter_key

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMUnavailable(RuntimeError):
    pass


def available() -> bool:
    return bool(openrouter_key())


def _cache_path(key: str):
    d = CACHE_DIR / "llm"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.json"


def _extract_json(text: str) -> Any:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
        if start < 0:
            raise
        end = max(text.rfind("}"), text.rfind("]"))
        return json.loads(text[start : end + 1])


def chat_json(
    system: str,
    user: str,
    *,
    temperature: float = 0.0,
    model: str | None = None,
    max_tokens: int = 4000,
    retries: int = 7,
    use_cache: bool = True,
) -> Any:
    """Send a chat request that must return JSON. Raises LLMUnavailable without a key."""
    key = openrouter_key()
    if not key:
        raise LLMUnavailable("OPENROUTER_API_KEY is not set")
    model = model or llm_model()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    h = hashlib.sha256(json.dumps([model, messages, temperature], sort_keys=True).encode()).hexdigest()[:24]
    cp = _cache_path(h)
    if use_cache and cp.exists():
        return json.loads(cp.read_text(encoding="utf-8"))["parsed"]

    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://github.com/athar-prototype",
        "X-Title": "Athar prototype",
    }
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = httpx.post(OPENROUTER_URL, json=body, headers=headers, timeout=120)
            if r.status_code in (429, 502, 503):
                # rate limited / overloaded: wait as long as the server asks, else back off exponentially
                wait = float(r.headers.get("retry-after") or 0) or min(60, 4 * 2 ** attempt)
                last_err = RuntimeError(f"HTTP {r.status_code}")
                time.sleep(wait + random.uniform(0, 2))
                continue
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            parsed = _extract_json(content)
            cp.write_text(json.dumps({"model": model, "raw": content, "parsed": parsed}), encoding="utf-8")
            return parsed
        except Exception as e:  # network error or malformed JSON
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")
