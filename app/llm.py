"""Groq client + safe JSON parsing (ARCHITECTURE.md §7.3).

When ``GROQ_API_KEY`` is unset (``config.OFFLINE_LLM``), a deterministic offline
stub is used instead so the app and tests run end-to-end without network access.
The offline stub is intentionally simple — production quality comes from Groq.
"""
from __future__ import annotations

import json
import re
from typing import Any

from app import config
from app.offline_llm import offline_chat_json, offline_chat_text

_client = None


def _get_client():
    global _client
    if _client is None:
        from groq import Groq

        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def _extract_json(text: str) -> dict:
    """Parse JSON defensively — strip code fences, else grab the first {...} block."""
    cleaned = _strip_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Could not parse JSON from LLM output: {text[:200]!r}")


def _raw_chat(system: str, user: str, model: str, json_mode: bool) -> str:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _get_client().chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def chat_json(system: str, user: str, model: str = config.MODEL) -> dict:
    """Call Groq, request JSON, strip fences, ``json.loads`` defensively.

    On parse failure, retry once with a strict-JSON nudge, else raise.
    """
    if config.OFFLINE_LLM:
        return offline_chat_json(system, user)

    try:
        return _extract_json(_raw_chat(system, user, model, json_mode=True))
    except Exception:
        nudge = user + "\n\nReturn ONLY valid JSON. No prose, no code fences."
        return _extract_json(_raw_chat(system, nudge, model, json_mode=True))


def chat_text(system: str, user: str, model: str = config.MODEL) -> str:
    if config.OFFLINE_LLM:
        return offline_chat_text(system, user)
    return _raw_chat(system, user, model, json_mode=False)
