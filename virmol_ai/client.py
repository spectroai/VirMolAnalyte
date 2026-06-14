# -*- coding: utf-8 -*-
"""OpenAI-compatible chat completions client."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .config import LLMConfig, load_llm_config, normalize_model_name


class LLMClientError(RuntimeError):
    pass


def is_llm_configured(cfg: Optional[LLMConfig] = None) -> bool:
    cfg = cfg or load_llm_config()
    return bool(cfg.api_key.strip())


def chat_completion(
    messages: List[Dict[str, str]],
    *,
    cfg: Optional[LLMConfig] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
) -> str:
    cfg = cfg or load_llm_config()
    model = normalize_model_name(cfg.model, cfg.base_url)
    if not cfg.api_key.strip():
        raise LLMClientError(
            "API Key not configured. Set it under Edit → Preferences or "
            "environment variable VIRMOL_LLM_API_KEY."
        )
    url = f"{cfg.base_url.rstrip('/')}/chat/completions"
    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": cfg.temperature if temperature is None else temperature,
        "max_tokens": cfg.max_tokens if max_tokens is None else max_tokens,
    }
    effort = (reasoning_effort if reasoning_effort is not None else cfg.reasoning_effort) or ""
    effort = effort.strip().lower()
    if effort in ("low", "medium", "high", "minimal"):
        body["reasoning_effort"] = effort

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.api_key.strip()}",
    }

    def _post(req_body: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(req_body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=cfg.timeout_sec) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        try:
            payload = _post(body)
        except urllib.error.HTTPError as e:
            # Some providers reject the optional reasoning_effort field — retry without it.
            if e.code == 400 and "reasoning_effort" in body:
                body.pop("reasoning_effort", None)
                payload = _post(body)
            else:
                raise
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise LLMClientError(f"HTTP {e.code}: {detail[:500]}") from e
    except urllib.error.URLError as e:
        raise LLMClientError(f"Network error: {e.reason}") from e

    try:
        choice = payload["choices"][0]
        message = choice.get("message") or {}
    except (KeyError, IndexError, TypeError) as e:
        raise LLMClientError(f"Could not parse API response: {payload!r}") from e

    text = _extract_message_text(message)
    finish = choice.get("finish_reason") or message.get("finish_reason") or "unknown"
    if text:
        if finish == "length":
            return text + (
                "\n\n[Note: reply was truncated by the model output token limit. "
                "Increase max output tokens or use a non-reasoning model such as deepseek-v4-flash.]"
            )
        return text

    refusal = message.get("refusal")
    detail = f"finish_reason={finish}"
    if refusal:
        detail += f"; refusal={refusal!r}"
    if finish == "length":
        detail += (
            ". The model hit max_tokens before producing visible answer text "
            "(common with reasoning models). Prefer **deepseek-v4-flash** for Direct LLM Top5, "
            "or increase output length in API settings."
        )
    raise LLMClientError(
        "The API returned an empty reply. "
        f"({detail}). "
        "Try a smaller Top N, check model/output limits, or switch model."
    )


def _extract_message_text(message: Dict[str, Any]) -> str:
    """Normalize OpenAI-compatible message content to a single string."""
    chunks: List[str] = []

    def _append_piece(raw: Any) -> None:
        if raw is None:
            return
        if isinstance(raw, str):
            s = raw.strip()
            if s:
                chunks.append(s)
            return
        if isinstance(raw, list):
            for block in raw:
                if isinstance(block, dict):
                    if block.get("type") == "text" and block.get("text"):
                        chunks.append(str(block["text"]).strip())
                    elif block.get("text"):
                        chunks.append(str(block["text"]).strip())
                elif isinstance(block, str) and block.strip():
                    chunks.append(block.strip())
            return
        s = str(raw).strip()
        if s:
            chunks.append(s)

    # Reasoning models (e.g. DeepSeek Pro) may put text only in reasoning_content.
    for key in ("content", "reasoning_content", "text"):
        _append_piece(message.get(key))

    return "\n\n".join(chunks).strip()
