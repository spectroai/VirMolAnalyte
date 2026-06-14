# -*- coding: utf-8 -*-
"""LLM provider presets — pick a vendor + model, URLs filled automatically."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

LLM_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "deepseek_flash": {
        "label": "DeepSeek · v4 Flash (recommended)",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-v4-flash",
        "models": ["deepseek-v4-flash", "deepseek-v4-pro"],
    },
    "deepseek_pro": {
        "label": "DeepSeek · v4 Pro",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-v4-pro",
        "models": ["deepseek-v4-pro", "deepseek-v4-flash"],
    },
    "openai": {
        "label": "OpenAI · GPT-4o mini",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
    },
    "dashscope": {
        "label": "Alibaba DashScope · qwen-plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "models": ["qwen-plus", "qwen-turbo", "qwen-max"],
    },
    "custom": {
        "label": "Custom (edit URL / model manually)",
        "base_url": "",
        "model": "",
        "models": [],
    },
}

DEFAULT_PROVIDER_ID = "deepseek_flash"


def provider_choices() -> List[Tuple[str, str]]:
    return [(pid, spec["label"]) for pid, spec in LLM_PROVIDERS.items()]


def get_provider(provider_id: str) -> Dict[str, Any]:
    return LLM_PROVIDERS.get(provider_id) or LLM_PROVIDERS[DEFAULT_PROVIDER_ID]


def guess_provider_id(base_url: str, model: str) -> str:
    base = (base_url or "").lower()
    mod = (model or "").lower()
    if "deepseek.com" in base:
        if "pro" in mod:
            return "deepseek_pro"
        return "deepseek_flash"
    if "openai.com" in base:
        return "openai"
    if "dashscope" in base or "aliyuncs.com" in base:
        return "dashscope"
    if base or model:
        return "custom"
    return DEFAULT_PROVIDER_ID


def apply_provider(provider_id: str, model_override: Optional[str] = None) -> Tuple[str, str]:
    spec = get_provider(provider_id)
    base = str(spec.get("base_url") or "").rstrip("/")
    if provider_id == "custom":
        return base, model_override or ""
    model = model_override or str(spec.get("model") or "")
    models = spec.get("models") or []
    if model and model not in models and models:
        model = models[0]
    return base, model
