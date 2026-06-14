# -*- coding: utf-8 -*-
"""LLM API configuration (environment variables + QSettings)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .providers import (
    DEFAULT_PROVIDER_ID,
    LLM_PROVIDERS,
    apply_provider,
    guess_provider_id,
)


@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 8192
    timeout_sec: float = 120.0
    # Optional reasoning-effort hint for reasoning models ("low"/"medium"/"high").
    # Empty means "do not send the field" (safe default for providers that reject it).
    reasoning_effort: str = ""


def _qsettings():
    try:
        from PyQt5.QtCore import QSettings
    except ImportError:
        return None
    return QSettings("VirMolAnalyte", "VirMolAnalyte")


# Common typos / old names → current API id (DeepSeek platform, 2025+)
_MODEL_ALIASES = {
    "deepseek": "deepseek-v4-flash",
    "deepseek-chat": "deepseek-v4-flash",
    "deepseek-coder": "deepseek-v4-flash",
    "deepseek-reasoner": "deepseek-v4-pro",
}


def normalize_model_name(model: str, base_url: str = "") -> str:
    """Map friendly labels to provider-supported model ids."""
    raw = (model or "").strip()
    if not raw:
        return raw
    if raw in ("DeepSeek", "DEEPSEEK"):
        return "deepseek-v4-flash"
    key = raw.lower()
    if key in _MODEL_ALIASES:
        return _MODEL_ALIASES[key]
    return raw


def load_provider_id() -> str:
    qs = _qsettings()
    pid = os.environ.get("VIRMOL_LLM_PROVIDER", "")
    if qs is not None:
        pid = pid or str(qs.value("llm/provider_id", "") or "")
    if pid and pid in LLM_PROVIDERS:
        return pid
    return ""


def load_llm_config() -> LLMConfig:
    qs = _qsettings()
    key = os.environ.get("VIRMOL_LLM_API_KEY", "")
    base = os.environ.get("VIRMOL_LLM_BASE_URL", "")
    model = os.environ.get("VIRMOL_LLM_MODEL", "")
    provider_id = load_provider_id()
    if qs is not None:
        key = key or str(qs.value("llm/api_key", "") or "")
        base = base or str(qs.value("llm/base_url", "") or "")
        model = model or str(qs.value("llm/model", "") or "")
    temp = os.environ.get("VIRMOL_LLM_TEMPERATURE", "")
    if qs is not None and not temp:
        temp = str(qs.value("llm/temperature", "") or "")
    effort = os.environ.get("VIRMOL_LLM_REASONING_EFFORT", "")
    if qs is not None and not effort:
        effort = str(qs.value("llm/reasoning_effort", "") or "")

    if not base and not model and provider_id:
        base, model = apply_provider(provider_id)
    elif not base and not model and not key:
        # First-run default: DeepSeek v4 Flash
        provider_id = provider_id or DEFAULT_PROVIDER_ID
        base, model = apply_provider(provider_id)
    else:
        if not provider_id:
            provider_id = guess_provider_id(base, model)

    resolved_base = (base or "https://api.openai.com/v1").strip().rstrip("/")
    resolved_model = (model or "gpt-4o-mini").strip()
    resolved_model = normalize_model_name(resolved_model, resolved_base)
    return LLMConfig(
        api_key=key.strip(),
        base_url=resolved_base,
        model=resolved_model,
        temperature=float(temp) if temp else 0.2,
        reasoning_effort=(effort or "").strip().lower(),
    )


def save_llm_config(cfg: LLMConfig, provider_id: Optional[str] = None) -> None:
    qs = _qsettings()
    if qs is None:
        return
    qs.setValue("llm/api_key", cfg.api_key)
    qs.setValue("llm/base_url", cfg.base_url.rstrip("/"))
    qs.setValue("llm/model", cfg.model)
    qs.setValue("llm/temperature", cfg.temperature)
    if (getattr(cfg, "reasoning_effort", "") or "").strip():
        qs.setValue("llm/reasoning_effort", cfg.reasoning_effort.strip().lower())
    if provider_id:
        qs.setValue("llm/provider_id", provider_id)
    elif not str(qs.value("llm/provider_id", "") or ""):
        qs.setValue("llm/provider_id", guess_provider_id(cfg.base_url, cfg.model))
    qs.sync()
