# VirMolAnalyte P0 AI assistant (cloud LLM, read-only context).

from .config import LLMConfig, load_llm_config, save_llm_config
from .client import chat_completion, is_llm_configured
from .context import build_analysis_context
from .fragment_context import build_fragment_analysis_context, has_fragment_data
from .preflight import (
    run_preflight,
    run_screening_diagnosis,
    format_diagnosis_text,
    format_preflight_text,
)
from .prompts import TASK_PROMPTS, system_prompt
from .reports import build_results_draft

__all__ = [
    "LLMConfig",
    "load_llm_config",
    "save_llm_config",
    "chat_completion",
    "is_llm_configured",
    "build_analysis_context",
    "build_fragment_analysis_context",
    "has_fragment_data",
    "run_preflight",
    "run_screening_diagnosis",
    "format_diagnosis_text",
    "format_preflight_text",
    "TASK_PROMPTS",
    "system_prompt",
    "build_results_draft",
]
