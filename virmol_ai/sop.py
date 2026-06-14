# -*- coding: utf-8 -*-
"""Predefined Standard Operating Procedures (SOPs) for VirMol Copilot.

A SOP is a static list of :class:`PlanStep` records derived from an
:class:`IntakePayload`. The runner consumes the list in order and invokes
the corresponding tools. SOPs are intentionally rule-based (no LLM
involvement) so that the P0 path works fully offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------- User-facing payload ------------------------------------------------

@dataclass
class IntakePayload:
    """Everything the runner needs to build a plan.

    All fields except ``peak_text`` have sane defaults that match the GUI
    defaults, so an end user usually only supplies a peak list.
    """

    peak_text: str = ""                   # canonical 'ppm, q/t/d/s' text
    database: str = "plant"               # 'plant' / 'human' / 'microbial' / 'drug' / 'all' / 'custom'
    custom_database_path: Optional[str] = None

    use_cnf: bool = True
    use_ctnf: bool = True
    use_mw: bool = False
    cnf_bias: int = 5
    ctnf_bias: int = 2
    mw_list: Optional[List[float]] = None

    evaluator: str = "FPAACS"             # CSS / AAS / FPS / FPAACS
    fpaacs_weights: Optional[List[float]] = None

    masking_top_n: int = 5
    do_masking: bool = False
    do_fragment_extraction: bool = False
    do_fusion: bool = False
    fusion_mode: str = "cross"            # 'cross' or 'intra'

    do_export: bool = True
    sample_note: str = ""                 # free-text annotation for the report

    def normalised_database(self) -> str:
        if self.database == "custom":
            return "custom"
        key = (self.database or "").strip().lower()
        if key in ("plant", "human", "microbial", "microorganism", "drug", "all", "alldb"):
            return key
        return "plant"


# ---------- Plan structures ----------------------------------------------------

@dataclass
class PlanStep:
    tool_name: str
    args: Dict[str, Any]
    label: str
    note: str = ""
    skippable: bool = False
    danger_level: str = "low"
    estimated_seconds: float = 2.0


@dataclass
class Plan:
    name: str
    description: str
    steps: List[PlanStep] = field(default_factory=list)

    @property
    def total_eta_seconds(self) -> float:
        return sum(step.estimated_seconds for step in self.steps)


# ---------- Internal helpers ---------------------------------------------------

def _filter_args(p: IntakePayload) -> Dict[str, Any]:
    args = {
        "CNF": p.use_cnf,
        "CTNF": p.use_ctnf,
        "MW": p.use_mw,
        "CNF_bias": p.cnf_bias,
        "CTNF_bias": p.ctnf_bias,
    }
    if p.use_mw and p.mw_list:
        args["MW_list"] = list(p.mw_list)
    return args


def _evaluator_args(p: IntakePayload) -> Dict[str, Any]:
    args: Dict[str, Any] = {"name": p.evaluator}
    if p.evaluator.upper() == "FPAACS" and p.fpaacs_weights:
        args["fpaacs_weights"] = list(p.fpaacs_weights)
    return args


def _db_step(p: IntakePayload) -> PlanStep:
    db = p.normalised_database()
    if db == "custom":
        return PlanStep(
            tool_name="load_custom_database",
            args={"path": p.custom_database_path or ""},
            label=f"Load custom database: {p.custom_database_path or '(missing)'}",
            danger_level="medium",
            estimated_seconds=5,
        )
    return PlanStep(
        tool_name="select_database",
        args={"name": db},
        label=f"Load built-in database: {db.capitalize()}",
        danger_level="medium",
        estimated_seconds=5,
    )


def _peaks_step(p: IntakePayload) -> Optional[PlanStep]:
    text = (p.peak_text or "").strip()
    if not text:
        return None
    n_lines = sum(1 for line in text.splitlines() if line.strip() and not line.startswith("#"))
    return PlanStep(
        tool_name="set_peaks",
        args={"text": text},
        label=f"Write {n_lines} experimental peaks into manual input",
        estimated_seconds=0.5,
    )


def _filters_step(p: IntakePayload) -> PlanStep:
    summary_bits = []
    if p.use_cnf: summary_bits.append(f"CNF±{p.cnf_bias}")
    if p.use_ctnf: summary_bits.append(f"CTNF±{p.ctnf_bias}")
    if p.use_mw: summary_bits.append(f"MW {p.mw_list or ''}")
    summary = ", ".join(summary_bits) if summary_bits else "no filters"
    return PlanStep(
        tool_name="set_filters",
        args=_filter_args(p),
        label=f"Configure filters ({summary})",
        estimated_seconds=0.5,
    )


def _evaluator_step(p: IntakePayload) -> PlanStep:
    weights = ""
    if p.evaluator.upper() == "FPAACS" and p.fpaacs_weights:
        weights = f" (weights={p.fpaacs_weights})"
    return PlanStep(
        tool_name="set_evaluator",
        args=_evaluator_args(p),
        label=f"Choose evaluator: {p.evaluator}{weights}",
        estimated_seconds=0.5,
    )


def _screening_step() -> PlanStep:
    return PlanStep(
        tool_name="run_screening",
        args={},
        label="Run database screening (Start Analysis)",
        danger_level="medium",
        estimated_seconds=60,
    )


def _diagnosis_step() -> PlanStep:
    return PlanStep(
        tool_name="screening_diagnosis",
        args={},
        label="Rule-based diagnosis of the hit list",
        estimated_seconds=1,
    )


def _masking_steps(p: IntakePayload) -> List[PlanStep]:
    steps: List[PlanStep] = [
        PlanStep(
            tool_name="set_masking_topn",
            args={"top_n": p.masking_top_n},
            label=f"Set masking Top-N = {p.masking_top_n}",
            estimated_seconds=0.5,
        ),
        PlanStep(
            tool_name="run_masking_attribution",
            args={},
            label=f"Run Monte Carlo masking on top {p.masking_top_n} candidates",
            danger_level="medium",
            estimated_seconds=120,
        ),
    ]
    if p.do_fragment_extraction or p.do_fusion:
        steps.append(PlanStep(
            tool_name="extract_fragments",
            args={},
            label="Extract positive fragments from masking",
            estimated_seconds=10,
        ))
    if p.do_fusion:
        mode = (p.fusion_mode or "cross").lower()
        steps.append(PlanStep(
            tool_name="run_fusion",
            args={"mode": mode},
            label=f"Fragment fusion ({mode})",
            estimated_seconds=20,
        ))
    return steps


def _export_step() -> PlanStep:
    return PlanStep(
        tool_name="export_results",
        args={},
        label="Export analysis folder (Result.csv, Top-20 grid, parameters.txt)",
        danger_level="medium",
        estimated_seconds=5,
    )


# ---------- Public SOP builders ------------------------------------------------

def build_quick_screening_plan(payload: IntakePayload) -> Plan:
    """SOP 1 — Quick database screening only (no masking).

    Assumes the user already has a peak list (or NMR-1D.csv from preprocessing)
    and just wants a ranked candidate hit list.
    """
    steps: List[PlanStep] = []
    peak_step = _peaks_step(payload)
    if peak_step is not None:
        steps.append(peak_step)
    steps.append(_db_step(payload))
    steps.append(_filters_step(payload))
    steps.append(_evaluator_step(payload))
    steps.append(_screening_step())
    steps.append(_diagnosis_step())
    if payload.do_export:
        steps.append(_export_step())
    return Plan(
        name="Quick screening",
        description="Peaks → Database → Filters/Evaluator → Screening → Diagnosis → (Export).",
        steps=steps,
    )


def build_full_pipeline_plan(payload: IntakePayload) -> Plan:
    """SOP 2 — Screening + Random masking attribution (+ optional fusion)."""
    forced = IntakePayload(**{**payload.__dict__, "do_masking": True})
    forced.do_fragment_extraction = forced.do_fragment_extraction or forced.do_fusion or True

    steps: List[PlanStep] = []
    peak_step = _peaks_step(forced)
    if peak_step is not None:
        steps.append(peak_step)
    steps.append(_db_step(forced))
    steps.append(_filters_step(forced))
    steps.append(_evaluator_step(forced))
    steps.append(_screening_step())
    steps.append(_diagnosis_step())
    steps.extend(_masking_steps(forced))
    if forced.do_export:
        steps.append(_export_step())
    return Plan(
        name="Full pipeline (screening + masking)",
        description=(
            "Standard end-to-end workflow: screening, masking attribution on "
            "Top-N, positive-fragment extraction, optional fragment fusion, export."
        ),
        steps=steps,
    )


def build_post_screening_masking_plan(payload: IntakePayload) -> Plan:
    """SOP 3 — Skip screening (user already has results in the GUI) and only
    run masking + fragment extraction + fusion. Useful when the user wants to
    re-run attribution with different masking settings.
    """
    forced = IntakePayload(**{**payload.__dict__, "do_masking": True})
    forced.do_fragment_extraction = True

    steps: List[PlanStep] = list(_masking_steps(forced))
    steps.insert(0, _diagnosis_step())
    if forced.do_export:
        steps.append(_export_step())
    return Plan(
        name="Re-run masking on existing hits",
        description="Skip screening; only run masking/extraction/fusion on the current hit list.",
        steps=steps,
    )


SOP_REGISTRY: Dict[str, Any] = {
    "quick_screening": {
        "label": "Quick screening",
        "builder": build_quick_screening_plan,
        "description": (
            "Fastest path: peaks → database → screening → ranked hits → diagnosis. "
            "Use when you just need the top candidates."
        ),
    },
    "full_pipeline": {
        "label": "Full pipeline (with masking + fusion)",
        "builder": build_full_pipeline_plan,
        "description": (
            "End-to-end: screening, masking attribution on Top-N, positive "
            "fragment extraction, optional cross-candidate fusion, export."
        ),
    },
    "post_screening_masking": {
        "label": "Re-run masking on existing hits",
        "builder": build_post_screening_masking_plan,
        "description": (
            "Skip screening; only run masking/extraction/fusion on the current "
            "hit list. Use when you want to re-attribute with new MC settings."
        ),
    },
}


def list_sops() -> List[Dict[str, Any]]:
    """Return ``[{id, label, description}, ...]`` for UI selectors."""
    return [
        {"id": sid, "label": entry["label"], "description": entry["description"]}
        for sid, entry in SOP_REGISTRY.items()
    ]


def build_plan(sop_id: str, payload: IntakePayload) -> Plan:
    entry = SOP_REGISTRY.get(sop_id)
    if entry is None:
        raise KeyError(f"Unknown SOP id: {sop_id}")
    return entry["builder"](payload)
