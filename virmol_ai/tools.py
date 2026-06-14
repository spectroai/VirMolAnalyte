# -*- coding: utf-8 -*-
"""Tool wrappers for the VirMol Copilot SOP runner.

Each tool encapsulates exactly one GUI action (e.g. "load Plant database",
"start screening", "run masking attribution"). Tools are described as
``ToolSpec`` records and invoked through a callable with the signature::

    fn(gui, args: dict, on_done: Callable[[ToolResult], None]) -> None

``fn`` must call ``on_done`` **exactly once**, either synchronously (for fast
actions) or via ``PyQt5.QtCore.QTimer.singleShot`` (for actions that depend on
a worker thread). The runner consumes ``on_done`` to advance the plan.

Tools intentionally **do not** depend on the LLM. They are the building blocks
both for predefined SOPs (P0) and for future LLM tool-calling planners.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:
    from PyQt5.QtCore import QTimer
except ImportError:  # pragma: no cover - allow unit tests without Qt
    QTimer = None  # type: ignore


# ---------- Public dataclasses --------------------------------------------------

@dataclass
class ToolResult:
    """Outcome reported back to the runner."""
    ok: bool
    summary: str
    detail: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


ToolCallable = Callable[[Any, Dict[str, Any], Callable[[ToolResult], None]], None]


@dataclass
class ToolSpec:
    """Static metadata used to render plan steps and check preconditions."""
    name: str
    label: str
    description: str
    fn: ToolCallable
    estimated_seconds: float = 2.0
    is_async: bool = False
    danger_level: str = "low"  # low / medium / high (drives confirmation UI)


# ---------- Internal helpers ---------------------------------------------------

_POLL_INTERVAL_MS = 200


def _later(callable_no_args, delay_ms: int = 0) -> None:
    """Schedule ``callable_no_args`` on the Qt event loop, falling back to
    immediate execution if PyQt is not importable (e.g. unit-test runs).
    """
    if QTimer is None:
        callable_no_args()
        return
    QTimer.singleShot(delay_ms, callable_no_args)


def _poll_until(
    predicate: Callable[[], bool],
    *,
    on_ready: Callable[[], None],
    on_timeout: Callable[[], None],
    timeout_ms: int,
) -> None:
    """Generic poll loop. Calls ``on_ready`` once ``predicate()`` is true, or
    ``on_timeout`` after ``timeout_ms`` elapsed.
    """
    if QTimer is None:
        if predicate():
            on_ready()
        else:
            on_timeout()
        return
    elapsed = [0]

    def _tick():
        try:
            if predicate():
                on_ready()
                return
        except Exception as exc:  # noqa: BLE001 - never let the loop die
            print(f"poll predicate raised: {exc}")
            on_timeout()
            return
        elapsed[0] += _POLL_INTERVAL_MS
        if elapsed[0] >= timeout_ms:
            on_timeout()
            return
        QTimer.singleShot(_POLL_INTERVAL_MS, _tick)

    QTimer.singleShot(_POLL_INTERVAL_MS, _tick)


def _ensure_main_thread(gui, fn):
    """Schedule ``fn`` on the Qt main thread via QTimer (single-shot 0ms)."""
    _later(fn, 0)


# ---------- Tool implementations -----------------------------------------------

# 1) set_peaks ------------------------------------------------------------------

def _tool_set_peaks(gui, args, on_done):
    text = (args.get("text") or "").strip()
    if not text:
        on_done(ToolResult(False, "No peak text provided.", {}, "empty"))
        return

    def _do():
        if not hasattr(gui, "manual_peak_input"):
            on_done(ToolResult(False, "GUI is missing manual_peak_input.", {}, "no_widget"))
            return
        gui.manual_peak_input.setPlainText(text)
        try:
            parsed = gui.parse_manual_peak_input_strict(text)
        except ValueError as exc:
            on_done(ToolResult(False, f"Peak parse failed: {exc}", {"raw_text": text}, str(exc)))
            return
        n = len(parsed[0]) if parsed else 0
        on_done(ToolResult(True, f"Loaded {n} experimental peaks.", {"n_peaks": n}))

    _ensure_main_thread(gui, _do)


# 2) select_database ------------------------------------------------------------

_DB_LABEL_MAP = {
    "plant":       "Plant Database",
    "human":       "Human Database",
    "microbial":   "Microbial Database",
    "microorganism": "Microbial Database",
    "drug":        "Drug Database",
    "all":         "All Database",
    "alldb":       "All Database",
}


def _tool_select_database(gui, args, on_done):
    raw = str(args.get("name", "")).strip().lower()
    target = _DB_LABEL_MAP.get(raw)
    if target is None:
        on_done(ToolResult(
            False,
            f"Unknown database name '{args.get('name')}'. "
            f"Use one of: plant, human, microbial, drug, all.",
            {}, "bad_name",
        ))
        return

    def _do():
        if not hasattr(gui, "db_combo"):
            on_done(ToolResult(False, "GUI is missing db_combo.", {}, "no_widget"))
            return
        for i in range(gui.db_combo.count()):
            if target in gui.db_combo.itemText(i):
                gui.db_combo.setCurrentIndex(i)
                break
        try:
            gui.load_database()
        except Exception as exc:  # noqa: BLE001
            on_done(ToolResult(False, f"Database load failed: {exc}", {}, str(exc)))
            return
        ok = gui.database1 is not None
        on_done(ToolResult(
            ok,
            f"Loaded {target}." if ok else f"Database load returned no data ({target}).",
            {"database": target},
            None if ok else "no_data",
        ))

    _ensure_main_thread(gui, _do)


# 3) load_custom_database -------------------------------------------------------

def _tool_load_custom_database(gui, args, on_done):
    path = (args.get("path") or "").strip()
    if not path or not os.path.isfile(path):
        on_done(ToolResult(False, f"Database file not found: {path}", {}, "missing"))
        return

    def _do():
        gui.other_db_input.setText(path)
        try:
            gui.load_other_database()
        except Exception as exc:  # noqa: BLE001
            on_done(ToolResult(False, f"Custom DB load failed: {exc}", {}, str(exc)))
            return
        ok = gui.database1 is not None
        on_done(ToolResult(
            ok,
            f"Loaded custom database: {os.path.basename(path)}",
            {"path": path},
            None if ok else "no_data",
        ))

    _ensure_main_thread(gui, _do)


# 4) set_filters ---------------------------------------------------------------

def _tool_set_filters(gui, args, on_done):
    def _do():
        if "CNF" in args:
            gui.cnf_checkbox.setChecked(bool(args["CNF"]))
        if "CTNF" in args:
            gui.ctnf_checkbox.setChecked(bool(args["CTNF"]))
        if "MW" in args:
            gui.mw_checkbox.setChecked(bool(args["MW"]))
        if "CNF_bias" in args:
            gui.cnf_bias.setValue(int(args["CNF_bias"]))
        if "CTNF_bias" in args:
            gui.ctnf_bias.setValue(int(args["CTNF_bias"]))
        if "MW_list" in args:
            mw_text = ",".join(str(x) for x in args["MW_list"])
            gui.mw_list.setText(mw_text)
        summary = (
            f"Filters: CNF={gui.cnf_checkbox.isChecked()}(±{gui.cnf_bias.value()})  "
            f"CTNF={gui.ctnf_checkbox.isChecked()}(±{gui.ctnf_bias.value()})  "
            f"MW={gui.mw_checkbox.isChecked()}"
        )
        on_done(ToolResult(True, summary, {
            "CNF": gui.cnf_checkbox.isChecked(),
            "CTNF": gui.ctnf_checkbox.isChecked(),
            "MW": gui.mw_checkbox.isChecked(),
        }))

    _ensure_main_thread(gui, _do)


# 5) set_evaluator -------------------------------------------------------------

_EVAL_RADIOS = {
    "css":    "evaluator_css",
    "aas":    "evaluator_aas",
    "fps":    "evaluator_fps",
    "fpaacs": "evaluator_fpaacs",
}


def _tool_set_evaluator(gui, args, on_done):
    name = str(args.get("name", "")).strip().lower()
    attr = _EVAL_RADIOS.get(name)
    if attr is None:
        on_done(ToolResult(
            False,
            f"Unknown evaluator '{args.get('name')}' (use CSS/AAS/FPS/FPAACS).",
            {}, "bad_name",
        ))
        return

    def _do():
        getattr(gui, attr).setChecked(True)
        if "fpaacs_weights" in args and hasattr(gui, "fpaacs_weights"):
            weights_text = ",".join(str(x) for x in args["fpaacs_weights"])
            gui.fpaacs_weights.setText(weights_text)
        on_done(ToolResult(True, f"Evaluator set to {name.upper()}.", {"evaluator": name.upper()}))

    _ensure_main_thread(gui, _do)


# 6) set_masking_topn ---------------------------------------------------------

def _tool_set_masking_topn(gui, args, on_done):
    n = int(args.get("top_n", 5))

    def _do():
        if not hasattr(gui, "frag_mask_topn_spin"):
            on_done(ToolResult(False, "GUI is missing frag_mask_topn_spin.", {}, "no_widget"))
            return
        spin = gui.frag_mask_topn_spin
        capped = min(max(1, n), spin.maximum())
        spin.setValue(capped)
        on_done(ToolResult(True, f"Masking Top-N set to {capped}.", {"top_n": capped}))

    _ensure_main_thread(gui, _do)


# 7) run_screening (async) ----------------------------------------------------

def _tool_run_screening(gui, args, on_done):
    timeout_ms = int(args.get("timeout_seconds", 600)) * 1000

    def _start():
        if gui.database1 is None:
            on_done(ToolResult(False, "No database loaded — call select_database first.", {}, "no_db"))
            return
        if gui._is_thread_running("analysis_thread"):
            on_done(ToolResult(False, "Screening is already running.", {}, "busy"))
            return
        try:
            gui.start_analysis()
        except Exception as exc:  # noqa: BLE001
            on_done(ToolResult(False, f"start_analysis failed: {exc}", {}, str(exc)))
            return

        def _ready():
            n = 0
            try:
                n = len(gui.result) if gui.result is not None else 0
            except Exception:
                n = 0
            on_done(ToolResult(
                ok=(n > 0),
                summary=(f"Screening finished — {n} hits." if n else "Screening finished but produced 0 hits."),
                detail={"n_hits": n},
                error=None if n > 0 else "zero_hits",
            ))

        def _timeout():
            on_done(ToolResult(False, "Screening timed out.", {}, "timeout"))

        _poll_until(
            lambda: not gui._is_thread_running("analysis_thread"),
            on_ready=_ready,
            on_timeout=_timeout,
            timeout_ms=timeout_ms,
        )

    _ensure_main_thread(gui, _start)


# 8) run_masking_attribution (async via batch callback) -----------------------

def _tool_run_masking(gui, args, on_done):
    timeout_ms = int(args.get("timeout_seconds", 1800)) * 1000

    def _start():
        if gui.result is None or len(gui.result) == 0:
            on_done(ToolResult(False, "No screening results — run screening first.", {}, "no_results"))
            return
        if gui._is_thread_running("masking_thread"):
            on_done(ToolResult(False, "Masking is already running.", {}, "busy"))
            return

        done_flag = {"fired": False}

        def _on_batch(ok: bool, error: Optional[str]):
            if done_flag["fired"]:
                return
            done_flag["fired"] = True
            n = len(getattr(gui, "masking_all_results", []) or [])
            if not ok:
                on_done(ToolResult(False, f"Masking failed: {error}", {"n_done": n}, error or "fail"))
                return
            on_done(ToolResult(True, f"Masking attribution completed on {n} candidates.", {"n_done": n}))

        gui.register_masking_batch_callback(_on_batch)

        try:
            gui.run_masking_attribution()
        except Exception as exc:  # noqa: BLE001
            if not done_flag["fired"]:
                done_flag["fired"] = True
                on_done(ToolResult(False, f"run_masking_attribution failed: {exc}", {}, str(exc)))
            return

        def _timeout():
            if not done_flag["fired"]:
                done_flag["fired"] = True
                on_done(ToolResult(False, "Masking timed out.", {}, "timeout"))

        # Defensive timeout in case the callback is never fired.
        if QTimer is not None:
            QTimer.singleShot(timeout_ms, _timeout)

    _ensure_main_thread(gui, _start)


# 9) extract_fragments --------------------------------------------------------

def _tool_extract_fragments(gui, args, on_done):
    def _do():
        if not getattr(gui, "masking_all_results", None):
            on_done(ToolResult(False, "No masking results — run masking first.", {}, "no_masking"))
            return
        try:
            gui.extract_masking_fragments()
        except Exception as exc:  # noqa: BLE001
            on_done(ToolResult(False, f"extract_masking_fragments failed: {exc}", {}, str(exc)))
            return
        all_frags = getattr(gui, "masking_fragments_all", []) or []
        n_total = sum(len(f) for f in all_frags)
        on_done(ToolResult(
            True,
            f"Extracted positive fragments for {len(all_frags)} candidates (total {n_total}).",
            {"n_candidates": len(all_frags), "n_fragments_total": n_total},
        ))

    _ensure_main_thread(gui, _do)


# 10) run_fusion (cross or intra) --------------------------------------------

def _tool_run_fusion(gui, args, on_done):
    mode = str(args.get("mode", "cross")).strip().lower()
    if mode not in ("cross", "intra"):
        on_done(ToolResult(False, f"Unknown fusion mode '{mode}' (use cross/intra).", {}, "bad_mode"))
        return

    def _do():
        if mode == "intra":
            if not getattr(gui, "masking_fragments", None):
                on_done(ToolResult(False, "No fragments for current candidate.", {}, "no_frag"))
                return
            try:
                gui.run_intra_fragment_fusion()
            except Exception as exc:  # noqa: BLE001
                on_done(ToolResult(False, f"intra fusion failed: {exc}", {}, str(exc)))
                return
        else:
            if not getattr(gui, "masking_fragments_all", None):
                on_done(ToolResult(False, "No fragments across candidates.", {}, "no_frag"))
                return
            try:
                gui.run_fragment_fusion_analysis()
            except Exception as exc:  # noqa: BLE001
                on_done(ToolResult(False, f"cross fusion failed: {exc}", {}, str(exc)))
                return
        fusion = getattr(gui, "fusion_results", []) or []
        on_done(ToolResult(
            True,
            f"Fusion ({mode}) finished — {len(fusion)} top combinations.",
            {"mode": mode, "n_combinations": len(fusion)},
        ))

    _ensure_main_thread(gui, _do)


# 11) export_results ----------------------------------------------------------

def _tool_export_results(gui, args, on_done):
    def _do():
        if gui.result is None or len(gui.result) == 0:
            on_done(ToolResult(False, "No results to export.", {}, "no_results"))
            return
        try:
            gui.export_molecular_analysis_folder()
        except Exception as exc:  # noqa: BLE001
            on_done(ToolResult(False, f"export failed: {exc}", {}, str(exc)))
            return
        on_done(ToolResult(True, "Exported analysis folder.", {}))

    _ensure_main_thread(gui, _do)


# 12) screening_diagnosis -----------------------------------------------------

def _tool_screening_diagnosis(gui, args, on_done):
    def _do():
        try:
            from .context import build_analysis_context
            from .preflight import run_screening_diagnosis, format_diagnosis_text
            ctx = build_analysis_context(gui, top_n=15)
            items = run_screening_diagnosis(ctx)
            text = format_diagnosis_text(items)
        except Exception as exc:  # noqa: BLE001
            on_done(ToolResult(False, f"diagnosis failed: {exc}", {}, str(exc)))
            return
        n_warn = sum(1 for it in items if it.get("level") in ("warn", "error"))
        on_done(ToolResult(
            True,
            f"Screening diagnosis: {len(items)} items ({n_warn} warning/error).",
            {"diagnosis_text": text, "items": items},
        ))

    _ensure_main_thread(gui, _do)


# ---------- Registry ---------------------------------------------------------

_SPECS: List[ToolSpec] = [
    ToolSpec(
        name="set_peaks",
        label="Write experimental peaks",
        description="Write a 'ppm, q/t/d/s' table into the Manual peak input box.",
        fn=_tool_set_peaks,
        estimated_seconds=0.5,
    ),
    ToolSpec(
        name="select_database",
        label="Load built-in database",
        description="Select one of Plant/Human/Microbial/Drug/All and load it.",
        fn=_tool_select_database,
        estimated_seconds=5,
        is_async=False,
        danger_level="medium",
    ),
    ToolSpec(
        name="load_custom_database",
        label="Load custom database (.npz)",
        description="Load a user-supplied database file (.npz).",
        fn=_tool_load_custom_database,
        estimated_seconds=5,
        danger_level="medium",
    ),
    ToolSpec(
        name="set_filters",
        label="Configure CNF / CTNF / MW filters",
        description="Toggle filter checkboxes and biases.",
        fn=_tool_set_filters,
        estimated_seconds=0.5,
    ),
    ToolSpec(
        name="set_evaluator",
        label="Choose evaluator (FPAACS/CSS/AAS/FPS)",
        description="Switch evaluator radio + optional FPAACS weights.",
        fn=_tool_set_evaluator,
        estimated_seconds=0.5,
    ),
    ToolSpec(
        name="set_masking_topn",
        label="Set masking Top-N",
        description="How many top candidates to run Random masking attribution on.",
        fn=_tool_set_masking_topn,
        estimated_seconds=0.5,
    ),
    ToolSpec(
        name="run_screening",
        label="Run database screening",
        description="Execute Start Analysis and wait for the ranked hit list.",
        fn=_tool_run_screening,
        estimated_seconds=60,
        is_async=True,
        danger_level="medium",
    ),
    ToolSpec(
        name="run_masking_attribution",
        label="Run Monte Carlo masking attribution",
        description="Run masking attribution for the configured Top-N candidates.",
        fn=_tool_run_masking,
        estimated_seconds=120,
        is_async=True,
        danger_level="medium",
    ),
    ToolSpec(
        name="extract_fragments",
        label="Extract positive fragments",
        description="Extract per-candidate positive fragments after masking.",
        fn=_tool_extract_fragments,
        estimated_seconds=10,
    ),
    ToolSpec(
        name="run_fusion",
        label="Fragment fusion (intra/cross)",
        description="Run fragment fusion analysis (intra: current candidate, cross: all candidates).",
        fn=_tool_run_fusion,
        estimated_seconds=20,
    ),
    ToolSpec(
        name="export_results",
        label="Export analysis folder",
        description="Write Result.csv, NMR-1D.csv, Top-20 grid, and parameters.txt.",
        fn=_tool_export_results,
        estimated_seconds=5,
        danger_level="medium",
    ),
    ToolSpec(
        name="screening_diagnosis",
        label="Run screening diagnosis (rule engine)",
        description="Apply rule-based diagnosis to the current hits.",
        fn=_tool_screening_diagnosis,
        estimated_seconds=1,
    ),
]


TOOL_REGISTRY: Dict[str, ToolSpec] = {spec.name: spec for spec in _SPECS}


def get_tool(name: str) -> Optional[ToolSpec]:
    return TOOL_REGISTRY.get(name)


def list_tools() -> List[ToolSpec]:
    return list(_SPECS)
