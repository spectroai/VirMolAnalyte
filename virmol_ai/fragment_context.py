# -*- coding: utf-8 -*-
"""Rich Fragment Analysis context for LLM (Phase A)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from .context import _experimental_peaks_from_gui, _json_safe


def _pairs_summary(frag: Dict[str, Any], max_pairs: int = 6) -> str:
    pairs = frag.get("pairs") or []
    parts = []
    for row in pairs[:max_pairs]:
        try:
            exp_p = float(row.get("exp_ppm", 0.0))
            pred_p = float(row.get("pred_ppm", 0.0))
            parts.append(f"{exp_p:.2f}<-{pred_p:.2f}")
        except (TypeError, ValueError):
            continue
    if len(pairs) > max_pairs:
        parts.append(f"(+{len(pairs) - max_pairs} more)")
    return "; ".join(parts) if parts else ""


def _serialize_fragment(frag: Dict[str, Any], rank_in_table: Optional[int] = None) -> Dict[str, Any]:
    mae = frag.get("mean_abs_err")
    mse = frag.get("match_mse")
    out = {
        "fragment_id": frag.get("fragment_id"),
        "n_carbons": frag.get("n_carbons"),
        "n_core_high_carbons": frag.get("n_core_high_carbons", frag.get("n_carbons")),
        "score_sum": round(float(frag.get("score_sum", 0.0)), 4),
        "score_mean": round(float(frag.get("score_mean", 0.0)), 4),
        "hit_rate_unique": round(float(frag.get("hit_rate_unique", 0.0)), 4),
        "matched_peaks": _pairs_summary(frag),
    }
    if rank_in_table is not None:
        out["table_rank"] = rank_in_table
    if mae is not None and not (isinstance(mae, float) and math.isnan(mae)):
        try:
            out["mean_abs_err_ppm"] = round(float(mae), 3)
        except (TypeError, ValueError):
            pass
    if mse is not None and not (isinstance(mse, float) and math.isnan(mse)):
        try:
            out["match_mse"] = round(float(mse), 4)
        except (TypeError, ValueError):
            pass
    return out


def _per_carbon_highlights(payload: Dict[str, Any], n: int = 8) -> Dict[str, Any]:
    scores = payload.get("mean_scores") or []
    shifts = payload.get("vir_shifts") or []
    atom_idx = payload.get("carbon_atom_indices") or []
    lib_types = payload.get("lib_types") or []
    entries = []
    for i, sc in enumerate(scores):
        try:
            sf = float(sc)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(sf):
            continue
        entry = {
            "carbon_slot": i,
            "rdkit_atom_index": atom_idx[i] if i < len(atom_idx) else None,
            "pred_ppm": round(float(shifts[i]), 2) if i < len(shifts) else None,
            "dept_type": str(lib_types[i]) if i < len(lib_types) else None,
            "mean_score": round(sf, 4),
        }
        entries.append(entry)
    entries.sort(key=lambda x: -x["mean_score"])
    return {
        "top_carbons": entries[:n],
        "bottom_carbons": sorted(entries, key=lambda x: x["mean_score"])[:n],
    }


def _masking_threshold(gui) -> Optional[float]:
    payload = getattr(gui, "masking_result_data", None)
    scores = (payload or {}).get("mean_scores") or []
    if not scores:
        return None
    try:
        t, _ = gui._resolve_fragment_score_threshold(scores)
        return float(t)
    except Exception:
        return None


def _count_carbons_above_threshold(mean_scores: List[Any], threshold: Optional[float]) -> int:
    if threshold is None:
        return 0
    n = 0
    for sc in mean_scores:
        try:
            if float(sc) > float(threshold) and math.isfinite(float(sc)):
                n += 1
        except (TypeError, ValueError):
            continue
    return n


def _masking_candidates_summary(gui, max_candidates: int = 15) -> List[Dict[str, Any]]:
    out = []
    all_res = list(getattr(gui, "masking_all_results", []) or [])
    thr = _masking_threshold(gui)
    for list_idx, payload in enumerate(all_res[:max_candidates]):
        mean_scores = payload.get("mean_scores") or []
        finite = [float(x) for x in mean_scores if math.isfinite(float(x))]
        rank_idx = int(payload.get("candidate_rank_index", list_idx))
        out.append({
            "list_index": list_idx,
            "candidate_rank": rank_idx + 1,
            "smiles": payload.get("smiles"),
            "aas_full": round(float(payload.get("aas_full", 0.0)), 4)
            if payload.get("aas_full") is not None
            else None,
            "mask_mode": payload.get("mask_mode"),
            "n_iterations": payload.get("n_iterations"),
            "n_carbons": payload.get("n_carbons"),
            "high_score_carbon_count": _count_carbons_above_threshold(mean_scores, thr),
            "mean_score_max": round(max(finite), 4) if finite else None,
        })
    return out


def _global_fragments_top(gui, n: int = 10) -> List[Dict[str, Any]]:
    rows = list(getattr(gui, "masking_fragments_global_rows", []) or [])
    top = []
    for i, rec in enumerate(rows[:n]):
        frag = rec.get("frag") or {}
        top.append({
            "global_rank": i + 1,
            "candidate_rank": rec.get("candidate_rank"),
            "fragment_id": frag.get("fragment_id", rec.get("fragment_idx", 0) + 1),
            "n_carbons": frag.get("n_carbons"),
            "score_sum": round(float(rec.get("score_sum", 0.0)), 4),
            "score_mean": round(float(frag.get("score_mean", 0.0)), 4),
            "mean_abs_err_ppm": round(float(frag.get("mean_abs_err", 0.0)), 3)
            if frag.get("mean_abs_err") is not None
            and math.isfinite(float(frag.get("mean_abs_err", float("nan"))))
            else None,
            "matched_peaks": _pairs_summary(frag, max_pairs=4),
        })
    return top


def _fragment_gui_parameters(gui) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if hasattr(gui, "frag_mask_mode_combo"):
        params["mask_mode"] = gui.frag_mask_mode_combo.currentData()
    if hasattr(gui, "frag_mask_fraction"):
        params["mask_fraction"] = float(gui.frag_mask_fraction.value())
    if hasattr(gui, "frag_mask_connected_k_spin"):
        params["connected_mask_size_k"] = int(gui.frag_mask_connected_k_spin.value())
    if hasattr(gui, "frag_mask_tau_spin"):
        params["tau_ppm"] = float(gui.frag_mask_tau_spin.value())
    if hasattr(gui, "frag_mask_iterations"):
        params["monte_carlo_iterations"] = int(gui.frag_mask_iterations.value())
    if hasattr(gui, "frag_mask_seed_edit"):
        seed = gui.frag_mask_seed_edit.text().strip()
        params["random_seed"] = seed or None
    if hasattr(gui, "frag_mask_dept_check"):
        params["dept_constraint"] = gui.frag_mask_dept_check.isChecked()
    if hasattr(gui, "frag_mask_unique_match_check"):
        params["greedy_unique_matching"] = gui.frag_mask_unique_match_check.isChecked()
    if hasattr(gui, "frag_mask_thresh_mode"):
        params["fragment_threshold_mode"] = gui.frag_mask_thresh_mode.currentData()
    if hasattr(gui, "frag_mask_frag_thresh"):
        params["manual_fragment_threshold"] = float(gui.frag_mask_frag_thresh.value())
    if hasattr(gui, "frag_mask_robust_k"):
        params["robust_z_k"] = float(gui.frag_mask_robust_k.value())
    if hasattr(gui, "frag_mask_frag_min_c"):
        params["min_carbons_per_fragment"] = int(gui.frag_mask_frag_min_c.value())
    if hasattr(gui, "frag_mask_bridge_check"):
        params["bridge_low_score_carbons"] = gui.frag_mask_bridge_check.isChecked()
    if hasattr(gui, "frag_mask_max_frags"):
        params["max_fragments_listed"] = int(gui.frag_mask_max_frags.value())
    if hasattr(gui, "frag_mask_topn_spin"):
        params["masking_top_n_candidates"] = int(gui.frag_mask_topn_spin.value())
    return params


def _resolve_candidate_list_index(gui, candidate_index: Optional[int]) -> Optional[int]:
    if candidate_index is not None:
        all_res = list(getattr(gui, "masking_all_results", []) or [])
        for i, p in enumerate(all_res):
            if int(p.get("candidate_rank_index", i)) == int(candidate_index):
                return i
        if 0 <= int(candidate_index) < len(all_res):
            return int(candidate_index)
        return int(candidate_index)
    if getattr(gui, "frag_mask_result_combo", None) and gui.frag_mask_result_combo.isEnabled():
        return int(gui.frag_mask_result_combo.currentIndex())
    payload = getattr(gui, "masking_result_data", None)
    if payload is not None:
        return int(payload.get("candidate_rank_index", 0))
    return None


def _current_candidate_detail(
    gui, list_index: Optional[int], fragment_row_index: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    all_res = list(getattr(gui, "masking_all_results", []) or [])
    if list_index is None or list_index < 0 or list_index >= len(all_res):
        payload = getattr(gui, "masking_result_data", None)
    else:
        payload = all_res[list_index]

    if not payload:
        return None

    all_frags = list(getattr(gui, "masking_fragments_all", []) or [])
    if list_index is not None and list_index < len(all_frags):
        frag_list = list(all_frags[list_index] or [])
    else:
        frag_list = list(getattr(gui, "masking_fragments", []) or [])

    sorted_frags = sorted(frag_list, key=lambda f: float(f.get("score_sum", 0.0)), reverse=True)
    fragments_full = [
        _serialize_fragment(f, rank_in_table=i + 1) for i, f in enumerate(sorted_frags)
    ]

    selected_fragment = None
    if fragment_row_index is not None and 0 <= fragment_row_index < len(sorted_frags):
        selected_fragment = _serialize_fragment(
            sorted_frags[fragment_row_index], rank_in_table=fragment_row_index + 1
        )

    rank_idx = int(payload.get("candidate_rank_index", list_index or 0))
    detail = {
        "list_index": list_index,
        "candidate_rank": rank_idx + 1,
        "smiles": payload.get("smiles"),
        "aas_full": round(float(payload.get("aas_full", 0.0)), 4)
        if payload.get("aas_full") is not None
        else None,
        "mask_mode": payload.get("mask_mode"),
        "mask_fraction": payload.get("mask_fraction"),
        "n_iterations": payload.get("n_iterations"),
        "n_carbons": payload.get("n_carbons"),
        "per_carbon_attribution": _per_carbon_highlights(payload),
        "fragments_all": fragments_full,
        "selected_fragment": selected_fragment,
    }
    return detail


def has_fragment_data(gui) -> bool:
    return bool(
        getattr(gui, "masking_result_data", None)
        or getattr(gui, "masking_all_results", None)
    )


def build_fragment_analysis_context(
    gui,
    *,
    candidate_list_index: Optional[int] = None,
    fragment_row_index: Optional[int] = None,
    max_masking_candidates: int = 15,
    max_global_fragments: int = 10,
) -> Optional[Dict[str, Any]]:
    """Phase A: full structured Fragment Analysis snapshot for LLM."""
    if not has_fragment_data(gui):
        return None

    list_idx = _resolve_candidate_list_index(gui, candidate_list_index)
    thr = _masking_threshold(gui)

    ctx = {
        "experimental_peaks": _experimental_peaks_from_gui(gui),
        "fragment_analysis_parameters": _fragment_gui_parameters(gui),
        "score_threshold_for_high_carbons": round(float(thr), 4) if thr is not None else None,
        "masking_candidates_summary": _masking_candidates_summary(gui, max_masking_candidates),
        "global_top_fragments": _global_fragments_top(gui, max_global_fragments),
        "current_candidate": _current_candidate_detail(gui, list_idx, fragment_row_index),
        "n_candidates_with_masking": len(getattr(gui, "masking_all_results", []) or []),
        "n_global_fragment_rows": len(getattr(gui, "masking_fragments_global_rows", []) or []),
    }
    return _json_safe(ctx)


def merge_fragment_into_analysis(ctx: Dict[str, Any], frag_ctx: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if frag_ctx:
        ctx["fragment_analysis"] = frag_ctx
        # Legacy key for older prompts
        ctx["fragment_attribution"] = frag_ctx.get("current_candidate")
    return ctx
