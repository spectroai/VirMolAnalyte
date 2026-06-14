# -*- coding: utf-8 -*-
"""Build AnalysisContext JSON from ModernVirMolAnalyteGUI."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# SMARTS for coarse functional groups (optional hints for LLM)
_SUBSTRUCTURE_PATTERNS = [
    ("phenyl", "[c]1[c][c][c][c][c]1"),
    ("hydroxyl", "[OX2H]"),
    ("carbonyl", "[CX3]=[OX1]"),
    ("ester", "[CX3](=O)[OX2]"),
    ("ether", "[OD2]([#6])[#6]"),
    ("amine", "[NX3]"),
]


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return str(obj)


_CARBON_TYPES = ("q", "t", "d", "s")


def _count_types(ctype: Sequence[str]) -> Dict[str, int]:
    out = {t: 0 for t in _CARBON_TYPES}
    for c in ctype:
        k = str(c).strip().lower()
        if k in out:
            out[k] += 1
    return out


def _experimental_carbon_summary(peaks: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = _count_types(p.get("type", "s") for p in peaks) if peaks else {t: 0 for t in _CARBON_TYPES}
    return {"n_carbons": len(peaks), "carbon_type_counts": counts}


def _candidate_carbon_fully_matches(exp_summary: Dict[str, Any], cand: Dict[str, Any]) -> bool:
    n_exp = int(exp_summary.get("n_carbons") or 0)
    if n_exp <= 0:
        return False
    if int(cand.get("n_carbons_predicted") or 0) != n_exp:
        return False
    exp_ct = exp_summary.get("carbon_type_counts") or {}
    pred_ct = cand.get("carbon_type_counts") or {}
    return all(int(exp_ct.get(t, 0)) == int(pred_ct.get(t, 0)) for t in _CARBON_TYPES)


def _candidate_pick_fields(c: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "rank": c.get("rank"),
        "dbindex": c.get("dbindex"),
        "smiles": c.get("smiles"),
        "score": c.get("score"),
        "n_carbons_predicted": c.get("n_carbons_predicted"),
        "carbon_type_counts": c.get("carbon_type_counts"),
        "predicted_shifts_ppm": c.get("predicted_shifts_ppm"),
    }


def compute_carbon_structure_match(
    candidates: List[Dict[str, Any]],
    experimental_peaks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Find candidates whose carbon count and q/t/d/s counts fully match experiment."""
    exp_summary = _experimental_carbon_summary(experimental_peaks)
    if int(exp_summary.get("n_carbons") or 0) <= 0:
        return _json_safe({
            "status": "no_experimental_peaks",
            "experimental": exp_summary,
            "fully_matching_candidates": [],
            "message": "No experimental peaks — cannot assess full carbon-count/type match.",
        })

    matched = [c for c in candidates if _candidate_carbon_fully_matches(exp_summary, c)]
    picks = [_candidate_pick_fields(c) for c in matched]

    if picks:
        return _json_safe({
            "status": "full_match_found",
            "experimental": exp_summary,
            "fully_matching_candidates": picks,
            "message": (
                f"{len(picks)} candidate(s) fully match experimental carbon count "
                f"(n={exp_summary['n_carbons']}) and DEPT-type distribution. "
                "Which is most likely correct must be judged from spectrum–structure "
                "evidence in top_candidates (not screening score alone)."
            ),
        })

    return _json_safe({
        "status": "no_full_match",
        "experimental": exp_summary,
        "fully_matching_candidates": [],
        "message": (
            "No candidate in the analyzed set has a predicted carbon count and "
            "DEPT-type distribution that fully matches the experimental spectrum — "
            "there is no completely correct structure in this set."
        ),
    })


def _structure_facts(smiles: str) -> Dict[str, Any]:
    facts: Dict[str, Any] = {"smiles": smiles}
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            facts["valid_smiles"] = False
            return facts
        facts["valid_smiles"] = True
        facts["molecular_formula"] = Chem.rdMolDescriptors.CalcMolFormula(mol)
        facts["heavy_atom_count"] = int(mol.GetNumHeavyAtoms())
        facts["molecular_weight"] = round(float(Descriptors.MolWt(mol)), 2)
        groups = []
        for name, smarts in _SUBSTRUCTURE_PATTERNS:
            pat = Chem.MolFromSmarts(smarts)
            if pat and mol.HasSubstructMatch(pat):
                groups.append(name)
        facts["substructure_hints"] = groups
    except Exception as e:
        facts["rdkit_error"] = str(e)
    return facts


def _row_to_candidate(gui, row_index: int, rank: int) -> Dict[str, Any]:
    cand: Dict[str, Any] = {"rank": rank, "row_index": row_index}
    if hasattr(gui, "results_table") and gui.results_table.rowCount() > row_index:
        cand["dbindex"] = gui.results_table.item(row_index, 0).text()
        cand["smiles"] = gui.results_table.item(row_index, 1).text()
        try:
            cand["score"] = float(gui.results_table.item(row_index, 2).text())
        except (TypeError, ValueError):
            cand["score"] = None
    smiles, shifts, ctypes = gui._get_compound_row_data(row_index)
    cand["smiles"] = smiles or cand.get("smiles", "")
    cand["n_carbons_predicted"] = len(shifts) if shifts else 0
    cand["carbon_type_counts"] = _count_types(ctypes) if ctypes else {}
    if shifts:
        cand["predicted_shifts_ppm"] = [round(float(x), 2) for x in shifts]
    return _json_safe(cand)


def _compact_candidate_for_llm(cand: Dict[str, Any], *, max_shifts: int) -> Dict[str, Any]:
    """Trim per-candidate payload so large Top-N requests stay within model limits."""
    out = dict(cand)
    shifts = out.get("predicted_shifts_ppm")
    if isinstance(shifts, list) and len(shifts) > max_shifts:
        out["predicted_shifts_ppm"] = shifts[:max_shifts]
        out["predicted_shifts_note"] = f"truncated to first {max_shifts} of {len(shifts)}"
    sf = out.get("structure_facts")
    if isinstance(sf, dict):
        out["structure_facts"] = {
            k: sf[k]
            for k in (
                "molecular_formula",
                "heavy_atom_count",
                "molecular_weight",
                "substructure_hints",
                "valid_smiles",
            )
            if k in sf
        }
    return _json_safe(out)


def _max_shifts_for_pool(pool_size: int) -> int:
    if pool_size <= 5:
        return 40
    if pool_size <= 10:
        return 28
    return 22


def _experimental_peaks_from_gui(gui) -> List[Dict[str, Any]]:
    exp = gui.get_experimental_data() if hasattr(gui, "get_experimental_data") else None
    if not exp:
        return []
    out = []
    for p in exp:
        out.append({
            "ppm": round(float(p.get("ppm", 0)), 2),
            "type": str(p.get("type", "s")).lower(),
        })
    return out


def _fragment_summary(gui, candidate_index: Optional[int] = None) -> Optional[Dict[str, Any]]:
    payload = None
    if candidate_index is not None:
        all_res = list(getattr(gui, "masking_all_results", []) or [])
        for p in all_res:
            if int(p.get("candidate_rank_index", -1)) == int(candidate_index):
                payload = p
                break
        if payload is None and all_res and 0 <= candidate_index < len(all_res):
            payload = all_res[candidate_index]
    else:
        payload = getattr(gui, "masking_result_data", None)

    if not payload:
        return None

    frags = []
    if candidate_index is not None:
        all_frags = list(getattr(gui, "masking_fragments_all", []) or [])
        if candidate_index < len(all_frags):
            frag_list = all_frags[candidate_index]
        else:
            frag_list = getattr(gui, "masking_fragments", []) or []
    else:
        frag_list = getattr(gui, "masking_fragments", []) or []

    sorted_frags = sorted(
        list(frag_list),
        key=lambda f: float(f.get("score_sum", 0.0)),
        reverse=True,
    )
    for i, frag in enumerate(sorted_frags[:5]):
        frags.append({
            "fragment_id": frag.get("fragment_id", i + 1),
            "n_carbons": frag.get("n_carbons"),
            "score_sum": round(float(frag.get("score_sum", 0.0)), 4),
            "score_mean": round(float(frag.get("score_mean", 0.0)), 4),
            "hit_rate_unique": round(float(frag.get("hit_rate_unique", 0.0)), 4),
        })

    mean_scores = payload.get("mean_scores") or []
    finite = [float(x) for x in mean_scores if x is not None and np.isfinite(float(x))]
    return _json_safe({
        "candidate_rank_index": payload.get("candidate_rank_index"),
        "smiles": payload.get("smiles"),
        "aas_full": payload.get("aas_full"),
        "mask_mode": payload.get("mask_mode"),
        "n_iterations": payload.get("n_iterations"),
        "top_fragments": frags,
        "mean_score_max": max(finite) if finite else None,
        "mean_score_min": min(finite) if finite else None,
    })


def build_analysis_context(
    gui,
    *,
    top_n: int = 15,
    include_fragment: bool = False,
    selected_row: Optional[int] = None,
    compare_rows: Optional[Tuple[int, int]] = None,
    fragment_row_index: Optional[int] = None,
    candidate_list_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Assemble P0 context dict from live GUI state."""
    base = gui._collect_analysis_params_base() if hasattr(gui, "_collect_analysis_params_base") else {}
    peaks = _experimental_peaks_from_gui(gui)
    peak_src = (base.get("peak_input") or {}).get("source", "none")

    wf = {
        "database_loaded": bool(getattr(gui, "database1", None)),
        "peak_source": peak_src,
        "n_experimental_peaks": len(peaks),
        "analysis_done": getattr(gui, "result", None) is not None,
        "result_rows": 0,
        "fragment_attribution_available": bool(
            getattr(gui, "masking_result_data", None)
            or getattr(gui, "masking_all_results", None)
        ),
    }
    res = getattr(gui, "result", None)
    if res is not None:
        try:
            wf["result_rows"] = len(res)
        except Exception:
            pass

    screening = {
        "database": base.get("database"),
        "filters": base.get("filters"),
        "evaluator": base.get("evaluator"),
        "run_params": base.get("run_params"),
    }

    top_candidates = []
    n_rows = gui.results_table.rowCount() if hasattr(gui, "results_table") else 0
    pool_size = 0
    if res is not None and n_rows > 0:
        pool_size = min(top_n, n_rows)
        shift_cap = _max_shifts_for_pool(pool_size)
        for i in range(pool_size):
            cand = _row_to_candidate(gui, i, i + 1)
            smiles = cand.get("smiles") or ""
            if smiles:
                cand["structure_facts"] = _structure_facts(smiles)
            top_candidates.append(_compact_candidate_for_llm(cand, max_shifts=shift_cap))

    ctx: Dict[str, Any] = {
        "workflow": wf,
        "experimental_peaks": peaks,
        "screening": screening,
        "top_candidates": top_candidates,
        "analysis_top_n": pool_size,
        "carbon_structure_match": compute_carbon_structure_match(top_candidates, peaks),
        "methodology_draft_en": None,
    }

    if hasattr(gui, "_build_methodology_paragraph"):
        try:
            ctx["methodology_draft_en"] = gui._build_methodology_paragraph()
        except Exception:
            ctx["methodology_draft_en"] = None

    if include_fragment:
        from .fragment_context import build_fragment_analysis_context, merge_fragment_into_analysis

        list_idx = candidate_list_index
        if list_idx is None and selected_row is not None:
            list_idx = selected_row
        frag_ctx = build_fragment_analysis_context(
            gui,
            candidate_list_index=list_idx,
            fragment_row_index=fragment_row_index,
        )
        merge_fragment_into_analysis(ctx, frag_ctx)

    if selected_row is not None and selected_row >= 0:
        ctx["selected_candidate"] = _row_to_candidate(gui, selected_row, selected_row + 1)
        ctx["selected_candidate"]["structure_facts"] = _structure_facts(
            ctx["selected_candidate"].get("smiles", "")
        )
        if peaks and ctx["selected_candidate"].get("predicted_shifts_ppm"):
            ctx["selected_candidate"]["shift_mismatch_top3"] = _top_shift_mismatches(
                peaks,
                ctx["selected_candidate"].get("predicted_shifts_ppm", []),
            )

    if compare_rows:
        a, b = compare_rows
        ctx["compare_candidates"] = [
            _row_to_candidate(gui, a, a + 1),
            _row_to_candidate(gui, b, b + 1),
        ]

    from .preflight import run_preflight

    ctx["preflight"] = run_preflight(ctx)
    return _json_safe(ctx)


def _top_shift_mismatches(
    exp: List[Dict[str, Any]],
    pred_ppm: List[float],
    pred_types: Optional[Dict[str, int]] = None,
    n: int = 3,
) -> List[Dict[str, float]]:
    """Greedy nearest-neighbor |Δδ| for diagnostic hints."""
    used = set()
    mismatches = []
    for pi, pp in enumerate(pred_ppm):
        best_j, best_d = None, 1e9
        for j, ep in enumerate(exp):
            if j in used:
                continue
            d = abs(float(pp) - float(ep["ppm"]))
            if d < best_d:
                best_d, best_j = d, j
        if best_j is not None:
            used.add(best_j)
            mismatches.append({
                "pred_ppm": round(float(pp), 2),
                "exp_ppm": round(float(exp[best_j]["ppm"]), 2),
                "delta_ppm": round(best_d, 2),
            })
    mismatches.sort(key=lambda x: -x["delta_ppm"])
    return mismatches[:n]
