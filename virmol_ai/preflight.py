# -*- coding: utf-8 -*-
"""Rule-based workflow checks and screening diagnosis (no LLM)."""

from __future__ import annotations

from typing import Any, Dict, List


def run_preflight(context: Dict[str, Any]) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    wf = context.get("workflow") or {}

    if not wf.get("database_loaded"):
        items.append({
            "level": "error",
            "message": "Compound library not loaded.",
            "action": "Molecular Analysis → Database Selection → Load Database.",
        })
    else:
        items.append({
            "level": "info",
            "message": "Compound library loaded.",
            "action": "",
        })

    peak_src = wf.get("peak_source", "none")
    n_peaks = int(wf.get("n_experimental_peaks") or 0)
    if peak_src == "none" or n_peaks == 0:
        items.append({
            "level": "error",
            "message": "No experimental peak list.",
            "action": "Finish spectral preprocessing and Merge, or fill Manual peak input.",
        })
    elif n_peaks < 5:
        items.append({
            "level": "warn",
            "message": f"Only {n_peaks} experimental peaks — may be too few.",
            "action": "Check peak-detection thresholds or complete the manual peak list.",
        })
    else:
        items.append({
            "level": "info",
            "message": f"{n_peaks} experimental peaks (source: {peak_src}).",
            "action": "",
        })

    if wf.get("analysis_done"):
        n = int(wf.get("result_rows") or 0)
        if n == 0:
            items.append({
                "level": "warn",
                "message": "Screening finished with zero hits.",
                "action": "Use Run screening diagnosis or relax CNF/CTNF/MW filters.",
            })
        else:
            items.append({
                "level": "info",
                "message": f"Database screening done — {n} hits in the results table.",
                "action": "Open a result row for details, or use Interpret Top hits in AI Assistant.",
            })
    else:
        items.append({
            "level": "info",
            "message": "Database Analysis (Start Analysis) not run yet.",
            "action": "Load library, confirm peaks, then click Start Analysis.",
        })

    if wf.get("fragment_attribution_available"):
        items.append({
            "level": "info",
            "message": "Fragment attribution data available.",
            "action": "Generate attribution narrative in Fragment Analysis or AI Assistant.",
        })

    return items


def run_screening_diagnosis(context: Dict[str, Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    wf = context.get("workflow") or {}
    screening = context.get("screening") or {}
    filters = screening.get("filters") or {}
    peaks = context.get("experimental_peaks") or []
    candidates = context.get("top_candidates") or []

    if not wf.get("database_loaded"):
        out.append({
            "level": "error",
            "message": "Database not loaded — screening cannot run.",
            "action": "Load Database / Load Other Database.",
        })
        return out

    if not peaks:
        out.append({
            "level": "error",
            "message": "No experimental peaks.",
            "action": "Fill Manual peak input or generate NMR-1D.csv.",
        })
        return out

    n_res = int(wf.get("result_rows") or 0)
    if n_res == 0:
        enabled = []
        if filters.get("CNF"):
            enabled.append(f"CNF (bias={filters.get('CNF_bias')})")
        if filters.get("CTNF"):
            enabled.append(f"CTNF (bias={filters.get('CTNF_bias')})")
        if filters.get("MW"):
            enabled.append(f"MW (list={filters.get('MW_list')})")
        out.append({
            "level": "error",
            "message": "No compounds passed filters (0 hits).",
            "action": (
                "Active filters: " + ("; ".join(enabled) if enabled else "none")
                + ". Try disabling MW, increasing CNF/CTNF bias, or switching to All Database."
            ),
        })
        return out

    if candidates:
        scores = [float(c.get("score") or 0) for c in candidates]
        avg = sum(scores) / len(scores) if scores else 0.0
        top = float(candidates[0].get("score") or 0)
        if top < 0.35:
            out.append({
                "level": "warn",
                "message": f"Low Top1 score ({top:.3f}) — weak match possible.",
                "action": "Verify peaks and solvent removal; try CSS/AAS or adjust FPAACS weights.",
            })
        if avg < 0.25 and len(candidates) >= 5:
            out.append({
                "level": "warn",
                "message": f"Low mean score among Top{len(candidates)} ({avg:.3f}).",
                "action": "Check peak ppm and DEPT types (q/t/d/s).",
            })

    exp_types = {}
    for p in peaks:
        t = str(p.get("type", "s")).lower()
        exp_types[t] = exp_types.get(t, 0) + 1
    if candidates:
        c0 = candidates[0]
        pred_ct = c0.get("carbon_type_counts") or {}
        for t in ("q", "t", "d", "s"):
            if abs(int(exp_types.get(t, 0)) - int(pred_ct.get(t, 0))) > int(
                filters.get("CTNF_bias") or 2
            ):
                out.append({
                    "level": "warn",
                    "message": (
                        f"Top1 carbon-type counts differ from experiment ({t}: "
                        f"exp {exp_types.get(t,0)} vs pred {pred_ct.get(t,0)})."
                    ),
                    "action": "If CTNF is on, filters may be too strict; verify DEPT labels.",
                })
                break

    if not out:
        out.append({
            "level": "info",
            "message": f"{n_res} hits — no obvious rule-level issues.",
            "action": "Use Interpret Top hits for an AI narrative.",
        })
    return out


def format_diagnosis_text(items: List[Dict[str, str]], title: str = "Diagnosis") -> str:
    lines = [f"## {title}", ""]
    for it in items:
        lvl = it.get("level", "info").upper()
        lines.append(f"- **[{lvl}]** {it.get('message', '')}")
        act = it.get("action", "").strip()
        if act:
            lines.append(f"  - Suggestion: {act}")
    lines.append("")
    lines.append(
        "*Rule-engine output. AI suggestions do not replace expert structure proof; "
        "confirm with 2D NMR and isolation.*"
    )
    return "\n".join(lines)


def format_preflight_text(items: List[Dict[str, str]]) -> str:
    return format_diagnosis_text(items, title="Workflow status")
