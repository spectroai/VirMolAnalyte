# -*- coding: utf-8 -*-
"""System and task prompts for P0."""

from __future__ import annotations

import json
from typing import Any, Dict, List


DISCLAIMER_EN = (
    "\n\n---\n*Disclaimer: The above is assistive guidance from current software data only; "
    "it is not a final structure assignment. Confirm with 2D NMR and isolation by an expert.*"
)

# Backward compatibility
DISCLAIMER_ZH = DISCLAIMER_EN


# Tasks where the LLM is EXPECTED to construct/propose new candidate SMILES.
_STRUCTURE_GENERATION_TASKS = ("fusion_infer_top5", "direct_top5", "infer_structure")


def system_prompt(*, allow_structure_generation: bool = False) -> str:
    if allow_structure_generation:
        rule1 = (
            "1. Do NOT fabricate DBindex, screening scores, or experimental ppm/DEPT values "
            "(use only those given in the JSON). However, GENERATING NEW candidate structures as "
            "SMILES is the PURPOSE of this task: you ARE required to construct and output new/modified "
            "SMILES strings — this is explicitly allowed and expected, not a violation.\n"
        )
        rule6 = (
            "6. Apply general organic / natural-products / ¹³C NMR knowledge to build, modify and "
            "rank plausible whole-molecule structures from the provided clues.\n"
        )
    else:
        rule1 = (
            "1. Answer only from the JSON context in the user message; do not invent "
            "DBindex, score, chemical shifts, or SMILES.\n"
        )
        rule6 = (
            "6. For topn tasks, you may apply general organic/NP chemistry knowledge to rank "
            "candidates when the JSON supports it; still do not invent DBindex, scores, or ppm values.\n"
        )
    return (
        "You are the VirMolAnalyte ¹³C NMR database-screening assistant.\n"
        "Rules:\n"
        + rule1
        + "2. When citing candidates, include #rank and the numeric score.\n"
        "3. Do not claim final structure identification; use phrasing like "
        "'consistent with' or 'suggested for priority verification'.\n"
        "4. If a field is missing, state that the GUI did not provide it.\n"
        "5. Respond in English, concise and professional; Methods paragraphs may stay in English.\n"
        + rule6
    )


def system_prompt_for_task(task: str) -> str:
    """Task-specific system instructions (extends base rules)."""
    gen = task in _STRUCTURE_GENERATION_TASKS
    base = system_prompt(allow_structure_generation=gen)
    if task == "fusion_infer_top5":
        return (
            base
            + "\n7. fusion_infer_top5 is DE NOVO structure hypothesis: assemble fragment "
            "motifs + experimental_peaks into five NEW full structures (valid SMILES each). "
            "Use organic / natural-products / 13C NMR reasoning.\n"
            "8. Matched fragments come from database ANALOGS — they may be only PART of the "
            "true molecule. You MUST account for unmatched_exp_peaks and may add/link/scaffold "
            "groups not present verbatim in the analog fragments.\n"
            "9. FORBIDDEN as your five answers: copying parent_database_reference entries, "
            "listing screening-hit SMILES, or returning fragment_smiles unchanged as the full "
            "structure. Those are clues only.\n"
        )
    if task == "direct_top5":
        return (
            base
            + "\n7. direct_top5: propose five plausible full structures from peaks only; "
            "valid SMILES required; no database metadata.\n"
        )
    if task == "infer_structure":
        return (
            base
            + "\n7. infer_structure is STRUCTURE GENERATION from one selected candidate's "
            "masking data + experimental peaks. You MUST construct and output plausible full "
            "structures as NEW valid SMILES (generating SMILES is the goal here). "
            "Use organic / natural-products / 13C NMR reasoning.\n"
            "8. candidate.smiles is the closest database ANALOG, not necessarily the real "
            "compound. Use per-carbon attribution_score (low = weakly supported) and vir_shift vs "
            "experimental_peaks mismatches to locate where the real structure likely differs.\n"
            "9. You are EXPECTED to modify, substitute or extend the analog scaffold and emit the "
            "resulting new SMILES; do not just echo candidate.smiles unchanged unless the evidence "
            "strongly supports it.\n"
        )
    if task == "signal_assignment":
        return (
            base
            + "\n7. signal_assignment is INTERPRETATION, not structure generation. The rule engine "
            "already matched each carbon to an experimental peak; you validate, rate confidence, and "
            "structure the result. Do NOT output a new full SMILES and do NOT invent shifts/scores.\n"
            "8. Base every confidence judgement on the provided attribution_score, Δδ (abs_err) and "
            "DEPT-type agreement only.\n"
        )
    if task == "fragment_evidence_review":
        return (
            base
            + "\n7. fragment_evidence_review is EVIDENCE ANALYSIS, not structure generation. "
            "Do NOT output new structures/SMILES. Focus on fragment quality, complementarity, "
            "unexplained peaks, and a concise readout of likely substructures / compound class.\n"
            "8. Use only values provided in Context JSON (scores, Δδ, hit_rate, matched peaks, "
            "carbon_profile, fragment_smiles, carbon counts). Infer motifs from ppm+DEPT patterns "
            "and fragment evidence; phrase as 'consistent with' / 'suggests', never as confirmed.\n"
            "9. Write the ENTIRE response in English only (all six sections).\n"
        )
    if task == "fusion_evidence_review":
        return (
            base
            + "\n7. fusion_evidence_review is COMBINATION-EVIDENCE ANALYSIS after fusion search, "
            "not de novo structure generation. Do NOT output new structures/SMILES.\n"
            "8. Base ranking rationale on provided metrics only (coverage, aas_best, score_final, "
            "sse, matched/unmatched peaks, fragment composition). Do not fabricate numbers.\n"
        )
    return base


TASK_PROMPTS = {
    "wizard": (
        "From workflow and preflight status, explain in 3–6 sentences:\n"
        "1) Current progress; 2) Next GUI steps (use exact English button names); "
        "3) One caution. Do not pretend analysis is complete if it is not."
    ),
    "topn": (
        "Interpret top_candidates, experimental_peaks, carbon_structure_match, and "
        "analysis_top_n (size of the screened pool: 5, 10, or 20). "
        "Use these Markdown sections (keep the same depth and style as before for all "
        "original sections; do not shorten or omit them):\n"
        "## Experimental peak overview\n"
        "## Most likely candidates\n"
        "## Five likely candidates by AI assistant\n"
        "## Priority verification\n## Main discrepancies\n"
        "## Lower priority\n## Next experimental steps\n"
        "## Most likely candidates: "
        "if fully_matching_candidates is non-empty, judge which rank(s) are most likely "
        "correct using experimental_peaks vs predicted_shifts_ppm (not screening score alone; "
        "may be more than one). If status is no_full_match, state that there is no completely "
        "correct structure in the analyzed set (carbon count and DEPT q/t/d/s must all match).\n"
        "## Five likely candidates by AI assistant: "
        "Pick exactly five distinct candidates from the full top_candidates pool "
        "(all analysis_top_n entries). If analysis_top_n < 5, list every pool member. "
        "Order them 1→5 by **your** assessed likelihood of being the true structure "
        "(1 = most likely) — this order MUST NOT simply follow screening score or #rank. "
        "Use your natural-products / ¹³C NMR knowledge plus structure_facts, "
        "experimental_peaks vs predicted_shifts_ppm, carbon_type fit, and "
        "substructure_hints; cite screening score only as secondary context. "
        "Format each item as: "
        "**AI rank N** — screening **#rank** (dbindex …, score …): one or two sentences why. "
        "At most 5 items under Priority verification; each must include #rank and score."
    ),
    "compound": (
        "Interpret selected_candidate, experimental_peaks, and structure_facts.\n"
        "Explain spectrum–structure correspondence and main mismatches (cite ppm). "
        "Do not state a final assignment."
    ),
    "fragment": (
        "From fragment_analysis (especially current_candidate), summarize whole-molecule masking "
        "fit (aas_full), per-carbon highlights, and top fragments by score_sum. "
        "Cite candidate_rank and fragment_id. Do not invent fragments."
    ),
    "infer_structure": (
        "Single-candidate structure inference. The user selected ONE screening candidate after "
        "masking attribution and wants the most plausible REAL structure(s) of the unknown compound.\n"
        "Inputs:\n"
        "- candidate.smiles = closest database analog (NOT guaranteed correct).\n"
        "- per_carbon_table: each carbon's vir_shift_ppm (predicted 13C), dept_type (q/t/d/s), and "
        "attribution_score (how well the experimental spectrum supports that carbon; LOW or missing "
        "score = weakly supported / likely different in the real molecule).\n"
        "- experimental_peaks: the ORIGINAL data used for database screening (ppm + DEPT type).\n"
        "How to reason:\n"
        "1) Carbons with HIGH attribution_score and small vir-vs-exp mismatch are well explained; "
        "keep those local environments.\n"
        "2) Carbons with LOW score or large mismatch, and experimental peaks not covered by the "
        "analog, mark regions where the real structure likely DIFFERS (substituent, ring size, "
        "oxidation/unsaturation, glycosylation, etc.).\n"
        "3) Match total carbon count and q/t/d/s distribution to experimental_peaks when reasonable.\n"
        "Output (strict Markdown):\n"
        "## Evidence assessment\n"
        "<well-supported vs weak carbons; key ppm/DEPT mismatches; unexplained experimental peaks>\n"
        "## Proposed structure(s)\n"
        "1) SMILES: <full structure>\n"
        "   Likelihood: <high|medium|low>\n"
        "   Rationale: <ONE sentence tied to specific shifts / DEPT / scores>\n"
        "...(up to 5, ranked most -> least likely)\n"
        "Rules:\n"
        "- Provide 1 to 5 DISTINCT candidate full structures, each a valid SMILES.\n"
        "- Do NOT simply echo candidate.smiles unless the evidence strongly supports it; modified or "
        "analog scaffolds are encouraged when they better explain the spectrum.\n"
        "- Keep under 500 words; do not invent dbindex or screening scores.\n"
        "- Remind that 2D NMR / MS / isolation is required to confirm."
    ),
    "fragment_explain": (
        "Explain only current_candidate.selected_fragment: why its score_sum/mean_abs_err and "
        "matched_peaks support or weaken attribution. Compare briefly to other fragments_all "
        "in the same candidate if helpful. Cite fragment_id and ppm values from JSON only."
    ),
    "fusion_infer_top5": (
        "You are an expert in natural-product structure elucidation. From ONE selected fragment "
        "combination plus the experimental ¹³C/DEPT data, infer the most likely CORRECT full "
        "structure(s) of the unknown compound. This is constrained structure GENERATION.\n"
        "OUTPUT DISCIPLINE (critical — do this or the answer is useless):\n"
        "- Do all analysis SILENTLY. Do NOT narrate your thinking, do NOT explore-and-backtrack, do "
        "NOT re-derive things in the visible text.\n"
        "- Emit the THREE sections below in order, '## Most likely structures' FIRST in content "
        "priority: produce the #1 SMILES before writing anything long. The reasoning sections are "
        "SHORT SUMMARIES (each ≤120 words), not derivations.\n"
        "HOW TO READ THE DATA (avoids wasted effort):\n"
        "- fragment_smiles is a CARBON-SKELETON approximation that MAY omit O/N. Do NOT try to "
        "reconcile it atom-by-atom. The AUTHORITATIVE per-carbon environment is each motif's "
        "carbon_profile (ctype + exp_ppm) and matched_pairs. Trust those for DEPT type and shift.\n"
        "FIXED vs FREE:\n"
        "- FIXED: each selected motif contributes exactly its carbon_profile carbons (count + q/t/d/s "
        "+ their ppm). Keep those environments.\n"
        "- FREE: (a) how the motifs are connected; (b) the identity of carbons behind unmatched_exp_peaks.\n"
        "HARD CONSTRAINTS:\n"
        "- Build ONLY from the selected fragment_motifs. Ignore any peer_fusion_combinations / "
        "supplementary_fragment_motifs / parent_database_reference if present.\n"
        "- CARBON BUDGET: final carbon count MUST equal experimental_summary.n_carbons and its q/t/d/s "
        "MUST match experimental_summary.carbon_type_counts. When motifs come from different analogs, "
        "do NOT double-count shared/overlapping atoms.\n"
        "- Derive functional groups from carbon_profile shifts (guide: C=O/ester s ~165-210; O-quaternary "
        "s ~70-85; sp2 quaternary s ~120-160; O-CH/O-CH2 ~60-85; anomeric d ~95-110; olefinic/aromatic "
        "CH d ~105-150; CH3 q ~10-25; OCH3 q ~50-60).\n"
        "- For each unmatched peak add only the smallest group whose ppm+DEPT match it. Add nothing extra.\n"
        "OUTPUT (Markdown, exactly these three headings):\n"
        "## Most likely structures\n"
        "1) SMILES: <best full structure>\n"
        "   Confidence: <high|medium|low>\n"
        "   Carbon check: <this SMILES carbon count & q/t/d/s vs experiment — match / where it differs>\n"
        "   Rationale: <1 sentence: key environments explained; any peak left unexplained>\n"
        "2) ... (alternatives differing in motif CONNECTIVITY and/or unmatched-carbon reading; only if "
        "genuinely ambiguous — do not pad with near-duplicates; 1 to 5 total, ranked best first)\n"
        "## Carbon budget\n"
        "<≤120 words: target n_carbons & q/t/d/s; carbons from each motif's carbon_profile; remaining "
        "carbons to infer; coverage %>\n"
        "## Assembly notes\n"
        "<≤120 words: how the motifs connect; for each unmatched peak the inferred group with ppm/DEPT>\n"
        "RULES:\n"
        "- Prefer the structure explaining the MOST peaks with correct DEPT and correct total carbon "
        "count, using the FEWEST unsupported atoms.\n"
        "- Every SMILES must be a valid, connected molecule.\n"
        "- Do not invent dbindex or screening scores; total answer ≤450 words.\n"
        "- Remind that 2D NMR / MS / isolation is required to confirm."
    ),
    "polish_methods": (
        "The user provides an English Methods draft. Polish wording only; do not add, remove, "
        "or change any numbers, parameter names, file names, TopN, bias, τ, iteration count R, etc. "
        "Output the full polished paragraph in English."
    ),
    "diagnosis_llm": (
        "The user provides rule_diagnosis entries. Restate in English and prioritize; "
        "do not introduce parameters or values not present in rule_diagnosis."
    ),
    "chat": (
        "The user is having a free-form conversation. Reply directly and "
        "conversationally — match the user's language (Chinese or English). "
        "Keep it short and natural. Do NOT use the 'Current progress / Next GUI "
        "steps / One caution' workflow template; do NOT add a disclaimer line. "
        "Only consult the Context JSON if the user's question actually refers to "
        "the screening data; otherwise simply answer the question."
    ),
    "direct_top5": (
        "The user asks for direct hypothesis generation from 13C/DEPT peaks only "
        "(no database screening, no ranking table). "
        "Use experimental_peaks + experimental_summary in Context JSON and propose "
        "exactly 5 plausible candidate structures as SMILES.\n"
        "Output format (strict):\n"
        "## Direct LLM Top-5 hypotheses (no database screening)\n"
        "1) SMILES: <...>\n"
        "   Reason: <ONE short sentence tied to peak pattern/carbon types>\n"
        "   Confidence: <low|medium|high>\n"
        "...\n"
        "5) ...\n"
        "Rules:\n"
        "- EXACTLY five candidates.\n"
        "- Every candidate MUST include a SMILES string.\n"
        "- Keep total answer under 350 words.\n"
        "- Do not invent database metadata (no dbindex, no screening score, no rank from DB).\n"
        "- This is hypothesis generation only; explicitly remind that 2D NMR/MS/isolation validation is required."
    ),
    "signal_assignment": (
        "You assign the experimental ¹³C/DEPT signals of an unknown to the carbons of the selected "
        "database analog, using Monte-Carlo per-carbon attribution scores and score-based fragment "
        "extraction. The rule engine ALREADY computed everything you need in the Context JSON:\n"
        "- assignment_table: one row per carbon → its best experimental peak "
        "(atom index, dept_type, vir_shift_ppm = δ_calc, exp_ppm = δ_exp, abs_err = Δδ, "
        "attribution_score, fragment_label, in_high_score_fragment).\n"
        "- high_score_fragments: the connected high-score motifs (label + SMILES + carbons).\n"
        "- unassigned_exp_peaks: experimental peaks no carbon was matched to.\n"
        "Your job is to VALIDATE and STRUCTURE these assignments — NOT to invent new shifts, scores, "
        "or a new molecule.\n"
        "Confidence rubric per carbon: HIGH = high attribution_score AND small Δδ (≲1.5 ppm) AND DEPT "
        "type matches; MEDIUM = exactly one of those is weak; LOW = low score, or Δδ large, or DEPT "
        "mismatch.\n"
        "Output STRICT Markdown with EXACTLY these five sections, in this order:\n"
        "## Assignment table\n"
        "A Markdown table | Atom | DEPT | δ_calc | δ_exp | Δδ | Score | Fragment | Confidence |, one "
        "row per carbon in assignment_table, ordered by DESCENDING δ_exp. Fill Confidence per the rubric.\n"
        "## Confident assignments\n"
        "<bullet list of the HIGH-confidence carbons / fragments that are reliably assigned, grouped "
        "by fragment_label when useful>\n"
        "## Uncertain or conflicting\n"
        "<carbons with DEPT mismatch, large Δδ, or low score; one short clause each on WHY it is doubtful>\n"
        "## Unassigned experimental peaks\n"
        "<for each peak in unassigned_exp_peaks: δ + DEPT, and the functional group it most likely "
        "represents — i.e. the part of the REAL molecule the analog lacks. If the list is empty, say "
        "'All experimental peaks were assigned.'>\n"
        "## Summary\n"
        "<2-3 sentences: which regions of the molecule are spectroscopically well supported vs which "
        "are uncertain and need 2D NMR>\n"
        "Rules:\n"
        "- Use ONLY values present in the Context JSON (atom indices, ppm, types, scores). Never invent "
        "shifts or scores.\n"
        "- Do NOT propose a new full SMILES here; this is signal assignment, not structure generation.\n"
        "- Prose (everything except the table) under 450 words.\n"
        "- End by reminding that final assignment requires 2D NMR (HSQC/HMBC) confirmation."
    ),
    "fragment_evidence_review": (
        "The user asks you to analyze extracted positive fragments and mapping evidence from "
        "Fragment Analysis. This is a decision-support review BEFORE/ALONGSIDE fusion, not "
        "new structure generation.\n"
        "Input fields include fragment_rows (per-fragment metrics, fragment_smiles, carbon_profile, "
        "pair_examples), experimental_summary, and unassigned_exp_peaks.\n"
        "Output STRICT Markdown with EXACTLY these six sections in order:\n"
        "## Likely structural motifs & compound class\n"
        "<FIRST and most user-facing: in ≤6 short bullets or lines, state (a) the most plausible "
        "compound CLASS (e.g. flavonoid glycoside, terpene glycoside, phenylpropanoid ester, "
        "alkaloid — pick what fits the data); (b) 2–5 likely SUBSTRUCTURE units the unknown may "
        "contain (name them plainly: e.g. β-D-glucopyranosyl, caffeoyl/feruloyl, aromatic ring, "
        "isopropyl/monoterpene skeleton, prenyl/alkyl chain), each tied to specific ppm+DEPT "
        "and/or high-confidence fragments; (c) one line on overall structural FEATURES "
        "(oxygenation, glycosylation, aromaticity). Write in English only. "
        "Keep this section under 120 words — intuitive, not exhaustive.>\n"
        "## Fragment evidence ranking\n"
        "<rank all fragments by evidence strength; for each: key supporting metrics "
        "(score_sum/score_mean/hit_rate_unique/mean_abs_err/match_mse), and one-line verdict "
        "(Keep / Medium priority / Low priority).>\n"
        "## Redundant vs complementary fragments\n"
        "<which fragments are likely redundant (same motif/evidence region) vs complementary "
        "(cover different peak regions/types)>\n"
        "## Unexplained peaks and likely missing motifs\n"
        "<interpret unassigned_exp_peaks by ppm+DEPT; suggest likely missing chemical motif(s) "
        "without inventing exact structures>\n"
        "## Recommended fusion-ready fragment set\n"
        "<propose a minimal, high-quality fragment subset for fusion, with short rationale>\n"
        "## Risks and verification priorities\n"
        "<top risks (conflicting assignment / weak fragment / overfitting) and what to verify first>\n"
        "Rules:\n"
        "- Entire answer in English only.\n"
        "- Use ONLY Context JSON values; do not invent scores/ppm.\n"
        "- Do NOT propose new full-molecule SMILES in this task.\n"
        "- Section 1 must be readable by a non-specialist; avoid long tables there.\n"
        "- Keep total answer concise, under 580 words.\n"
        "- End with: final confirmation requires 2D NMR (HSQC/HMBC) and expert validation."
    ),
    "fusion_evidence_review": (
        "Analyze fusion combinations generated by 'Run fusion analysis (Top combinations)'. "
        "This is a review/rationalization task, not structure generation.\n"
        "Inputs include fusion_rows (Top combinations with metrics), optional selected_combination, "
        "experimental_summary and unmatched_exp_peaks.\n"
        "Output STRICT Markdown with EXACTLY these five sections:\n"
        "## Combination ranking rationale\n"
        "<for top combinations: explain ranking using coverage_pct, aas_best, score_final, sse, and k>\n"
        "## Fragment-role interpretation\n"
        "<for each priority combination: identify core-supporting vs compensating vs likely noisy fragments>\n"
        "## Unexplained / conflicting signals\n"
        "<summarize unmatched peaks and high-conflict mapping patterns from provided data>\n"
        "## Recommended combinations for verification\n"
        "<recommend 2-3 combinations to verify first, with short why>\n"
        "## Next experimental checks\n"
        "<targeted 2D NMR checks (HSQC/HMBC/COSY/NOESY as applicable) that can discriminate top options>\n"
        "Rules:\n"
        "- Use ONLY Context JSON values; do not invent metrics or peaks.\n"
        "- Do NOT output new full-molecule SMILES in this task.\n"
        "- Keep concise, under 540 words.\n"
        "- End with: final confirmation requires 2D NMR (HSQC/HMBC) and expert validation."
    ),
}


def build_messages(
    task: str,
    context: Dict[str, Any],
    extra_user: str = "",
    *,
    compact_context: bool = False,
) -> List[Dict[str, str]]:
    task_instruction = TASK_PROMPTS.get(task, "Answer from the context.")
    compact_json = compact_context or task == "topn"
    user_parts = [
        f"Task: {task}\n{task_instruction}",
        "\nContext JSON:\n"
        + json.dumps(
            context,
            ensure_ascii=False,
            indent=None if compact_json else 2,
            separators=(",", ":") if compact_json else (", ", ": "),
        ),
    ]
    if extra_user.strip():
        user_parts.append("\nUser note:\n" + extra_user.strip())
    return [
        {"role": "system", "content": system_prompt_for_task(task)},
        {"role": "user", "content": "\n".join(user_parts)},
    ]
