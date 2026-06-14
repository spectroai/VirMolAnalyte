# -*- coding: utf-8 -*-
"""
Fragment-level analysis for virtual 13C shifts vs experimental peaks.

Pipeline (see 虚拟谱片段挖掘_多方案技术说明.txt):
  B — Hungarian assignment (library carbon ↔ experimental peak)
  C — Optional median bias correction on virtual shifts, 1–2 iterations
  E — Optional DEPT / carbon-type constraint on allowed pairs
  D — Top-K fragment consensus over BRICS/Murcko-derived atom sets

Maintainers: edit parameters via FragmentAnalysisOptions or run_fragment_analysis kwargs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as e:  # pragma: no cover
    raise ImportError("fragment_analysis requires scipy (linear_sum_assignment).") from e

from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold


BIG = 1.0e7


@dataclass
class FragmentAnalysisOptions:
    use_median_correction: bool = True
    median_max_iters: int = 2
    use_dept_constraint: bool = False
    cost_cap_ppm: float = 50.0
    unmatched_penalty_ppm: float = 100.0
    hit_threshold_ppm: float = 0.8
    consensus_min_count: int = 2
    fragment_method: str = "murcko_only"  # default: larger chunks; see get_fragment_atom_sets
    # Drop fragments with too few carbons (BRICS often yields C / CC only — not chemically informative).
    min_carbons_per_fragment: int = 4


@dataclass
class CarbonAssignment:
    carbon_slot: int
    atom_idx: int
    vir_shift: float
    lib_type: str
    matched_peak_index: Optional[int]
    exp_ppm: Optional[float]
    exp_type: Optional[str]
    residual_ppm: Optional[float]
    hit: bool


@dataclass
class FragmentRecord:
    fragment_key: str
    atom_indices: Tuple[int, ...]
    carbon_slots: List[int]
    n_carbons: int
    n_matched: int
    n_hit: int
    mean_abs_residual: float
    median_abs_residual: float
    hit_rate: float


@dataclass
class CandidateFragmentResult:
    rank: int
    smiles: str
    dbindex: Any
    score: float
    median_corrections: List[float]
    assignments: List[CarbonAssignment]
    fragments: List[FragmentRecord]


@dataclass
class ConsensusRecord:
    fragment_key: str
    count: int
    avg_hit_rate: float
    avg_mean_abs_residual: float
    ranks: List[int]


@dataclass
class FragmentAnalysisResult:
    options: FragmentAnalysisOptions
    candidates: List[CandidateFragmentResult]
    consensus: List[ConsensusRecord]

    def to_serializable(self) -> Dict[str, Any]:
        def assign_dict(a: CarbonAssignment) -> Dict[str, Any]:
            d = asdict(a)
            return d

        def frag_dict(f: FragmentRecord) -> Dict[str, Any]:
            d = asdict(f)
            d["atom_indices"] = list(f.atom_indices)
            return d

        return {
            "options": asdict(self.options),
            "candidates": [
                {
                    "rank": c.rank,
                    "smiles": c.smiles,
                    "dbindex": c.dbindex,
                    "score": float(c.score) if c.score is not None else None,
                    "median_corrections": [float(x) for x in c.median_corrections],
                    "assignments": [assign_dict(a) for a in c.assignments],
                    "fragments": [frag_dict(f) for f in c.fragments],
                }
                for c in self.candidates
            ],
            "consensus": [asdict(x) for x in self.consensus],
        }


def normalize_peak_type(raw: Any) -> Optional[str]:
    """Map CSV / GUI carbon labels to q,t,d,s (DEPT). Unknown → None (wildcard)."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in ("q", "ch3", "methyl"):
        return "q"
    if s in ("t", "ch2", "methylene"):
        return "t"
    if s in ("d", "ch", "methine"):
        return "d"
    if s in ("s", "c", "quat", "quaternary"):
        return "s"
    if len(s) == 1 and s in "qtds":
        return s
    return None


def carbon_atom_indices_in_shift_order(mol: Chem.Mol) -> List[int]:
    """Same traversal order as GetCarbonType in DataPrepare.py."""
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6]


def _listify_shifts(x: Any) -> List[float]:
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return [float(v) for v in x.flatten()]
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    return [float(x)]


def _listify_ctype(x: Any, n: int) -> List[str]:
    if x is None:
        return ["s"] * n
    if isinstance(x, np.ndarray):
        seq = [str(v).strip() for v in x.flatten()]
    elif isinstance(x, (list, tuple)):
        seq = [str(v).strip() for v in x]
    elif isinstance(x, str):
        seq = [p.strip() for p in x.replace(";", ",").split(",") if p.strip()]
    else:
        seq = [str(x)]
    seq = [s[:1].lower() if s else "s" for s in seq]
    out = []
    for i in range(n):
        c = seq[i] if i < len(seq) else "s"
        if c not in ("q", "t", "d", "s"):
            c = "s"
        out.append(c)
    return out


def _pair_cost(
    vir: float,
    lib_t: str,
    exp_ppm: float,
    exp_t: Optional[str],
    cap: float,
    use_dept: bool,
) -> float:
    if use_dept and exp_t is not None and exp_t != lib_t:
        return BIG
    d = abs(vir - exp_ppm)
    return float(min(d, cap))


def assign_hungarian(
    vir_shifts: np.ndarray,
    lib_types: Sequence[str],
    exp_ppms: np.ndarray,
    exp_types: Sequence[Optional[str]],
    *,
    use_dept: bool,
    cost_cap_ppm: float,
    unmatched_penalty: float,
) -> np.ndarray:
    """
    Returns col_for_row of shape (n_carbons,): column index for each carbon row.
    Column in [0, n_peaks) is a real peak; >= n_peaks means dummy (unmatched).
    """
    n_c = len(vir_shifts)
    n_p = len(exp_ppms)
    n_cols = n_p + n_c
    cost = np.full((n_c, n_cols), BIG, dtype=np.float64)
    for i in range(n_c):
        for j in range(n_p):
            cost[i, j] = _pair_cost(
                float(vir_shifts[i]),
                lib_types[i],
                float(exp_ppms[j]),
                exp_types[j],
                cost_cap_ppm,
                use_dept,
            )
        cost[i, n_p + i] = unmatched_penalty
    ri, ci = linear_sum_assignment(cost)
    col_for_row = np.zeros(n_c, dtype=int)
    col_for_row[ri] = ci
    return col_for_row


def build_assignments(
    vir_shifts: np.ndarray,
    lib_types: Sequence[str],
    exp_peaks: List[Dict[str, Any]],
    col_for_row: np.ndarray,
    hit_threshold_ppm: float,
) -> List[CarbonAssignment]:
    n_p = len(exp_peaks)
    exp_ppms = np.array([float(p["ppm"]) for p in exp_peaks], dtype=np.float64)
    exp_types = [normalize_peak_type(p.get("type")) for p in exp_peaks]
    out: List[CarbonAssignment] = []
    for i in range(len(vir_shifts)):
        ccol = int(col_for_row[i])
        if ccol < n_p:
            ppm_e = float(exp_ppms[ccol])
            res = ppm_e - float(vir_shifts[i])
            hit = abs(res) <= hit_threshold_ppm
            out.append(
                CarbonAssignment(
                    carbon_slot=i,
                    atom_idx=-1,
                    vir_shift=float(vir_shifts[i]),
                    lib_type=lib_types[i],
                    matched_peak_index=ccol,
                    exp_ppm=ppm_e,
                    exp_type=exp_types[ccol],
                    residual_ppm=float(res),
                    hit=hit,
                )
            )
        else:
            out.append(
                CarbonAssignment(
                    carbon_slot=i,
                    atom_idx=-1,
                    vir_shift=float(vir_shifts[i]),
                    lib_type=lib_types[i],
                    matched_peak_index=None,
                    exp_ppm=None,
                    exp_type=None,
                    residual_ppm=None,
                    hit=False,
                )
            )
    return out


def iterative_median_assign(
    vir0: np.ndarray,
    lib_types: Sequence[str],
    exp_peaks: List[Dict[str, Any]],
    options: FragmentAnalysisOptions,
) -> Tuple[np.ndarray, List[float], np.ndarray]:
    """Returns (adjusted vir shifts, list of median corrections per iter, col_for_row)."""
    exp_ppms = np.array([float(p["ppm"]) for p in exp_peaks], dtype=np.float64)
    exp_types = [normalize_peak_type(p.get("type")) for p in exp_peaks]
    vir = np.array(vir0, dtype=np.float64, copy=True)
    medians: List[float] = []
    col_for_row = np.zeros(len(vir), dtype=int)

    max_it = options.median_max_iters if options.use_median_correction else 1
    for it in range(max_it):
        col_for_row = assign_hungarian(
            vir,
            lib_types,
            exp_ppms,
            exp_types,
            use_dept=options.use_dept_constraint,
            cost_cap_ppm=options.cost_cap_ppm,
            unmatched_penalty=options.unmatched_penalty_ppm,
        )
        n_p = len(exp_peaks)
        residuals = []
        for i, ccol in enumerate(col_for_row):
            if ccol < n_p:
                residuals.append(float(exp_ppms[ccol] - vir[i]))
        if not residuals or not options.use_median_correction:
            break
        m = float(np.median(residuals))
        medians.append(m)
        vir = vir + m
        if abs(m) < 1e-6:
            break
    return vir, medians, col_for_row


def get_fragment_atom_sets(mol: Chem.Mol, method: str) -> List[Tuple[int, ...]]:
    """Return disjoint atom-index tuples (heavy atoms) covering the molecule."""
    if method == "whole":
        atoms = [a.GetIdx() for a in mol.GetAtoms()]
        return [tuple(atoms)] if atoms else []

    frags: List[Tuple[int, ...]] = []
    if method == "brics_murcko":
        try:
            nm = BRICS.BreakBRICSBonds(Chem.Mol(mol))
            if nm is None:
                nm = Chem.Mol(mol)
            parts = Chem.GetMolFrags(nm, asMols=False)
            frags = [tuple(sorted(t)) for t in parts if len(t) > 0]
        except Exception:
            frags = []

    if method == "murcko_only" or (method == "brics_murcko" and len(frags) <= 1):
        try:
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            if scaf is not None and scaf.GetNumAtoms() > 0:
                match = mol.GetSubstructMatch(scaf)
                if match:
                    scaffold_atoms = set(match)
                    all_atoms = set(a.GetIdx() for a in mol.GetAtoms())
                    side = all_atoms - scaffold_atoms
                    if side and scaffold_atoms:
                        return [tuple(sorted(scaffold_atoms)), tuple(sorted(side))]
        except Exception:
            pass

    if len(frags) <= 1:
        atoms = [a.GetIdx() for a in mol.GetAtoms()]
        return [tuple(atoms)] if atoms else []
    return frags


def fragment_key_for_atoms(mol: Chem.Mol, atom_indices: Sequence[int]) -> str:
    atoms = sorted(set(int(i) for i in atom_indices))
    if not atoms:
        return ""
    try:
        return Chem.MolFragmentToSmiles(mol, atoms, canonical=True)
    except Exception:
        return "|".join(str(i) for i in atoms)


def aggregate_fragments(
    mol: Chem.Mol,
    assignments: List[CarbonAssignment],
    carbon_atom_idx: List[int],
    options: FragmentAnalysisOptions,
) -> List[FragmentRecord]:
    atom_sets = get_fragment_atom_sets(mol, options.fragment_method)
    carbon_to_atom = {i: carbon_atom_idx[i] for i in range(len(carbon_atom_idx))}

    records: List[FragmentRecord] = []
    for aset in atom_sets:
        aset_s = set(aset)
        slots = [i for i, ai in carbon_to_atom.items() if ai in aset_s]
        if not slots:
            continue
        res_list = [
            abs(a.residual_ppm)
            for a in assignments
            if a.carbon_slot in slots and a.residual_ppm is not None
        ]
        matched = sum(
            1 for a in assignments if a.carbon_slot in slots and a.matched_peak_index is not None
        )
        hits = sum(1 for a in assignments if a.carbon_slot in slots and a.hit)
        n = len(slots)
        mean_abs = float(np.mean(res_list)) if res_list else float("nan")
        med_abs = float(np.median(res_list)) if res_list else float("nan")
        hit_rate = hits / n if n else 0.0
        key = fragment_key_for_atoms(mol, [carbon_to_atom[i] for i in slots])
        records.append(
            FragmentRecord(
                fragment_key=key,
                atom_indices=tuple(sorted(carbon_to_atom[i] for i in slots)),
                carbon_slots=slots,
                n_carbons=n,
                n_matched=matched,
                n_hit=hits,
                mean_abs_residual=mean_abs,
                median_abs_residual=med_abs,
                hit_rate=hit_rate,
            )
        )
    return _filter_fragments_by_size(records, options)


def _filter_fragments_by_size(
    records: List[FragmentRecord], options: FragmentAnalysisOptions
) -> List[FragmentRecord]:
    m = max(1, int(options.min_carbons_per_fragment))
    if m <= 1:
        return records
    return [fr for fr in records if fr.n_carbons >= m]


def analyze_one_candidate(
    rank: int,
    smiles: str,
    dbindex: Any,
    score: float,
    vir_shifts_raw: Any,
    ctype_raw: Any,
    exp_peaks: List[Dict[str, Any]],
    options: FragmentAnalysisOptions,
) -> CandidateFragmentResult:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    vir_list = _listify_shifts(vir_shifts_raw)
    carbon_idx = carbon_atom_indices_in_shift_order(mol)
    n = min(len(vir_list), len(carbon_idx))
    vir_list = vir_list[:n]
    carbon_idx = carbon_idx[:n]
    lib_types = _listify_ctype(ctype_raw, n)

    if options.use_dept_constraint and not exp_peaks:
        raise ValueError("DEPT constraint requires experimental peaks with types.")

    vir0 = np.array(vir_list, dtype=np.float64)
    vir_adj, medians, col_for_row = iterative_median_assign(
        vir0, lib_types, exp_peaks, options
    )
    assigns = build_assignments(
        vir_adj, lib_types, exp_peaks, col_for_row, options.hit_threshold_ppm
    )
    for a in assigns:
        if 0 <= a.carbon_slot < len(carbon_idx):
            a.atom_idx = int(carbon_idx[a.carbon_slot])

    frags = aggregate_fragments(mol, assigns, carbon_idx, options)
    return CandidateFragmentResult(
        rank=rank,
        smiles=smiles,
        dbindex=dbindex,
        score=float(score) if score is not None else float("nan"),
        median_corrections=medians,
        assignments=assigns,
        fragments=frags,
    )


def build_consensus(
    candidates: List[CandidateFragmentResult],
    min_count: int,
) -> List[ConsensusRecord]:
    bucket: Dict[str, List[Tuple[int, FragmentRecord]]] = {}
    for c in candidates:
        for f in c.fragments:
            bucket.setdefault(f.fragment_key, []).append((c.rank, f))

    out: List[ConsensusRecord] = []
    for key, items in bucket.items():
        # One candidate may contribute several disjoint pieces with the same fragment SMILES (e.g. two methyls → "C").
        by_rank: Dict[int, List[FragmentRecord]] = {}
        for rank, fr in items:
            by_rank.setdefault(rank, []).append(fr)
        uniq_ranks = sorted(by_rank.keys())
        if len(uniq_ranks) < min_count:
            continue

        per_rank_hr: List[float] = []
        per_rank_mean: List[float] = []
        for rank in uniq_ranks:
            frs = by_rank[rank]
            nc = sum(f.n_carbons for f in frs)
            nh = sum(f.n_hit for f in frs)
            if nc > 0:
                per_rank_hr.append(nh / nc)
            res_vals: List[float] = []
            for f in frs:
                if f.n_carbons > 0 and not np.isnan(f.mean_abs_residual):
                    res_vals.extend([f.mean_abs_residual] * f.n_carbons)
            if res_vals:
                per_rank_mean.append(float(np.mean(res_vals)))

        out.append(
            ConsensusRecord(
                fragment_key=key,
                count=len(uniq_ranks),
                avg_hit_rate=float(np.mean(per_rank_hr)) if per_rank_hr else 0.0,
                avg_mean_abs_residual=float(np.mean(per_rank_mean)) if per_rank_mean else float("nan"),
                ranks=uniq_ranks,
            )
        )
    out.sort(key=lambda x: (-x.count, -x.avg_hit_rate))
    return out


def run_fragment_analysis(
    result_df: Any,
    experimental_peaks: List[Dict[str, Any]],
    top_k: int,
    options: Optional[FragmentAnalysisOptions] = None,
) -> FragmentAnalysisResult:
    """
    Parameters
    ----------
    result_df : pandas.DataFrame
        Columns: smiles, Vir_shifts, Ctype, score, DBindex (as from ShowTopN).
    experimental_peaks : list of dict
        Each dict: ppm, type (optional), intensity (ignored).
    top_k : int
        Number of top rows to use for per-candidate + consensus.
    """
    import pandas as pd

    if options is None:
        options = FragmentAnalysisOptions()

    if not isinstance(result_df, pd.DataFrame):
        raise TypeError("result_df must be a pandas DataFrame")
    if result_df.empty:
        raise ValueError("result_df is empty")
    if not experimental_peaks:
        raise ValueError("experimental_peaks is empty; load NMR-1D.csv first")

    n = min(top_k, len(result_df))
    rows = result_df.iloc[:n]

    candidates: List[CandidateFragmentResult] = []
    for i in range(n):
        r = rows.iloc[i]
        smiles = str(r.get("smiles", ""))
        if not smiles:
            continue
        cfr = analyze_one_candidate(
            rank=i,
            smiles=smiles,
            dbindex=r.get("DBindex"),
            score=r.get("score", float("nan")),
            vir_shifts_raw=r.get("Vir_shifts"),
            ctype_raw=r.get("Ctype"),
            exp_peaks=experimental_peaks,
            options=options,
        )
        candidates.append(cfr)

    consensus = build_consensus(candidates, options.consensus_min_count)
    return FragmentAnalysisResult(options=options, candidates=candidates, consensus=consensus)
