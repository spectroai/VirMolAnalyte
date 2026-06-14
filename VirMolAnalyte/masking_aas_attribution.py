# -*- coding: utf-8 -*-
"""
Random-subset AAS attribution for virtual 13C shifts vs experimental peaks.

Base idea per iteration:
  S = subset of carbon indices (random ceil(f*n), or k mutually bonded carbons)
  Δ = AAS(predicted shifts restricted to S) - AAS(full molecule)
  For each i in S: acc[i] += Δ / |S|

Final:
  mean_score[i] = acc[i] / n_iterations

Optional matching modes for AAS:
  - use_dept_constraint: only compare d_i to experimental peaks of matching q/t/d/s
  - greedy_unique_matching: match sequentially and remove matched experimental peak
    (to avoid repeated matching to the same peak when possible)
"""

from __future__ import annotations

import math
import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from rdkit import Chem


def normalize_peak_type(raw: Any) -> Optional[str]:
    """Map various labels to q/t/d/s. Unknown returns None (wildcard)."""
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
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6]


def _carbon_slot_neighbors(
    mol: Chem.Mol,
    atom_idx: Sequence[int],
    *,
    allow_hetero_bridge_neighbors: bool = True,
) -> List[List[int]]:
    """
    Carbon-slot adjacency for connected masking (same order as shift list).

    By default, both of the following are treated as adjacent:
      - direct C-C bond
      - C-X-C via one non-hydrogen atom X (e.g., O/N/S)
    """
    n = len(atom_idx)
    slot_of = {int(aid): i for i, aid in enumerate(atom_idx)}
    neigh_sets: List[Set[int]] = [set() for _ in range(n)]
    for i in range(n):
        a = mol.GetAtomWithIdx(int(atom_idx[i]))
        for nb in a.GetNeighbors():
            j = slot_of.get(nb.GetIdx())
            if j is not None and j != i:
                neigh_sets[i].add(int(j))
                continue
            if not bool(allow_hetero_bridge_neighbors):
                continue
            if int(nb.GetAtomicNum()) == 1:
                continue
            # One-hetero bridge: C(i)-X-C(k)
            for nb2 in nb.GetNeighbors():
                if int(nb2.GetIdx()) == int(a.GetIdx()):
                    continue
                k = slot_of.get(nb2.GetIdx())
                if k is not None and int(k) != int(i):
                    neigh_sets[i].add(int(k))
    return [sorted(list(s)) for s in neigh_sets]


def _carbon_connected_components(neighbors: Sequence[Sequence[int]], n: int) -> List[List[int]]:
    """Connected components on the carbon-only adjacency graph (slot indices)."""
    seen: Set[int] = set()
    comps: List[List[int]] = []
    for s in range(n):
        if s in seen:
            continue
        stack = [s]
        comp: List[int] = []
        while stack:
            u = int(stack.pop())
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            for v in neighbors[u]:
                if v not in seen:
                    stack.append(int(v))
        comps.append(comp)
    return comps


def _sample_connected_mask_indices(
    rng: np.random.Generator,
    neighbors: Sequence[Sequence[int]],
    n: int,
    k: int,
    max_retries: int = 400,
) -> Optional[np.ndarray]:
    """
    Sample k distinct slot indices that induce a connected subgraph on the carbon graph.
    Picks a component with at least k nodes, then grows a connected k-set inside it.
    Returns None only if no component has size >= k.
    """
    if n <= 0 or k <= 0:
        return None
    k = min(int(k), n)
    if k == n:
        return np.arange(n, dtype=int)
    if k == 1:
        return np.array([int(rng.integers(0, n))], dtype=int)

    comps = _carbon_connected_components(neighbors, n)
    big = [c for c in comps if len(c) >= k]
    if not big:
        return None

    for _ in range(max_retries):
        comp = list(big[int(rng.integers(0, len(big)))])
        comp_set = set(comp)
        start = int(rng.choice(np.asarray(comp, dtype=int)))
        chosen: List[int] = [start]
        chosen_set: Set[int] = {start}
        while len(chosen) < k:
            cand: List[int] = []
            seen_c: Set[int] = set()
            for i in chosen_set:
                for j in neighbors[i]:
                    if j not in comp_set or j in chosen_set:
                        continue
                    if j not in seen_c:
                        seen_c.add(int(j))
                        cand.append(int(j))
            if not cand:
                break
            pick = int(rng.choice(np.asarray(cand, dtype=int)))
            chosen.append(pick)
            chosen_set.add(pick)
        if len(chosen) == k:
            return np.asarray(chosen, dtype=int)
    return None


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
    out: List[str] = []
    for i in range(n):
        c = seq[i] if i < len(seq) else "s"
        c = normalize_peak_type(c) or "s"
        out.append(c)
    return out


def compute_aas(
    virtual_shifts: Sequence[float],
    experimental_ppms: Sequence[float],
    *,
    lib_types: Optional[Sequence[str]] = None,
    exp_types: Optional[Sequence[Optional[str]]] = None,
    use_dept_constraint: bool = False,
    greedy_unique_matching: bool = False,
) -> float:
    """
    AAS = (2/pi)*atan(1/MSE)*100
    MSE averages squared nearest distance for each virtual shift.

    Matching mode:
      - default: independent nearest (no removal)
      - greedy_unique_matching=True: sequential nearest with peak removal
    """
    d_list = [float(x) for x in virtual_shifts]
    c_list = [float(x) for x in experimental_ppms]
    n = len(d_list)
    if n == 0 or len(c_list) == 0:
        return float("nan")

    if lib_types is None:
        lib_types = ["s"] * n
    else:
        lib_types = [normalize_peak_type(x) or "s" for x in lib_types[:n]]

    if exp_types is None:
        exp_types = [None] * len(c_list)
    else:
        exp_types = [normalize_peak_type(x) for x in exp_types[: len(c_list)]]

    def eligible(peak_idx: int, lib_t: str) -> bool:
        if not use_dept_constraint:
            return True
        t = exp_types[peak_idx]
        return (t is None) or (t == lib_t)

    min_delta_sum = 0.0
    if not greedy_unique_matching:
        for i, d in enumerate(d_list):
            lib_t = lib_types[i]
            cand = [abs(d - c_list[j]) for j in range(len(c_list)) if eligible(j, lib_t)]
            if not cand:
                # If DEPT removes all peaks, degrade gracefully by falling back to all peaks.
                cand = [abs(d - c) for c in c_list]
            min_delta_sum += min(cand) ** 2
    else:
        available = list(range(len(c_list)))
        for i, d in enumerate(d_list):
            lib_t = lib_types[i]
            elig = [j for j in available if eligible(j, lib_t)]
            if not elig:
                # If no eligible available peak, relax DEPT first, then availability.
                if available:
                    elig = list(available)
                else:
                    elig = list(range(len(c_list)))
            best_j = min(elig, key=lambda j: abs(d - c_list[j]))
            min_delta_sum += abs(d - c_list[best_j]) ** 2
            if best_j in available:
                available.remove(best_j)

    mse = min_delta_sum / n
    if mse <= 0.0:
        mse = 1e-12
    return float((2.0 * math.atan(1.0 / mse) / math.pi) * 100.0)


@dataclass
class MaskingAttributionOptions:
    mask_fraction: float = 0.5
    n_iterations: int = 100
    random_seed: Optional[int] = None
    use_dept_constraint: bool = False
    greedy_unique_matching: bool = False
    #: When greedy_unique_matching is True, permute virtual shift / ctype order before each AAS call.
    shuffle_before_each_aas_if_greedy: bool = True
    #: "random_fraction": mask ceil(f*n) arbitrary carbons per iteration.
    #: "random_connected": mask k mutually bonded carbons (connected subgraph) per iteration.
    mask_mode: str = "random_fraction"
    #: Used when mask_mode == "random_connected"; must be >= 1.
    connected_mask_size: int = 3
    #: In random_connected mode, treat C-X-C (X non-hydrogen) as connected.
    allow_hetero_bridge_neighbors: bool = True


@dataclass
class MaskingAttributionResult:
    smiles: str
    candidate_rank_index: int
    mean_scores: List[float]
    carbon_atom_indices: List[int]
    vir_shifts: List[float]
    lib_types: List[str]
    aas_full: float
    n_carbons: int
    n_iterations: int
    mask_fraction: float
    mask_mode: str = "random_fraction"
    connected_mask_size: int = 3

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "smiles": self.smiles,
            "candidate_rank_index": self.candidate_rank_index,
            "mean_scores": [float(x) for x in self.mean_scores],
            "carbon_atom_indices": [int(x) for x in self.carbon_atom_indices],
            "vir_shifts": [float(x) for x in self.vir_shifts],
            "lib_types": [str(x) for x in self.lib_types],
            "aas_full": float(self.aas_full),
            "n_carbons": self.n_carbons,
            "n_iterations": self.n_iterations,
            "mask_fraction": float(self.mask_fraction),
            "mask_mode": str(self.mask_mode),
            "connected_mask_size": int(self.connected_mask_size),
        }


def _match_pred_to_exp_pairs(
    pred_ppms: Sequence[float],
    pred_types: Sequence[str],
    experimental_peaks: Sequence[Dict[str, Any]],
    *,
    use_dept_constraint: bool = False,
    greedy_unique_matching: bool = True,
) -> List[Dict[str, Any]]:
    """Return per-prediction assignment pairs to experimental peaks."""
    exp_ppms = [float(p["ppm"]) for p in experimental_peaks]
    exp_types = [normalize_peak_type(p.get("type")) for p in experimental_peaks]
    if not exp_ppms:
        return []

    pred_vals = [float(x) for x in pred_ppms]
    pred_t = [normalize_peak_type(x) or "s" for x in pred_types[: len(pred_vals)]]
    if len(pred_t) < len(pred_vals):
        pred_t.extend(["s"] * (len(pred_vals) - len(pred_t)))

    def eligible(peak_idx: int, lib_t: str) -> bool:
        if not use_dept_constraint:
            return True
        t = exp_types[peak_idx]
        return (t is None) or (t == lib_t)

    out: List[Dict[str, Any]] = []
    if not greedy_unique_matching:
        for i, d in enumerate(pred_vals):
            lib_t = pred_t[i]
            cand = [j for j in range(len(exp_ppms)) if eligible(j, lib_t)]
            if not cand:
                cand = list(range(len(exp_ppms)))
            best_j = min(cand, key=lambda j: abs(d - exp_ppms[j]))
            out.append(
                {
                    "pred_local_index": i,
                    "exp_index": int(best_j),
                    "pred_ppm": float(d),
                    "exp_ppm": float(exp_ppms[best_j]),
                    "abs_err": float(abs(d - exp_ppms[best_j])),
                }
            )
        return out

    available = list(range(len(exp_ppms)))
    for i, d in enumerate(pred_vals):
        lib_t = pred_t[i]
        elig = [j for j in available if eligible(j, lib_t)]
        if not elig:
            if available:
                elig = list(available)
            else:
                elig = list(range(len(exp_ppms)))
        best_j = min(elig, key=lambda j: abs(d - exp_ppms[j]))
        out.append(
            {
                "pred_local_index": i,
                "exp_index": int(best_j),
                "pred_ppm": float(d),
                "exp_ppm": float(exp_ppms[best_j]),
                "abs_err": float(abs(d - exp_ppms[best_j])),
            }
        )
        if best_j in available:
            available.remove(best_j)
    return out


def _pairs_raw_to_fragment_rows(
    pairs_raw: Sequence[Dict[str, Any]],
    order: np.ndarray,
    comp: Sequence[int],
    comp_atom_idx: Sequence[int],
    comp_types: Sequence[str],
) -> List[Dict[str, Any]]:
    """Map greedy output (order of processing) back to fragment-local atom rows."""
    pair_rows: List[Dict[str, Any]] = []
    for pair in pairs_raw:
        step_i = int(pair["pred_local_index"])
        orig_pos = int(order[step_i])
        pair_rows.append(
            {
                "shift_index": int(comp[orig_pos]),
                "atom_index": int(comp_atom_idx[orig_pos]),
                "ctype": str(comp_types[orig_pos]),
                "pred_ppm": float(pair["pred_ppm"]),
                "exp_index": int(pair["exp_index"]),
                "exp_ppm": float(pair["exp_ppm"]),
                "abs_err": float(pair["abs_err"]),
            }
        )
    return pair_rows


def _best_greedy_shuffle_pairs(
    comp_shifts: Sequence[float],
    comp_types: Sequence[str],
    experimental_peaks: Sequence[Dict[str, Any]],
    *,
    use_dept_constraint: bool,
    greedy_unique_matching: bool,
    greedy_shuffle_repeats: int,
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, Any]], np.ndarray, float, int]:
    """
    For greedy unique matching, optionally repeat with random permutations of
    virtual-shift processing order; keep the assignment with lowest Σ(Δδ)²
    (same MSE term as AAS uses for nearest matching).

    Returns (pairs_raw, order, mse_sum, trials_used).
    """
    k = len(comp_shifts)
    if k == 0:
        return [], np.zeros(0, dtype=int), float("nan"), 0

    if not greedy_unique_matching:
        pairs = _match_pred_to_exp_pairs(
            comp_shifts,
            comp_types,
            experimental_peaks,
            use_dept_constraint=use_dept_constraint,
            greedy_unique_matching=False,
        )
        order = np.arange(k, dtype=int)
        mse = float(sum(float(p["abs_err"]) ** 2 for p in pairs))
        return pairs, order, mse, 1

    n_rep = max(1, int(greedy_shuffle_repeats))
    if n_rep <= 1:
        order = np.arange(k, dtype=int)
        ord_sh = [float(comp_shifts[i]) for i in range(k)]
        ord_ty = [str(comp_types[i]) for i in range(k)]
        pairs = _match_pred_to_exp_pairs(
            ord_sh,
            ord_ty,
            experimental_peaks,
            use_dept_constraint=use_dept_constraint,
            greedy_unique_matching=True,
        )
        mse = float(sum(float(p["abs_err"]) ** 2 for p in pairs))
        return pairs, order, mse, 1

    best_mse = float("inf")
    best_pairs: Optional[List[Dict[str, Any]]] = None
    best_order: Optional[np.ndarray] = None
    trials_used = 0
    for _ in range(n_rep):
        order = rng.permutation(k)
        ord_sh = [float(comp_shifts[int(order[i])]) for i in range(k)]
        ord_ty = [str(comp_types[int(order[i])]) for i in range(k)]
        pairs = _match_pred_to_exp_pairs(
            ord_sh,
            ord_ty,
            experimental_peaks,
            use_dept_constraint=use_dept_constraint,
            greedy_unique_matching=True,
        )
        mse = float(sum(float(p["abs_err"]) ** 2 for p in pairs))
        trials_used += 1
        if mse < best_mse - 1e-15:
            best_mse = mse
            best_pairs = pairs
            best_order = order.copy()
    assert best_pairs is not None and best_order is not None
    return best_pairs, best_order, best_mse, trials_used


def extract_positive_fragments(
    smiles: str,
    carbon_atom_indices: Sequence[int],
    mean_scores: Sequence[float],
    vir_shifts: Sequence[float],
    lib_types: Optional[Sequence[str]],
    experimental_peaks: Sequence[Dict[str, Any]],
    *,
    score_threshold: float = 0.0,
    min_carbons: int = 2,
    use_dept_constraint: bool = False,
    greedy_unique_matching: bool = True,
    bridge_max_low_carbons: int = 0,
    allow_hetero_bridge_neighbors: bool = True,
    max_fragments: int = 50,
    greedy_shuffle_repeats: int = 0,
    greedy_shuffle_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Extract connected carbon fragments from positive-score carbons and map each
    fragment's predicted shifts to experimental peaks.

    If bridge_max_low_carbons > 0, merge high-score clusters that can be
    connected through up to N low-score carbons (N = bridge_max_low_carbons),
    and include those bridge carbons in fragment-level spectrum matching.

    If allow_hetero_bridge_neighbors is True (default), carbon nodes are also
    considered adjacent when they are connected as C-X-C through one hetero
    atom (e.g., O/N/S). This helps fragment connectivity span hetero links.

    If greedy_unique_matching is True and greedy_shuffle_repeats > 1, repeat
    greedy matching with random permutations of virtual-shift order; keep the
    assignment with minimum Σ(Δδ)² (MSE term aligned with AAS).
    """
    rng = np.random.default_rng(greedy_shuffle_seed)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    atom_idx = [int(x) for x in carbon_atom_indices]
    scores = [float(x) for x in mean_scores]
    shifts = [float(x) for x in vir_shifts]
    n = min(len(atom_idx), len(scores), len(shifts))
    if n == 0:
        return []

    atom_idx = atom_idx[:n]
    scores = scores[:n]
    shifts = shifts[:n]
    ctypes = (
        _listify_ctype(lib_types, n) if lib_types is not None else ["s"] * n
    )
    slot_of_atom = {aid: i for i, aid in enumerate(atom_idx)}

    selected_slots: Set[int] = {
        i for i, s in enumerate(scores) if np.isfinite(s) and float(s) > float(score_threshold)
    }
    if not selected_slots:
        return []

    low_slots: Set[int] = set(range(n)) - selected_slots

    # Neighbors among tracked carbons (full subgraph). By default, include
    # C-X-C (one hetero atom) as adjacency for extraction connectivity.
    full_neigh: Dict[int, List[int]] = {i: [] for i in range(n)}
    neigh_sets: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        a = mol.GetAtomWithIdx(int(atom_idx[i]))
        for nb in a.GetNeighbors():
            j = slot_of_atom.get(nb.GetIdx())
            if j is not None:
                neigh_sets[i].add(int(j))
            elif bool(allow_hetero_bridge_neighbors):
                # Through one hetero atom: C(i)-X-C(k) -> connect i and k.
                if int(nb.GetAtomicNum()) == 1:
                    continue
                for nb2 in nb.GetNeighbors():
                    if int(nb2.GetIdx()) == int(a.GetIdx()):
                        continue
                    k = slot_of_atom.get(nb2.GetIdx())
                    if k is not None and int(k) != int(i):
                        neigh_sets[i].add(int(k))
    for i in range(n):
        full_neigh[i] = sorted(neigh_sets[i])

    parent = {i: i for i in selected_slots}

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for i in selected_slots:
        for j in full_neigh[i]:
            if j in selected_slots and i < j:
                _union(i, j)

    bmax = max(0, int(bridge_max_low_carbons))
    bridge_paths: List[Tuple[int, int, Set[int]]] = []
    if bmax > 0:
        # BFS from each high-score node through at most bmax low-score nodes.
        high_list = sorted(selected_slots)
        for src in high_list:
            q: List[Tuple[int, int, Tuple[int, ...]]] = [(src, 0, tuple())]
            seen: Set[Tuple[int, int]] = {(src, 0)}
            while q:
                cur, used_low, low_path = q.pop(0)
                for nb in full_neigh[cur]:
                    if nb in selected_slots:
                        if nb != src:
                            _union(src, nb)
                            bridge_paths.append((src, nb, set(low_path)))
                        continue
                    # traverse low-score node
                    if used_low >= bmax:
                        continue
                    st = (nb, used_low + 1)
                    if st in seen:
                        continue
                    seen.add(st)
                    q.append((nb, used_low + 1, low_path + (nb,)))

    comp_map: Dict[int, List[int]] = {}
    for i in selected_slots:
        r = _find(i)
        comp_map.setdefault(r, []).append(i)
    components = [sorted(v) for v in comp_map.values()]
    bridge_low_by_root: Dict[int, Set[int]] = {}
    for a, b, lows in bridge_paths:
        if not lows:
            continue
        ra, rb = _find(a), _find(b)
        if ra != rb:
            continue
        bridge_low_by_root.setdefault(ra, set()).update(int(x) for x in lows)

    out: List[Dict[str, Any]] = []
    for comp in components:
        if len(comp) < max(1, int(min_carbons)):
            continue
        root = _find(comp[0])
        bridge_slots = sorted(int(x) for x in bridge_low_by_root.get(root, set()))
        match_slots = sorted(set(comp) | set(bridge_slots))

        comp_atom_idx = [int(atom_idx[i]) for i in match_slots]
        comp_shifts = [float(shifts[i]) for i in match_slots]
        comp_scores = [float(scores[i]) for i in match_slots]
        comp_types = [ctypes[i] for i in match_slots]

        pairs_raw, order, match_mse, shuffle_trials = _best_greedy_shuffle_pairs(
            comp_shifts,
            comp_types,
            experimental_peaks,
            use_dept_constraint=use_dept_constraint,
            greedy_unique_matching=greedy_unique_matching,
            greedy_shuffle_repeats=greedy_shuffle_repeats,
            rng=rng,
        )
        pair_rows = _pairs_raw_to_fragment_rows(
            pairs_raw,
            order,
            match_slots,
            comp_atom_idx,
            comp_types,
        )

        abs_errs = [float(p["abs_err"]) for p in pair_rows]
        unique_exp_idx = sorted({int(p["exp_index"]) for p in pair_rows})
        out.append(
            {
                "fragment_id": 0,
                "atom_indices": comp_atom_idx,
                "shift_indices": [int(i) for i in match_slots],
                "core_high_shift_indices": [int(i) for i in comp],
                "bridge_shift_indices": [int(i) for i in bridge_slots],
                "n_carbons": int(len(match_slots)),
                "n_core_high_carbons": int(len(comp)),
                "score_sum": float(np.sum(comp_scores)),
                "score_mean": float(np.mean(comp_scores)),
                "pred_shifts": comp_shifts,
                "types": comp_types,
                "pairs": pair_rows,
                "matched_exp_indices": unique_exp_idx,
                "matched_exp_ppms": [float(experimental_peaks[j]["ppm"]) for j in unique_exp_idx],
                "mean_abs_err": float(np.mean(abs_errs)) if abs_errs else float("nan"),
                "hit_rate_unique": float(len(unique_exp_idx) / max(1, len(comp))),
                "match_mse": float(match_mse),
                "greedy_shuffle_trials": int(shuffle_trials),
            }
        )

    out.sort(key=lambda d: (d["score_sum"], d["n_carbons"]), reverse=True)
    mf = max(1, int(max_fragments))
    out = out[:mf]
    for i, row in enumerate(out, start=1):
        row["fragment_id"] = i
    return out


def run_fragment_fusion(
    fragments_all: Sequence[Sequence[Dict[str, Any]]],
    experimental_peaks: Sequence[Dict[str, Any]],
    *,
    top_fragment_pool: int = 30,
    max_merge_fragments: int = 5,
    greedy_shuffle_repeats: int = 10,
    coverage_weight: float = 0.3,
    top_k_results: int = 10,
    use_dept_constraint: bool = True,
    random_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Enumerate merged fragment combinations, score merged virtual shifts vs experimental
    peaks (greedy shuffle matching + AAS + coverage), and return top combinations.

    Pass ``fragments_all`` with one element to restrict the pool to fragments from a
    single compound (intra-molecular fusion); multiple elements allow cross-compound
    combinations. Caller should tag ``compound_idx`` in outputs if a single-compound
    pool must map to a global candidate index (see GUI intra-molecular fusion).
    """
    exp = list(experimental_peaks)
    nb = len(exp)
    if nb <= 0:
        return []

    flat: List[Dict[str, Any]] = []
    for ci, frags in enumerate(fragments_all):
        for frag in frags:
            pred = [float(x) for x in frag.get("pred_shifts", [])]
            types = [str(x) for x in frag.get("types", [])]
            shift_idx = [int(x) for x in frag.get("shift_indices", [])]
            atom_idx = [int(x) for x in frag.get("atom_indices", [])]
            n = min(len(pred), len(types), len(shift_idx), len(atom_idx))
            if n <= 0:
                continue
            flat.append(
                {
                    "compound_idx": int(ci),
                    "fragment_id": int(frag.get("fragment_id", 0)),
                    "score_sum": float(frag.get("score_sum", 0.0)),
                    "pred_shifts": pred[:n],
                    "types": types[:n],
                    "shift_indices": shift_idx[:n],
                    "atom_indices": atom_idx[:n],
                }
            )
    if not flat:
        return []
    flat.sort(key=lambda x: x["score_sum"], reverse=True)
    pool = flat[: max(1, int(top_fragment_pool))]

    rng = np.random.default_rng(random_seed)
    max_merge = max(1, int(max_merge_fragments))
    top_k = max(1, int(top_k_results))
    cov_w = float(coverage_weight)
    out: List[Dict[str, Any]] = []

    for k in range(1, min(max_merge, len(pool)) + 1):
        for combo in itertools.combinations(range(len(pool)), k):
            merged_pred: List[float] = []
            merged_types: List[str] = []
            meta: List[Tuple[int, int, int, int, str]] = []
            combo_info: List[Dict[str, Any]] = []
            for pi in combo:
                pf = pool[pi]
                combo_info.append(
                    {
                        "compound_idx": int(pf["compound_idx"]),
                        "fragment_id": int(pf["fragment_id"]),
                        "score_sum": float(pf["score_sum"]),
                        "n_carbons": int(len(pf["pred_shifts"])),
                        "atom_indices": [int(x) for x in pf["atom_indices"]],
                        "shift_indices": [int(x) for x in pf["shift_indices"]],
                    }
                )
                for j, d in enumerate(pf["pred_shifts"]):
                    merged_pred.append(float(d))
                    merged_types.append(str(pf["types"][j]))
                    meta.append(
                        (
                            int(pf["compound_idx"]),
                            int(pf["fragment_id"]),
                            int(pf["shift_indices"][j]),
                            int(pf["atom_indices"][j]),
                            str(pf["types"][j]),
                        )
                    )

            na = len(merged_pred)
            if na <= 0 or na > nb:
                continue

            pairs_raw, order, sse, trials = _best_greedy_shuffle_pairs(
                merged_pred,
                merged_types,
                exp,
                use_dept_constraint=use_dept_constraint,
                greedy_unique_matching=True,
                greedy_shuffle_repeats=max(1, int(greedy_shuffle_repeats)),
                rng=rng,
            )
            pair_rows: List[Dict[str, Any]] = []
            for pair in pairs_raw:
                step_i = int(pair["pred_local_index"])
                orig_i = int(order[step_i])
                ci, fid, sidx, aidx, ct = meta[orig_i]
                pair_rows.append(
                    {
                        "compound_idx": ci,
                        "fragment_id": fid,
                        "shift_index": sidx,
                        "atom_index": aidx,
                        "ctype": ct,
                        "pred_ppm": float(pair["pred_ppm"]),
                        "exp_index": int(pair["exp_index"]),
                        "exp_ppm": float(pair["exp_ppm"]),
                        "abs_err": float(pair["abs_err"]),
                    }
                )

            mse = float(sse) / float(max(1, na))
            if mse <= 0.0:
                mse = 1e-12
            aas = float((2.0 * math.atan(1.0 / mse) / math.pi) * 100.0)
            cover = float(na) / float(nb)
            final_score = aas + cov_w * (cover * 100.0)
            out.append(
                {
                    "combo_size": int(k),
                    "na": int(na),
                    "nb": int(nb),
                    "coverage": float(cover),
                    "aas_best": float(aas),
                    "score_final": float(final_score),
                    "sse": float(sse),
                    "shuffle_trials": int(trials),
                    "fragments": combo_info,
                    "pairs": pair_rows,
                }
            )

    out.sort(key=lambda d: d["score_final"], reverse=True)
    return out[:top_k]


def run_masking_attribution(
    smiles: str,
    vir_shifts_raw: Any,
    experimental_peaks: List[Dict[str, Any]],
    options: Optional[MaskingAttributionOptions] = None,
    candidate_rank_index: int = 0,
    ctype_raw: Any = None,
) -> MaskingAttributionResult:
    if options is None:
        options = MaskingAttributionOptions()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    vir = np.array(_listify_shifts(vir_shifts_raw), dtype=np.float64)
    atom_idx = carbon_atom_indices_in_shift_order(mol)
    n = min(len(vir), len(atom_idx))
    if n == 0:
        raise ValueError("No carbon shifts or no carbons in molecule.")
    vir = vir[:n]
    atom_idx = atom_idx[:n]
    lib_types = _listify_ctype(ctype_raw, n)

    exp_ppms = [float(p["ppm"]) for p in experimental_peaks]
    exp_types = [normalize_peak_type(p.get("type")) for p in experimental_peaks]
    if not exp_ppms:
        raise ValueError("No experimental peaks.")

    rng = np.random.default_rng(options.random_seed)
    gu = bool(options.greedy_unique_matching)
    shuf = bool(getattr(options, "shuffle_before_each_aas_if_greedy", True))

    def _shuffle_aligned_for_greedy(
        shift_vals: Sequence[float], ctypes: Sequence[str]
    ) -> Tuple[List[float], List[str]]:
        sl = [float(x) for x in shift_vals]
        lt = [str(ctypes[i]) for i in range(len(sl))]
        if not gu or not shuf or len(sl) <= 1:
            return sl, lt
        perm = rng.permutation(len(sl))
        return [sl[int(i)] for i in perm], [lt[int(i)] for i in perm]

    v_full, lt_full = _shuffle_aligned_for_greedy(vir.tolist(), lib_types)
    aas_full = compute_aas(
        v_full,
        exp_ppms,
        lib_types=lt_full,
        exp_types=exp_types,
        use_dept_constraint=options.use_dept_constraint,
        greedy_unique_matching=options.greedy_unique_matching,
    )
    if math.isnan(aas_full):
        raise ValueError("AAS(full) is undefined.")

    acc = np.zeros(n, dtype=np.float64)
    frac = float(np.clip(options.mask_fraction, 0.01, 0.99))
    R = max(1, int(options.n_iterations))
    mode = str(getattr(options, "mask_mode", "random_fraction") or "random_fraction").strip().lower()
    if mode not in ("random_fraction", "random_connected"):
        mode = "random_fraction"
    k_conn = max(1, int(getattr(options, "connected_mask_size", 3)))
    slot_neighbors = _carbon_slot_neighbors(
        mol,
        atom_idx,
        allow_hetero_bridge_neighbors=bool(
            getattr(options, "allow_hetero_bridge_neighbors", True)
        ),
    )

    for _ in range(R):
        if mode == "random_connected":
            idxs_arr = _sample_connected_mask_indices(rng, slot_neighbors, n, k_conn)
            if idxs_arr is None:
                continue
            idxs = idxs_arr
        else:
            k_sub = max(1, int(math.ceil(frac * n)))
            k_sub = min(k_sub, n)
            idxs = rng.choice(n, size=k_sub, replace=False)
        subset = vir[idxs]
        subset_types = [lib_types[int(i)] for i in idxs]
        vs, lts = _shuffle_aligned_for_greedy(subset.tolist(), subset_types)
        aas_sel = compute_aas(
            vs,
            exp_ppms,
            lib_types=lts,
            exp_types=exp_types,
            use_dept_constraint=options.use_dept_constraint,
            greedy_unique_matching=options.greedy_unique_matching,
        )
        if math.isnan(aas_sel):
            continue
        delta = aas_sel - aas_full
        k_mask = int(len(idxs))
        share = delta / float(k_mask)
        for i in idxs:
            acc[int(i)] += share

    mean_scores = (acc / R).tolist()
    return MaskingAttributionResult(
        smiles=smiles,
        candidate_rank_index=candidate_rank_index,
        mean_scores=mean_scores,
        carbon_atom_indices=[int(x) for x in atom_idx],
        vir_shifts=[float(x) for x in vir],
        lib_types=list(lib_types),
        aas_full=aas_full,
        n_carbons=n,
        n_iterations=R,
        mask_fraction=frac,
        mask_mode=mode,
        connected_mask_size=int(getattr(options, "connected_mask_size", 3)),
    )


def scores_to_atom_rgb(
    mol: Chem.Mol,
    carbon_atom_indices: Sequence[int],
    mean_scores: Sequence[float],
    cmap_name: str = "RdYlGn",
    *,
    score_vmin: Optional[float] = None,
    score_vmax: Optional[float] = None,
) -> Dict[int, Tuple[float, float, float]]:
    """
    Map mean attribution scores to RGB in [0,1] for RDKit highlightAtomColors.

    If score_vmin and score_vmax are both set (score_vmin < score_vmax), use this
    fixed range for all molecules (cross-molecule comparable).
    """
    import matplotlib.colors as mcolors

    try:
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap(cmap_name)
    except Exception:
        import matplotlib.cm as cm

        cmap = cm.get_cmap(cmap_name)

    scores = np.array(mean_scores, dtype=np.float64)
    if len(scores) != len(carbon_atom_indices):
        raise ValueError("scores and carbon_atom_indices length mismatch")

    use_fixed = (
        score_vmin is not None
        and score_vmax is not None
        and float(score_vmax) > float(score_vmin)
    )
    if use_fixed:
        lo, hi = float(score_vmin), float(score_vmax)
        cnorm = mcolors.Normalize(vmin=lo, vmax=hi)
    else:
        finite = scores[np.isfinite(scores)]
        if finite.size == 0:
            cnorm = mcolors.Normalize(vmin=0.0, vmax=1.0)
        else:
            if finite.size >= 5:
                lo, hi = float(np.percentile(finite, 5)), float(np.percentile(finite, 95))
                if hi - lo < 1e-9:
                    lo, hi = float(np.min(finite)), float(np.max(finite))
            else:
                lo, hi = float(np.min(finite)), float(np.max(finite))
            if hi - lo < 1e-12:
                cnorm = mcolors.Normalize(vmin=lo - 1.0, vmax=hi + 1.0)
            else:
                cnorm = mcolors.Normalize(vmin=lo, vmax=hi)

    out: Dict[int, Tuple[float, float, float]] = {}
    for aid, s in zip(carbon_atom_indices, scores):
        if not np.isfinite(s):
            out[int(aid)] = (0.75, 0.75, 0.75)
        else:
            t = float(np.clip(s, lo, hi)) if use_fixed else float(s)
            rgba = cmap(cnorm(t))
            out[int(aid)] = (float(rgba[0]), float(rgba[1]), float(rgba[2]))
    return out


def draw_mol_attribution_png(
    smiles: str,
    carbon_atom_indices: Sequence[int],
    mean_scores: Sequence[float],
    vir_shifts: Sequence[float],
    size: Tuple[int, int] = (520, 420),
    *,
    score_vmin: Optional[float] = None,
    score_vmax: Optional[float] = None,
    show_rdkit_atom_index: bool = False,
    show_pred_shift_label: bool = False,
    show_matched_shift_label: bool = False,
    matched_ppm_by_atom: Optional[Dict[int, float]] = None,
    selected_atom_indices: Optional[Sequence[int]] = None,
    selected_atom_color: Optional[Tuple[float, float, float]] = None,
    fragment_highlight_groups: Optional[
        Sequence[Tuple[Sequence[int], Tuple[float, float, float]]]
    ] = None,
    annotation_font_scale: float = 0.65,
    use_score_coloring: bool = True,
    rotate_deg: float = 0.0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
) -> Any:
    """Return PIL Image with per-carbon highlight colors and optional atom labels.

    If ``fragment_highlight_groups`` is set, each ``(atom_indices, rgb)`` group paints
    those carbons with the given RGB (0–1) and a larger highlight radius; used for
    intra-molecular fusion (multiple fragments on one structure). Mutually exclusive
    in practice with ``selected_atom_indices`` / ``selected_atom_color`` (groups win).
    """
    import io
    from PIL import Image
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    if use_score_coloring:
        hm = scores_to_atom_rgb(
            mol,
            carbon_atom_indices,
            mean_scores,
            score_vmin=score_vmin,
            score_vmax=score_vmax,
        )
    else:
        hm = {int(aid): (0.92, 0.92, 0.92) for aid in carbon_atom_indices}
    highlight_atoms = sorted(int(k) for k in hm.keys())
    if not highlight_atoms:
        raise ValueError("No atoms to highlight")

    mol_draw = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol_draw)
    if abs(float(rotate_deg)) > 1e-9 or bool(flip_horizontal) or bool(flip_vertical):
        conf = mol_draw.GetConformer()
        xs = [conf.GetAtomPosition(i).x for i in range(mol_draw.GetNumAtoms())]
        ys = [conf.GetAtomPosition(i).y for i in range(mol_draw.GetNumAtoms())]
        cx = float(np.mean(xs)) if xs else 0.0
        cy = float(np.mean(ys)) if ys else 0.0
        ang = math.radians(-float(rotate_deg))  # clockwise positive
        ca, sa = math.cos(ang), math.sin(ang)
        for i in range(mol_draw.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            x = p.x - cx
            y = p.y - cy
            if flip_horizontal:
                x = -x
            if flip_vertical:
                y = -y
            xr = x * ca - y * sa
            yr = x * sa + y * ca
            conf.SetAtomPosition(i, Point3D(float(xr), float(yr), float(p.z)))
    w, h = int(size[0]), int(size[1])
    d2d = rdMolDraw2D.MolDraw2DCairo(w, h)
    opts = d2d.drawOptions()
    opts.clearBackground = True
    try:
        opts.useBWAtomPalette()
    except Exception:
        pass
    try:
        opts.setBackgroundColour((1.0, 1.0, 1.0))
    except Exception:
        pass

    if show_rdkit_atom_index or show_pred_shift_label or show_matched_shift_label:
        # RDKit expects IntStringMap, not a Python dict (dict assignment fails silently
        # if caught — labels would never appear).
        vir_list = list(vir_shifts)
        matched_map = matched_ppm_by_atom or {}
        amap = rdMolDraw2D.IntStringMap()
        for slot, aid in enumerate(int(x) for x in carbon_atom_indices):
            parts: List[str] = []
            if show_rdkit_atom_index:
                parts.append(str(aid))
            if show_pred_shift_label and slot < len(vir_list):
                parts.append(f"{float(vir_list[slot]):.1f}")
            if show_matched_shift_label and int(aid) in matched_map:
                parts.append(f"{float(matched_map[int(aid)]):.1f}")
            if parts:
                amap[int(aid)] = "\n".join(parts)
        opts.atomLabels = amap
        opts.annotationFontScale = float(annotation_font_scale)

    if fragment_highlight_groups:
        radii: Dict[int, float] = {aid: 0.35 for aid in highlight_atoms}
        for atom_list, rgb in fragment_highlight_groups:
            r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
            for aid in atom_list:
                ia = int(aid)
                if ia in hm:
                    hm[ia] = (r, g, b)
                    radii[ia] = 0.55
        rdMolDraw2D.PrepareAndDrawMolecule(
            d2d,
            mol_draw,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=hm,
            highlightAtomRadii=radii,
        )
    elif selected_atom_indices:
        sel = {int(x) for x in selected_atom_indices}
        radii = {}
        for aid in highlight_atoms:
            radii[aid] = 0.55 if aid in sel else 0.35
        sel_color = selected_atom_color if selected_atom_color is not None else (0.10, 0.40, 0.95)
        for aid in sel:
            if aid in hm:
                hm[aid] = (
                    float(sel_color[0]),
                    float(sel_color[1]),
                    float(sel_color[2]),
                )
        rdMolDraw2D.PrepareAndDrawMolecule(
            d2d,
            mol_draw,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=hm,
            highlightAtomRadii=radii,
        )
    else:
        rdMolDraw2D.PrepareAndDrawMolecule(
            d2d,
            mol_draw,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=hm,
        )
    d2d.FinishDrawing()
    png = d2d.GetDrawingText()
    return Image.open(io.BytesIO(png))


def draw_mol_attribution_svg(
    smiles: str,
    carbon_atom_indices: Sequence[int],
    mean_scores: Sequence[float],
    vir_shifts: Sequence[float],
    size: Tuple[int, int] = (1200, 900),
    *,
    score_vmin: Optional[float] = None,
    score_vmax: Optional[float] = None,
    show_rdkit_atom_index: bool = False,
    show_pred_shift_label: bool = False,
    show_matched_shift_label: bool = False,
    matched_ppm_by_atom: Optional[Dict[int, float]] = None,
    selected_atom_indices: Optional[Sequence[int]] = None,
    selected_atom_color: Optional[Tuple[float, float, float]] = None,
    fragment_highlight_groups: Optional[
        Sequence[Tuple[Sequence[int], Tuple[float, float, float]]]
    ] = None,
    annotation_font_scale: float = 0.65,
    atom_symbol_font_scale: float = 2.0,
    use_score_coloring: bool = True,
    rotate_deg: float = 0.0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
) -> str:
    """Return SVG text for the attribution structure (vector export)."""
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Geometry import Point3D

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    if use_score_coloring:
        hm = scores_to_atom_rgb(
            mol,
            carbon_atom_indices,
            mean_scores,
            score_vmin=score_vmin,
            score_vmax=score_vmax,
        )
    else:
        hm = {int(aid): (0.92, 0.92, 0.92) for aid in carbon_atom_indices}
    highlight_atoms = sorted(int(k) for k in hm.keys())
    if not highlight_atoms:
        raise ValueError("No atoms to highlight")

    mol_draw = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol_draw)
    if abs(float(rotate_deg)) > 1e-9 or bool(flip_horizontal) or bool(flip_vertical):
        conf = mol_draw.GetConformer()
        xs = [conf.GetAtomPosition(i).x for i in range(mol_draw.GetNumAtoms())]
        ys = [conf.GetAtomPosition(i).y for i in range(mol_draw.GetNumAtoms())]
        cx = float(np.mean(xs)) if xs else 0.0
        cy = float(np.mean(ys)) if ys else 0.0
        ang = math.radians(-float(rotate_deg))
        ca, sa = math.cos(ang), math.sin(ang)
        for i in range(mol_draw.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            x = p.x - cx
            y = p.y - cy
            if flip_horizontal:
                x = -x
            if flip_vertical:
                y = -y
            xr = x * ca - y * sa
            yr = x * sa + y * ca
            conf.SetAtomPosition(i, Point3D(float(xr), float(yr), float(p.z)))

    w, h = int(size[0]), int(size[1])
    # Prefer text nodes in SVG (instead of glyph paths) so vector editors
    # like Adobe Illustrator can recognize atom labels as editable text.
    try:
        d2d = rdMolDraw2D.MolDraw2DSVG(w, h, -1, -1, True)
    except Exception:
        d2d = rdMolDraw2D.MolDraw2DSVG(w, h)
    opts = d2d.drawOptions()
    opts.clearBackground = True
    try:
        opts.baseFontSize = float(opts.baseFontSize) * float(atom_symbol_font_scale)
    except Exception:
        pass
    try:
        opts.fixedFontSize = float(opts.fixedFontSize) * float(atom_symbol_font_scale)
    except Exception:
        pass
    try:
        opts.useBWAtomPalette()
    except Exception:
        pass
    try:
        opts.setBackgroundColour((1.0, 1.0, 1.0))
    except Exception:
        pass

    if show_rdkit_atom_index or show_pred_shift_label or show_matched_shift_label:
        vir_list = list(vir_shifts)
        matched_map = matched_ppm_by_atom or {}
        amap = rdMolDraw2D.IntStringMap()
        for slot, aid in enumerate(int(x) for x in carbon_atom_indices):
            parts: List[str] = []
            if show_rdkit_atom_index:
                parts.append(str(aid))
            if show_pred_shift_label and slot < len(vir_list):
                parts.append(f"{float(vir_list[slot]):.1f}")
            if show_matched_shift_label and int(aid) in matched_map:
                parts.append(f"{float(matched_map[int(aid)]):.1f}")
            if parts:
                amap[int(aid)] = "\n".join(parts)
        opts.atomLabels = amap
        opts.annotationFontScale = float(annotation_font_scale)

    if fragment_highlight_groups:
        radii: Dict[int, float] = {aid: 0.35 for aid in highlight_atoms}
        for atom_list, rgb in fragment_highlight_groups:
            r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
            for aid in atom_list:
                ia = int(aid)
                if ia in hm:
                    hm[ia] = (r, g, b)
                    radii[ia] = 0.55
        rdMolDraw2D.PrepareAndDrawMolecule(
            d2d,
            mol_draw,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=hm,
            highlightAtomRadii=radii,
        )
    elif selected_atom_indices:
        sel = {int(x) for x in selected_atom_indices}
        radii = {}
        for aid in highlight_atoms:
            radii[aid] = 0.55 if aid in sel else 0.35
        sel_color = selected_atom_color if selected_atom_color is not None else (0.10, 0.40, 0.95)
        for aid in sel:
            if aid in hm:
                hm[aid] = (
                    float(sel_color[0]),
                    float(sel_color[1]),
                    float(sel_color[2]),
                )
        rdMolDraw2D.PrepareAndDrawMolecule(
            d2d,
            mol_draw,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=hm,
            highlightAtomRadii=radii,
        )
    else:
        rdMolDraw2D.PrepareAndDrawMolecule(
            d2d,
            mol_draw,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=hm,
        )
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()
    return svg if isinstance(svg, str) else svg.decode("utf-8")


def _rwmol_induced_subgraph(mol: Chem.Mol, env: Set[int]) -> Tuple[Chem.Mol, Dict[int, int]]:
    """Build induced subgraph on ``env`` (atom indices); sanitize caller."""
    env_list = sorted(int(x) for x in env)
    rw = Chem.RWMol()
    old_to_new: Dict[int, int] = {}
    for o in env_list:
        old_to_new[o] = rw.AddAtom(mol.GetAtomWithIdx(o))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in old_to_new and j in old_to_new:
            ni, nj = old_to_new[i], old_to_new[j]
            if rw.GetBondBetweenAtoms(ni, nj) is None:
                rw.AddBond(ni, nj, bond.GetBondType())
    return rw.GetMol(), old_to_new


def submol_first_neighbor_shell(
    mol: Chem.Mol, seed_atoms: Sequence[int]
) -> Tuple[Chem.Mol, Dict[int, int]]:
    """
    Induced subgraph: all ``seed_atoms`` plus every atom directly bonded to them
    (one bond shell). Returns sanitized submol and map RDKit parent index -> submol index.
    """
    env: Set[int] = {int(x) for x in seed_atoms}
    for aid in list(env):
        atom = mol.GetAtomWithIdx(aid)
        for nb in atom.GetNeighbors():
            env.add(nb.GetIdx())
    sub, old_to_new = _rwmol_induced_subgraph(mol, env)
    Chem.SanitizeMol(sub)
    return sub, old_to_new


def submol_fragment_atoms_only(
    mol: Chem.Mol, frag_atom_indices: Sequence[int]
) -> Tuple[Chem.Mol, Dict[int, int]]:
    """Only fragment atoms and bonds between them (no neighbor shell)."""
    env = {int(x) for x in frag_atom_indices}
    sub, old_to_new = _rwmol_induced_subgraph(mol, env)
    Chem.SanitizeMol(sub)
    return sub, old_to_new


def draw_fusion_pair_fragment_env_png(
    smiles: str,
    fragment_atom_indices: Sequence[int],
    highlight_atom_idx: int,
    pair_rgb: Tuple[float, float, float],
    size: Tuple[int, int] = (120, 90),
) -> Any:
    """
    Small 2D structure: fragment carbons + directly bonded atoms; ``highlight_atom_idx``
    (parent mol index) is drawn with ``pair_rgb``; other fragment atoms pale blue-gray,
    neighbor-only atoms light gray.
    """
    import io
    from PIL import Image
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    hi = int(highlight_atom_idx)
    if hi < 0 or hi >= mol.GetNumAtoms():
        raise ValueError("highlight_atom_idx out of range")

    frag_old = {int(x) for x in fragment_atom_indices}
    try:
        sub, old_to_new = submol_first_neighbor_shell(mol, sorted(frag_old))
    except Exception:
        sub, old_to_new = submol_fragment_atoms_only(mol, sorted(frag_old))
    if hi not in old_to_new:
        raise ValueError("highlight atom outside submol")

    AllChem.Compute2DCoords(sub)
    h_new = old_to_new[hi]
    pr, pg, pb = float(pair_rgb[0]), float(pair_rgb[1]), float(pair_rgb[2])

    hm: Dict[int, Tuple[float, float, float]] = {}
    radii: Dict[int, float] = {}
    for old_a, new_i in old_to_new.items():
        if new_i == h_new:
            hm[new_i] = (pr, pg, pb)
            radii[new_i] = 0.55
        elif old_a in frag_old:
            hm[new_i] = (0.72, 0.78, 0.90)
            radii[new_i] = 0.38
        else:
            hm[new_i] = (0.88, 0.88, 0.88)
            radii[new_i] = 0.35

    highlight_atoms = sorted(hm.keys())
    w, h = int(size[0]), int(size[1])
    d2d = rdMolDraw2D.MolDraw2DCairo(w, h)
    opts = d2d.drawOptions()
    opts.clearBackground = True
    try:
        opts.useBWAtomPalette()
    except Exception:
        pass
    try:
        opts.setBackgroundColour((1.0, 1.0, 1.0))
    except Exception:
        pass
    mol_draw = Chem.Mol(sub)
    rdMolDraw2D.PrepareAndDrawMolecule(
        d2d,
        mol_draw,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=hm,
        highlightAtomRadii=radii,
    )
    d2d.FinishDrawing()
    return Image.open(io.BytesIO(d2d.GetDrawingText()))


def prepare_fusion_fragment_draw_mol(
    mol: Chem.Mol, frag_old: Set[int]
) -> Tuple[Chem.Mol, Dict[int, int]]:
    """
    Submol with 1-bond shell, else fragment-only subgraph, else full molecule + identity map.
    """
    from rdkit.Chem import AllChem

    if not frag_old:
        raise ValueError("empty fragment atom set")
    try:
        return submol_first_neighbor_shell(mol, sorted(frag_old))
    except Exception:
        try:
            return submol_fragment_atoms_only(mol, sorted(frag_old))
        except Exception:
            m2 = Chem.Mol(mol)
            AllChem.Compute2DCoords(m2)
            return m2, {i: i for i in range(m2.GetNumAtoms())}


def _pixel_map_from_conformer(
    sub: Chem.Mol, old_to_new: Dict[int, int], frag_old: Set[int], w: int, h: int
) -> Dict[int, Tuple[float, float]]:
    """Map parent atom index -> (px, py) with top-left origin, approximating RDKit draw box."""
    conf = sub.GetConformer()
    if sub.GetNumAtoms() == 0:
        return {}
    axs = [conf.GetAtomPosition(i).x for i in range(sub.GetNumAtoms())]
    ays = [conf.GetAtomPosition(i).y for i in range(sub.GetNumAtoms())]
    xmin, xmax = min(axs), max(axs)
    ymin, ymax = min(ays), max(ays)
    dx = xmax - xmin + 1e-9
    dy = ymax - ymin + 1e-9
    margin = 0.1
    span = 1.0 - 2.0 * margin
    out: Dict[int, Tuple[float, float]] = {}
    for old_a in frag_old:
        if old_a not in old_to_new:
            continue
        ni = old_to_new[old_a]
        p = conf.GetAtomPosition(int(ni))
        u = (float(p.x) - xmin) / dx
        v = (float(p.y) - ymin) / dy
        px = w * (margin + span * u)
        py = h * (margin + span * (1.0 - v))
        out[int(old_a)] = (px, py)
    return out


def draw_fusion_fragment_env_group_image_and_atom_pixels(
    smiles: str,
    fragment_atom_indices: Sequence[int],
    frag_rgb: Tuple[float, float, float],
    size: Tuple[int, int] = (240, 180),
    *,
    font_scale: float = 1.0,
) -> Tuple[Any, Dict[int, Tuple[float, float]], int, int]:
    """
    Draw one fragment thumbnail (neighbor shell when possible). Returns
    ``(PIL.Image RGBA, parent_atom_idx -> (px, py) top-left canvas coords, width, height)``.
    Pixel coords match RDKit MolDraw2D when ``GetDrawCoords`` is available, else conformer fallback.
    """
    import io
    from PIL import Image
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    frag_old = {int(x) for x in fragment_atom_indices}
    if not frag_old:
        raise ValueError("empty fragment_atom_indices")

    sub, old_to_new = prepare_fusion_fragment_draw_mol(mol, frag_old)
    AllChem.Compute2DCoords(sub)

    pr, pg, pb = float(frag_rgb[0]), float(frag_rgb[1]), float(frag_rgb[2])
    hm: Dict[int, Tuple[float, float, float]] = {}
    radii: Dict[int, float] = {}
    for old_a, new_i in old_to_new.items():
        if old_a in frag_old:
            hm[new_i] = (pr, pg, pb)
            radii[new_i] = 0.52
        else:
            hm[new_i] = (0.88, 0.88, 0.88)
            radii[new_i] = 0.35

    highlight_atoms = sorted(hm.keys())
    w, h = int(size[0]), int(size[1])
    d2d = rdMolDraw2D.MolDraw2DCairo(w, h)
    opts = d2d.drawOptions()
    opts.clearBackground = True
    try:
        opts.useBWAtomPalette()
    except Exception:
        pass
    try:
        opts.setBackgroundColour((1.0, 1.0, 1.0))
    except Exception:
        pass
    if float(font_scale) != 1.0:
        try:
            fs = max(0.75, float(font_scale))
            mf = int(round(opts.minFontSize * fs))
            opts.minFontSize = max(mf, 7)
            mf_max = getattr(opts, "maxFontSize", None)
            if mf_max is not None:
                opts.maxFontSize = max(int(round(mf_max * fs)), opts.minFontSize + 2)
        except Exception:
            pass
    mol_draw = Chem.Mol(sub)
    rdMolDraw2D.PrepareAndDrawMolecule(
        d2d,
        mol_draw,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=hm,
        highlightAtomRadii=radii,
    )

    pixel_map: Dict[int, Tuple[float, float]] = {}
    gc = getattr(d2d, "GetDrawCoords", None)
    if gc is not None:
        for old_a in frag_old:
            if old_a not in old_to_new:
                continue
            ni = int(old_to_new[old_a])
            try:
                pt = gc(ni)
                pixel_map[int(old_a)] = (float(pt.x), float(pt.y))
            except Exception:
                pass
    if len(pixel_map) < len([a for a in frag_old if a in old_to_new]):
        fb = _pixel_map_from_conformer(sub, old_to_new, frag_old, w, h)
        for k, v in fb.items():
            pixel_map.setdefault(k, v)

    d2d.FinishDrawing()
    img = Image.open(io.BytesIO(d2d.GetDrawingText()))
    return img, pixel_map, w, h


def draw_fusion_fragment_env_group_png(
    smiles: str,
    fragment_atom_indices: Sequence[int],
    frag_rgb: Tuple[float, float, float],
    size: Tuple[int, int] = (240, 180),
) -> Any:
    """
    One thumbnail per fragment: neighbor shell (or fragment-only / full-mol fallback).
    """
    img, _, _, _ = draw_fusion_fragment_env_group_image_and_atom_pixels(
        smiles, fragment_atom_indices, frag_rgb, size=size
    )
    return img
