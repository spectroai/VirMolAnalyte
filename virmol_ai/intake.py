# -*- coding: utf-8 -*-
"""Intake layer: normalize various peak-list inputs into the strict
'ppm, q/t/d/s' format consumed by ModernVirMolAnalyteGUI.parse_manual_peak_input_strict.

Supported inputs:
  - Paper-style inline text: "172.5 (s), 145.2 (d), 130.1 (s, 2C), 21.3 (q)"
  - Newline / TSV text:      "172.5\\tCH3\\n145.2\\tCH\\n..."
  - CSV / Excel file paths   (column auto-detection)
  - List[dict]               (already-parsed peaks: [{'ppm': 172.5, 'type': 's'}, ...])

The module is **pure Python** (only depends on stdlib + pandas when reading
CSV/XLSX). No Qt, no LLM. The GUI layer feeds the resulting text into
``manual_peak_input.setPlainText(text)`` and then triggers Start Analysis.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence, Tuple


# ---------- DEPT / carbon-type normalization -----------------------------------

_TYPE_ALIASES = {
    # canonical short codes (already correct)
    "q": "q", "t": "t", "d": "d", "s": "s",
    # DEPT spelled out
    "ch3": "q", "ch2": "t", "ch": "d", "c": "s",
    "ch_3": "q", "ch_2": "t",
    # words
    "methyl":     "q",
    "methylene":  "t",
    "methine":    "d",
    "methine_ch": "d",
    "quaternary": "s",
    "quat":       "s",
    "carbonyl":   "s",
}

_VALID_SHORT = {"q", "t", "d", "s"}

# Regex used for paper-style inline peaks: "172.5 (s, 2C)" / "21.3 (q)"
_PAPER_PEAK_RE = re.compile(
    r"""
    (?P<ppm>-?\d+(?:\.\d+)?)        # ppm value (allows negative)
    \s*
    \(\s*                           # opening paren
    (?P<type>[A-Za-z][A-Za-z_0-9]*) # type token (CH3 / s / quaternary)
    \s*
    (?:,\s*(?P<multi>\d+)\s*C\s*)?  # optional ", 2C"
    \)
    """,
    re.VERBOSE,
)

# Lower-case header keywords we accept for the two columns
_PPM_HEADERS  = {"ppm", "shift", "chemical shift", "delta", "δ", "chem_shift", "chemshift"}
_TYPE_HEADERS = {"type", "carbon type", "dept", "dept type", "mult", "multiplicity"}


@dataclass
class IntakeReport:
    """Result of a normalization pass."""
    text: str                              # canonical "ppm, type\n..." (may be empty)
    n_peaks: int = 0
    warnings: List[str] = field(default_factory=list)
    source: str = "unknown"                # 'paper_text' / 'lines_text' / 'csv' / 'xlsx' / 'list'
    detected_columns: Optional[Tuple[str, str]] = None  # ('ppm', 'type') header names

    @property
    def ok(self) -> bool:
        return self.n_peaks > 0


# ---------- Public API ----------------------------------------------------------

def normalize_carbon_type(token: Any) -> Optional[str]:
    """Map DEPT / multiplicity labels to canonical q/t/d/s, or None if unrecognised."""
    if token is None:
        return None
    key = str(token).strip().lower()
    if not key:
        return None
    if key in _VALID_SHORT:
        return key
    return _TYPE_ALIASES.get(key)


def to_canonical_text(peaks: Sequence[Tuple[float, str]]) -> str:
    """Join ``[(ppm, type), ...]`` into the strict 'ppm, type\\n' format."""
    return "\n".join(f"{float(p):.2f}, {t}" for p, t in peaks)


def from_text(text: str) -> IntakeReport:
    """Parse any reasonable text representation. Tries paper-style first,
    then falls back to newline/TSV/CSV-like rows.
    """
    if not isinstance(text, str) or not text.strip():
        return IntakeReport(text="", warnings=["empty text"], source="text")

    paper = _parse_paper_style(text)
    if paper.ok and paper.n_peaks >= 2:
        return paper

    lines = _parse_lines_style(text)
    if lines.ok:
        return lines

    if paper.ok:
        return paper

    return IntakeReport(
        text="",
        warnings=[
            "Could not parse text. Examples that work:\n"
            "  172.5, s\n  21.3, q\n"
            "or paper-style:\n  172.5 (s), 145.2 (d), 21.3 (q)"
        ],
        source="text",
    )


def from_file(path: str) -> IntakeReport:
    """Read a CSV or Excel file and normalize. Detects ppm/type columns automatically."""
    if not path or not os.path.isfile(path):
        return IntakeReport(text="", warnings=[f"file not found: {path}"], source="file")

    try:
        import pandas as pd
    except ImportError:
        return IntakeReport(
            text="",
            warnings=["pandas is required to read CSV/XLSX files"],
            source="file",
        )

    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".xls", ".xlsx"):
            df = pd.read_excel(path)
            source = "xlsx"
        else:
            df = pd.read_csv(path)
            source = "csv"
    except Exception as exc:  # noqa: BLE001
        return IntakeReport(text="", warnings=[f"file read error: {exc}"], source="file")

    return _from_dataframe(df, source=source)


def from_list(items: Iterable[Any]) -> IntakeReport:
    """Normalize an already-parsed iterable like ``[{'ppm': 172.5, 'type': 's'}, ...]``
    or ``[(172.5, 's'), ...]``.
    """
    peaks: List[Tuple[float, str]] = []
    warnings: List[str] = []
    for idx, item in enumerate(items, 1):
        ppm: Any = None
        ctype: Any = None
        if isinstance(item, dict):
            ppm = item.get("ppm") or item.get("shift") or item.get("delta")
            ctype = item.get("type") or item.get("dept") or item.get("multiplicity")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            ppm, ctype = item[0], item[1]
        else:
            warnings.append(f"item {idx}: unsupported shape ({type(item).__name__})")
            continue

        try:
            ppm_f = float(ppm)
        except (TypeError, ValueError):
            warnings.append(f"item {idx}: invalid ppm {ppm!r}")
            continue
        norm = normalize_carbon_type(ctype)
        if norm is None:
            warnings.append(f"item {idx}: unknown DEPT type {ctype!r}")
            continue
        peaks.append((ppm_f, norm))

    return IntakeReport(
        text=to_canonical_text(peaks),
        n_peaks=len(peaks),
        warnings=warnings,
        source="list",
    )


# ---------- Internal helpers ----------------------------------------------------

def _parse_paper_style(text: str) -> IntakeReport:
    """Extract peaks from a paper-style inline list like '172.5 (s), 21.3 (q)'."""
    peaks: List[Tuple[float, str]] = []
    warnings: List[str] = []
    for m in _PAPER_PEAK_RE.finditer(text):
        try:
            ppm_f = float(m.group("ppm"))
        except (TypeError, ValueError):
            continue
        ctype = normalize_carbon_type(m.group("type"))
        if ctype is None:
            warnings.append(f"unknown DEPT label '{m.group('type')}' near {m.group('ppm')}")
            continue
        multi = m.group("multi")
        n_copies = int(multi) if multi else 1
        for _ in range(n_copies):
            peaks.append((ppm_f, ctype))
    peaks.sort(key=lambda x: -x[0])  # high ppm first (matches NMR convention)
    return IntakeReport(
        text=to_canonical_text(peaks),
        n_peaks=len(peaks),
        warnings=warnings,
        source="paper_text",
    )


_LINE_SPLIT_RE = re.compile(r"[,\t]|\s{2,}|\s+")


def _parse_lines_style(text: str) -> IntakeReport:
    """Parse one-peak-per-line text. Handles commas, tabs, and whitespace separators."""
    peaks: List[Tuple[float, str]] = []
    warnings: List[str] = []
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = [t for t in _LINE_SPLIT_RE.split(line) if t]
        if len(tokens) < 2:
            warnings.append(f"line {line_no}: need at least 'ppm type' (got {line!r})")
            continue
        ppm_token = tokens[0]
        type_token = tokens[1]
        try:
            ppm_f = float(ppm_token)
        except ValueError:
            warnings.append(f"line {line_no}: invalid ppm '{ppm_token}'")
            continue
        norm = normalize_carbon_type(type_token)
        if norm is None:
            warnings.append(f"line {line_no}: unknown DEPT label '{type_token}'")
            continue
        peaks.append((ppm_f, norm))
    return IntakeReport(
        text=to_canonical_text(peaks),
        n_peaks=len(peaks),
        warnings=warnings,
        source="lines_text",
    )


def _from_dataframe(df, *, source: str) -> IntakeReport:
    """Normalize a pandas DataFrame, auto-detecting the two columns."""
    if df is None or df.empty:
        return IntakeReport(text="", warnings=["empty dataframe"], source=source)

    ppm_col, type_col = _detect_columns(df)
    if ppm_col is None or type_col is None:
        return IntakeReport(
            text="",
            warnings=[
                "Could not detect ppm/type columns. "
                f"Found headers: {list(df.columns)}. "
                "Rename one column to 'ppm' and another to 'type'."
            ],
            source=source,
            detected_columns=(str(ppm_col) if ppm_col else "?", str(type_col) if type_col else "?"),
        )

    peaks: List[Tuple[float, str]] = []
    warnings: List[str] = []
    for idx, row in df.iterrows():
        ppm_raw = row[ppm_col]
        type_raw = row[type_col]
        try:
            ppm_f = float(ppm_raw)
        except (TypeError, ValueError):
            warnings.append(f"row {idx}: invalid ppm {ppm_raw!r}")
            continue
        norm = normalize_carbon_type(type_raw)
        if norm is None:
            warnings.append(f"row {idx}: unknown DEPT label {type_raw!r}")
            continue
        peaks.append((ppm_f, norm))

    return IntakeReport(
        text=to_canonical_text(peaks),
        n_peaks=len(peaks),
        warnings=warnings,
        source=source,
        detected_columns=(str(ppm_col), str(type_col)),
    )


def _detect_columns(df) -> Tuple[Optional[Any], Optional[Any]]:
    """Best-effort auto-detection of (ppm_column, type_column) in a DataFrame."""
    ppm_col: Optional[Any] = None
    type_col: Optional[Any] = None

    for col in df.columns:
        key = str(col).strip().lower()
        if ppm_col is None and key in _PPM_HEADERS:
            ppm_col = col
        elif type_col is None and key in _TYPE_HEADERS:
            type_col = col

    if ppm_col is not None and type_col is not None:
        return ppm_col, type_col

    # Fallback: inspect column dtypes / sample values
    for col in df.columns:
        sample = df[col].dropna()
        if sample.empty:
            continue
        first = sample.iloc[0]
        try:
            float(first)
            if ppm_col is None:
                ppm_col = col
                continue
        except (TypeError, ValueError):
            pass
        if type_col is None and normalize_carbon_type(first) is not None:
            type_col = col

    return ppm_col, type_col
