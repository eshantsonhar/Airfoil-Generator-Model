"""
Shared parsing helpers for SU2-style CSV output (``history.csv``, ``surface_flow.csv``).

SU2 writes CSV files whose header names are quoted and padded with whitespace, and
whose trailing rows may be blank or contain a lone comma. These helpers centralise
the tokenisation, last-row lookup and column-trace extraction used by the CFD,
ASO and UI layers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "split_csv_line",
    "read_csv_table",
    "last_row_mapping",
    "column_traces",
    "lookup_float",
]


def split_csv_line(line: str) -> List[str]:
    """Split a CSV line into cells with surrounding whitespace and quotes removed."""
    return [cell.strip().strip('"').strip("'") for cell in line.split(",")]


def _is_data_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and stripped != "," and not stripped.startswith("#")


def read_csv_table(path: Path) -> Tuple[List[str], List[List[str]]]:
    """
    Read an SU2 CSV file into its header cells and data rows.

    Blank rows, lone-comma rows and ``#`` comment rows are dropped. Returns
    ``([], [])`` when the file holds fewer than two lines (i.e. no data rows).
    Propagates :class:`OSError` if the file cannot be read.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 2:
        return [], []
    headers = split_csv_line(lines[0])
    rows = [split_csv_line(line) for line in lines[1:] if _is_data_line(line)]
    return headers, rows


def last_row_mapping(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    pad: bool = False,
) -> Optional[Dict[str, str]]:
    """
    Map header names to the cells of the final data row.

    With ``pad`` the row is padded with ``"0.0"`` up to the header count, so short
    trailing rows still yield a full mapping.
    """
    if not headers or not rows:
        return None
    values = list(rows[-1])
    if pad and len(values) < len(headers):
        values.extend(["0.0"] * (len(headers) - len(values)))
    return dict(zip(headers, values))


def column_traces(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> Dict[str, List[float]]:
    """Map each header name to its column of parseable float values."""
    traces: Dict[str, List[float]] = {header: [] for header in headers}
    for row in rows:
        for index, header in enumerate(headers):
            if index >= len(row):
                continue
            try:
                traces[header].append(float(row[index]))
            except (ValueError, TypeError):
                pass
    return traces


def lookup_float(
    mapping: Dict[str, str],
    candidates: Sequence[str],
    default: Optional[float] = None,
) -> Optional[float]:
    """Return the first candidate column parsed as a float, else ``default``."""
    for candidate in candidates:
        if candidate not in mapping:
            continue
        try:
            return float(mapping[candidate])
        except (ValueError, TypeError):
            return default
    return default
