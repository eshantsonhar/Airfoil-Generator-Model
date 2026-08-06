"""Tests for the shared SU2 CSV parsing helpers."""

from pathlib import Path

from airfoil_discovery.cfd.su2_csv import (
    column_traces,
    last_row_mapping,
    lookup_float,
    read_csv_table,
    split_csv_line,
)

HISTORY = '''"Inner_Iter",   "CL"  ,  "CD"
0, 0.10, 0.010
# comment row
,
1, 0.20, 0.020

'''


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "history.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_split_csv_line_strips_quotes_and_space():
    assert split_csv_line('  "CL" , \'CD\' ,CMz') == ["CL", "CD", "CMz"]


def test_read_csv_table_skips_blank_comment_and_comma_rows(tmp_path):
    headers, rows = read_csv_table(_write(tmp_path, HISTORY))
    assert headers == ["Inner_Iter", "CL", "CD"]
    assert rows == [["0", "0.10", "0.010"], ["1", "0.20", "0.020"]]


def test_read_csv_table_header_only(tmp_path):
    assert read_csv_table(_write(tmp_path, '"CL","CD"\n')) == ([], [])


def test_last_row_mapping(tmp_path):
    headers, rows = read_csv_table(_write(tmp_path, HISTORY))
    assert last_row_mapping(headers, rows) == {"Inner_Iter": "1", "CL": "0.20", "CD": "0.020"}
    assert last_row_mapping([], []) is None


def test_last_row_mapping_pads_short_rows(tmp_path):
    headers, rows = read_csv_table(_write(tmp_path, '"A","B","C"\n1,2\n'))
    assert last_row_mapping(headers, rows) == {"A": "1", "B": "2"}
    assert last_row_mapping(headers, rows, pad=True) == {"A": "1", "B": "2", "C": "0.0"}


def test_column_traces_ignores_unparseable_and_missing_cells(tmp_path):
    headers, rows = read_csv_table(_write(tmp_path, '"A","B"\n1,nan_value\n2\n'))
    traces = column_traces(headers, rows)
    assert traces["A"] == [1.0, 2.0]
    assert traces["B"] == []


def test_lookup_float_uses_first_present_candidate():
    mapping = {"CD": "0.02", "DRAG": "0.05"}
    assert lookup_float(mapping, ["CD", "DRAG"]) == 0.02
    assert lookup_float(mapping, ["LIFT", "DRAG"]) == 0.05
    assert lookup_float(mapping, ["MISSING"], 1.5) == 1.5
    assert lookup_float({"CD": "bad"}, ["CD"], 0.0) == 0.0
