"""scripts/compare_outputs.py on two tiny synthetic stages."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kappaccino.io import store

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_outputs.py"


@pytest.fixture(scope="module")
def compare():
    spec = importlib.util.spec_from_file_location("compare_outputs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage(root: Path, *, value=1.0, flag=True, seconds=1.0, exposure=10.0, note=True) -> Path:
    root.mkdir(parents=True)
    store.write_table(root, "table", pd.DataFrame({"x": [1.0, 2.0], "exposure_seconds": [exposure, 5.0]}))
    store.write_record(root, "record", {"value": value, "flag": flag, "module": {"seconds": seconds}})
    store.write_arrays(root, "arrays", grid=np.array([[1.0, 2.0], [3.0, value]]))
    if note:
        (root / "note.txt").write_text("same\n")
    return root


def run(compare, a, b, capsys):
    code = compare.main([str(a), str(b), "--no-figures"])
    return code, capsys.readouterr().out


def test_identical_stages_and_ignored_timings(compare, tmp_path, capsys):
    code, out = run(compare, stage(tmp_path / "a"), stage(tmp_path / "b", seconds=9.0), capsys)
    assert code == 0 and "0 difference(s)" in out           # a record's wall time is not a difference


@pytest.mark.parametrize("change, expected", [
    (dict(value=2.0), ["record", "value: 1.0 != 2.0", "arrays", "grid: 1 of 4 values differ"]),
    (dict(flag=1), ["flag: bool 'True' != int '1'"]),
    (dict(exposure=11.0), ["column exposure_seconds"]),      # a data column named like a timing key is compared
    (dict(note=False), ["note.txt: only in A"]),
])
def test_each_difference_is_reported(compare, tmp_path, capsys, change, expected):
    code, out = run(compare, stage(tmp_path / "a"), stage(tmp_path / "b", **change), capsys)
    assert code == 1
    for text in expected:
        assert text in out, out


def test_table_indexes_are_compared(compare):
    # The store keeps no index (tables read back with a RangeIndex), but the comparison does not rely on that.
    a = pd.DataFrame({"x": [1.0, 2.0]})
    assert compare.compare_tables(a, a.copy(), ()) == []
    assert compare.compare_tables(a, a.set_axis([0, 2]), ()) == ["index RangeIndex (int64) != Index (int64)"]
    assert compare.compare_tables(a.set_axis([0, 1]), a.set_axis([0, 2]), ()) == ["index: 1 of 2 values differ, max abs 1"]
