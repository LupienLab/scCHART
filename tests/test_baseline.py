"""Numerical regression against the scCHART baseline (``tests/data/scchart_baseline.json``).

The baseline is a regression guard recorded from scCHART 0.1.0; see
``tests/baseline_workload.py`` for what it contains, the provenance of each
entry ("established" or "new") and how it may be regenerated. Null chains,
seeds and population orders must match exactly; floating-point outputs within
the stored tolerance (``rtol=1e-9, atol=1e-11``), which allows for last-digit
differences of the SVD between BLAS/LAPACK builds.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from tests.baseline_workload import ATOL, RTOL, provenance, workload

BASELINE = Path(__file__).parent / "data" / "scchart_baseline.json"


@pytest.fixture(scope="module")
def outputs():
    return workload(), json.loads(BASELINE.read_text())


def test_baseline_metadata(outputs):
    _, base = outputs
    assert base["generated_with"].startswith("scchart ")
    assert base["tolerance"] == {"rtol": RTOL, "atol": ATOL}
    assert {e["provenance"] for e in base["entries"].values()} == {"established", "new"}
    for name, entry in base["entries"].items():
        assert entry["provenance"] == provenance(name)


def test_same_entries_are_produced(outputs):
    got, base = outputs
    assert set(got) == set(base["entries"])


ENTRIES = json.loads(BASELINE.read_text())["entries"] if BASELINE.exists() else {}
EXACT = sorted(n for n, e in ENTRIES.items() if not isinstance(e["value"], dict))
TABLES = sorted(n for n, e in ENTRIES.items() if isinstance(e["value"], dict))


@pytest.mark.parametrize("name", EXACT)
def test_exact_entries(outputs, name):
    got, base = outputs
    assert got[name] == base["entries"][name]["value"]


@pytest.mark.parametrize("name", TABLES)
def test_tables(outputs, name):
    got, base = outputs
    g, w = got[name], base["entries"][name]["value"]
    assert g["index"] == w["index"]
    assert g["columns"] == w["columns"]
    np.testing.assert_allclose(np.asarray(g["values"], float), np.asarray(w["values"], float),
                               rtol=RTOL, atol=ATOL, equal_nan=True)
