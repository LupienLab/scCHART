"""The minimal example and the PBMC tutorial notebook run on the current API."""

import json
import os
import re
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MINIMAL = ROOT / "examples" / "minimal_example.py"
NOTEBOOK = ROOT / "examples" / "pbmc_tutorial" / "pbmc_tutorial.ipynb"
DOCS_NOTEBOOK = ROOT / "docs" / "tutorial.ipynb"
OBSOLETE = ["cluster_key", "run_scchart", "ScCHARTResult", "plot_", "save_figure", ".savefig(",
            '"structural_heterogeneity"', '"programme_mixing"', '"programme_heterogeneity"',
            "J_structural_bits", "J_P_bits", "J_projected"]

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def _needs(path):
    if not path.exists():
        pytest.skip(f"{path.name} not present (source distribution without examples)")


def _code_cells(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_minimal_example_runs_and_matches_its_description(tmp_path, monkeypatch, capsys):
    _needs(MINIMAL)
    out = tmp_path / "calibration.png"
    monkeypatch.setattr(sys, "argv", [str(MINIMAL), str(out)])
    ns = runpy.run_path(str(MINIMAL), run_name="__main__")
    assert out.exists()
    printed = capsys.readouterr().out
    assert "Structural axis" in printed and "Programme axis" in printed
    s, p = ns["model"].structural_, ns["model"].programme_
    # The docstring's statements about the output.
    assert s.loc["two_states", "Delta_G"] > s.loc["homogeneous", "Delta_G"]
    assert p["Delta_P"].idxmax() == "programme_mix"
    assert p.loc["programme_mix", "Delta_P"] > 5 * p.drop("programme_mix")["Delta_P"].max()
    assert (p.drop("programme_mix")["scored_fraction"] < 1).all()
    source = MINIMAL.read_text()
    assert not [w for w in OBSOLETE if w in source and w != ".savefig("]


def test_tutorial_notebook_is_canonical_and_fully_executed():
    _needs(NOTEBOOK)
    nb = json.loads(NOTEBOOK.read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    counts = [c["execution_count"] for c in code]
    assert counts == list(range(1, len(code) + 1)), "stored outputs must come from one full run"
    errors = [o for c in code for o in c["outputs"] if o["output_type"] == "error"]
    assert not errors
    text = "\n".join(_code_cells(nb))
    assert not [w for w in OBSOLETE if w in text]
    assert "scchart.Heterogeneity(" in text and "scchart.pl." in text and "write_h5ad" in text
    markdown = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert "PLACEHOLDER" not in markdown
    if DOCS_NOTEBOOK.exists() or DOCS_NOTEBOOK.is_symlink():
        assert DOCS_NOTEBOOK.resolve() == NOTEBOOK.resolve(), "the docs page must show the same notebook"


def test_tutorial_code_runs_on_a_small_synthetic_input(tmp_path, monkeypatch):
    """Execute every code cell of the notebook on a small PBMC-shaped AnnData."""
    _needs(NOTEBOOK)
    import anndata as ad
    from scipy import sparse

    rng = np.random.default_rng(0)
    names = ["CD4 Naive", "Naive B", "CD14 Mono"]   # PBMC cell types, so the tutorial's lineage table applies
    groups = np.repeat(names, [120, 100, 90])
    n_feat = 600
    rates = np.full((groups.size, n_feat), 0.05)
    for k, g in enumerate(names):
        rates[groups == g, k * 100:(k + 1) * 100] = 0.25
    depth_scale = rng.uniform(0.6, 1.6, size=(groups.size, 1))
    X = sparse.csr_matrix(rng.binomial(2, np.clip(rates * depth_scale, 0, 1)).astype(np.uint32))
    adata = ad.AnnData(X=X, obs=pd.DataFrame({"cell_type": pd.Categorical(groups)},
                                             index=[f"c{i}" for i in range(groups.size)]),
                       var=pd.DataFrame(index=[f"tile{j}" for j in range(n_feat)]))
    adata.obsm["X_umap"] = rng.normal(size=(groups.size, 2))
    data = tmp_path / "input.h5ad"
    adata.write_h5ad(data)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCCHART_PBMC_DATA", str(data))
    monkeypatch.setenv("SCCHART_N_JOBS", "1")
    monkeypatch.setenv("SCCHART_TUTORIAL_MATCHED_DEPTH", "0")
    import matplotlib.pyplot as plt

    ns = {"display": print}
    for i, cell in enumerate(_code_cells(json.loads(NOTEBOOK.read_text()))):
        try:
            exec(compile(cell, f"notebook cell {i}", "exec"), ns)
        finally:
            plt.close("all")
    assert (tmp_path / "pbmc5k_scchart_results.h5ad").exists()
    assert list(ns["model"].population_order_) == ns["ORDER"] == names
    assert np.isfinite(ns["model"].structural_["Delta_G"]).all()
    assert np.isfinite(ns["model"].programme_["Delta_P"]).all()
