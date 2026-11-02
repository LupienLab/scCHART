"""Workload of the scCHART numerical baseline, ``tests/data/scchart_baseline.json``.

``workload()`` runs a fixed set of scCHART calls on deterministic synthetic data
and returns their numeric outputs; ``tests/test_baseline.py`` compares them
with the stored baseline.

What the baseline is
    A regression guard. Its values are the outputs of the implementation at the
    time the baseline was recorded; they are not derived independently, so they
    detect changes but do not prove correctness. Independently derived values
    (hand calculations and exact enumeration of small fixed-margin state spaces)
    are tested in ``test_worked_example.py``, ``test_null_sampler.py``,
    ``test_structural_divergence.py`` and ``test_programme_entropy.py``.

Provenance of each entry (``entries[name]["provenance"]``)
    ``"established"``: before recording, the values were compared with values
    established before scCHART 0.1.0 and agreed within the stored tolerance;
    those earlier values are kept outside this repository.
    ``"new"``: no earlier counterpart (population-order and seed assignment
    with unsorted labels); recorded from scCHART 0.1.0 only.

Regenerating the baseline is a deliberate act; the tests never write it::

    uv run python -m tests.baseline_workload --write tests/data/scchart_baseline.json

A regenerated baseline must be justified in CHANGELOG.md.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

RTOL, ATOL = 1e-9, 1e-11   # the SVD in LSIProjection differs in the last digits between BLAS builds
SAMPLERS = ("proposal_clock", "accepted_swap")
CELL_COLUMNS = ["scchart_structural_d", "scchart_structural_entropy", "scchart_programme_P",
                "scchart_programme_d", "scchart_programme_background_depth"]
NEW_ENTRIES = ("shuffled/",)


def synthetic_data():
    """Three populations (homogeneous, two-state, programme mixture) x 400 features."""
    rng = np.random.default_rng(0)
    n, f = 80, 400
    base = rng.random(f) * 0.06
    alt = np.roll(base, 100)
    prog = np.repeat(np.arange(4), f // 4)
    mix = np.vstack([np.where(prog == (k % 3), base * 4, base) for k in range(n)])

    def cells(p):
        return rng.binomial(3, np.clip(p, 0, 1)).astype(np.int64)

    def sample():
        X = sparse.csr_matrix(np.vstack([
            cells(np.tile(base, (n, 1))),
            cells(np.vstack([np.tile(base, (n // 2, 1)), np.tile(alt, (n - n // 2, 1))])),
            cells(mix)]))
        g = np.repeat(["a_homog", "b_two", "c_mix"], n)
        keep = np.asarray(X.sum(1)).ravel() > 0
        return X[keep], g[keep]

    X_ref, _ = sample()
    X, g = sample()
    external = np.array(["Shared"] * f, dtype=object)
    for k in range(3):
        external[prog == k] = f"P{k}"
    return X_ref, X, g, np.array([f"p{j}" for j in range(f)]), external


def _table(df: pd.DataFrame) -> dict:
    num = df.select_dtypes("number")
    return {"index": [str(i) for i in num.index], "columns": list(num.columns),
            "values": num.to_numpy(dtype=float).tolist()}


def workload() -> dict:
    """Numeric outputs: both axes, both samplers, calibration, matched depth, seeds."""
    import anndata as ad

    import scchart
    from scchart import _scoring

    X_ref, X, g, features, external = synthetic_data()
    proj = scchart.LSIProjection.fit(X_ref, n_components=8, random_state=0)

    def adata(X, g):
        a = ad.AnnData(X=X.copy(), obs=pd.DataFrame({"population": g}, index=np.arange(g.size).astype(str)),
                       var=pd.DataFrame({"programme": external}, index=features))
        a.obsm["X_lsi"] = proj(X)
        return a

    out: dict = {}
    for sampler in SAMPLERS:
        a = adata(X, g)
        model = scchart.Heterogeneity(use_rep="X_lsi", projection=proj, n_folds=3, null_sampler=sampler,
                                      verbose=False).fit(a, groupby="population")
        out[f"fit/{sampler}/cells"] = _table(a.obs[CELL_COLUMNS])
        out[f"fit/{sampler}/structural"] = _table(model.structural_)
        out[f"fit/{sampler}/programme"] = _table(model.programme_)
        e = adata(X, g)
        ext = scchart.Heterogeneity(structural=False, programme_key="programme", n_folds=3, null_sampler=sampler,
                                    verbose=False).fit(e, groupby="population")
        out[f"external/{sampler}/programme"] = _table(ext.programme_)
        ds = scchart.depth_standardized_structural_comparison(X, g, proj, 20.0, n_replicates=2, n_null=4,
                                                              sampler=sampler)
        out[f"matched_depth/{sampler}/structural"] = _table(ds.replicates)
        dp = scchart.depth_standardized_programme_comparison(X, g, model.references_, 20.0, n_replicates=2,
                                                             n_null=4, sampler=sampler)
        out[f"matched_depth/{sampler}/programme"] = _table(dp.replicates)
        null, _ = _scoring.swap_randomise(X[:50], 7, sampler=sampler)
        out[f"null_chain/{sampler}/sha256"] = hashlib.sha256(np.ascontiguousarray(null).tobytes()).hexdigest()
    out["seeds/seeds_from_key"] = np.asarray(_scoring.seeds_from_key((3, 1), 5)).tolist()

    # Unsorted labels: default (sorted) population order and the seeds it assigns.
    perm = np.random.default_rng(5).permutation(X.shape[0])
    a = adata(X[perm].tocsr(), g[perm])
    model = scchart.Heterogeneity(use_rep="X_lsi", projection=proj, n_folds=3, verbose=False)
    model.fit(a, groupby="population")
    out["shuffled/proposal_clock/structural"] = _table(model.structural_)
    out["shuffled/proposal_clock/programme"] = _table(model.programme_)
    out["shuffled/population_order"] = list(model.population_order_)
    out["shuffled/structural_seed_keys"] = np.asarray(a.uns["scchart"]["structural"]["seed_keys"]).tolist()
    out["shuffled/programme_base_seeds"] = np.asarray(a.uns["scchart"]["programme"]["base_seeds"]).tolist()
    return out


def provenance(name: str) -> str:
    return "new" if name.startswith(NEW_ENTRIES) else "established"


def _source_digest() -> str:
    """SHA-256 over the package sources, to identify the code that recorded the baseline."""
    pkg = Path(importlib.import_module("scchart").__file__).parent
    h = hashlib.sha256()
    for path in sorted(pkg.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def record() -> dict:
    """The baseline document: metadata and every workload entry with its provenance."""
    import scipy

    import scchart

    return {
        "description": "scCHART numerical baseline: a regression guard, not independently derived values. "
                       "See tests/baseline_workload.py.",
        "generated_with": f"scchart {scchart.__version__}",
        "source_sha256": _source_digest(),
        "generated_on": date.today().isoformat(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "pandas": pd.__version__, "platform": platform.platform()},
        "tolerance": {"rtol": RTOL, "atol": ATOL},
        "provenance_legend": {
            "established": "compared before recording with values established before scCHART 0.1.0 "
                           "(kept outside this repository); agreed within the tolerance",
            "new": "no earlier counterpart; recorded from scCHART 0.1.0 only"},
        "entries": {name: {"provenance": provenance(name), "value": value} for name, value in workload().items()},
    }


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--write":
        raise SystemExit("usage: python -m tests.baseline_workload --write PATH")
    with open(sys.argv[2], "w") as fh:
        json.dump(record(), fh, indent=1)
        fh.write("\n")
    print(f"wrote {sys.argv[2]}")
