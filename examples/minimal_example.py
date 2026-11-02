"""Minimal scCHART example on synthetic data (runs in a few seconds).

Three populations of 150 cells x 2,000 peaks. Peaks 0-1,499 form three
regulatory programmes of 500 peaks (P0, P1, P2); the rest are background.

* ``homogeneous``:   every cell is drawn from one accessibility profile.
* ``two_states``:    half the cells use one profile, half a different one
                     (structural heterogeneity).
* ``programme_mix``: each cell opens the peaks of one of the three
                     programmes four times more often (programme heterogeneity).

What the output shows: ``two_states`` has a larger structural excess
``Delta_G`` than ``homogeneous``, and ``programme_mix`` has by far the largest
programme excess ``Delta_P``. Many cells of the other two populations show no
programme evidence above background; they are unscorable (``scored_fraction``
< 1) and are never imputed.

Run from the repository root::

    uv run python examples/minimal_example.py                 # show the figure
    uv run python examples/minimal_example.py calibration.png # save it instead
"""

import sys

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

import scchart

# --- 1. A small AnnData: integer counts in X, populations in obs, programmes in var
rng = np.random.default_rng(0)
n, f = 150, 2000
base = rng.random(f) * 0.04                              # per-peak accessibility rate
programme = np.repeat(["P0", "P1", "P2", "Shared"], f // 4)
other = np.roll(base, f // 5)                            # a second, different profile


def sample():
    rates = np.vstack([
        np.tile(base, (n, 1)),                                                          # homogeneous
        np.vstack([np.tile(base, (n // 2, 1)), np.tile(other, (n - n // 2, 1))]),       # two_states
        np.vstack([np.where(programme == f"P{k % 3}", 4 * base, base) for k in range(n)]),  # programme_mix
    ])
    return sparse.csr_matrix((rng.random(rates.shape) < rates).astype(np.int64))


X_ref = sample()       # independent cells from the same populations, used only to fit the projection
adata = ad.AnnData(
    X=sample(),
    obs=pd.DataFrame({"population": np.repeat(["homogeneous", "two_states", "programme_mix"], n)},
                     index=[f"cell{i}" for i in range(3 * n)]),
    var=pd.DataFrame({"programme": programme}, index=[f"peak{j}" for j in range(f)]),
)

# The structural axis needs a representation Z and, for the null, the fixed
# projection that produced it. Fitting it on independent cells avoids fitting
# the noise of the cells being scored.
proj = scchart.LSIProjection.fit(X_ref, n_components=10, random_state=0)
adata.obsm["X_lsi"] = proj(adata.X)

# --- 2. Configure and fit both axes, with null calibration
model = scchart.Heterogeneity(
    use_rep="X_lsi",
    projection=proj,
    programme_key="programme",    # adata.var column with the programme of each peak
    n_folds=3,
    verbose=False,                # True prints the configuration, stages and output locations
)
model.fit(adata, groupby="population")

# --- 3. Population results (bits): observed J, null expectation E0[J], excess Delta
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 10)
print("Structural axis")
print(model.structural_[["n_cells", "J_obs", "E0_J", "Delta_G"]].round(4))
print("\nProgramme axis (median_P = median within-cell mixing P_i)")
print(model.programme_[["scored_fraction", "J_P_obs", "E0_J_P", "Delta_P", "median_P"]].round(4))

# Per-cell outputs are in adata.obs. The divergences d_i average to their
# population's J (programme: over scored cells; unscorable cells are NaN).
per_cell = ["scchart_structural_d", "scchart_programme_d", "scchart_programme_P"]
print("\nPopulation means of the per-cell outputs")
print(adata.obs.groupby("population")[per_cell].mean().round(4))

# --- 4. Figures from the results stored on adata
scchart.pl.set_style()        # Arial, thin lines, editable PDF/SVG text, 600 dpi PNG
fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.0))
fig.subplots_adjust(wspace=0.75)
scchart.pl.calibration(adata, "structural", ax=axes[0])
scchart.pl.calibration(adata, "programme", ax=axes[1])
if len(sys.argv) > 1:
    fig.savefig(sys.argv[1])
    print(f"\nsaved {sys.argv[1]}")
else:
    plt.show()
