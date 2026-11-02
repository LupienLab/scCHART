# scCHART

**scCHART** (single-cell Chromatin Heterogeneity Assessment and Ranking Tool) is a Python package based on an information-theoretic framework for quantifying heterogeneity in single-cell chromatin accessibility data.

scCHART measures variation within user-defined cell populations along two complementary axes:

- **Structural heterogeneity:** Quantifies variation in chromatin accessibility representations among cells within a defined population.
- **Regulatory programme heterogeneity:** Quantifies diversity in regulatory programme usage among cells within a defined population.

Additionally, scCHART:

- Provides both per-cell and population-level measurements of heterogeneity.
- Uses sampling-based null models to estimate heterogeneity expected from sparse observations.
- Supports depth-standardized comparisons of heterogeneity across cell populations.

The two axes are reported independently and are not combined into a single heterogeneity score.

## Installation

scCHART supports Python 3.10–3.13 and integrates with the [scverse ecosystem](https://scverse.org/) through [AnnData](https://anndata.readthedocs.io/).

Clone the repository and install scCHART with optional plotting dependencies:

```bash
git clone https://github.com/LupienLab/scCHART.git
cd scCHART
pip install -e ".[plot]"
```

Alternatively, install using [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/LupienLab/scCHART.git
cd scCHART
uv sync
```

Verify the installation:

```bash
python -c "import scchart; print(scchart.__version__)"
```

If using `uv`, verify with:

```bash
uv run python -c "import scchart; print(scchart.__version__)"
```

## Quick start

A toy data set with three populations of 150 cells and 2,000 peaks. Peaks 0–1,499 form three
programmes (`P0`, `P1`, `P2`); the rest are background. `homogeneous` cells share one
accessibility profile, half of the `two_states` cells use a second profile, and each
`programme_mix` cell opens one of the three programmes four times more often.

```python
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

import scchart

rng = np.random.default_rng(0)
base = rng.random(2000) * 0.04                            # accessibility rate of each peak
programme = np.repeat(["P0", "P1", "P2", "Shared"], 500)


def sample():
    homogeneous = np.tile(base, (150, 1))
    two_states = np.repeat([base, np.roll(base, 400)], 75, axis=0)
    programme_mix = np.array([np.where(programme == f"P{k % 3}", 4 * base, base) for k in range(150)])
    rates = np.vstack([homogeneous, two_states, programme_mix])
    return sparse.csr_matrix((rng.random(rates.shape) < rates).astype(np.int64))


reference = sample()                                      # independent cells, used to fit the projection
adata = ad.AnnData(
    sample(),                                             # integer counts, cells x peaks
    obs=pd.DataFrame({"population": np.repeat(["homogeneous", "two_states", "programme_mix"], 150)},
                     index=[f"cell{i}" for i in range(450)]),
    var=pd.DataFrame({"programme": programme}, index=[f"peak{j}" for j in range(2000)]),
)
```

The structural axis needs a representation in `adata.obsm` and, for the null model, the fixed
projection that produced it. The programme axis reads the programme of each peak from
`adata.var` (without `programme_key`, scCHART derives one programme per population).

```python
proj = scchart.LSIProjection.fit(reference, n_components=10, random_state=0)
adata.obsm["X_lsi"] = proj(adata.X)

model = scchart.Heterogeneity(use_rep="X_lsi", projection=proj, programme_key="programme", n_folds=3)
print(model)                                  # configuration and planned analyses
model.fit(adata, groupby="population")        # progress, then where the results are stored
```

Population results: observed divergence `J_obs`, null expectation `E0_J` and calibrated
excess `Delta_G` (structural), and the same for the programme axis. Here `two_states` has a
larger structural excess than `homogeneous`, and `programme_mix` has by far the largest
programme excess.

```python
print(model.structural_[["J_obs", "E0_J", "Delta_G"]])
print(model.programme_[["scored_fraction", "J_P_obs", "E0_J_P", "Delta_P", "median_P"]])
print(adata.obs[["scchart_structural_d", "scchart_programme_d", "scchart_programme_P"]].head())
```

Plots read the results stored on `adata` and return a Matplotlib `Axes`:

```python
ax = scchart.pl.calibration(adata, "programme")
# ax.figure.savefig("calibration.pdf")
```

[`examples/minimal_example.py`](examples/minimal_example.py) is a runnable version of this example.

## Examples

- [`examples/minimal_example.py`](examples/minimal_example.py): the Quick start above as a script that runs in seconds.
- [`examples/pbmc_tutorial/`](examples/pbmc_tutorial/): the [PBMC tutorial notebook](examples/pbmc_tutorial/pbmc_tutorial.ipynb), stored with the outputs of a full run, contains a complete analysis of the 10x Genomics PBMC 5k scATAC-seq dataset, including data preparation, structural and programme heterogeneity estimation, null calibration, visualization, and depth-standardized comparisons. Instructions to download the data and run it are in its [README](examples/pbmc_tutorial/README.md).

## Issues and feedback

Found a bug, encountered an installation problem, or have a suggestion for improving scCHART? Please [open an issue](https://github.com/LupienLab/scCHART/issues) on GitHub.

## Contributing

Contributions to scCHART are welcome, including bug fixes, documentation improvements, tests, and new features.

Please open a [GitHub issue](https://github.com/LupienLab/scCHART/issues) to discuss substantial changes before submitting a [pull request](https://github.com/LupienLab/scCHART/pulls). Contributions should include relevant tests and documentation updates where applicable.

## AI assistance acknowledgment

Generative AI tools were used to assist with code development, debugging, testing, and documentation preparation. The authors reviewed and remain responsible for the scientific methodology, implementation, and released software.

## Citation

Iyer A, Lupien M. *Quantifying Chromatin Heterogeneity from Population-Conditioned Structural Divergence and Regulatory Programme Diversity.* Manuscript in preparation.

## License

scCHART is distributed under the MIT License. See [LICENSE](LICENSE).