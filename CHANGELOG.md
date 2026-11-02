
# Changelog

All notable changes to scCHART are documented here.

## 0.1.0 (unreleased)

Initial release of **scCHART** (single-cell Chromatin Heterogeneity Assessment and Ranking Tool), an information-theoretic framework for quantifying structural and regulatory programme heterogeneity from single-cell chromatin accessibility data.

### Features

- **Unified interface:** `scchart.Heterogeneity` provides a common interface for estimating structural and regulatory programme heterogeneity from an `AnnData` object using `model.fit(adata, groupby=...)`. Both axes can also be analyzed independently.

- **Structural heterogeneity:** `StructuralHeterogeneity` quantifies variation in chromatin accessibility representations through per-cell divergence `d_i(G)` and population-level divergence `J_G`. `LSIProjection` supports fixed projections for null calibration.

- **Regulatory programme heterogeneity:** `ProgrammeHeterogeneity` quantifies within-cell programme mixing `P_i`, per-cell programme divergence `d_i^(P)`, and population-level divergence `J_G^(P)`. Supports endogenous cross-fitted, external, and fixed programme references.

- **Sampling-based null calibration:** Estimates observed heterogeneity `J`, null expectation `E0[J]`, and calibrated excess `Delta` for each population using fixed-margin sampling. Supports proposal-clock and accepted-swap samplers.

- **Depth-standardized comparisons:** Provides `depth_standardized_structural_comparison` and `depth_standardized_programme_comparison` for evaluating heterogeneity across populations at matched sequencing depth.

- **Per-cell and population-level outputs:** Stores results directly in `AnnData` using the `scchart_` prefix across `obs`, `obsm`, and `uns`, including population summaries, null calibration results, and relevant analysis metadata. Results are compatible with `AnnData.write_h5ad()`.

- **Visualization:** Introduces the `scchart.pl` plotting namespace with:
  - `embedding`
  - `programme_composition`
  - `calibration`
  - `null_distribution`
  - `structural_vs_programme`
  - `set_style`

- **Examples:** Includes a runnable minimal example and a PBMC 5k scATAC-seq tutorial notebook.

### Validation

- Unit tests covering structural and programme heterogeneity, null calibration, population ordering, and reproducibility.
- Analytical checks using hand-calculated divergence values for small populations.
- Exact enumeration of small fixed-margin state spaces to examine null expectations and sampler behavior.
- Numerical regression tests using deterministic synthetic datasets, including both heterogeneity axes, null sampling, calibration, and depth-standardized comparisons.

### Availability

- **Source code:** https://github.com/LupienLab/scCHART
- **License:** MIT
- **Supported Python versions:** 3.10–3.13
