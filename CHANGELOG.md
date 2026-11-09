
# Changelog

All notable changes to scCHART are documented here.

## 0.2.0 (2026-11-09)

Performance and memory release. The methods are unchanged: all scores, null distributions, seeds and matched-depth results are byte-identical to 0.1.0 in every configuration checked:

- the PBMC 5k tutorial, at `n_jobs` 8 and 24;
- a synthetic regression harness, at `n_jobs` 1, 4 and 7;
- the test suite.

Each null chain keeps its own seed, so results do not depend on `n_jobs`.

### Measured improvement

PBMC 5k tutorial workload: 3,921 cells x 250,000 tiles, with the tutorial's seeds and null budgets, `n_jobs=24`, on an Apple M3 Ultra. Each figure is the mean of 3 runs per version:

| | 0.1.0 | 0.2.0 | change |
|---|---|---|---|
| Structural null calibration | 664 s | 400 s | 1.66x faster |
| Whole scCHART computation | 775 s | 472 s | 1.64x faster |
| Peak memory (RSS) | 10.64 GB | 4.06 GB | 62% lower |

Gains depend on the data, the hardware and `n_jobs`.

### Performance

- The fixed-margin chain keeps its matrix bit-packed (1 bit per cell x feature instead of 1 byte), stores each entry's row and column side by side as 32-bit integers, and draws its proposals a few steps ahead so that their entries are already cached when they are evaluated. Same draws, in the same order, and the same tests and updates as 0.1.0.
- The structural and programme nulls receive each null replicate as a sparse matrix built straight from the chain's bit-packed matrix (the same CSR matrix as before), and the chain input and diagnostics are read from the sparse input and the bit-packed matrix. No dense cells x features block is made on these paths. `swap_randomise` and `swap_replicates` still return dense blocks.
- The dense null block is built directly from the sparse input, without a float copy of the whole cells x features matrix (about 10 bytes per cell x feature before, 1 byte now).
- `binarize` makes one copy instead of two, and the programme axis no longer re-binarizes matrices that are already binary.
- The programme axis computes background depth, background rates and group detection rates without copying the background columns or converting the reference matrix to CSC.
- `ProgrammeReference` computes `background_mask`, `programme_index`, `membership_matrix()` and `sizes()` once instead of on every access.

### Added

- The programme null runs its chains in parallel. `n_jobs` is added to `programme_null_replicates`, `programme_null_calibration`, `crossfit_population_calibration` and `depth_standardized_programme_comparison`, and `Heterogeneity(n_jobs=...)` now applies to both nulls. Each chain keeps its own seed, so results do not depend on `n_jobs`.

### Changed

- `n_jobs` is validated: a positive integer, or `-1` for all numba threads. `0`, values below `-1` and non-integers (including `bool` and `float`) now raise `ValueError` / `TypeError`; 0.1.0 silently ran them serially (non-positive values) or truncated them with `int()`. Values above `numba.get_num_threads()` are capped at it, which only reduces memory.
- `ProgrammeReference.programme_of_feature` (the reference's own copy of the labels) is read-only; editing it in place raises. Assign a new array, or build a new reference, to change the labels.

### Documentation

- The PBMC tutorial was rerun with 0.2.0. Its results are identical to the 0.1.0 run, and the programme matched-depth comparison now passes `n_jobs`. The tutorial README reports the benchmark above.

## 0.1.0 (2026-11-02)

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
