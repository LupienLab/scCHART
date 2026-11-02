"""Population-conditioned structural heterogeneity by information divergence.

This module implements the scCHART structural axis: each cell's divergence
from the consensus distribution of its declared population ``G``, the
population divergence ``J_G``, its technical null calibration ``Delta_G``
and the matched-depth cross-population comparison ``Delta_G(D*)``.

Representation Z (caller-declared)
----------------------------------
The scorer consumes a pre-computed representation from
``adata.obsm[use_rep]`` and uses every supplied dimension exactly as
provided. It does not compute, select, drop, reweight or rescale
components. Representation construction and feature selection are upstream
choices; scCHART does not validate whether Z preserves biology from the
accessibility matrix. All scores are conditional on the declared Z and
feature universe.

Simplex adapters
----------------
A simplex adapter maps each cell's ``z_i`` in R^D to a probability vector
``p_i``. Two adapters are implemented; both feed the same KL /
generalized-JSD engine.

``"signed_channel"`` (default)
    ``2D`` channels, one positive and one negative per component::

        q_i = [max(z_i1,0)^2, ..., max(z_iD,0)^2, max(-z_i1,0)^2, ..., max(-z_iD,0)^2]
        p_i = q_i / sum(q_i)

    Keeps the relative sign between cells (``z`` and ``-z`` have disjoint
    support). A global sign flip of a component permutes its two channels
    for every cell, so ``d_i`` and ``J_G`` are unchanged. Invariant to
    per-cell and global positive rescaling; **not** rotation invariant.

``"spectral_energy"``
    ``D`` channels, ``p_ik = z_ik^2 / sum_l z_il^2``. Invariant to positive
    rescaling and to the sign of every coordinate, so ``z`` and ``-z`` map
    to the same ``p``; antipodal states therefore coincide.

No pseudocount is used. A cell with ``z_i = 0`` has no defined ``p_i`` and
raises an error; it is never assigned an artificial distribution.

Structural estimand
-------------------
For a declared population ``G``::

    pbar_G = mean_{i in G} p_i
    d_i(G) = KL(p_i || pbar_G)
    J_G    = mean_{i in G} d_i(G) = H(pbar_G) - mean_{i in G} H(p_i)

``J_G`` is the equal-weight generalized Jensen-Shannon divergence, with
log base 2 (bits). The grouping is part of the estimand. ``d_i`` is the
exact per-cell decomposition of ``J_G``; it is **not** a calibrated
per-cell heterogeneity score. :func:`structural_chain_rule` gives the
exact within/between decomposition for a finer partition of ``G``.

Technical null and Delta_G
--------------------------
:func:`structural_null_calibration` evaluates, for the cells of one
population:

* ``B = 1[X > 0]``;
* 2 x 2 degree-preserving swaps within ``G`` (default proposal clock,
  ``100 * nnz`` proposals per replicate; 20 replicates, deterministic
  seeds, exact margins verified; see
  :func:`scchart._scoring.swap_randomise`);
* every null cell projected into the **same fixed** representation as the
  observed cells (for example :class:`LSIProjection`; no refit);
* ``Delta_G = J_G,obs - E0[J_G]``, where ``E0[J_G]`` is the mean null
  ``J_G``.

Observed and null quantities must share one representation: a supplied
``Z`` is accepted for calibration only if it equals ``project(B)``.
``Delta_G / sd0`` and ``J_obs / E0[J]`` are diagnostics only, not
heterogeneity magnitudes.

Cross-population comparison
---------------------------
:func:`depth_standardized_structural_comparison` evaluates every
population at a declared common effective depth ``D*`` by downsampling
only (never upsampling).

Outputs of :class:`StructuralHeterogeneity`
-------------------------------------------
``adata.obs['scchart_structural_d']``
    Raw ``d_i(G)`` in bits (decomposition term of ``J_G``).
``adata.obs['scchart_structural_entropy']``
    Shannon entropy ``H(p_i)`` in bits.
``adata.obs['scchart_structural_population_J_obs']``
    Raw ``J_obs`` (``J_G``) of the cell's population, repeated per cell.
``adata.obsm['scchart_structural_probabilities']``
    Cell-by-channel probabilities ``p_i``, when
    ``store_probabilities=True``.
``adata.uns['scchart_structural']``
    Provenance: grouping, representation key and dimensionality, adapter,
    log base, units, calibration status, population summaries and
    consensus.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from scipy.sparse.linalg import svds

from ._scoring import (
    N_NULL_REPLICATES,
    DEFAULT_SAMPLER,
    DepthStandardizedResult,
    binarize,
    binomial_downsample,
    check_labels,
    count_depth,
    downsampling_fractions,
    entropy_bits,
    generalized_jsd,
    kl_bits,
    pairwise_replicate_agreement,
    seed_key as _seed_key,
    seeds_from_key,
    swap_replicates,
)


DEFAULT_ADAPTER = "signed_channel"
ADAPTERS = ("signed_channel", "spectral_energy")
PROBABILITY_DEFINITIONS = {
    "signed_channel": "q_i = [max(z_i,0)^2, max(-z_i,0)^2] (2D channels); p_i = q_i / sum q_i",
    "spectral_energy": "p_ik = z_ik^2 / sum_l z_il^2",
}
REPRESENTATION_STATUS = "caller-declared; used as supplied"
METHOD_NAME = "scchart_structural_information_divergence"

Projector = Callable[[object], np.ndarray]


@dataclass(frozen=True)
class StructuralHeterogeneityResult:
    """Container returned by ``StructuralHeterogeneity.fit_transform``.

    Returned when ``fit_transform`` is called with ``inplace=False``.

    Attributes
    ----------
    scores : pandas.DataFrame
        Per-cell structural outputs indexed by ``adata.obs_names``.
    population_summary : pandas.DataFrame
        One row per declared population (sorted) with ``J_obs``,
        consensus entropy, mean cell entropy and the generalized-JSD
        identity check.
    probabilities : pandas.DataFrame
        Cell-by-channel probability matrix ``p_i``.
    """

    scores: pd.DataFrame
    population_summary: pd.DataFrame
    probabilities: pd.DataFrame


# ---------------------------------------------------------------------------
# Simplex adapters
# ---------------------------------------------------------------------------


def _check_representation(Z) -> np.ndarray:
    """Return ``Z`` as a float64 matrix, rejecting non-2D or non-finite input."""
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError("Representation must be a two-dimensional matrix.")
    if not np.isfinite(Z).all():
        raise ValueError("Representation contains non-finite values.")
    return Z


def _normalise_energy(E: np.ndarray, name: str) -> np.ndarray:
    """Normalize non-negative channel energies to probabilities, row by row.

    Raises ``ValueError`` if any row has zero total energy (``z_i = 0``);
    ``name`` labels the adapter in the error message.
    """
    total = E.sum(axis=1, keepdims=True)
    bad = np.flatnonzero(total.ravel() <= 0)
    if bad.size:
        raise ValueError(f"{bad.size} cell(s) have z_i = 0 (zero {name} energy); p_i is undefined")
    return E / total


def spectral_energy_probability(Z: np.ndarray) -> np.ndarray:
    """Map a representation to the simplex with the spectral-energy adapter.

        p_ik = z_ik^2 / sum_l z_il^2

    Parameters
    ----------
    Z : numpy.ndarray
        ``(n_cells, D)`` representation matrix.

    Returns
    -------
    numpy.ndarray
        ``(n_cells, D)`` probability matrix.

    Raises
    ------
    ValueError
        If ``Z`` is not two-dimensional, contains non-finite values, or has
        a row with ``z_i = 0`` (no defined ``p_i``; never replaced).

    Notes
    -----
    The adapter is invariant to positive rescaling of each cell and to the
    sign of every coordinate, so ``z`` and ``-z`` give the same ``p``.
    """
    Z = _check_representation(Z)
    return _normalise_energy(Z * Z, "spectral")


def signed_channel_probability(Z: np.ndarray) -> np.ndarray:
    """Map a representation to the simplex with the signed-channel adapter.

        q_i = [max(z_i, 0)^2, max(-z_i, 0)^2]
        p_i = q_i / sum(q_i)

    This is the default structural adapter.

    Parameters
    ----------
    Z : numpy.ndarray
        ``(n_cells, D)`` representation matrix.

    Returns
    -------
    numpy.ndarray
        ``(n_cells, 2D)`` probability matrix. Columns ``0..D-1`` are the
        positive channels of components ``1..D``; columns ``D..2D-1`` are
        the negative channels.

    Raises
    ------
    ValueError
        If ``Z`` is not two-dimensional, contains non-finite values, or has
        a row with ``z_i = 0`` (no defined ``p_i``; never replaced).

    Notes
    -----
    The adapter is invariant to positive rescaling of each cell. It keeps
    the sign of each coordinate, so ``z`` and ``-z`` have disjoint support.
    It is not rotation invariant.
    """
    Z = _check_representation(Z)
    Q = np.hstack([np.maximum(Z, 0.0) ** 2, np.maximum(-Z, 0.0) ** 2])
    return _normalise_energy(Q, "signed-channel")


def adapter_probability(Z: np.ndarray, adapter: str = DEFAULT_ADAPTER) -> np.ndarray:
    """Apply the named simplex adapter.

    Parameters
    ----------
    Z : numpy.ndarray
        ``(n_cells, D)`` representation matrix.
    adapter : {"signed_channel", "spectral_energy"}, optional
        Simplex adapter. Default ``"signed_channel"``.

    Returns
    -------
    numpy.ndarray
        Cell-by-channel probability matrix.

    Raises
    ------
    ValueError
        If ``adapter`` is unknown, or the adapter rejects ``Z``.
    """
    if adapter == "signed_channel":
        return signed_channel_probability(Z)
    if adapter == "spectral_energy":
        return spectral_energy_probability(Z)
    raise ValueError(f"adapter must be one of {ADAPTERS}, got {adapter!r}")


def _probability_columns(adapter: str, n_dimensions: int) -> list[str]:
    """Return the channel column names for an adapter and dimensionality."""
    if adapter == "signed_channel":
        return ([f"component_{i + 1}_pos" for i in range(n_dimensions)]
                + [f"component_{i + 1}_neg" for i in range(n_dimensions)])
    return [f"component_{i + 1}" for i in range(n_dimensions)]


# ---------------------------------------------------------------------------
# Population KL / generalized JSD engine (shared by both adapters)
# ---------------------------------------------------------------------------


def _validate_probabilities(P: np.ndarray, atol: float = 1e-10) -> np.ndarray:
    """Check that ``P`` is a 2D matrix of finite, non-negative rows summing to 1.

    Rows must sum to 1 within absolute tolerance ``atol``; a
    ``ValueError`` is raised otherwise.
    """
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional matrix.")
    if not np.isfinite(P).all() or np.any(P < 0):
        raise ValueError("probabilities must be finite and non-negative.")
    if not np.allclose(P.sum(axis=1), 1.0, rtol=0.0, atol=atol):
        raise ValueError("every probability row must sum to 1.")
    return P


def _population_divergence(
    P: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[str, np.ndarray]]:
    """Evaluate the structural estimand population by population.

    Parameters
    ----------
    P : numpy.ndarray
        Cell-by-channel probability matrix.
    labels : numpy.ndarray
        Population label of each cell.

    Returns
    -------
    cell_kl : numpy.ndarray
        ``d_i(G) = KL(p_i || pbar_G)`` per cell, in bits.
    cell_entropy : numpy.ndarray
        ``H(p_i)`` per cell, in bits.
    summary : pandas.DataFrame
        One row per population, in order of first appearance, with
        ``n_cells``, ``J_structural_bits``, ``consensus_entropy_bits``,
        ``mean_cell_entropy_bits``, ``gjds_identity_bits`` and
        ``identity_error``.
    population_means : dict
        Consensus ``pbar_G`` of each population.
    """
    n = P.shape[0]
    cell_kl = np.empty(n, dtype=np.float64)
    cell_entropy = entropy_bits(P)
    population_means: Dict[str, np.ndarray] = {}
    rows = []

    for group in pd.unique(labels):
        mask = labels == group
        out = generalized_jsd(P[mask])
        population_means[str(group)] = out["pbar"]
        cell_kl[mask] = out["d"]
        rows.append(
            {
                "population": str(group),
                "n_cells": int(mask.sum()),
                "J_structural_bits": out["J"],
                "consensus_entropy_bits": out["H_consensus"],
                "mean_cell_entropy_bits": out["H_cells_mean"],
                "gjds_identity_bits": out["H_consensus"] - out["H_cells_mean"],
                "identity_error": out["identity_error"],
            }
        )

    summary = pd.DataFrame(rows).set_index("population")
    return cell_kl, cell_entropy, summary, population_means


def structural_J(Z: np.ndarray, *, adapter: str = DEFAULT_ADAPTER) -> dict:
    """Compute the generalized JSD of one population's representation.

    Parameters
    ----------
    Z : numpy.ndarray
        ``(n_cells, D)`` representation of the population's cells.
    adapter : {"signed_channel", "spectral_energy"}, optional
        Simplex adapter. Default ``"signed_channel"``.

    Returns
    -------
    dict
        Output of :func:`scchart._scoring.generalized_jsd`: ``d``, ``J``,
        ``pbar``, ``H_consensus``, ``H_cells_mean`` and ``identity_error``.
    """
    return generalized_jsd(adapter_probability(Z, adapter))


def structural_chain_rule(
    probabilities: np.ndarray,
    groups: Sequence,
    subgroups: Sequence,
) -> pd.DataFrame:
    """Decompose ``J_G`` exactly into within- and between-subgroup terms.

    Each population ``G`` is partitioned into the sub-populations ``c``
    given by ``subgroups`` among the cells of ``G``. With
    ``w_c = N_c / N_G``::

        J_G = sum_c w_c J_c  +  sum_c w_c KL(pbar_c || pbar_G)
              (within)          (between)

    Parameters
    ----------
    probabilities : numpy.ndarray
        Cell-by-channel probability matrix ``p_i`` (for example
        ``adata.obsm['scchart_structural_probabilities']``). Rows must be
        non-negative and sum to one.
    groups : sequence
        Population ``G`` of each cell.
    subgroups : sequence
        Finer label of each cell defining the partition of its ``G``.

    Returns
    -------
    pandas.DataFrame
        One row per population with ``n_cells``, ``n_subgroups``,
        ``J_structural_bits``, ``within_bits``, ``between_bits`` and
        ``decomposition_error = |J - (within + between)|``. Values are in
        bits.

    Raises
    ------
    ValueError
        If ``probabilities`` is not a valid probability matrix, or
        ``groups`` / ``subgroups`` do not have one entry per cell.
    """
    P = _validate_probabilities(probabilities)
    groups = np.asarray(groups).astype(str)
    subgroups = np.asarray(subgroups).astype(str)
    if len(groups) != P.shape[0] or len(subgroups) != P.shape[0]:
        raise ValueError("groups and subgroups must have one entry per cell.")

    rows = []
    for group in pd.unique(groups):
        mask = groups == group
        Pg = P[mask]
        sub = subgroups[mask]
        n_g = Pg.shape[0]
        total = generalized_jsd(Pg)

        within = 0.0
        between = 0.0
        subs = pd.unique(sub)
        for c in subs:
            part = generalized_jsd(Pg[sub == c])
            w = int((sub == c).sum()) / n_g
            within += w * part["J"]
            between += w * float(kl_bits(part["pbar"][None, :], total["pbar"])[0])

        rows.append(
            {
                "population": str(group),
                "n_cells": int(n_g),
                "n_subgroups": int(len(subs)),
                "J_structural_bits": total["J"],
                "within_bits": within,
                "between_bits": between,
                "decomposition_error": float(abs(total["J"] - (within + between))),
            }
        )
    return pd.DataFrame(rows).set_index("population")


# ---------------------------------------------------------------------------
# Fixed projection B -> Z
# ---------------------------------------------------------------------------


@dataclass
class LSIProjection:
    """Fixed TF-IDF / LSI projection ``B -> Z`` (no refit on transform).

    The structural null projects every null cell into the **same fixed**
    representation as the observed cells: same feature universe, IDF,
    loadings and retained components. With binary TF and
    ``B = 1[X > 0]``::

        tf_ij  = B_ij / sum_j B_ij
        idf_j  = ln(1 + N_ref / f_j)     (f_j = reference cells with feature j open)
        Z      = (tf * idf) @ U[:, components]

    ``U`` holds the top right-singular vectors of the reference TF-IDF
    matrix. Choosing the reference cells, the number of components and
    which components to retain is an upstream representation choice.
    Because the null preserves column sums, the IDF is unchanged by
    construction. Any callable ``project(B) -> Z`` with the same
    fixed-projection property can be used in its place.

    Attributes
    ----------
    idf : numpy.ndarray
        ``(n_features,)`` inverse document frequency from the reference
        cells.
    loadings : numpy.ndarray
        ``(n_features, n_fitted)`` right-singular vectors ``U``.
    components : numpy.ndarray
        0-based indices of the retained columns of ``loadings``.
    singular_values : numpy.ndarray, optional
        Singular values of the fitted components (descending), for
        provenance.
    provenance : dict
        Description of how the projection was fitted.

    Raises
    ------
    ValueError
        If ``loadings`` and ``idf`` do not have the same number of features.

    Examples
    --------
    >>> proj = LSIProjection.fit(adata.X, n_components=50, random_state=1)
    >>> adata.obsm["X_lsi"] = proj(adata.X)
    >>> proj = proj.with_components([2, 3, 4])      # keep a subset, still fixed
    >>> scchart.Heterogeneity(use_rep="X_lsi", projection=proj)
    """

    idf: np.ndarray
    loadings: np.ndarray
    components: np.ndarray
    singular_values: Optional[np.ndarray] = None
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.idf = np.asarray(self.idf, dtype=np.float64).ravel()
        self.loadings = np.asarray(self.loadings, dtype=np.float64)
        self.components = np.asarray(self.components, dtype=np.int64).ravel()
        if self.loadings.shape[0] != self.idf.size:
            raise ValueError("loadings and idf must have one row per feature")

    @property
    def n_features(self) -> int:
        """Number of features in the projection's feature universe."""
        return int(self.idf.size)

    @property
    def n_components(self) -> int:
        """Number of retained components (columns of ``Z``)."""
        return int(self.components.size)

    def transform(self, B) -> np.ndarray:
        """Project cells into the fixed representation ``Z``.

        Parameters
        ----------
        B : array-like or scipy.sparse matrix
            Binary (or count; binarized as ``B > 0``) cells x features matrix
            over the projection's feature universe.

        Returns
        -------
        numpy.ndarray
            ``(n_cells, n_components)`` representation.

        Raises
        ------
        ValueError
            If the feature dimension does not match, or a cell has no open
            feature (no TF is defined; never imputed).
        """
        already_binary = sparse.issparse(B) and B.dtype == np.float64 and np.all(B.data == 1.0)
        B = sparse.csr_matrix(B) if already_binary else binarize(B)
        if B.shape[1] != self.n_features:
            raise ValueError(f"B has {B.shape[1]} features, projection expects {self.n_features}")
        n_open = np.asarray(B.sum(axis=1)).ravel()
        if np.any(n_open <= 0):
            raise ValueError(f"{int((n_open <= 0).sum())} cell(s) have no open feature; "
                             "they have no defined projection")
        T = sparse.diags(1.0 / n_open) @ B @ sparse.diags(self.idf)
        return np.asarray(T @ self.loadings)[:, self.components]

    __call__ = transform

    @classmethod
    def fit(cls, X, *, n_components: int = 50, components: Optional[Sequence[int]] = None,
            align_to: Optional[np.ndarray] = None, random_state: int = 1) -> "LSIProjection":
        """Fit the fixed projection on reference cells.

        Parameters
        ----------
        X : scipy.sparse matrix
            Reference cells x features counts or binary matrix (binarized).
        n_components : int, optional
            Number of singular vectors fitted with
            ``scipy.sparse.linalg.svds``. Default 50.
        components : sequence of int, optional
            0-based components to retain. Default: all fitted components.
            Dropping depth-associated components is an upstream choice made by
            the caller.
        align_to : numpy.ndarray, optional
            ``(n_ref, n_components)`` stored coordinates. The sign of each
            fitted component is aligned to it, since the SVD sign is arbitrary.
        random_state : int, optional
            Seed of the ``svds`` start vector. Default 1.

        Returns
        -------
        LSIProjection
            Fitted projection, with components in descending order of singular
            value.

        Raises
        ------
        ValueError
            If a reference cell has no open feature.
        """
        B = binarize(X)
        n_open = np.asarray(B.sum(axis=1)).ravel()
        if np.any(n_open <= 0):
            raise ValueError("reference cells must each have at least one open feature")
        freq = np.asarray(B.sum(axis=0)).ravel()
        idf = np.log(1.0 + B.shape[0] / np.maximum(freq, 1))
        T = (sparse.diags(1.0 / n_open) @ B @ sparse.diags(idf)).tocsr()
        U, s, _ = svds(T.T.tocsc(), k=n_components, random_state=random_state)
        order = np.argsort(-s)
        U, s = U[:, order], s[order]
        if align_to is not None:
            Zp = np.asarray(T @ U)
            sign = np.sign(np.einsum("ij,ij->j", Zp, np.asarray(align_to, dtype=np.float64)))
            sign[sign == 0] = 1.0
            U = U * sign
        comps = np.arange(n_components) if components is None else np.asarray(components, dtype=np.int64)
        return cls(idf=idf, loadings=U, components=comps, singular_values=s,
                   provenance={"recipe": "TF (binary / n_open) x IDF log(1 + N/f), truncated SVD",
                               "n_reference_cells": int(B.shape[0]), "n_fitted": int(n_components),
                               "svds_random_state": int(random_state)})

    def with_components(self, components: Sequence[int]) -> "LSIProjection":
        """Return the same fitted projection with a different component subset.

        Parameters
        ----------
        components : sequence of int
            0-based indices of the components to retain.

        Returns
        -------
        LSIProjection
            New projection sharing the IDF, loadings and singular values.
        """
        return LSIProjection(self.idf, self.loadings, np.asarray(components, dtype=np.int64),
                             self.singular_values, dict(self.provenance))


# ---------------------------------------------------------------------------
# Technical null calibration
# ---------------------------------------------------------------------------


@dataclass
class StructuralCalibration:
    """Calibrated structural heterogeneity of one population.

    Attributes
    ----------
    J_obs : float
        Observed ``J_G`` (bits) in the fixed projection.
    E0_J : float
        Technical null expectation ``E0[J_G]`` (mean over replicates).
    Delta : float
        ``Delta_G = J_obs - E0_J``, the primary calibrated quantity.
    sd0 : float
        Standard deviation of the null ``J_G`` (diagnostic; NaN with one
        replicate).
    J_null : numpy.ndarray
        Null ``J_G`` per replicate.
    d_obs : numpy.ndarray
        Observed per-cell ``d_i(G)`` (decomposition terms, not calibrated).
    n_cells : int
        Number of cells in the population.
    nnz : int
        Number of open entries in the binary block.
    median_n_open : float
        Median number of open features per cell.
    margins_exact_all : bool
        True when every null replicate preserved the margins exactly.
    chains_complete_all : bool
        True when every chain used its full budget.
    acceptance_mean : float
        Mean fraction of accepted proposals across replicates.
    retained_mean : float
        Mean fraction of observed open entries still open in the null.
    seeds : numpy.ndarray
        Chain seed of each replicate.
    adapter : str
        Simplex adapter used for observed and null cells.
    extra : dict
        Additional information.
    sampler : str
        Fixed-margin null sampler.
    proposals_mean : float
        Mean number of proposals per replicate.
    """

    J_obs: float
    E0_J: float
    Delta: float
    sd0: float
    J_null: np.ndarray
    d_obs: np.ndarray
    n_cells: int
    nnz: int
    median_n_open: float
    margins_exact_all: bool
    chains_complete_all: bool
    acceptance_mean: float
    retained_mean: float
    seeds: np.ndarray
    adapter: str = DEFAULT_ADAPTER
    extra: dict = field(default_factory=dict)
    sampler: str = DEFAULT_SAMPLER
    proposals_mean: float = np.nan

    @property
    def R_G(self) -> float:
        """Ratio ``J_obs / E0[J_G]``; a secondary diagnostic, NaN if ``E0_J <= 0``."""
        return self.J_obs / self.E0_J if self.E0_J > 0 else np.nan

    @property
    def Z_G_cal(self) -> float:
        """Detection diagnostic ``Delta / sd0``; not a heterogeneity magnitude.

        NaN if ``sd0`` is not positive.
        """
        return self.Delta / self.sd0 if self.sd0 > 0 else np.nan

    def summary(self) -> dict:
        """Return the calibration and its diagnostics as a flat dictionary."""
        return {"n_cells": self.n_cells, "median_n_open": self.median_n_open, "J_obs": self.J_obs,
                "E0_J": self.E0_J, "Delta_G": self.Delta, "sd0_diagnostic": self.sd0,
                "R_G_diagnostic": self.R_G, "Z_G_cal_diagnostic": self.Z_G_cal,
                "n_null": int(self.J_null.size), "nnz": self.nnz,
                "margins_exact_all_nulls": self.margins_exact_all,
                "chains_complete_all": self.chains_complete_all,
                "acceptance_mean": self.acceptance_mean, "retained_fraction_mean": self.retained_mean,
                "null_sampler": self.sampler, "proposals_mean": self.proposals_mean}


def check_same_representation(Z: np.ndarray, Z_projected: np.ndarray, *, rtol: float = 1e-8,
                              atol: float = 1e-10) -> None:
    """Check that a supplied ``Z`` is the fixed projection of the same cells.

    Observed and null structural quantities must be evaluated in the same
    declared representation, so a supplied ``Z`` is accepted for
    calibration only if it equals ``project(B)``.

    Parameters
    ----------
    Z : numpy.ndarray
        Supplied representation.
    Z_projected : numpy.ndarray
        ``project(B)`` for the same cells.
    rtol, atol : float, optional
        Tolerances passed to ``numpy.allclose``.

    Raises
    ------
    ValueError
        If the shapes differ or the values are not close.
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.shape != Z_projected.shape or not np.allclose(Z, Z_projected, rtol=rtol, atol=atol):
        raise ValueError("the supplied Z is not project(B): observed and null structural quantities "
                         "must be evaluated in the same representation. Pass Z = project(X), or "
                         "report raw J_G on Z without calibration.")


def structural_null_calibration(B, project: Projector, *, n_replicates: int = N_NULL_REPLICATES,
                                sampler: str = DEFAULT_SAMPLER, n_mult: Optional[float] = None, seed_key=0,
                                seeds: Optional[Sequence[int]] = None, n_jobs: int = 1,
                                adapter: str = DEFAULT_ADAPTER,
                                Z_obs: Optional[np.ndarray] = None) -> StructuralCalibration:
    """Calibrate ``J_G`` against the technical null for one population.

    The observed ``J_G`` is computed on ``project(B)``. Each null replicate
    randomizes ``B`` with a fixed-margin swap chain, projects it with the
    same ``project`` and recomputes ``J_G``. ``E0[J_G]`` is the mean null
    value and ``Delta_G = J_obs - E0[J_G]``.

    Parameters
    ----------
    B : array-like or scipy.sparse matrix
        Cells of **one** population x declared feature universe; binarized
        as ``B > 0``. Every cell needs at least one open feature.
    project : callable
        Fixed projection ``B -> Z`` (for example :class:`LSIProjection`).
        It is applied unchanged to observed and null cells; it must not
        refit.
    n_replicates : int, optional
        Number of null replicates. Default 20.
    sampler : {"proposal_clock", "accepted_swap"}, optional
        Fixed-margin chain (see :func:`scchart._scoring.swap_randomise`).
        Default ``"proposal_clock"``.
    n_mult : float, optional
        Budget per occupied entry. Default 100 proposals (proposal clock)
        or 10 accepted swaps (accepted-swap sampler).
    seed_key : int or tuple of int, optional
        Replicate seeds are ``default_rng((*seed_key, b)).integers(2**31 - 1)``.
        Ignored when ``seeds`` is given. Default 0.
    seeds : sequence of int, optional
        Explicit chain seeds, one per replicate.
    n_jobs : int, optional
        Chains run in parallel batches of this size (memory about
        ``n_jobs`` dense copies of ``B``). Default 1.
    adapter : {"signed_channel", "spectral_energy"}, optional
        Simplex adapter, used for both observed and null cells.
    Z_obs : numpy.ndarray, optional
        Observed representation, checked against ``project(B)``.

    Returns
    -------
    StructuralCalibration
        Observed and null ``J_G``, ``Delta_G`` and chain diagnostics.

    Raises
    ------
    ValueError
        If a cell has no open feature, or ``Z_obs`` differs from
        ``project(B)``.
    """
    Bb = binarize(B)
    n_open = np.asarray(Bb.sum(axis=1)).ravel()
    if np.any(n_open <= 0):
        raise ValueError(f"{int((n_open <= 0).sum())} cell(s) have no open feature; remove them "
                         "explicitly before calibration")
    Z = np.asarray(project(Bb), dtype=np.float64)
    if Z_obs is not None:
        check_same_representation(Z_obs, Z)
    obs = structural_J(Z, adapter=adapter)
    seeds = seeds_from_key(seed_key, n_replicates) if seeds is None else np.asarray(seeds, np.int64)
    jn, diags = [], []
    for D, dg in swap_replicates(Bb, seeds, sampler=sampler, n_mult=n_mult, n_jobs=n_jobs):
        jn.append(structural_J(project(sparse.csr_matrix(D, dtype=np.float64)), adapter=adapter)["J"])
        diags.append(dg)
    jn = np.asarray(jn)
    mu = float(jn.mean())
    sd = float(jn.std(ddof=1)) if jn.size > 1 else np.nan
    return StructuralCalibration(
        J_obs=obs["J"], E0_J=mu, Delta=obs["J"] - mu, sd0=sd, J_null=jn, d_obs=obs["d"],
        n_cells=int(Bb.shape[0]), nnz=int(Bb.nnz), median_n_open=float(np.median(n_open)),
        margins_exact_all=all(d.margins_exact for d in diags),
        chains_complete_all=all(d.complete for d in diags),
        acceptance_mean=float(np.mean([d.acceptance for d in diags])),
        retained_mean=float(np.mean([d.retained for d in diags])), seeds=seeds, adapter=adapter,
        sampler=sampler, proposals_mean=float(np.mean([d.attempts for d in diags])))


def structural_population_calibration(X, groups, project: Projector, *,
                                      populations: Optional[Sequence[str]] = None,
                                      seed=0, n_replicates: int = N_NULL_REPLICATES,
                                      sampler: str = DEFAULT_SAMPLER, n_mult: Optional[float] = None, n_jobs: int = 1,
                                      adapter: str = DEFAULT_ADAPTER,
                                      verbose: bool = False) -> tuple[pd.DataFrame, dict]:
    """Calibrate every declared population separately (within-G null).

    Parameters
    ----------
    X : array-like or scipy.sparse matrix
        Cells x declared feature universe; binarized as ``X > 0``.
    groups : sequence
        Population label of each cell.
    project : callable
        Fixed projection ``B -> Z``, shared by all populations.
    populations : sequence of str, optional
        Populations to calibrate, in order. Default: all labels, sorted.
    seed : int or tuple of int, optional
        Base seed key. Population ``k`` (its position in ``populations``)
        uses ``seed_key = (*seed, k)``. Default 0.
    n_replicates, sampler, n_mult, n_jobs, adapter
        As in :func:`structural_null_calibration`.
    verbose : bool, optional
        Print one progress line per population. Default False.

    Returns
    -------
    table : pandas.DataFrame
        One row per population with ``J_obs``, ``E0_J``, ``Delta_G`` and
        diagnostics.
    calibrations : dict
        :class:`StructuralCalibration` of each population.
    """
    B = binarize(X)
    g = check_labels(groups, B.shape[0])
    pops = list(populations) if populations is not None else sorted(set(g))
    base = _seed_key(seed)
    rows, objs = [], {}
    for k, p in enumerate(pops):
        idx = np.flatnonzero(g == p)
        cal = structural_null_calibration(B[idx], project, n_replicates=n_replicates, sampler=sampler, n_mult=n_mult,
                                          seed_key=(*base, k), n_jobs=n_jobs, adapter=adapter)
        objs[p] = cal
        rows.append({"population": p, **cal.summary()})
        if verbose:
            print(f"[scCHART structural null] {p}: J={cal.J_obs:.4g} E0={cal.E0_J:.4g} "
                  f"Delta={cal.Delta:.4g}", flush=True)
    return pd.DataFrame(rows).set_index("population"), objs


# ---------------------------------------------------------------------------
# Matched-depth cross-population comparison
# ---------------------------------------------------------------------------


def depth_standardized_structural_comparison(
        X, groups, project: Projector, D_star: float, *, populations: Optional[Sequence[str]] = None,
        n_replicates: int = 5, seed=0, dstar_index: int = 0,
        n_null: int = N_NULL_REPLICATES, sampler: str = DEFAULT_SAMPLER, n_mult: Optional[float] = None, n_jobs: int = 1,
        adapter: str = DEFAULT_ADAPTER,
        seed_index: Optional[Sequence[int]] = None, verbose: bool = False) -> DepthStandardizedResult:
    """Compute ``Delta_G(D*)`` for every population at a common depth ``D*``.

    For each population ``G`` (index ``k``):

    1. ``q_G = min(1, D* / median count depth_G)``.
    2. If ``q_G < 1``, the counts are binomially downsampled
       ``n_replicates`` times with seeds ``(*seed, 160, k, dstar_index, t)``;
       otherwise the population is evaluated once, unchanged (never
       upsampled).
    3. Each replicate is re-binarized. Cells left with no open feature are
       removed and counted in ``zero_open_excluded``.
    4. The remaining cells are projected with the same fixed ``project``
       and calibrated with :func:`structural_null_calibration`, using seed
       key ``(*seed, 161, k, dstar_index, t)``.

    Parameters
    ----------
    X : scipy.sparse matrix
        Integer counts (cells x declared features). Binary input is thinned
        by independent retention.
    groups : sequence
        Population label of each cell.
    project : callable
        Fixed projection ``B -> Z``.
    D_star : float
        Declared common effective depth, in count-depth units (row sums of
        ``X``).
    populations : sequence of str, optional
        Populations to compare, in order. Default: all labels, sorted.
    n_replicates : int, optional
        Downsampling replicates per population with ``q_G < 1``. Default 5.
    seed : int or tuple of int, optional
        Base seed key. Default 0.
    dstar_index : int, optional
        Index of this ``D*`` in a series of depths, used in the seeds.
        Default 0.
    n_null : int, optional
        Null replicates per calibration. Default 20.
    sampler, n_mult, n_jobs, adapter
        As in :func:`structural_null_calibration`.
    seed_index : sequence of int, optional
        Seed index ``k`` of each population. Default: its position in
        ``populations``. Lets a subset reproduce the seeds of a larger run.
    verbose : bool, optional
        Print one progress line per population. Default False.

    Returns
    -------
    DepthStandardizedResult
        ``summary`` has ``Delta_G_mean``, ``Delta_G_sd``, ``q``,
        ``reaches_D_star`` and related columns.

    Warns
    -----
    UpsamplingWarning
        If a population's median depth is below ``D*``.

    Notes
    -----
    Populations evaluated once (``q_G = 1``) repeat their single value
    across replicates in the pairwise agreement table.
    """
    X = sparse.csr_matrix(X)
    g = check_labels(groups, X.shape[0])
    pops = list(populations) if populations is not None else sorted(set(g))
    depth = count_depth(X)
    frac = downsampling_fractions(depth, g, D_star, populations=pops)
    base = _seed_key(seed)
    rows = []
    ks = list(range(len(pops))) if seed_index is None else [int(x) for x in seed_index]
    for k, p in zip(ks, pops):
        Xg = X[np.flatnonzero(g == p)].tocsr()
        q = float(frac.loc[p, "q"])
        for t in range(n_replicates if q < 1 else 1):
            Xt = Xg.copy()
            if q < 1:
                Xt = binomial_downsample(Xt, q, np.random.default_rng((*base, 160, k, dstar_index, t)))
            dep_t = count_depth(Xt)
            keep = dep_t > 0
            cal = structural_null_calibration(Xt[np.flatnonzero(keep)], project, n_replicates=n_null,
                                              sampler=sampler, n_mult=n_mult,
                                              seed_key=(*base, 161, k, dstar_index, t),
                                              n_jobs=n_jobs, adapter=adapter)
            rows.append({"population": p, "replicate": t, "q": q, "D_star": D_star,
                         "zero_open_excluded": int((~keep).sum()),
                         "median_count_depth_after": float(np.median(dep_t[keep])), **cal.summary()})
        if verbose:
            print(f"[scCHART structural D*] {p}: q={q:.3g} done", flush=True)
    rep = pd.DataFrame(rows)
    summ = rep.groupby("population", sort=False).agg(
        n_replicates=("replicate", "size"), q=("q", "first"),
        median_count_depth_after=("median_count_depth_after", "mean"),
        J_obs_mean=("J_obs", "mean"), E0_J_mean=("E0_J", "mean"),
        Delta_G_mean=("Delta_G", "mean"), Delta_G_sd=("Delta_G", "std"),
        zero_open_excluded=("zero_open_excluded", "sum")).reindex(pops)
    summ = summ.join(frac[["median_depth", "reaches_D_star"]])
    wide = rep.pivot(index="replicate", columns="population", values="Delta_G").reindex(columns=pops)
    wide = wide.ffill()  # populations evaluated once (q = 1) repeat their single value
    return DepthStandardizedResult(float(D_star), rep, summ, pairwise_replicate_agreement(wide), "structural")


# ---------------------------------------------------------------------------
# AnnData-facing scorer
# ---------------------------------------------------------------------------


class StructuralHeterogeneity:
    """Population-conditioned structural information-divergence scorer.

    Computes the structural estimand, ``d_i(G) = KL(p_i || pbar_G)`` and
    ``J_G``, on a caller-declared representation ``Z`` in
    ``adata.obsm[use_rep]`` through a simplex adapter. The representation
    is used exactly as supplied. The scorer does not infer populations and
    does not calibrate: the technical null and ``Delta_G`` are computed by
    :func:`structural_null_calibration` /
    :func:`structural_population_calibration`.

    Parameters
    ----------
    adapter : {"signed_channel", "spectral_energy"}, optional
        Simplex adapter. Default ``"signed_channel"``.
    store_probabilities : bool, optional
        Store ``p_i`` in ``adata.obsm['scchart_structural_probabilities']``.
        Default True.
    verbose : bool, optional
        Print ``[scCHART xx/xx]`` progress messages. Default True.

    Attributes
    ----------
    use_rep_ : str
        Representation key used by the last fit.
    groupby_ : str
        Population key used by the last fit.
    n_dimensions_ : int
        Number of representation dimensions supplied to the scorer.
    probabilities_ : numpy.ndarray
        Cell-by-channel probability matrix ``p_i``.
    population_summary_ : pandas.DataFrame
        Population-level raw structural heterogeneity summaries.
    population_consensus_ : dict
        Consensus distribution ``pbar_G`` of each population.
    scores_ : pandas.DataFrame
        Per-cell structural score table after :meth:`fit`.
    timings_ : dict
        Wall-clock duration of each pipeline stage, in seconds.

    Raises
    ------
    ValueError
        If ``adapter`` is unknown.

    Examples
    --------
    >>> model = StructuralHeterogeneity()
    >>> scores = model.fit(
    ...     adata,
    ...     groupby="BioClassification",
    ...     use_rep="X_lsi",
    ... )

    >>> result = StructuralHeterogeneity(adapter="spectral_energy").fit_transform(
    ...     adata,
    ...     groupby="BioClassification",
    ...     use_rep="X_lsi",
    ...     inplace=False,
    ... )

    Notes
    -----
    The grouping supplied by ``groupby`` is part of the structural
    estimand. Scores under different groupings, representations or
    adapters answer different questions and are not interchangeable.
    """

    def __init__(
        self,
        *,
        adapter: str = DEFAULT_ADAPTER,
        store_probabilities: bool = True,
        verbose: bool = True,
    ) -> None:
        if adapter not in ADAPTERS:
            raise ValueError(f"adapter must be one of {ADAPTERS}, got {adapter!r}")
        self.adapter = adapter
        self.store_probabilities = bool(store_probabilities)
        self.verbose = bool(verbose)

        self.use_rep_: Optional[str] = None
        self.groupby_: Optional[str] = None
        self.n_dimensions_: Optional[int] = None
        self.probabilities_: Optional[np.ndarray] = None
        self.population_summary_: Optional[pd.DataFrame] = None
        self.population_consensus_: Optional[Dict[str, np.ndarray]] = None
        self.scores_: Optional[pd.DataFrame] = None
        self.timings_: Optional[Dict[str, float]] = None
        self._stage_started: Optional[float] = None
        self._last_stage_title: Optional[str] = None

    def _log(self, msg: str) -> None:
        """Print ``msg`` when verbose."""
        if self.verbose:
            print(msg, flush=True)

    def _log_stage(self, step: int, total: int, title: str) -> None:
        """Record the duration of the previous stage and announce the next one."""
        now = time.perf_counter()
        if (
            self.timings_ is not None
            and self._stage_started is not None
            and self._last_stage_title is not None
        ):
            self.timings_[self._last_stage_title] = now - self._stage_started
        self._stage_started = now
        self._last_stage_title = title
        if self.verbose:
            print(f"[scCHART {step:02d}/{total:02d}] {title}", flush=True)

    def _log_detail(self, label: str, value: object) -> None:
        """Print one aligned detail line when verbose."""
        if self.verbose:
            print(f"    {label:<32} {value}", flush=True)

    @staticmethod
    def _resolve_representation(adata: AnnData, use_rep: str) -> np.ndarray:
        """Return ``adata.obsm[use_rep]`` as a validated float64 matrix.

        Raises ``KeyError`` if the key is missing and ``ValueError`` if the
        matrix does not have one row per cell, has fewer than two components
        or contains non-finite values.
        """
        if use_rep not in adata.obsm:
            raise KeyError(
                f"use_rep={use_rep!r} not found in adata.obsm. "
                f"Available: {list(adata.obsm.keys())}"
            )
        Z = np.asarray(adata.obsm[use_rep], dtype=np.float64)
        if Z.ndim != 2 or Z.shape[0] != adata.n_obs:
            raise ValueError(
                f"adata.obsm[{use_rep!r}] must have shape "
                f"(adata.n_obs, n_components)."
            )
        if Z.shape[1] < 2:
            raise ValueError("At least two representation components are required.")
        if not np.isfinite(Z).all():
            raise ValueError(f"adata.obsm[{use_rep!r}] contains non-finite values.")
        return Z

    @staticmethod
    def _resolve_labels(adata: AnnData, groupby: str) -> np.ndarray:
        """Return ``adata.obs[groupby]`` as strings, rejecting missing labels."""
        if groupby not in adata.obs.columns:
            raise KeyError(f"groupby={groupby!r} not found in adata.obs.")
        labels = adata.obs[groupby]
        if labels.isna().any():
            raise ValueError(f"adata.obs[{groupby!r}] contains missing labels.")
        return labels.astype(str).to_numpy()

    def fit(
        self,
        adata: AnnData,
        groupby: str,
        use_rep: str = "X_lsi",
    ) -> pd.DataFrame:
        """Compute raw structural heterogeneity and write it to ``adata``.

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData containing the declared representation in ``adata.obsm``.
            Modified in place.
        groupby : str
            Population/group column in ``adata.obs`` (the declared ``G``).
            The grouping is part of the structural estimand.
        use_rep : str, optional
            Representation key in ``adata.obsm`` (the declared ``Z``). All
            supplied dimensions are used as provided. Default ``"X_lsi"``.

        Returns
        -------
        pandas.DataFrame
            Per-cell structural score table. The same columns are also written
            to ``adata.obs``.

        Raises
        ------
        KeyError
            If ``groupby`` or ``use_rep`` is missing.
        ValueError
            If labels are missing, the representation is malformed or has
            fewer than two components, or a cell has ``z_i = 0``.

        Notes
        -----
        Writes ``adata.obs`` score columns, ``adata.uns['scchart_structural']``
        and, when ``store_probabilities`` is True,
        ``adata.obsm['scchart_structural_probabilities']``.
        """
        fit_started = time.perf_counter()
        self.timings_ = {}
        self._stage_started = None
        self._last_stage_title = None
        self.use_rep_ = use_rep
        self.groupby_ = groupby

        total_stages = 4

        self._log_stage(1, total_stages, "Reading declared representation")
        Z = self._resolve_representation(adata, use_rep)
        labels = self._resolve_labels(adata, groupby)
        self.n_dimensions_ = int(Z.shape[1])
        self._log_detail(
            "Representation shape",
            f"{Z.shape[0]:,} cells × {Z.shape[1]} dimensions",
        )
        self._log_detail("Representation source", use_rep)
        self._log_detail("Component selection", "none (used as supplied)")
        self._log_detail("Population key", groupby)
        self._log_detail("Populations", len(pd.unique(labels)))

        self._log_stage(2, total_stages, "Applying simplex adapter")
        P = adapter_probability(Z, self.adapter)
        self.probabilities_ = P
        self._log_detail("Simplex adapter", self.adapter)
        self._log_detail("Channels", P.shape[1])
        self._log_detail("Pseudocount", "none")

        self._log_stage(3, total_stages, "Computing population KL / generalized JSD")
        cell_kl, cell_entropy, population_summary, population_means = (
            _population_divergence(P, labels)
        )
        # Population tables are in sorted label order; J_obs is J_G in bits.
        population_summary = population_summary.rename(
            columns={"J_structural_bits": "J_obs"}
        ).reindex(sorted(population_summary.index))
        self.population_summary_ = population_summary
        self.population_consensus_ = population_means
        self._log_detail("Divergence", "KL(p_i || p̄_G)")
        self._log_detail("Log base / units", "2 / bits")
        self._log_detail("Consensus", "inclusive arithmetic mean")
        self._log_detail(
            "Max JSD identity error",
            f"{population_summary['identity_error'].max():.3e}",
        )

        self._log_stage(4, total_stages, "Writing structural scores and metadata")
        population_J = population_summary.loc[
            pd.Index(labels, name="population"), "J_obs"
        ].to_numpy()

        scores = pd.DataFrame(
            {
                "scchart_structural_d": cell_kl,
                "scchart_structural_entropy": cell_entropy,
                "scchart_structural_population_J_obs": population_J,
            },
            index=adata.obs_names,
        )
        self.scores_ = scores

        for column in scores:
            adata.obs[column] = scores[column].to_numpy()

        if self.store_probabilities:
            adata.obsm["scchart_structural_probabilities"] = P

        adata.uns["scchart_structural"] = {
            "class_name": self.__class__.__name__,
            "method": METHOD_NAME,
            "groupby": groupby,
            "representation": use_rep,
            "representation_status": REPRESENTATION_STATUS,
            "n_dimensions": self.n_dimensions_,
            "component_selection": "none",
            "simplex_adapter": self.adapter,
            "n_channels": int(P.shape[1]),
            "probability_definition": PROBABILITY_DEFINITIONS[self.adapter],
            "cell_score": "KL(p_i || pbar_G)",
            "population_score": "mean_i_in_G KL(p_i || pbar_G)",
            "consensus": "inclusive arithmetic mean",
            "log_base": 2,
            "units": "bits",
            "pseudocount": 0.0,
            "population_summary": population_summary.reset_index().to_dict(
                orient="list"
            ),
            "population_consensus": {
                key: value.tolist() for key, value in population_means.items()
            },
            "primary_score": "scchart_structural_d",
            "calibration": "none",
            "calibration_note": "raw J_G; Delta_G via scchart.Heterogeneity(calibrate=True)",
        }
        # Optional entries are omitted rather than set to None, because
        # AnnData silently drops None-valued ``uns`` entries on write_h5ad.
        if self.store_probabilities:
            adata.uns["scchart_structural"]["probability_key"] = (
                "scchart_structural_probabilities"
            )

        self._log_detail("Primary score", "scchart_structural_d")
        self._log_detail("Population summary", "adata.uns['scchart_structural']")
        if self.store_probabilities:
            self._log_detail(
                "Probability matrix",
                "adata.obsm['scchart_structural_probabilities']",
            )

        now = time.perf_counter()
        if self._stage_started is not None and self._last_stage_title is not None:
            self.timings_[self._last_stage_title] = now - self._stage_started
        self.timings_["total"] = now - fit_started
        self._log("[scCHART] Fit complete.")
        return scores

    def fit_transform(
        self,
        adata: AnnData,
        *,
        groupby: str,
        use_rep: str = "X_lsi",
        inplace: bool = True,
    ) -> StructuralHeterogeneityResult | AnnData:
        """Fit the scorer and return either ``adata`` or a result container.

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData containing the declared representation. Scores are always
            written to ``adata`` (see :meth:`fit`).
        groupby : str
            Population/group column in ``adata.obs``.
        use_rep : str, optional
            Representation key in ``adata.obsm``. Default ``"X_lsi"``.
        inplace : bool, optional
            If True (default), return ``adata``; otherwise return a
            :class:`StructuralHeterogeneityResult`.

        Returns
        -------
        anndata.AnnData or StructuralHeterogeneityResult
            ``adata`` when ``inplace`` is True, otherwise copies of the scores,
            population summary and probabilities.
        """
        scores = self.fit(adata, groupby=groupby, use_rep=use_rep)
        if inplace:
            return adata

        return StructuralHeterogeneityResult(
            scores=scores.copy(),
            population_summary=self.population_summary_.copy(),
            probabilities=pd.DataFrame(
                self.probabilities_,
                index=adata.obs_names,
                columns=_probability_columns(self.adapter, self.n_dimensions_),
            ),
        )

    def __repr__(self) -> str:
        """Summarize the configuration and fit status."""
        def row(label: str, value) -> str:
            return f"  {label:<40}{value}"

        fitted = self.scores_ is not None
        lines = [
            self.__class__.__name__,
            "Population-conditioned structural information divergence.",
            "declared Z → adapter A → KL to population consensus → generalized JSD",
            "",
            "Core Configuration",
            row("Representation", "caller-declared, used as supplied"),
            row("Simplex Adapter", self.adapter),
            row("Divergence", "KL, base 2"),
            row("Consensus", "inclusive arithmetic mean"),
            row("Calibration", "none (see structural_population_calibration)"),
            "",
            "Fit Status",
            row("Fitted", "Yes" if fitted else "Not Yet"),
        ]
        if fitted:
            lines.extend(
                [
                    row("Representation Key", self.use_rep_),
                    row("Population Key", self.groupby_),
                    row("Cells", len(self.scores_)),
                    row("Populations", len(self.population_summary_)),
                    row("Dimensions", self.n_dimensions_),
                    row("Primary Score", "scchart_structural_d"),
                ]
            )
        return "\n".join(lines)
