"""Fitted-model interface: both heterogeneity axes on one AnnData.

:class:`Heterogeneity` runs the structural and/or programme axis on an
``AnnData`` object. It calls the existing scorers
(:class:`~scchart.StructuralHeterogeneity`,
:class:`~scchart.ProgrammeHeterogeneity`) and calibration functions
(:func:`~scchart.structural_divergence.structural_population_calibration`,
:func:`~scchart.programme_entropy.crossfit_population_calibration`) and adds no statistics
of its own. The axes are reported separately and never combined.

Population order and seeds
--------------------------
Null seeds depend on each population's position in the calibration order:

* structural: population ``k`` uses seed key ``(*structural_seed, k)``;
* programme: population ``k`` in fold ``f`` uses base seed
  ``programme_seed + 10_000 f + 100 k``.

With ``populations=None`` both axes use the sorted population labels, the
default of the calibration functions. Pass ``populations=`` to
:meth:`Heterogeneity.fit` to declare another order. The order and the seed
of every population are stored in ``adata.uns["scchart"]`` and in
:attr:`Heterogeneity.population_order_` / :attr:`Heterogeneity.seeds_`.
"""

from __future__ import annotations

import operator
import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from ._scoring import DEFAULT_SAMPLER, binarize, check_n_jobs, check_sampler, seed_key
from .programme_entropy import (
    ProgrammeHeterogeneity,
    ProgrammeReference,
    crossfit_population_calibration,
)
from .structural_divergence import (
    ADAPTERS,
    DEFAULT_ADAPTER,
    StructuralHeterogeneity,
    check_same_representation,
    structural_population_calibration,
)

COMBINED_SCORE_STATEMENT = "not defined: structural and programme heterogeneity are reported separately"
STRUCTURAL_NOT_CALIBRATED = "not run: needs X and a fixed projection of B into Z"
STRUCTURAL_SEED_SCHEME = "population k (position in population_order) uses seed key (*structural_seed, k)"
PROGRAMME_SEED_SCHEME = ("population k (position in population_order) in fold f uses base seed "
                         "programme_seed + 10_000 f + 100 k")
ORDER_DEFAULT = "sorted population labels"
ORDER_DECLARED = "declared with populations="

# Calibration columns joined onto the population tables.
_STRUCTURAL_CAL_COLUMNS = ["E0_J", "Delta_G", "sd0_diagnostic", "R_G_diagnostic",
                           "margins_exact_all_nulls", "median_n_open"]
_PROGRAMME_CAL_COLUMNS = ["E0_J_P", "Delta_P", "sd0_diagnostic", "null_scored_fraction",
                          "margins_exact_all_nulls"]


def _h5ad_safe(value):
    """Return ``value`` with None entries removed and tuples made lists.

    AnnData drops or rejects None-valued ``uns`` entries on ``write_h5ad``.
    """
    if isinstance(value, dict):
        return {str(k): _h5ad_safe(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_h5ad_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _table_to_uns(table: pd.DataFrame) -> dict:
    """Store a population table as an h5ad-safe dict of columns."""
    return table.reset_index().to_dict(orient="list")


class Heterogeneity:
    """Structural and programme heterogeneity of declared populations.

    Fits either axis or both on an ``AnnData`` object and reports them
    separately. Per-cell outputs are written to ``adata.obs`` /
    ``adata.obsm`` by the axis scorers; population tables, calibration
    results, population orders and declarations go to ``adata.uns`` and to
    the fitted attributes.

    Parameters
    ----------
    structural, programme : bool, optional
        Axes to fit. Default both.
    use_rep : str, optional
        ``adata.obsm`` key of the declared structural representation ``Z``.
        Default ``"X_lsi"``.
    layer : str, optional
        ``adata.layers`` key of the accessibility counts (cells x features).
        Default None, which uses ``adata.X``. Needed by the programme axis
        and by structural calibration.
    projection : callable, optional
        Fixed projection ``B -> Z`` (for example
        :class:`scchart.LSIProjection`) used by the structural null. Without
        it the structural axis reports raw ``J_G`` only. When given, ``Z``
        must equal ``projection(B)``.
    programme_key : str, optional
        ``adata.var`` column with external programme labels (``"Shared"``
        or missing = background). Default: endogenous cross-fitted
        programmes.
    calibrate : bool, optional
        Run the fixed-margin technical nulls. Default True.
    n_folds : int, optional
        Programme cross-fit folds. Default 5.
    random_state : int, optional
        Programme fold-split seed. Default 0.
    adapter : {"signed_channel", "spectral_energy"}, optional
        Structural simplex adapter. Default ``"signed_channel"``.
    structural_seed : int or tuple of int, optional
        Structural null base seed key. Default 0.
    programme_seed : int, optional
        Programme null base seed. Default 10_000_000.
    null_cells : {"scored", "all"}, optional
        Cells entering the programme null. Default ``"scored"``.
    null_sampler : {"proposal_clock", "accepted_swap"}, optional
        Fixed-margin null sampler for both axes. Default
        ``"proposal_clock"``.
    n_jobs : int, optional
        Null chains run in parallel, for the structural and the programme
        null (results do not depend on it): a positive integer, or -1 for
        all numba threads. Default 1.
    store_probabilities : bool, optional
        Store the structural ``p_i`` in ``adata.obsm``. Default True.
    verbose : bool, optional
        Print the fitting stages and output locations. Default True.

    Attributes
    ----------
    groupby_ : str
        Population column used by the last fit.
    population_order_ : list of str
        Population order of the tables and of the null seeds.
    seeds_ : dict
        Null seeds of every population (calibrated axes only):
        ``seeds_["structural"][G]`` is the seed key of ``G`` and
        ``seeds_["programme"][G]`` maps each fold to its base seed.
    structural_ : pandas.DataFrame or None
        Per-population structural table (``J_obs`` and, if calibrated,
        ``E0_J``, ``Delta_G`` and diagnostics), in ``population_order_``.
    programme_ : pandas.DataFrame or None
        Per-population programme table (``J_P_obs``, ``scored_fraction``,
        ``median_P`` and, if calibrated, ``E0_J_P``, ``Delta_P`` and
        diagnostics), in ``population_order_``.
    structural_model_, programme_model_ :
        The fitted :class:`StructuralHeterogeneity` /
        :class:`ProgrammeHeterogeneity`.
    structural_calibrations_, programme_calibrations_ : dict or None
        Calibration object of each population (includes the null
        replicates ``J_null``).
    references_ : list of ProgrammeReference or None
        Programme references used to score the cells.
    declarations_ : dict
        What the scores are conditional on; report it with the results.
    timings_ : dict
        Wall-clock duration of each fitting stage, in seconds.

    Examples
    --------
    >>> model = scchart.Heterogeneity(use_rep="X_lsi", projection=proj, layer="counts")
    >>> print(model)
    >>> model.fit(adata, groupby="cell_type")
    >>> model.structural_[["J_obs", "E0_J", "Delta_G"]]
    """

    def __init__(
        self,
        *,
        structural: bool = True,
        programme: bool = True,
        use_rep: str = "X_lsi",
        layer: Optional[str] = None,
        projection: Optional[Callable] = None,
        programme_key: Optional[str] = None,
        calibrate: bool = True,
        n_folds: int = 5,
        random_state: int = 0,
        adapter: str = DEFAULT_ADAPTER,
        structural_seed=0,
        programme_seed: int = 10_000_000,
        null_cells: str = "scored",
        null_sampler: str = DEFAULT_SAMPLER,
        n_jobs: int = 1,
        store_probabilities: bool = True,
        verbose: bool = True,
    ) -> None:
        if not (structural or programme):
            raise ValueError("at least one of structural and programme must be True")
        if adapter not in ADAPTERS:
            raise ValueError(f"adapter must be one of {ADAPTERS}, got {adapter!r}")
        if n_folds < 2:
            raise ValueError("n_folds must be >= 2")
        if null_cells not in ("scored", "all"):
            raise ValueError("null_cells must be 'scored' or 'all'")
        check_sampler(null_sampler)
        check_n_jobs(n_jobs)
        self.structural = bool(structural)
        self.programme = bool(programme)
        self.use_rep = use_rep
        self.layer = layer
        self.projection = projection
        self.programme_key = programme_key
        self.calibrate = bool(calibrate)
        self.n_folds = int(n_folds)
        self.random_state = int(random_state)
        self.adapter = adapter
        self.structural_seed = structural_seed
        self.programme_seed = int(programme_seed)
        self.null_cells = null_cells
        self.null_sampler = null_sampler
        self.n_jobs = operator.index(n_jobs)
        self.store_probabilities = bool(store_probabilities)
        self.verbose = bool(verbose)

        self.groupby_: Optional[str] = None
        self.population_order_: Optional[List[str]] = None
        self.seeds_: Optional[Dict[str, dict]] = None
        self.structural_: Optional[pd.DataFrame] = None
        self.programme_: Optional[pd.DataFrame] = None
        self.structural_model_: Optional[StructuralHeterogeneity] = None
        self.programme_model_: Optional[ProgrammeHeterogeneity] = None
        self.structural_calibrations_: Optional[dict] = None
        self.programme_calibrations_: Optional[dict] = None
        self.references_: Optional[List[ProgrammeReference]] = None
        self.declarations_: Optional[dict] = None
        self.timings_: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _structural_calibration_planned(self) -> bool:
        return self.structural and self.calibrate and self.projection is not None

    def _planned(self) -> List[str]:
        """Analyses that :meth:`fit` will run, in order."""
        steps = ["Validate inputs"]
        if self.structural:
            steps.append("Structural scores (d_i, J_G)")
            if self._structural_calibration_planned():
                steps.append("Structural null calibration (E0[J_G], Delta_G)")
        if self.programme:
            steps.append("Programme references and scores (P_i, d_i^(P), J_G^(P))")
            if self.calibrate:
                steps.append("Programme null calibration (E0[J_G^(P)], Delta_G^(P))")
        steps.append("Store results")
        return steps

    def _config(self) -> dict:
        projection = None if self.projection is None else type(self.projection).__name__
        return {
            "structural": self.structural,
            "programme": self.programme,
            "use_rep": self.use_rep if self.structural else None,
            "layer": self.layer,
            "projection": projection,
            "programme_key": self.programme_key,
            "calibrate": self.calibrate,
            "n_folds": self.n_folds,
            "random_state": self.random_state,
            "adapter": self.adapter,
            "structural_seed": list(seed_key(self.structural_seed)),
            "programme_seed": self.programme_seed,
            "null_cells": self.null_cells,
            "null_sampler": self.null_sampler,
            "n_jobs": self.n_jobs,
        }

    def __repr__(self) -> str:
        def row(label: str, value) -> str:
            return f"  {label:<34}{value}"

        X_source = f"adata.layers[{self.layer!r}]" if self.layer else "adata.X"
        lines = [self.__class__.__name__,
                 "Population-conditioned structural and programme heterogeneity (reported separately).",
                 "", "Axes"]
        if self.structural:
            lines += [row("Structural", f"Z = adata.obsm[{self.use_rep!r}], adapter {self.adapter}")]
            if self.calibrate and self.projection is None:
                lines += [row("", "calibration not possible: no fixed projection (raw J_G only)")]
            elif self.calibrate:
                lines += [row("", f"null through fixed projection {type(self.projection).__name__}")]
        else:
            lines += [row("Structural", "off")]
        if self.programme:
            ref = (f"external, adata.var[{self.programme_key!r}]" if self.programme_key
                   else "endogenous group-specific (provisional)")
            lines += [row("Programme", f"{ref}; {self.n_folds}-fold cross-fit, random_state={self.random_state}")]
        else:
            lines += [row("Programme", "off")]
        lines += ["", "Settings",
                  row("Accessibility matrix", X_source),
                  row("Calibration", "fixed-margin null" if self.calibrate else "off (raw J only)")]
        if self.calibrate:
            lines += [row("Null sampler", self.null_sampler)]
            if self.structural:
                lines += [row("Structural seed", seed_key(self.structural_seed))]
            if self.programme:
                lines += [row("Programme seed", self.programme_seed),
                          row("Programme null cells", self.null_cells)]
        lines += ["", "Planned analyses"]
        lines += [f"  {i:02d}. {s}" for i, s in enumerate(self._planned(), start=1)]
        lines += ["", "Fit status", row("Fitted", "Yes" if self.declarations_ is not None else "Not yet")]
        if self.declarations_ is not None:
            lines += [row("Population key", self.groupby_)]
            lines += [row("Populations", len(self.population_order_))]
        return "\n".join(lines)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _detail(self, label: str, value) -> None:
        if self.verbose:
            print(f"    {label:<32} {value}", flush=True)

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    @staticmethod
    def _check_adata(adata) -> None:
        if not isinstance(adata, AnnData):
            hint = " Call .to_memory() first (SnapATAC2 backed objects)." if hasattr(adata, "to_memory") else ""
            raise TypeError(f"adata must be an in-memory anndata.AnnData, got {type(adata).__name__}.{hint}")
        if adata.isbacked:
            raise ValueError("backed AnnData is not supported; load it with adata.to_memory()")

    def _matrix(self, adata: AnnData) -> sparse.csr_matrix:
        if self.layer is None:
            X = adata.X
        elif self.layer in adata.layers:
            X = adata.layers[self.layer]
        else:
            raise KeyError(f"layer={self.layer!r} not found in adata.layers. Available: {list(adata.layers)}")
        if X is None:
            raise ValueError("adata has no accessibility matrix")
        return sparse.csr_matrix(X)

    @staticmethod
    def _order(labels: np.ndarray, populations) -> tuple[List[str], str]:
        if populations is None:
            return sorted(set(labels)), ORDER_DEFAULT
        order = [str(p) for p in populations]
        if len(set(order)) != len(order):
            raise ValueError("populations contains duplicates")
        if set(order) != set(labels):
            missing, extra = sorted(set(labels) - set(order)), sorted(set(order) - set(labels))
            raise ValueError(f"populations must list every label once; missing {missing}, unknown {extra}")
        return order, ORDER_DECLARED

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, adata: AnnData, groupby: str, *, populations: Optional[Sequence[str]] = None,
            reference: Optional[ProgrammeReference] = None) -> "Heterogeneity":
        """Fit the requested axes on ``adata`` and store the results.

        Parameters
        ----------
        adata : anndata.AnnData
            In-memory AnnData; modified in place.
        groupby : str
            ``adata.obs`` column with the declared populations ``G``.
        populations : sequence of str, optional
            Population order for both axes; must list every label once.
            It sets the null seeds (see the module notes). Default: sorted
            labels.
        reference : ProgrammeReference, optional
            Fixed programme reference scoring every cell (takes precedence
            over ``programme_key``).

        Returns
        -------
        Heterogeneity
            The fitted model.

        Raises
        ------
        TypeError
            If ``adata`` is not an in-memory ``AnnData``.
        KeyError
            If ``groupby``, ``use_rep``, ``layer`` or ``programme_key`` is
            missing.
        ValueError
            If labels are missing, ``populations`` is not a permutation of
            the labels, or ``Z`` differs from ``projection(B)``.
        """
        started = time.perf_counter()
        timings: Dict[str, float] = {}
        steps = self._planned()
        step = iter(range(1, len(steps) + 1))
        clock = [time.perf_counter()]

        def stage(title: str) -> None:
            self._log(f"[scCHART {next(step):02d}/{len(steps):02d}] {title}")

        def done(title: str) -> None:
            now = time.perf_counter()
            timings[title] = now - clock[0]
            clock[0] = now

        # 1. Inputs -----------------------------------------------------
        stage(steps[0])
        self._check_adata(adata)
        labels = StructuralHeterogeneity._resolve_labels(adata, groupby)
        if self.structural:
            Z = StructuralHeterogeneity._resolve_representation(adata, self.use_rep)
        if self.programme and self.programme_key is not None and reference is None \
                and self.programme_key not in adata.var.columns:
            raise KeyError(f"programme_key={self.programme_key!r} not found in adata.var.")
        need_X = self.programme or self._structural_calibration_planned()
        X = self._matrix(adata) if need_X else None
        order, order_rule = self._order(labels, populations)
        self._detail("Cells x features", f"{adata.n_obs:,} x {adata.n_vars:,}")
        self._detail("Population key", f"adata.obs[{groupby!r}] ({len(order)} populations)")
        self._detail("Population order", f"{order_rule}: {', '.join(order)}")
        if self.structural and self.calibrate and self.projection is None:
            self._detail("Structural calibration", "not run: no fixed projection given")
        done("inputs")

        decl: dict = {}
        structural_table = programme_table = None
        s_cal = p_cal = refs = None
        s_model = p_model = None

        # 2. Structural axis -------------------------------------------
        if self.structural:
            stage("Structural scores (d_i, J_G)")
            s_model = StructuralHeterogeneity(adapter=self.adapter, store_probabilities=self.store_probabilities,
                                              verbose=False)
            s_model.fit(adata, groupby=groupby, use_rep=self.use_rep)
            structural_table = s_model.population_summary_.reindex(order)
            decl["structural_adapter"] = self.adapter
            decl["structural_representation_dims"] = int(Z.shape[1])
            self._detail("Representation", f"adata.obsm[{self.use_rep!r}], {Z.shape[1]} dimensions")
            done("structural scores")
            if self.calibrate:
                if self.projection is None:
                    structural_table["calibration"] = STRUCTURAL_NOT_CALIBRATED
                else:
                    stage("Structural null calibration (E0[J_G], Delta_G)")
                    # J_G above and Delta_G below must refer to the same representation.
                    check_same_representation(Z, self.projection(binarize(X)))
                    tab, s_cal = structural_population_calibration(
                        X, labels, self.projection, populations=order, seed=self.structural_seed,
                        n_jobs=self.n_jobs, sampler=self.null_sampler, adapter=self.adapter,
                        verbose=self.verbose)
                    structural_table = structural_table.join(tab[_STRUCTURAL_CAL_COLUMNS])
                    done("structural null")

        # 3. Programme axis --------------------------------------------
        if self.programme:
            stage("Programme references and scores (P_i, d_i^(P), J_G^(P))")
            p_model = ProgrammeHeterogeneity(programme_key=self.programme_key, layer=self.layer,
                                             n_folds=self.n_folds, random_state=self.random_state,
                                             verbose=False)
            p_model.fit(adata, groupby=groupby, reference=reference)
            refs = p_model.references_
            programme_table = p_model.population_summary_.reindex(order)
            decl["programme_reference"] = refs[0].rule
            decl["programme_K"] = len(p_model.programmes_)
            self._detail("Reference mode", p_model.reference_mode_)
            self._detail("Programmes (K)", len(p_model.programmes_))
            self._detail("Scored cells", f"{int(p_model.scores_['scchart_programme_scored'].sum()):,} / {adata.n_obs:,}")
            done("programme scores")
            if self.calibrate:
                stage("Programme null calibration (E0[J_G^(P)], Delta_G^(P))")
                tab, p_cal = crossfit_population_calibration(
                    X, labels, refs, populations=order, seed=self.programme_seed,
                    null_cells=self.null_cells, sampler=self.null_sampler, n_jobs=self.n_jobs,
                    verbose=self.verbose)
                programme_table = programme_table.join(tab[_PROGRAMME_CAL_COLUMNS])
                done("programme null")

        if self.calibrate:
            decl["null_sampler"] = self.null_sampler
        decl["combined_score"] = COMBINED_SCORE_STATEMENT

        # 4. Store -------------------------------------------------------
        stage("Store results")
        seeds = self._seeds(order, s_cal is not None, refs if p_cal is not None else None)
        self.groupby_ = groupby
        self.population_order_ = order
        self.seeds_ = seeds
        self.structural_, self.programme_ = structural_table, programme_table
        self.structural_model_, self.programme_model_ = s_model, p_model
        self.structural_calibrations_, self.programme_calibrations_ = s_cal, p_cal
        self.references_ = refs
        self.declarations_ = decl

        uns: dict = {"groupby": groupby, "config": self._config(), "population_order": order,
                     "population_order_rule": order_rule, "declarations": decl}
        if self.structural:
            entry = {"table": _table_to_uns(structural_table), "calibrated": s_cal is not None}
            if s_cal is not None:
                entry["seed_scheme"] = STRUCTURAL_SEED_SCHEME
                entry["seed_keys"] = np.array([seeds["structural"][p] for p in order])
                entry["null_J"] = np.vstack([s_cal[p].J_null for p in order])
                adata.uns["scchart_structural"]["calibration"] = "fixed-margin null"
                adata.uns["scchart_structural"]["calibration_note"] = "E0_J, Delta_G in adata.uns['scchart']"
            uns["structural"] = entry
        if self.programme:
            entry = {"table": _table_to_uns(programme_table), "calibrated": p_cal is not None}
            if p_cal is not None:
                folds = sorted(next(iter(seeds["programme"].values())))
                entry["seed_scheme"] = PROGRAMME_SEED_SCHEME
                entry["seed_folds"] = folds
                entry["base_seeds"] = np.array([[seeds["programme"][p][f] for f in folds] for p in order])
                entry["null_J"] = np.vstack([p_cal[p].J_null for p in order])
                adata.uns["scchart_programme"]["calibration"] = "fixed-margin null"
                adata.uns["scchart_programme"]["calibration_note"] = "E0_J_P, Delta_P in adata.uns['scchart']"
            uns["programme"] = entry
        adata.uns["scchart"] = _h5ad_safe(uns)
        done("store")
        timings["total"] = time.perf_counter() - started
        self.timings_ = timings

        if self.verbose:
            self._report_locations()
            self._log(f"[scCHART] Fit complete in {timings['total']:.1f} s.")
        return self

    def _seeds(self, order: List[str], structural: bool, refs) -> Dict[str, dict]:
        """Null seed of every population, from the seed schemes of the calibration functions."""
        seeds: Dict[str, dict] = {}
        if structural:
            base = seed_key(self.structural_seed)
            seeds["structural"] = {p: [*base, k] for k, p in enumerate(order)}
        if refs is not None:
            folds = sorted({ref.fold or 0 for ref in refs})
            seeds["programme"] = {p: {f: self.programme_seed + 10_000 * f + 100 * k for f in folds}
                                  for k, p in enumerate(order)}
        return seeds

    def _report_locations(self) -> None:
        """Print where the fitted outputs are stored."""
        self._log("    Outputs")
        if self.structural:
            m = self.structural_model_
            self._detail("Structural per-cell", "adata.obs: " + ", ".join(m.scores_.columns))
            if m.store_probabilities:
                self._detail("Structural probabilities", "adata.obsm['scchart_structural_probabilities']")
            self._detail("Structural metadata", "adata.uns['scchart_structural']")
        if self.programme:
            self._detail("Programme per-cell", "adata.obs: " + ", ".join(self.programme_model_.scores_.columns))
            self._detail("Programme composition, evidence",
                         "adata.obsm['scchart_programme_composition'], ['scchart_programme_evidence']")
            self._detail("Programme metadata", "adata.uns['scchart_programme']")
        self._detail("Population tables, nulls, seeds", "adata.uns['scchart']")
        attrs = [a for a, on in (("model.structural_", self.structural), ("model.programme_", self.programme)) if on]
        self._detail("Fitted attributes", ", ".join(attrs))
