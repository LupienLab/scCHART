"""Programme heterogeneity by information divergence (scCHART programme axis).

The computation has three separate layers.

A. **Reference R** (:class:`ProgrammeReference`): programme peak sets
   ``R_1..R_K`` (exclusive feature -> programme labels; every other
   feature belongs to the background block) and the background rates
   ``beta~``.

   * Endogenous, group-derived references **must be cross-fitted**
     (:func:`crossfit_programme_references`): a scored cell never
     contributes to the R or ``beta~`` used to score it.
   * External fixed references (:func:`external_programme_reference`, or
     ``external=`` in :func:`crossfit_programme_references`) need no
     cross-fitting of R.
   * The group-specific top-300 rule (:func:`group_specific_programmes`)
     is a **provisional**, declared choice of R and must be reported.

B. **Background-corrected evidence** and membership::

       D_i       = sum_{j not in union_k R_k} B_ij           (background depth)
       beta~_j   = sum_{i' in Ref, g(i') != k(j)} B_i'j / sum_{i' in Ref, g(i') != k(j)} D_i'
       a_ik      = max(0, (1/|R_k|) sum_{j in R_k} (B_ij - D_i beta~_j))
       r_ik      = a_ik / sum_k a_ik ;   sum_k a_ik = 0  ->  unscorable (NA)

   Here ``k(j)`` is the group that defines the programme containing peak
   ``j``. The clip ``max(0, .)`` is applied **after** aggregation over the
   programme's peaks. Peaks have unit weights; there is no IDF, no
   Dirichlet prior and no programme balancing. A cell with zero total
   evidence is unscorable: nothing is imputed.

C. **Heterogeneity** (base 2, bits), over scored cells::

       P_i         = H(r_i) / log2 K                  (within-cell mixing)
       d_i^(P)     = KL(r_i || rbar_G)                 (decomposition term, not calibrated)
       J_G^(P)     = mean_i d_i^(P) = H(rbar_G) - mean_i H(r_i)
       Delta_G^(P) = J_G,obs^(P) - E0[J_G^(P)]         (primary calibrated quantity)

   ``E0`` is the within-population technical null
   (:func:`programme_null_calibration`): the programme-peak block is
   randomized by 2 x 2 degree-preserving swaps (default proposal clock,
   100 * nnz proposals, 20 replicates). The background block can be
   randomized separately (``randomise_background=True``); the evidence
   uses it only through ``D_i``, which the swaps preserve, so the scores
   are identical. The null preserves every cell's programme-peak burden
   and ``D_i`` exactly, and every peak's frequency; R and ``beta~`` are
   fixed. ``Delta_G^(P)`` is conservative, attenuates as depth falls, and
   is not an unbiased estimate of a latent true J. Null z-scores are not
   magnitudes. Every population statistic is reported with its scored
   fraction.

Cross-population comparison uses ``Delta_G^(P)(D*)`` at a declared common
effective depth (:func:`depth_standardized_programme_comparison`).

Outputs of :class:`ProgrammeHeterogeneity`
------------------------------------------
``adata.obs['scchart_programme_P']``
    ``P_i`` in [0, 1]; NaN for unscorable cells.
``adata.obs['scchart_programme_d']``
    ``d_i^(P)(G)`` in bits (decomposition term of ``J_G^(P)``); NaN if
    unscorable.
``adata.obs['scchart_programme_population_J_P_obs']``
    ``J_P_obs`` (``J_G^(P)``) of the cell's population, repeated per cell.
``adata.obs['scchart_programme_scored']``
    Whether the cell has non-zero total evidence.
``adata.obs['scchart_programme_dominant']``, ``adata.obs['scchart_programme_dominant_weight']``
    ``argmax_k r_ik`` and ``max_k r_ik`` (NaN if unscorable).
``adata.obs['scchart_programme_total_evidence']``, ``adata.obs['scchart_programme_background_depth']``
    ``sum_k a_ik`` and ``D_i``.
``adata.obsm['scchart_programme_composition']``, ``adata.obsm['scchart_programme_evidence']``
    ``r`` and ``a`` (cells x K).
``adata.uns['scchart_programme']``
    Programmes, K, reference mode and rule, cross-fit settings and the
    population summary.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from ._scoring import (
    N_NULL_REPLICATES,
    check_n_jobs,
    _as_binary,
    DEFAULT_SAMPLER,
    DepthStandardizedResult,
    binarize,
    binomial_downsample,
    check_labels,
    count_depth,
    downsampling_fractions,
    entropy_bits,
    kl_bits,
    pairwise_replicate_agreement,
    swap_replicates,
)

SHARED_LABEL = "Shared"

#: Default parameters of the provisional group-specific reference rule.
PROVISIONAL_RULE = dict(pseudocount=0.002, min_in=0.01, min_delta=0.005, min_cells=50,
                        top_n=300, min_features_per_programme=25)

EVIDENCE_DEFINITION = "a_ik = max(0, (1/|R_k|) sum_{j in R_k} (B_ij - D_i beta~_j)); unit weights"
METHOD_NAME = "scchart_programme_information_divergence"


# =============================================================================
# Layer A: programme reference
# =============================================================================

@dataclass
class ProgrammeReference:
    """A declared programme reference (layer A) with its background rates.

    Attributes
    ----------
    programme_of_feature : numpy.ndarray
        One label per feature: a programme name, or another label such as
        ``"Shared"`` for the background block.
    programmes : list of str
        Programme order (columns of ``a`` and ``r``).
    beta_tilde : numpy.ndarray
        ``(n_features,)`` background rate ``beta~_j`` of each programme
        peak: the expected open probability of peak ``j`` per unit of
        background depth. Zero for background features.
    source : str
        ``"endogenous_crossfit"`` or ``"external"``.
    fold : int, optional
        Cross-fit fold index.
    train_index, score_index : numpy.ndarray, optional
        Cells used to build R and ``beta~``, and the held-out cells this
        reference scores.
    summary : pandas.DataFrame, optional
        Per-group provenance of the reference rule.
    rule : dict
        Declared reference rule and cross-fit settings.

    Raises
    ------
    ValueError
        If ``beta_tilde`` does not have one entry per feature, a programme
        has no features, or fewer than two programmes are given.

    Notes
    -----
    ``programme_of_feature`` is stored as a read-only copy of the labels.
    The derived arrays (``background_mask``, ``programme_index``,
    ``membership_matrix()``, ``sizes()``) are computed once and reused while
    ``programme_of_feature`` and ``programmes`` are unchanged; assigning a
    new ``programme_of_feature`` or changing ``programmes`` recomputes them.
    Each call returns a new copy.
    """

    programme_of_feature: np.ndarray
    programmes: List[str]
    beta_tilde: np.ndarray
    source: str = "external"
    fold: Optional[int] = None
    train_index: Optional[np.ndarray] = None
    score_index: Optional[np.ndarray] = None
    summary: Optional[pd.DataFrame] = None
    rule: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.programme_of_feature = np.asarray(self.programme_of_feature).astype(str).astype(object)
        self.programmes = [str(p) for p in self.programmes]
        self.beta_tilde = np.asarray(self.beta_tilde, dtype=np.float64).ravel()
        if self.beta_tilde.size != self.programme_of_feature.size:
            raise ValueError("beta_tilde needs one entry per feature")
        missing = [p for p in self.programmes if not np.any(self.programme_of_feature == p)]
        if missing:
            raise ValueError(f"programmes without features: {missing}")
        if len(self.programmes) < 2:
            raise ValueError("at least two programmes are required (P_i = H / log K)")
        self.programme_of_feature.flags.writeable = False

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state.pop("_derived", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.programme_of_feature.flags.writeable = False

    def _derived_arrays(self) -> dict:
        """Arrays derived from the labels, computed once per (labels, programmes).

        The cache is keyed on the identity of ``programme_of_feature`` (a
        read-only array) and on the tuple of ``programmes``. A writeable
        ``programme_of_feature`` (assigned after construction) is never
        cached.
        """
        key = (self.programme_of_feature, tuple(self.programmes))
        cached = self.__dict__.get("_derived")
        if cached is not None and cached["key"][0] is key[0] and cached["key"][1] == key[1]:
            return cached
        lab = self.programme_of_feature
        k_of = np.full(lab.size, -1)
        for k, p in enumerate(self.programmes):
            k_of[lab == p] = k
        rows = np.flatnonzero(k_of >= 0)
        M = sparse.csr_matrix((np.ones(rows.size), (rows, k_of[rows])), shape=(k_of.size, len(self.programmes)))
        cached = {"key": key, "background_mask": ~np.isin(lab, self.programmes), "programme_index": k_of,
                  "membership_matrix": M,
                  "sizes": pd.Series({p: int((lab == p).sum()) for p in self.programmes})}
        if not lab.flags.writeable:  # an array assigned after construction may change in place: not cached
            self.__dict__["_derived"] = cached
        return cached

    @property
    def K(self) -> int:
        """Number of programmes."""
        return len(self.programmes)

    @property
    def background_mask(self) -> np.ndarray:
        """Boolean mask of features outside every programme (the background block)."""
        return self._derived_arrays()["background_mask"].copy()

    @property
    def programme_index(self) -> np.ndarray:
        """Programme column of each feature, or -1 for background features."""
        return self._derived_arrays()["programme_index"].copy()

    def membership_matrix(self) -> sparse.csr_matrix:
        """Return the features x K indicator matrix ``M_jk = 1[j in R_k]``.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse programme membership matrix.
        """
        return self._derived_arrays()["membership_matrix"].copy()

    def sizes(self) -> pd.Series:
        """Return the number of peaks ``|R_k|`` in each programme.

        Returns
        -------
        pandas.Series
            Peak count indexed by programme.
        """
        return self._derived_arrays()["sizes"].copy()


def _group_detection_rates(B, labels, groups):
    """Compute in-group and out-group detection rates of every feature.

    Returns
    -------
    inside, outside : numpy.ndarray
        ``(n_groups, n_features)`` float32 fractions of cells with the
        feature open, inside and outside each group.
    """
    labels = np.asarray(labels).astype(str)
    # Already binary: count from B itself, without copying it. Counts below 2**24 are exact
    # in float32 and float64 alike, so the rates equal those of the float32 copy below.
    binary = (isinstance(B, sparse.csr_matrix) and B.dtype == np.float64 and bool(np.all(B.data == 1.0))
              and B.shape[0] < 2 ** 24)
    if binary:
        X = B
    else:
        X = sparse.csr_matrix(B, copy=True)
        X.data = np.ones_like(X.data, dtype=np.float32)
    index = {g: i for i, g in enumerate(groups)}
    rows = np.flatnonzero(np.isin(labels, groups))
    cols = np.array([index[l] for l in labels[rows]], dtype=np.int64)
    ind = sparse.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(labels.size, len(groups)))
    if binary:  # X.T is a view; only the small indicator is converted (same exact counts)
        group_sum = np.asarray((X.T @ ind).T.todense(), dtype=np.float64)
    else:
        group_sum = np.asarray((ind.T @ X).todense(), dtype=np.float64)
    total_sum = np.asarray(X.sum(axis=0), dtype=np.float64).ravel()
    n_in = np.asarray(ind.sum(axis=0)).ravel()
    n_out = labels.size - n_in
    inside = group_sum / np.maximum(n_in, 1)[:, None]
    outside = (total_sum[None, :] - group_sum) / np.maximum(n_out, 1)[:, None]
    return inside.astype(np.float32), outside.astype(np.float32)


def group_specific_programmes(B_ref, labels_ref, feature_names=None, *,
                              top_n: int = PROVISIONAL_RULE["top_n"],
                              pseudocount: float = PROVISIONAL_RULE["pseudocount"],
                              min_in: float = PROVISIONAL_RULE["min_in"],
                              min_delta: float = PROVISIONAL_RULE["min_delta"],
                              min_cells: int = PROVISIONAL_RULE["min_cells"],
                              min_features: int = PROVISIONAL_RULE["min_features_per_programme"],
                              all_groups: Optional[Sequence[str]] = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Derive provisional group-specific peak programmes from reference cells.

    The provisional rule, applied to reference cells only, is:

    1. Keep groups with at least ``min_cells`` reference cells.
    2. For each feature and group, compute the detection rates ``in`` and
       ``out`` and the enrichment ``log2((in + pseudocount) / (out + pseudocount))``.
    3. A (feature, group) pair is a candidate if ``in >= min_in`` and
       ``in - out >= min_delta``.
    4. Assign each feature exclusively to the group with the highest
       candidate enrichment.
    5. Keep the ``top_n`` most enriched features of each group.
    6. A group defines a programme only if at least ``min_features``
       features are selected; otherwise its features return to the
       background.

    Parameters
    ----------
    B_ref : scipy.sparse matrix
        Binary reference cells x features matrix.
    labels_ref : sequence
        Group label of each reference cell.
    feature_names : sequence, optional
        Accepted for API compatibility; not used.
    top_n : int, optional
        Maximum number of features per programme. Default 300.
    pseudocount : float, optional
        Pseudocount in the enrichment ratio. Default 0.002.
    min_in : float, optional
        Minimum in-group detection rate. Default 0.01.
    min_delta : float, optional
        Minimum difference between in-group and out-group detection.
        Default 0.005.
    min_cells : int, optional
        Minimum number of reference cells for an eligible group. Default 50.
    min_features : int, optional
        Minimum number of selected features for a defined programme.
        Default 25.
    all_groups : sequence of str, optional
        Groups to list in the summary in addition to those in
        ``labels_ref``.

    Returns
    -------
    programme_of_feature : numpy.ndarray
        Programme label of each feature (``"Shared"`` outside R).
    summary : pandas.DataFrame
        Per-group ``n_ref_cells``, ``eligible``, ``n_candidates``,
        ``n_selected`` (before the ``min_features`` check) and
        ``programme_defined``.
    """
    lref = check_labels(labels_ref, B_ref.shape[0], name="labels_ref")
    n_feat = B_ref.shape[1]
    counts = pd.Series(lref).value_counts()
    groups = sorted(counts[counts >= min_cells].index.tolist())
    prog = np.full(n_feat, SHARED_LABEL, dtype=object)
    n_cand = pd.Series(0, index=groups, dtype=int)
    if groups:
        inside, outside = _group_detection_rates(B_ref, lref, groups)
        spec = np.log2((inside + pseudocount) / (outside + pseudocount))
        spec[(inside < min_in) | ((inside - outside) < min_delta)] = -np.inf
        best = np.argmax(spec, axis=0)
        best_score = spec[best, np.arange(n_feat)]
        feats = np.flatnonzero(np.isfinite(best_score))
        cand = pd.DataFrame({"feature_index": feats, "group": np.asarray(groups, dtype=object)[best[feats]],
                             "specificity": best_score[feats]})
        cand = cand.sort_values(["group", "specificity"], ascending=[True, False], kind="mergesort")
        cand["rank"] = cand.groupby("group").cumcount() + 1
        n_cand = cand.groupby("group").size().reindex(groups).fillna(0).astype(int)
        sel = cand[cand["rank"] <= top_n]
        prog[sel["feature_index"].to_numpy()] = sel["group"].to_numpy()
    sizes = pd.Series(prog).value_counts()
    summary = pd.DataFrame({"group": sorted(set(all_groups) | set(lref)) if all_groups is not None
                            else sorted(set(lref))})
    summary["n_ref_cells"] = [int((lref == g).sum()) for g in summary.group]
    summary["eligible"] = summary.group.isin(groups)
    summary["n_candidates"] = summary.group.map(n_cand).fillna(0).astype(int)
    summary["n_selected"] = summary.group.map(sizes).fillna(0).astype(int)
    summary["programme_defined"] = summary.eligible & (summary.n_selected >= min_features)
    for g in summary.group[summary.eligible & ~summary.programme_defined]:
        prog[prog == g] = SHARED_LABEL
    return prog, summary.set_index("group")


def reference_background_rates(B_ref, ref_groups, programme_of_feature, programmes: Sequence[str], *,
                               programme_groups: Optional[Mapping[str, str]] = None) -> np.ndarray:
    """Estimate background rates ``beta~_j`` from reference cells.

    For a peak ``j`` in programme ``R_k`` defined by group ``g_k``,

        beta~_j = sum_{i' in Ref, g(i') != g_k} B_i'j / sum_{i' in Ref, g(i') != g_k} D_i'

    where ``D_i'`` is the background depth of reference cell ``i'``.
    Background features receive 0.

    Parameters
    ----------
    B_ref : scipy.sparse matrix
        Binary reference cells x features matrix.
    ref_groups : sequence
        Group label of each reference cell.
    programme_of_feature : sequence of str
        Programme label of each feature.
    programmes : sequence of str
        Programmes; features with any other label form the background.
    programme_groups : mapping, optional
        Programme -> defining group. Default: the programme's own name.

    Returns
    -------
    numpy.ndarray
        ``(n_features,)`` background rates.

    Notes
    -----
    If the defining group has no reference cells (for example an external
    programme), all reference cells are used.
    """
    g = np.asarray(ref_groups).astype(str)
    lab = np.asarray(programme_of_feature).astype(str)
    bgm = ~np.isin(lab, list(programmes))
    beta_t = np.zeros(B_ref.shape[1])
    if _integer_valued_sparse(B_ref):
        # Only the programme columns are copied. With integer entries every sum below is
        # exact in any order, so the rates equal the general path below exactly.
        B_ref = sparse.csr_matrix(B_ref)
        prog_cols = np.flatnonzero(~bgm)
        Bp = sparse.csc_matrix(B_ref[:, prog_cols])
        D_ref = np.asarray(B_ref.sum(axis=1)).ravel() - np.asarray(Bp.sum(axis=1)).ravel()
        local = np.full(lab.size, -1)
        local[prog_cols] = np.arange(prog_cols.size)

        def numerator(out, cols):
            return np.asarray(Bp[:, local[cols]][out].sum(axis=0)).ravel()
    else:
        D_ref = np.asarray(B_ref[:, bgm].sum(axis=1)).ravel()
        Bc = sparse.csc_matrix(B_ref)

        def numerator(out, cols):
            return np.asarray(Bc[out][:, cols].sum(axis=0)).ravel()
    for p in programmes:
        cols = np.flatnonzero(lab == p)
        if cols.size == 0:
            continue
        defining = (programme_groups or {}).get(p, p)
        out = g != defining
        beta_t[cols] = numerator(out, cols) / max(D_ref[out].sum(), 1e-12)
    return beta_t


def _integer_valued_sparse(B) -> bool:
    """True for a sparse matrix of finite integer values whose absolute sum is below 2**53.

    Sums of such entries are exact in float64 (and in integer dtypes) whatever
    the order of summation.
    """
    if not sparse.issparse(B):
        return False
    d = B.tocsr().data if B.format not in ("csr", "csc", "coo") else B.data
    if d.dtype.kind in "biu":
        return True
    if d.dtype.kind != "f":
        return False
    if bool(np.all(d == 1.0)):  # binary: the common case
        return d.size < 2 ** 53
    total = 0.0
    for start in range(0, d.size, 1 << 20):  # chunked, to avoid a full-size temporary
        c = d[start:start + (1 << 20)]
        if not (np.all(np.isfinite(c)) and np.all(c == np.trunc(c))):
            return False
        total += float(np.abs(c).sum())
    return total < 2.0 ** 53


def external_programme_reference(programme_of_feature, B_ref=None, ref_groups=None, *,
                                 programmes: Optional[Sequence[str]] = None,
                                 programme_groups: Optional[Mapping[str, str]] = None,
                                 beta_tilde: Optional[np.ndarray] = None,
                                 train_index=None, score_index=None) -> ProgrammeReference:
    """Wrap an externally supplied fixed programme reference R.

    R is not derived from the data, so it needs no cross-fitting.
    ``beta~`` is either supplied, or estimated from ``B_ref`` and
    ``ref_groups``. The reference cells should not be the cells scored with
    this reference; use :func:`crossfit_programme_references` with
    ``external=`` to cross-fit ``beta~``.

    Parameters
    ----------
    programme_of_feature : sequence of str
        Programme label of each feature (``"Shared"`` for background).
    B_ref : array-like or scipy.sparse matrix, optional
        Reference cells x features (binarized), used to estimate ``beta~``.
    ref_groups : sequence, optional
        Group label of each reference cell.
    programmes : sequence of str, optional
        Programme order. Default: all labels other than ``"Shared"``,
        sorted.
    programme_groups : mapping, optional
        Programme -> defining group, for ``beta~``.
    beta_tilde : numpy.ndarray, optional
        Pre-computed background rates.
    train_index, score_index : array-like, optional
        Cells used for ``beta~`` and cells scored by this reference.

    Returns
    -------
    ProgrammeReference
        Reference with ``source="external"``.

    Raises
    ------
    ValueError
        If neither ``beta_tilde`` nor both ``B_ref`` and ``ref_groups`` are
        supplied.
    """
    lab = np.asarray(programme_of_feature).astype(str)
    progs = sorted(set(lab) - {SHARED_LABEL}) if programmes is None else list(programmes)
    if beta_tilde is None:
        if B_ref is None or ref_groups is None:
            raise ValueError("supply beta_tilde, or B_ref and ref_groups to estimate it")
        beta_tilde = reference_background_rates(_as_binary(B_ref), ref_groups, lab, progs,
                                                programme_groups=programme_groups)
    return ProgrammeReference(lab, progs, beta_tilde, source="external",
                              train_index=None if train_index is None else np.asarray(train_index),
                              score_index=None if score_index is None else np.asarray(score_index),
                              rule={"reference": "external fixed R"})


def crossfit_programme_references(X, labels, feature_names=None, *, n_folds: int = 5,
                                  random_state: int = 0, external: Optional[Sequence[str]] = None,
                                  programme_groups: Optional[Mapping[str, str]] = None,
                                  harmonise: str = "intersection",
                                  rule: Optional[dict] = None) -> List[ProgrammeReference]:
    """Build cross-fitted programme references, one per fold.

    Cells are split by ``StratifiedKFold(n_folds, shuffle=True,
    random_state)``, stratified by ``labels``. For fold ``f``, the
    reference R (unless ``external`` is given) and ``beta~`` are built
    **only** from the training cells; the held-out cells (``score_index``)
    are scored with them. No scored cell contributes to its own R or
    ``beta~``.

    Parameters
    ----------
    X : array-like or scipy.sparse matrix
        Cells x declared feature universe (binarized).
    labels : sequence
        Population labels. They define both the groups of R and the strata.
    feature_names : sequence, optional
        Accepted for API compatibility; not used.
    n_folds : int, optional
        Number of cross-fit folds. Default 5.
    random_state : int, optional
        Fold-split seed. Default 0.
    external : sequence of str, optional
        Fixed external programme label per feature. R is then not
        re-derived; only ``beta~`` is cross-fitted.
    programme_groups : mapping, optional
        Programme -> defining group, for ``beta~``.
    harmonise : {"intersection", "error"}, optional
        If the set of defined programmes differs between folds (for example
        for a group near the eligibility threshold), ``"intersection"``
        (default) keeps only programmes defined in every fold; the others
        become background in the folds that had them and are recorded in
        ``rule``. ``"error"`` raises.
    rule : dict, optional
        Overrides of :data:`PROVISIONAL_RULE` for the group-specific rule.

    Returns
    -------
    list of ProgrammeReference
        One reference per fold, sharing the same programme order.

    Raises
    ------
    ValueError
        If ``harmonise`` is invalid, or the programme sets differ between
        folds and ``harmonise="error"``.

    Warns
    -----
    UserWarning
        If programmes are demoted to background because they are not
        defined in every fold.
    """
    from sklearn.model_selection import StratifiedKFold

    if harmonise not in ("intersection", "error"):
        raise ValueError("harmonise must be 'intersection' or 'error'")
    B = _as_binary(X)
    lab = check_labels(labels, B.shape[0], name="labels")
    params = dict(PROVISIONAL_RULE, **(rule or {}))
    folds = list(StratifiedKFold(n_folds, shuffle=True, random_state=random_state).split(np.zeros(lab.size), lab))
    built = []
    for f, (tr, te) in enumerate(folds):
        if external is not None:
            prog = np.asarray(external).astype(str).astype(object)
            summ = None
        else:
            prog, summ = group_specific_programmes(
                B[tr], lab[tr], feature_names, top_n=params["top_n"], pseudocount=params["pseudocount"],
                min_in=params["min_in"], min_delta=params["min_delta"], min_cells=params["min_cells"],
                min_features=params["min_features_per_programme"], all_groups=sorted(set(lab)))
        built.append((tr, te, prog, summ))
    sets = [set(p) - {SHARED_LABEL} for _, _, p, _ in built]
    common = set.intersection(*sets)
    dropped = sorted(set.union(*sets) - common)
    if dropped:
        if harmonise == "error":
            raise ValueError(f"programme sets differ between folds: {dropped}")
        warnings.warn(f"programmes not defined in every fold are demoted to background: {dropped}",
                      UserWarning, stacklevel=2)
    progs = sorted(common)
    refs = []
    for f, (tr, te, prog, summ) in enumerate(built):
        prog = prog.copy()
        prog[~np.isin(prog.astype(str), progs)] = SHARED_LABEL
        bt = reference_background_rates(B[tr], lab[tr], prog, progs, programme_groups=programme_groups)
        refs.append(ProgrammeReference(
            prog, progs, bt, source="external" if external is not None else "endogenous_crossfit",
            fold=f, train_index=tr, score_index=te, summary=summ,
            rule={"reference": "external fixed R" if external is not None else "group-specific (provisional)",
                  **({} if external is not None else params), "n_folds": n_folds,
                  "split_random_state": random_state, "programmes_demoted_not_in_all_folds": dropped}))
    return refs


# =============================================================================
# Layer B: background-corrected evidence and membership
# =============================================================================

def _background_depth_binary(B: sparse.csr_matrix, background_mask: np.ndarray) -> np.ndarray:
    """``D_i`` of a binary matrix without copying its background columns.

    ``D_i`` = open features - open programme features. Both are counts of
    1.0 entries, exact in float64, so the result equals
    ``B[:, background_mask].sum(axis=1)`` exactly.
    """
    total = np.asarray(B.sum(axis=1)).ravel()
    in_programmes = np.asarray(B @ (~background_mask).astype(np.float64)).ravel()
    return total - in_programmes


def background_depth(B, reference: ProgrammeReference) -> np.ndarray:
    """Compute the background depth ``D_i`` of each cell.

    ``D_i`` is the number of open features outside the union of all
    programme peak sets.

    Parameters
    ----------
    B : array-like or scipy.sparse matrix
        Cells x features (binarized).
    reference : ProgrammeReference
        Reference defining the background block.

    Returns
    -------
    numpy.ndarray
        Background depth of each cell.
    """
    return _background_depth_binary(_as_binary(B), reference.background_mask)


def _evidence_from_blocks(O: np.ndarray, D: np.ndarray, M: sparse.csr_matrix, beta_tilde: np.ndarray) -> np.ndarray:
    """Compute clipped evidence from programme counts and background depth.

    Returns ``max(0, (O_ik - D_i * sum_{j in R_k} beta~_j) / |R_k|)``,
    where ``O_ik`` is the number of open peaks of cell ``i`` in ``R_k``.
    The clip is applied after the programme mean.
    """
    size = np.asarray(M.sum(axis=0)).ravel()
    Ek = D[:, None] * np.asarray(M.T @ beta_tilde).ravel()[None, :]
    return np.maximum(0.0, (O - Ek) / size)


def programme_evidence(B, reference: ProgrammeReference) -> np.ndarray:
    """Compute background-corrected programme evidence.

        a_ik = max(0, (1/|R_k|) sum_{j in R_k} (B_ij - D_i beta~_j))

    Peaks have unit weights, and the clip is applied after aggregation over
    the programme's peaks.

    Parameters
    ----------
    B : array-like or scipy.sparse matrix
        Cells x features (binarized).
    reference : ProgrammeReference
        Programme reference with its background rates.

    Returns
    -------
    numpy.ndarray
        Non-negative ``(n_cells, K)`` evidence matrix ``a``.

    Raises
    ------
    ValueError
        If ``B`` and the reference have different feature universes.
    """
    B = _as_binary(B)
    if B.shape[1] != reference.programme_of_feature.size:
        raise ValueError("B and the reference have different feature universes")
    M = reference.membership_matrix()
    O = B @ M
    O = O.toarray() if sparse.issparse(O) else np.asarray(O)
    D = _background_depth_binary(B, reference.background_mask)
    return _evidence_from_blocks(O, D, M, reference.beta_tilde)


def programme_membership(a: np.ndarray) -> np.ndarray:
    """Normalize programme evidence to a composition.

        r_ik = a_ik / sum_k a_ik

    Parameters
    ----------
    a : numpy.ndarray
        Non-negative ``(n_cells, K)`` evidence matrix.

    Returns
    -------
    numpy.ndarray
        ``(n_cells, K)`` composition ``r``. Rows with zero total evidence
        are NaN (unscorable; never imputed).

    Raises
    ------
    ValueError
        If ``a`` has negative entries.
    """
    a = np.asarray(a, dtype=np.float64)
    if np.any(a < 0):
        raise ValueError("programme evidence must be non-negative")
    s = a.sum(axis=1, keepdims=True)
    r = np.full_like(a, np.nan)
    ok = s[:, 0] > 0
    r[ok] = a[ok] / s[ok]
    return r


# =============================================================================
# Layer C: heterogeneity
# =============================================================================

def programme_mixing(r: np.ndarray) -> np.ndarray:
    """Compute normalized within-cell programme mixing.

        P_i = H(r_i) / log2 K

    Parameters
    ----------
    r : numpy.ndarray
        ``(n_cells, K)`` programme composition.

    Returns
    -------
    numpy.ndarray
        ``P_i`` in [0, 1]; NaN for unscorable cells.
    """
    r = np.asarray(r, dtype=np.float64)
    return entropy_bits(r) / np.log2(r.shape[1])


def programme_divergence(r: np.ndarray, groups) -> np.ndarray:
    """Compute each cell's divergence from its population's consensus.

        d_i^(P)(G) = KL(r_i || rbar_G)

    where ``rbar_G`` is the mean composition of the scored cells of ``G``.
    The mean of ``d_i^(P)`` over scored cells equals ``J_G^(P)``. The value
    is a decomposition term, not a calibrated per-cell score.

    Parameters
    ----------
    r : numpy.ndarray
        ``(n_cells, K)`` programme composition (NaN rows unscorable).
    groups : sequence
        Population label of each cell.

    Returns
    -------
    numpy.ndarray
        ``d_i^(P)`` in bits; NaN for unscorable cells.
    """
    r = np.asarray(r, dtype=np.float64)
    g = np.asarray(groups).astype(str)
    ok = ~np.isnan(r).any(axis=1)
    d = np.full(r.shape[0], np.nan)
    for G in np.unique(g):
        m = (g == G) & ok
        if m.any():
            d[m] = kl_bits(r[m], r[m].mean(axis=0))
    return d


def programme_population_table(r: np.ndarray, groups, *, order: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Summarize programme heterogeneity per population over scored cells.

    ``J_G^(P)`` is computed as ``H(rbar_G) - mean_i H(r_i)`` over the scored
    cells of each population.

    Parameters
    ----------
    r : numpy.ndarray
        ``(n_cells, K)`` programme composition (NaN rows unscorable).
    groups : sequence
        Population label of each cell.
    order : sequence of str, optional
        Population order. Default: all labels, sorted.

    Returns
    -------
    pandas.DataFrame
        One row per population with ``n_cells``, ``n_scored``,
        ``scored_fraction``, ``H_rbar_bits``, ``mean_H_bits``, ``J_P_bits``,
        ``J_P_norm`` (``J / log2 K``, descriptive), ``median_P``
        (descriptive; not the population estimand) and ``mean_P``.
        Populations without scored cells have NaN in the score columns.
    """
    r = np.asarray(r, dtype=np.float64)
    K = r.shape[1]
    g = np.asarray(groups).astype(str)
    order = list(order) if order is not None else sorted(set(g))
    ok = ~np.isnan(r).any(axis=1)
    H = entropy_bits(r)
    rows = []
    for G in order:
        m = g == G
        ms = m & ok
        n, ns = int(m.sum()), int(ms.sum())
        if ns == 0:
            rows.append(dict(population=G, n_cells=n, n_scored=0, scored_fraction=0.0 if n else np.nan))
            continue
        rbar = r[ms].mean(axis=0, keepdims=True)
        h_rbar = float(entropy_bits(rbar)[0])
        mean_h = float(H[ms].mean())
        rows.append(dict(population=G, n_cells=n, n_scored=ns, scored_fraction=ns / n,
                         H_rbar_bits=h_rbar, mean_H_bits=mean_h, J_P_bits=h_rbar - mean_h,
                         J_P_norm=(h_rbar - mean_h) / np.log2(K),
                         median_P=float(np.median(H[ms] / np.log2(K))), mean_P=mean_h / np.log2(K)))
    return pd.DataFrame(rows).set_index("population")


@dataclass
class ProgrammeScores:
    """Per-cell programme outputs of one scoring pass.

    Attributes
    ----------
    programmes : list of str
        Programme order.
    evidence : numpy.ndarray
        Evidence ``a`` (cells x K).
    r : numpy.ndarray
        Composition ``r`` (cells x K; NaN rows unscorable).
    P : numpy.ndarray
        Within-cell mixing ``P_i``.
    scored : numpy.ndarray
        Boolean mask of scorable cells.
    D : numpy.ndarray
        Background depth ``D_i``.
    """

    programmes: List[str]
    evidence: np.ndarray
    r: np.ndarray
    P: np.ndarray
    scored: np.ndarray
    D: np.ndarray


def score_programmes(B, reference: ProgrammeReference) -> ProgrammeScores:
    """Score cells with one programme reference.

    Parameters
    ----------
    B : array-like or scipy.sparse matrix
        Cells x features (binarized).
    reference : ProgrammeReference
        Reference used for every cell.

    Returns
    -------
    ProgrammeScores
        Evidence, composition, ``P_i``, scored mask and background depth.
    """
    Bb = _as_binary(B)
    a = programme_evidence(Bb, reference)
    r = programme_membership(a)
    return ProgrammeScores(list(reference.programmes), a, r, programme_mixing(r),
                           ~np.isnan(r).any(axis=1), background_depth(Bb, reference))


def score_crossfit(X, references: Sequence[ProgrammeReference]) -> ProgrammeScores:
    """Score every cell with the reference of the fold that holds it out.

    Parameters
    ----------
    X : array-like or scipy.sparse matrix
        Cells x features (binarized).
    references : sequence of ProgrammeReference
        Fold references with ``score_index`` and a shared programme order.

    Returns
    -------
    ProgrammeScores
        Outputs for all cells, assembled across folds.

    Raises
    ------
    ValueError
        If the references have different programme orders, a reference
        lacks ``score_index``, or a cell is scored by no fold or by more
        than one fold.
    """
    B = _as_binary(X)
    K = references[0].K
    progs = references[0].programmes
    if any(ref.programmes != progs for ref in references):
        raise ValueError("all fold references must share the same programme order")
    n = B.shape[0]
    a = np.full((n, K), np.nan)
    D = np.full(n, np.nan)
    seen = np.zeros(n, bool)
    for ref in references:
        te = ref.score_index
        if te is None:
            raise ValueError("cross-fitted references need score_index")
        if seen[te].any():
            raise ValueError("a cell is scored by more than one fold")
        seen[te] = True
        s = score_programmes(B[te], ref)
        a[te] = s.evidence
        D[te] = s.D
    if not seen.all():
        raise ValueError("some cells are not held out in any fold")
    r = programme_membership(np.nan_to_num(a))
    return ProgrammeScores(list(progs), a, r, programme_mixing(r), ~np.isnan(r).any(axis=1), D)


# =============================================================================
# Technical null
# =============================================================================

def programme_null_replicates(B, reference: ProgrammeReference, *, n_replicates: int = N_NULL_REPLICATES,
                              sampler: str = DEFAULT_SAMPLER, n_mult: Optional[float] = None, seed: int = 0,
                              randomise_background: bool = False, n_jobs: int = 1) -> tuple[list, list]:
    """Generate null compositions for the cells of one population and fold.

    In replicate ``b`` the programme block (features in the union of the
    ``R_k``) is randomized by 2 x 2 swaps with seed ``seed + 2b``. If
    ``randomise_background`` is True, the background block is randomized
    separately with seed ``seed + 2b + 1``. R and ``beta~`` are fixed.

    Parameters
    ----------
    B : array-like or scipy.sparse matrix
        Cells x features (binarized).
    reference : ProgrammeReference
        Fixed reference.
    n_replicates : int, optional
        Number of null replicates. Default 20.
    sampler : {"proposal_clock", "accepted_swap"}, optional
        Fixed-margin sampler.
    n_mult : float, optional
        Budget per occupied entry.
    seed : int, optional
        Base seed. Default 0.
    randomise_background : bool, optional
        Also randomize the background block. Default False.
    n_jobs : int, optional
        Number of replicate chains run in parallel per batch (see
        :func:`scchart._scoring.swap_replicates`). The results do not depend
        on it. Default 1.

    Returns
    -------
    compositions : list of numpy.ndarray
        Null composition ``r`` of each replicate.
    diagnostics : list of tuple
        ``(block, SwapDiagnostics)`` pairs, where ``block`` is
        ``"programme"`` or ``"background"``.

    Notes
    -----
    The evidence uses the background block only through ``D_i``, which the
    swaps preserve exactly, so ``randomise_background=False`` gives
    identical scores.

    Every chain is seeded by its own replicate seed, so replicate ``b`` is
    the same whatever ``n_jobs`` is. With ``randomise_background=True`` the
    background chains run on the dense background block, which needs about
    ``n_jobs * n_cells * n_background_features`` bytes.
    """
    check_n_jobs(n_jobs)
    Bb = _as_binary(B)
    bgm = reference.background_mask
    prog_cols = np.flatnonzero(~bgm)
    bg_cols = np.flatnonzero(bgm)
    M = reference.membership_matrix()[prog_cols]
    Bp = Bb[:, prog_cols]
    # D_i = open features - open programme features (exact counts); the background
    # block itself is only built when it is randomized.
    D0 = np.asarray(Bb.sum(axis=1)).ravel() - np.asarray(Bp.sum(axis=1)).ravel()
    beta_p = reference.beta_tilde[prog_cols]
    rs, diags = [], []
    programme_chains = swap_replicates(Bp, [seed + 2 * b for b in range(n_replicates)], sampler=sampler,
                                       n_mult=n_mult, n_jobs=n_jobs)
    if randomise_background:
        background_chains = swap_replicates(Bb[:, bg_cols], [seed + 2 * b + 1 for b in range(n_replicates)],
                                            sampler=sampler, n_mult=n_mult, n_jobs=n_jobs)
    for b in range(n_replicates):
        Np, dp = next(programme_chains)
        diags.append(("programme", dp))
        D = D0
        if randomise_background:
            Nb, db = next(background_chains)
            diags.append(("background", db))
            D = Nb.sum(axis=1).astype(np.float64)
        O = sparse.csr_matrix(Np, dtype=np.float64) @ M
        O = O.toarray() if sparse.issparse(O) else np.asarray(O)
        rs.append(programme_membership(_evidence_from_blocks(O, D, M, beta_p)))
    return rs, diags


@dataclass
class ProgrammeCalibration:
    """Calibrated programme heterogeneity of one population.

    Attributes
    ----------
    J_obs : float
        Observed ``J_G^(P)`` (bits) over the scored cells.
    E0_J : float
        Technical null expectation ``E0[J_G^(P)]`` (mean over replicates).
    Delta : float
        ``Delta_G^(P) = J_obs - E0_J``, the primary calibrated quantity.
    sd0 : float
        Standard deviation of the null ``J_G^(P)`` (diagnostic).
    J_null : numpy.ndarray
        Null ``J_G^(P)`` per replicate.
    n_cells : int
        Number of cells in the population.
    n_scored : int
        Number of scorable observed cells.
    scored_fraction : float
        Fraction of observed cells that are scorable.
    null_scored_fraction : float
        Mean fraction of scorable cells among the null cells.
    median_P : float
        Median observed ``P_i`` over scored cells.
    margins_exact_all : bool
        True when every null chain preserved the margins exactly.
    chains_complete_all : bool
        True when every chain used its full budget.
    sampler : str
        Fixed-margin null sampler.
    """

    J_obs: float
    E0_J: float
    Delta: float
    sd0: float
    J_null: np.ndarray
    n_cells: int
    n_scored: int
    scored_fraction: float
    null_scored_fraction: float
    median_P: float
    margins_exact_all: bool
    chains_complete_all: bool
    sampler: str = DEFAULT_SAMPLER

    def summary(self) -> dict:
        """Return the calibration and its diagnostics as a flat dictionary."""
        return {"n_cells": self.n_cells, "n_scored": self.n_scored, "scored_fraction": self.scored_fraction,
                "median_P": self.median_P, "J_P_obs": self.J_obs, "E0_J_P": self.E0_J,
                "Delta_P": self.Delta, "sd0_diagnostic": self.sd0, "n_null": int(self.J_null.size),
                "null_scored_fraction": self.null_scored_fraction,
                "margins_exact_all_nulls": self.margins_exact_all,
                "chains_complete_all": self.chains_complete_all, "null_sampler": self.sampler}


def _pooled_J(r: np.ndarray) -> float:
    """Return ``J_G^(P)`` of all scored rows of ``r`` pooled as one population.

    NaN if no row is scorable.
    """
    ok = ~np.isnan(r).any(axis=1)
    if not ok.any():
        return np.nan
    t = programme_population_table(r[ok], np.zeros(ok.sum(), dtype=int).astype(str))
    return float(t["J_P_bits"].iloc[0])


def programme_null_calibration(parts: Sequence[tuple], *, n_replicates: int = N_NULL_REPLICATES,
                               sampler: str = DEFAULT_SAMPLER, n_mult: Optional[float] = None, null_cells: str = "scored",
                               randomise_background: bool = False, n_jobs: int = 1) -> ProgrammeCalibration:
    """Calibrate one population's ``J_G^(P)``, pooled over its cross-fit folds.

    Parameters
    ----------
    parts : sequence of (B_part, reference, seed)
        The population's cells held out in each fold, with that fold's
        reference and base seed. The randomization unit is
        (population, fold); replicate ``b`` of the population pools
        replicate ``b`` of every fold, and ``J_G^(P)`` is computed over the
        pooled scored null cells.
    n_replicates : int, optional
        Number of null replicates. Default 20.
    sampler : {"proposal_clock", "accepted_swap"}, optional
        Fixed-margin chain (see :func:`scchart._scoring.swap_randomise`).
        Default ``"proposal_clock"``.
    n_mult : float, optional
        Budget per occupied entry. Default 100 proposals (proposal clock)
        or 10 accepted swaps (accepted-swap sampler).
    null_cells : {"scored", "all"}, optional
        Cells entering the swaps. ``"scored"`` (default) uses only cells
        with programme evidence; ``"all"`` uses every cell. The two are
        identical when the scored fraction is 1.
    randomise_background : bool, optional
        Also randomize the background block (does not change the scores).
        Default False.
    n_jobs : int, optional
        Null chains run in parallel per batch. The results do not depend on
        it. Default 1.

    Returns
    -------
    ProgrammeCalibration
        Observed and null ``J_G^(P)``, ``Delta_G^(P)`` and diagnostics.

    Raises
    ------
    ValueError
        If ``null_cells`` is invalid.
    """
    check_n_jobs(n_jobs)
    if null_cells not in ("scored", "all"):
        raise ValueError("null_cells must be 'scored' or 'all'")
    r_obs, null_r = [], [[] for _ in range(n_replicates)]
    margins, complete, n_total = True, True, 0
    for B_part, ref, seed in parts:
        s = score_programmes(B_part, ref)
        r_obs.append(s.r)
        n_total += s.r.shape[0]
        Bn = _as_binary(B_part)
        if null_cells == "scored":
            Bn = Bn[np.flatnonzero(s.scored)]
        if Bn.shape[0] == 0:
            continue
        rs, dg = programme_null_replicates(Bn, ref, n_replicates=n_replicates, sampler=sampler, n_mult=n_mult, seed=seed,
                                           randomise_background=randomise_background, n_jobs=n_jobs)
        for b in range(n_replicates):
            null_r[b].append(rs[b])
        margins &= all(d.margins_exact for _, d in dg)
        complete &= all(d.complete for _, d in dg)
    r_obs = np.vstack(r_obs)
    ok = ~np.isnan(r_obs).any(axis=1)
    J_obs = _pooled_J(r_obs)
    jn, nsf = [], []
    for b in range(n_replicates):
        if not null_r[b]:
            continue
        rb = np.vstack(null_r[b])
        jn.append(_pooled_J(rb))
        nsf.append(float((~np.isnan(rb).any(axis=1)).mean()))
    jn = np.asarray(jn, dtype=np.float64)
    mu = float(np.nanmean(jn)) if jn.size else np.nan
    Pm = programme_mixing(r_obs)
    return ProgrammeCalibration(
        J_obs=J_obs, E0_J=mu, Delta=J_obs - mu, sd0=float(np.nanstd(jn, ddof=1)) if jn.size > 1 else np.nan,
        J_null=jn, n_cells=int(n_total), n_scored=int(ok.sum()), scored_fraction=float(ok.mean()) if n_total else np.nan,
        null_scored_fraction=float(np.mean(nsf)) if nsf else np.nan,
        median_P=float(np.nanmedian(Pm)) if ok.any() else np.nan,
        margins_exact_all=bool(margins), chains_complete_all=bool(complete), sampler=sampler)


def _score_index(reference: ProgrammeReference, n_cells: int) -> np.ndarray:
    """Return the cells scored by ``reference`` (all cells if not set)."""
    return reference.score_index if reference.score_index is not None else np.arange(n_cells)


def crossfit_population_calibration(X, labels, references: Sequence[ProgrammeReference], *,
                                    populations: Optional[Sequence[str]] = None,
                                    seed: int = 10_000_000, n_replicates: int = N_NULL_REPLICATES,
                                    sampler: str = DEFAULT_SAMPLER, n_mult: Optional[float] = None, null_cells: str = "scored",
                                    randomise_background: bool = False, n_jobs: int = 1,
                                    verbose: bool = False) -> tuple[pd.DataFrame, Dict[str, ProgrammeCalibration]]:
    """Calibrate every population's programme heterogeneity, ``Delta_G^(P)``.

    Each cell is scored with the reference whose ``score_index`` holds it;
    a reference without ``score_index`` (such as a single external
    reference) scores every cell. Population ``gi`` (its position in
    ``populations``) in fold ``f`` uses base seed
    ``seed + 10_000 f + 100 gi``.

    Parameters
    ----------
    X : array-like or scipy.sparse matrix
        Cells x features (binarized).
    labels : sequence
        Population label of each cell.
    references : sequence of ProgrammeReference
        Fold references, or one fixed reference.
    populations : sequence of str, optional
        Populations to calibrate, in order. Default: all labels, sorted.
    seed : int, optional
        Base seed. Default 10_000_000.
    n_replicates, sampler, n_mult, null_cells, randomise_background, n_jobs
        As in :func:`programme_null_calibration`.
    verbose : bool, optional
        Print one progress line per population. Default False.

    Returns
    -------
    table : pandas.DataFrame
        One row per population with ``J_P_obs``, ``E0_J_P``, ``Delta_P``,
        scored fractions and diagnostics.
    calibrations : dict
        :class:`ProgrammeCalibration` of each population.
    """
    check_n_jobs(n_jobs)
    B = _as_binary(X)
    lab = check_labels(labels, B.shape[0], name="labels")
    pops = list(populations) if populations is not None else sorted(set(lab))
    rows, objs = [], {}
    for gi, G in enumerate(pops):
        parts = []
        for ref in references:
            te = _score_index(ref, B.shape[0])
            idx = te[lab[te] == G]
            if idx.size:
                parts.append((B[idx], ref, seed + 10_000 * (ref.fold or 0) + 100 * gi))
        cal = programme_null_calibration(parts, n_replicates=n_replicates, sampler=sampler, n_mult=n_mult,
                                         null_cells=null_cells, randomise_background=randomise_background,
                                         n_jobs=n_jobs)
        objs[G] = cal
        rows.append({"population": G, **cal.summary()})
        if verbose:
            print(f"[scCHART programme null] {G}: J={cal.J_obs:.4g} E0={cal.E0_J:.4g} "
                  f"Delta={cal.Delta:.4g} scored={cal.scored_fraction:.3f}", flush=True)
    return pd.DataFrame(rows).set_index("population"), objs


# =============================================================================
# Matched-depth comparison
# =============================================================================

def depth_standardized_programme_comparison(
        X, labels, references: Sequence[ProgrammeReference], D_star: float, *,
        populations: Optional[Sequence[str]] = None, n_replicates: int = 5, seed: int = 900_000,
        null_seed: int = 20_000_000, dstar_index: int = 0, n_null: int = N_NULL_REPLICATES,
        sampler: str = DEFAULT_SAMPLER, n_mult: Optional[float] = None, null_cells: str = "scored",
        n_jobs: int = 1, verbose: bool = False) -> DepthStandardizedResult:
    """Compute ``Delta_G^(P)(D*)`` at a declared common effective depth.

    For replicate ``t``:

    1. A generator ``default_rng(seed + 1000 dstar_index + t)`` downsamples
       the held-out cells of each fold in turn by binomial thinning with
       ``q_G = min(1, D* / median count depth_G)`` (never upsampled).
    2. Cells are re-binarized; ``D_i`` and the evidence are recomputed with
       the fold's **fixed** R and ``beta~``.
    3. The null for (population ``gi``, fold ``f``) uses base seed
       ``null_seed + 1_000_000 t + 10_000 f + 100 gi + 7 dstar_index``.

    Parameters
    ----------
    X : scipy.sparse matrix
        Integer counts (cells x declared features).
    labels : sequence
        Population label of each cell.
    references : sequence of ProgrammeReference
        Fold references, or one fixed reference.
    D_star : float
        Declared common effective depth, in count-depth units (row sums of
        ``X``).
    populations : sequence of str, optional
        Populations to compare, in order. Default: all labels, sorted.
    n_replicates : int, optional
        Number of downsampling replicates. Default 5.
    seed : int, optional
        Base downsampling seed. Default 900_000.
    null_seed : int, optional
        Base null seed. Default 20_000_000.
    dstar_index : int, optional
        Index of this ``D*`` in a series of depths, used in the seeds.
        Default 0.
    n_null : int, optional
        Null replicates per calibration. Default 20.
    sampler, n_mult, null_cells, n_jobs
        As in :func:`programme_null_calibration`.
    verbose : bool, optional
        Print one progress line per replicate. Default False.

    Returns
    -------
    scchart._scoring.DepthStandardizedResult
        ``summary`` has ``Delta_P_mean``, ``Delta_P_sd``, ``q``,
        ``reaches_D_star`` and related columns.

    Warns
    -----
    UpsamplingWarning
        If a population's median depth is below ``D*``.

    Notes
    -----
    Unlike the structural comparison, every population is evaluated in all
    ``n_replicates`` replicates, including populations with ``q_G = 1``.
    Cells left with no open feature are kept and become unscorable.
    """
    check_n_jobs(n_jobs)
    X = sparse.csr_matrix(X)
    lab = check_labels(labels, X.shape[0], name="labels")
    pops = list(populations) if populations is not None else sorted(set(lab))
    depth = count_depth(X)
    frac = downsampling_fractions(depth, lab, D_star, populations=pops)
    q_pop = frac["q"].to_dict()
    rows = []
    for t in range(n_replicates):
        rng = np.random.default_rng(seed + 1000 * dstar_index + t)
        parts: Dict[str, list] = {G: [] for G in pops}
        open_after: Dict[str, list] = {G: [] for G in pops}
        for ref in references:
            te = _score_index(ref, X.shape[0])
            q_cell = np.array([q_pop.get(G, 1.0) for G in lab[te]])
            Bte = _as_binary(binomial_downsample(X[te], q_cell, rng))
            for gi, G in enumerate(pops):
                loc = np.flatnonzero(lab[te] == G)
                if loc.size:
                    parts[G].append((Bte[loc], ref, null_seed + 1_000_000 * t + 10_000 * (ref.fold or 0)
                                     + 100 * gi + 7 * dstar_index))
                    open_after[G].append(np.asarray(Bte[loc].sum(axis=1)).ravel())
        for G in pops:
            cal = programme_null_calibration(parts[G], n_replicates=n_null, sampler=sampler, n_mult=n_mult,
                                             null_cells=null_cells, n_jobs=n_jobs)
            rows.append({"population": G, "replicate": t, "q": q_pop[G], "D_star": D_star,
                         "median_open_after": float(np.median(np.concatenate(open_after[G]))),
                         **cal.summary()})
        if verbose:
            print(f"[scCHART programme D*] replicate {t} done", flush=True)
    rep = pd.DataFrame(rows)
    summ = rep.groupby("population", sort=False).agg(
        n_replicates=("replicate", "size"), q=("q", "first"), median_open_after=("median_open_after", "mean"),
        scored_fraction_min=("scored_fraction", "min"), median_P=("median_P", "mean"),
        J_P_obs_mean=("J_P_obs", "mean"), E0_J_P_mean=("E0_J_P", "mean"),
        Delta_P_mean=("Delta_P", "mean"), Delta_P_sd=("Delta_P", "std")).reindex(pops)
    summ = summ.join(frac[["median_depth", "reaches_D_star"]])
    wide = rep.pivot(index="replicate", columns="population", values="Delta_P").reindex(columns=pops)
    return DepthStandardizedResult(float(D_star), rep, summ, pairwise_replicate_agreement(wide), "programme")


# =============================================================================
# AnnData-facing scorer
# =============================================================================

@dataclass(frozen=True)
class ProgrammeHeterogeneityResult:
    """Container returned by ``ProgrammeHeterogeneity.fit_transform``.

    Returned when ``fit_transform`` is called with ``inplace=False``.

    Attributes
    ----------
    scores : pandas.DataFrame
        Per-cell programme outputs indexed by ``adata.obs_names``.
    population_summary : pandas.DataFrame
        One row per population (:func:`programme_population_table`).
    composition : pandas.DataFrame
        Cell-by-programme membership ``r`` (NaN rows are unscorable).
    evidence : pandas.DataFrame
        Cell-by-programme evidence ``a``.
    """

    scores: pd.DataFrame
    population_summary: pd.DataFrame
    composition: pd.DataFrame
    evidence: pd.DataFrame


class ProgrammeHeterogeneity:
    """Background-corrected programme heterogeneity scorer on AnnData.

    Scores every cell's programme composition ``r_i`` from binary
    accessibility ``B = 1[X > 0]`` and a programme reference R, then
    reports the within-cell mixing ``P_i``, the per-cell divergence
    ``d_i^(P)`` and the population divergence ``J_G^(P)``. The reference is
    chosen as follows:

    * ``reference=`` passed to :meth:`fit`: a fixed
      :class:`ProgrammeReference` (with its ``beta~``) scores every cell;
    * ``programme_key`` set: an external programme label per feature in
      ``adata.var[programme_key]`` (``"Shared"`` or missing = background);
      R is fixed and ``beta~`` is cross-fitted;
    * otherwise: endogenous group-specific programmes (provisional rule),
      with R and ``beta~`` cross-fitted over ``n_folds`` stratified folds.

    The scorer does not calibrate: ``Delta_G^(P)`` is computed from
    :attr:`references_` by :func:`crossfit_population_calibration` and, at
    a common depth, :func:`depth_standardized_programme_comparison`.

    Parameters
    ----------
    programme_key : str, optional
        ``adata.var`` column with external programme labels.
    layer : str, optional
        ``adata.layers`` key of the accessibility matrix. Default
        ``adata.X``.
    n_folds : int, optional
        Cross-fit folds. Default 5.
    random_state : int, optional
        Fold-split seed. Default 0.
    rule : dict, optional
        Overrides of :data:`PROVISIONAL_RULE` (endogenous mode).
    programme_groups : mapping, optional
        Programme -> defining group, for ``beta~``. Default: the
        programme's name.
    verbose : bool, optional
        Print ``[scCHART xx/xx]`` progress messages. Default True.

    Attributes
    ----------
    groupby_ : str
        Population key used by the last fit.
    reference_mode_ : str
        ``"fixed"``, ``"external_crossfit_background"`` or
        ``"endogenous_crossfit"``.
    references_ : list of ProgrammeReference
        References used to score the cells (one per fold, or one fixed).
    programmes_ : list of str
        Scored programmes; ``K = len(programmes_)``.
    scores_ : pandas.DataFrame
        Per-cell programme outputs after :meth:`fit`.
    population_summary_ : pandas.DataFrame
        ``J_G^(P)``, scored fraction and ``P_i`` summaries per population.
    composition_, evidence_ : numpy.ndarray
        ``r`` and ``a`` (cells x K).
    timings_ : dict
        Total wall-clock duration of the last fit, in seconds.

    Raises
    ------
    ValueError
        If ``n_folds < 2``.

    Examples
    --------
    >>> model = ProgrammeHeterogeneity(n_folds=5, random_state=0)
    >>> model.fit(adata, groupby="cell_type")
    >>> adata.obs[["scchart_programme_P", "scchart_programme_d"]]
    >>> model.population_summary_[["scored_fraction", "J_P_obs", "median_P"]]

    With external programme labels in ``adata.var["programme"]``:

    >>> ProgrammeHeterogeneity(programme_key="programme").fit(adata, groupby="cell_type")
    """

    def __init__(
        self,
        *,
        programme_key: Optional[str] = None,
        layer: Optional[str] = None,
        n_folds: int = 5,
        random_state: int = 0,
        rule: Optional[dict] = None,
        programme_groups: Optional[Mapping[str, str]] = None,
        verbose: bool = True,
    ) -> None:
        if n_folds < 2:
            raise ValueError("n_folds must be >= 2")
        self.programme_key = programme_key
        self.layer = layer
        self.n_folds = int(n_folds)
        self.random_state = int(random_state)
        self.rule = dict(rule or {})
        self.programme_groups = dict(programme_groups or {})
        self.verbose = bool(verbose)

        self.groupby_: Optional[str] = None
        self.reference_mode_: Optional[str] = None
        self.references_: Optional[List[ProgrammeReference]] = None
        self.programmes_: Optional[List[str]] = None
        self.scores_: Optional[pd.DataFrame] = None
        self.population_summary_: Optional[pd.DataFrame] = None
        self.composition_: Optional[np.ndarray] = None
        self.evidence_: Optional[np.ndarray] = None
        self.timings_: Optional[Dict[str, float]] = None

    def _log_stage(self, step: int, total: int, title: str) -> None:
        """Announce a pipeline stage when verbose."""
        if self.verbose:
            print(f"[scCHART {step:02d}/{total:02d}] {title}", flush=True)

    def _log_detail(self, label: str, value: object) -> None:
        """Print one aligned detail line when verbose."""
        if self.verbose:
            print(f"    {label:<32} {value}", flush=True)

    def _matrix(self, adata: AnnData):
        """Return the accessibility matrix (``adata.X`` or the layer) as CSR."""
        if self.layer is None:
            X = adata.X
        elif self.layer in adata.layers:
            X = adata.layers[self.layer]
        else:
            raise KeyError(f"layer={self.layer!r} not found in adata.layers")
        if X is None:
            raise ValueError("adata has no accessibility matrix")
        return sparse.csr_matrix(X)

    @staticmethod
    def _labels(adata: AnnData, groupby: str) -> np.ndarray:
        """Return ``adata.obs[groupby]`` as strings, rejecting missing labels."""
        if groupby not in adata.obs.columns:
            raise KeyError(f"groupby={groupby!r} not found in adata.obs.")
        labels = adata.obs[groupby]
        if labels.isna().any():
            raise ValueError(f"adata.obs[{groupby!r}] contains missing labels.")
        return labels.astype(str).to_numpy()

    def _external_labels(self, adata: AnnData) -> np.ndarray:
        """Return the external programme labels, with missing values as ``"Shared"``."""
        if self.programme_key not in adata.var.columns:
            raise KeyError(f"programme_key={self.programme_key!r} not found in adata.var.")
        lab = adata.var[self.programme_key].astype(object)
        return lab.where(lab.notna(), SHARED_LABEL).astype(str).to_numpy()

    def fit(
        self,
        adata: AnnData,
        groupby: str,
        *,
        reference: Optional[ProgrammeReference] = None,
    ) -> pd.DataFrame:
        """Score programme heterogeneity and write it to ``adata``.

        Parameters
        ----------
        adata : anndata.AnnData
            Cells x declared features (peaks). Modified in place.
        groupby : str
            Population/group column in ``adata.obs`` (the declared ``G``; in
            endogenous mode it also defines the programme groups and the
            folds).
        reference : ProgrammeReference, optional
            Fixed reference scoring every cell. Takes precedence over
            ``programme_key``.

        Returns
        -------
        pandas.DataFrame
            Per-cell programme table; the same columns are written to
            ``adata.obs``.

        Raises
        ------
        KeyError
            If ``groupby``, ``layer`` or ``programme_key`` is missing.
        ValueError
            If labels are missing, there is no accessibility matrix, or the
            reference has a different feature universe.

        Notes
        -----
        Writes ``adata.obs`` score columns,
        ``adata.obsm['scchart_programme_composition']``,
        ``adata.obsm['scchart_programme_evidence']`` and
        ``adata.uns['scchart_programme']``.
        """
        started = time.perf_counter()
        total = 4
        self.groupby_ = groupby

        self._log_stage(1, total, "Reading accessibility and populations")
        X = self._matrix(adata)
        labels = self._labels(adata, groupby)
        self._log_detail("Matrix", f"{X.shape[0]:,} cells × {X.shape[1]:,} features")
        self._log_detail("Population key", groupby)

        self._log_stage(2, total, "Building programme reference")
        if reference is not None:
            if reference.programme_of_feature.size != X.shape[1]:
                raise ValueError("reference and adata have different feature universes")
            refs = [reference]
            self.reference_mode_ = "fixed"
        elif self.programme_key is not None:
            refs = crossfit_programme_references(
                X, labels, n_folds=self.n_folds, random_state=self.random_state,
                external=self._external_labels(adata), programme_groups=self.programme_groups)
            self.reference_mode_ = "external_crossfit_background"
        else:
            refs = crossfit_programme_references(
                X, labels, adata.var_names.to_numpy(), n_folds=self.n_folds,
                random_state=self.random_state, programme_groups=self.programme_groups, rule=self.rule)
            self.reference_mode_ = "endogenous_crossfit"
        self.references_ = refs
        self.programmes_ = list(refs[0].programmes)
        K = len(self.programmes_)
        self._log_detail("Reference mode", self.reference_mode_)
        self._log_detail("Programmes (K)", K)

        self._log_stage(3, total, "Scoring background-corrected evidence")
        if reference is not None:
            s = score_programmes(X, reference)
        else:
            s = score_crossfit(X, refs)
        r, a = s.r, s.evidence
        self.composition_, self.evidence_ = r, a
        self._log_detail("Evidence", EVIDENCE_DEFINITION)
        self._log_detail("Scored cells", f"{int(s.scored.sum()):,} / {r.shape[0]:,}")

        self._log_stage(4, total, "Computing P_i, d_i^(P) and J_G^(P)")
        d = programme_divergence(r, labels)
        # Population tables are in sorted label order; J_P_obs is J_G^(P) in bits.
        table = programme_population_table(r, labels).rename(columns={"J_P_bits": "J_P_obs"})
        self.population_summary_ = table
        dominant = np.full(r.shape[0], np.nan, dtype=object)
        dominant_w = np.full(r.shape[0], np.nan)
        dominant[s.scored] = np.asarray(self.programmes_, dtype=object)[np.argmax(r[s.scored], axis=1)]
        dominant_w[s.scored] = np.max(r[s.scored], axis=1)
        pop_J = table["J_P_obs"].reindex(labels).to_numpy() if "J_P_obs" in table else np.full(r.shape[0], np.nan)

        scores = pd.DataFrame(
            {
                "scchart_programme_P": s.P,
                "scchart_programme_d": d,
                "scchart_programme_population_J_P_obs": pop_J,
                "scchart_programme_scored": s.scored,
                "scchart_programme_dominant": pd.Categorical(dominant, categories=self.programmes_),
                "scchart_programme_dominant_weight": dominant_w,
                "scchart_programme_total_evidence": np.nansum(a, axis=1),
                "scchart_programme_background_depth": s.D,
            },
            index=adata.obs_names,
        )
        self.scores_ = scores
        for column in scores:
            adata.obs[column] = scores[column].values
        adata.obsm["scchart_programme_composition"] = r
        adata.obsm["scchart_programme_evidence"] = a

        rule = dict(refs[0].rule)
        rule.pop("programmes_demoted_not_in_all_folds", None)
        adata.uns["scchart_programme"] = {
            "class_name": self.__class__.__name__,
            "method": METHOD_NAME,
            "groupby": groupby,
            "programmes": list(self.programmes_),
            "K": K,
            "reference_mode": self.reference_mode_,
            "reference_rule": {k: v for k, v in rule.items() if v is not None},
            "n_folds": len(refs),
            "evidence": EVIDENCE_DEFINITION,
            "membership": "r_i = a_i / sum_k a_ik; zero total evidence -> unscorable (NaN)",
            "cell_mixing": "P_i = H(r_i) / log2 K",
            "cell_divergence": "d_i^(P) = KL(r_i || rbar_G) over scored cells",
            "population_score": "J_G^(P) = mean_i d_i^(P) = H(rbar_G) - mean_i H(r_i)",
            "log_base": 2,
            "units": "bits",
            "composition_key": "scchart_programme_composition",
            "evidence_key": "scchart_programme_evidence",
            "population_summary": table.reset_index().to_dict(orient="list"),
            "calibration": "none",
            "calibration_note": "raw J_G^(P); Delta_G^(P) via scchart.Heterogeneity(calibrate=True)",
        }
        if self.programme_key is not None:
            adata.uns["scchart_programme"]["programme_key"] = self.programme_key
        self._log_detail("Primary outputs", "scchart_programme_P, scchart_programme_d")
        self.timings_ = {"total": time.perf_counter() - started}
        if self.verbose:
            print("[scCHART] Fit complete.", flush=True)
        return scores

    def fit_transform(
        self,
        adata: AnnData,
        *,
        groupby: str,
        reference: Optional[ProgrammeReference] = None,
        inplace: bool = True,
    ) -> ProgrammeHeterogeneityResult | AnnData:
        """Fit the scorer and return either ``adata`` or a result container.

        Parameters
        ----------
        adata : anndata.AnnData
            Cells x declared features. Scores are always written to ``adata``
            (see :meth:`fit`).
        groupby : str
            Population/group column in ``adata.obs``.
        reference : ProgrammeReference, optional
            Fixed reference scoring every cell.
        inplace : bool, optional
            If True (default), return ``adata``; otherwise return a
            :class:`ProgrammeHeterogeneityResult`.

        Returns
        -------
        anndata.AnnData or ProgrammeHeterogeneityResult
            ``adata`` when ``inplace`` is True, otherwise copies of the scores,
            population summary, composition and evidence.
        """
        scores = self.fit(adata, groupby=groupby, reference=reference)
        if inplace:
            return adata
        return ProgrammeHeterogeneityResult(
            scores=scores.copy(),
            population_summary=self.population_summary_.copy(),
            composition=pd.DataFrame(self.composition_, index=adata.obs_names, columns=self.programmes_),
            evidence=pd.DataFrame(self.evidence_, index=adata.obs_names, columns=self.programmes_),
        )

    def __repr__(self) -> str:
        """Summarize the configuration and fit status."""
        def row(label: str, value) -> str:
            return f"  {label:<40}{value}"

        fitted = self.scores_ is not None
        mode = ("external (adata.var[%r])" % self.programme_key if self.programme_key
                else "endogenous group-specific (provisional)")
        lines = [
            self.__class__.__name__,
            "Background-corrected programme information divergence.",
            "reference R → evidence a → composition r → P_i, KL to population consensus",
            "",
            "Core Configuration",
            row("Reference", mode),
            row("Cross-fit folds", self.n_folds),
            row("Evidence", "background-corrected, unit weights, clip after mean"),
            row("Unscorable cells", "zero total evidence -> NaN"),
            row("Divergence", "KL, base 2"),
            row("Calibration", "none (see crossfit_population_calibration)"),
            "",
            "Fit Status",
            row("Fitted", "Yes" if fitted else "Not Yet"),
        ]
        if fitted:
            lines.extend([
                row("Population Key", self.groupby_),
                row("Reference Mode", self.reference_mode_),
                row("Cells", len(self.scores_)),
                row("Scored Cells", int(self.scores_["scchart_programme_scored"].sum())),
                row("Programmes (K)", len(self.programmes_)),
            ])
        return "\n".join(lines)
