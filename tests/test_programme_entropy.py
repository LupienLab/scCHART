"""Tests of the programme axis (``scchart.programme_entropy``).

Background-corrected evidence (clip after the programme mean), unscorable
cells, P_i, d_i^(P), J_G^(P) identity and decomposition, cross-fitting leakage
control, external references, the block-swap technical null and Delta_G^(P),
matched-depth comparison, and the AnnData-facing ProgrammeHeterogeneity.
"""

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from scchart._scoring import binarize, entropy_bits, kl_bits, swap_randomise
from scchart.programme_entropy import (
    ProgrammeHeterogeneity,
    ProgrammeHeterogeneityResult,
    ProgrammeReference,
    crossfit_population_calibration,
    crossfit_programme_references,
    depth_standardized_programme_comparison,
    external_programme_reference,
    programme_divergence,
    background_depth,
    programme_evidence,
    programme_membership,
    programme_mixing,
    programme_null_calibration,
    programme_null_replicates,
    programme_population_table,
    reference_background_rates,
    score_crossfit,
    score_programmes,
)


def _toy_reference(beta=0.05):
    # features: p0 x2, p1 x2, p2 x2, background x4
    lab = np.array(["p0", "p0", "p1", "p1", "p2", "p2", "Shared", "Shared", "Shared", "Shared"])
    return ProgrammeReference(lab, ["p0", "p1", "p2"], np.where(lab == "Shared", 0.0, beta))


def _synthetic_programme_data(seed=10, n_per=120, K=3, S=30, n_bg=150):
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"g{k}" for k in range(K)], n_per)
    p = np.full((groups.size, K * S + n_bg), 0.04)
    for k in range(K):
        p[groups == f"g{k}", k * S:(k + 1) * S] = 0.35
    return sparse.csr_matrix((rng.random(p.shape) < p).astype(np.float64)), groups


RULE = dict(min_cells=20, top_n=20, min_features_per_programme=5)
EXTERNAL = np.array(["g0"] * 30 + ["g1"] * 30 + ["g2"] * 30 + ["Shared"] * 150)


def _adata(seed=10, counts=False):
    X, g = _synthetic_programme_data(seed=seed)
    if counts:
        X = X * 2
    a = AnnData(X=X)
    a.obs["group"] = pd.Categorical(g)
    a.obs_names = [f"cell_{i}" for i in range(a.n_obs)]
    a.var_names = [f"peak_{j}" for j in range(a.n_vars)]
    return a


# ============================================================================= reference
def test_reference_validates_inputs():
    lab = np.array(["p0", "p1", "Shared"])
    with pytest.raises(ValueError, match="one entry per feature"):
        ProgrammeReference(lab, ["p0", "p1"], np.zeros(2))
    with pytest.raises(ValueError, match="without features"):
        ProgrammeReference(lab, ["p0", "p9"], np.zeros(3))
    with pytest.raises(ValueError, match="two programmes"):
        ProgrammeReference(lab, ["p0"], np.zeros(3))
    ref = _toy_reference()
    assert ref.K == 3
    np.testing.assert_array_equal(ref.background_mask, [False] * 6 + [True] * 4)
    assert ref.sizes().tolist() == [2, 2, 2]


def _derived_v010(lab, programmes):
    """v0.1.0 computation of the derived reference arrays (no cache), kept as the reference."""
    lab = np.asarray(lab).astype(str).astype(object)
    k_of = np.full(lab.size, -1)
    for k, p in enumerate(programmes):
        k_of[lab == p] = k
    rows = np.flatnonzero(k_of >= 0)
    M = sparse.csr_matrix((np.ones(rows.size), (rows, k_of[rows])), shape=(k_of.size, len(programmes)))
    return ~np.isin(lab, programmes), k_of, M, pd.Series({p: int((lab == p).sum()) for p in programmes})


def _assert_derived_equal(ref, lab, programmes):
    bg, k_of, M, sizes = _derived_v010(lab, programmes)
    assert ref.background_mask.tobytes() == bg.tobytes()
    assert ref.programme_index.dtype == k_of.dtype and ref.programme_index.tobytes() == k_of.tobytes()
    got = ref.membership_matrix()
    assert got.dtype == M.dtype and (got != M).nnz == 0 and got.has_canonical_format == M.has_canonical_format
    pd.testing.assert_series_equal(ref.sizes(), sizes, check_exact=True)


def test_reference_derived_arrays_are_cached_and_match_v010():
    import copy
    import pickle

    ref = _toy_reference()
    lab = ref.programme_of_feature.copy()
    for _ in range(2):  # computed, then served from the cache
        _assert_derived_equal(ref, lab, ref.programmes)
    # returned arrays are copies: changing them does not change the reference
    ref.background_mask[:] = False
    ref.programme_index[:] = 7
    ref.membership_matrix().data[:] = 5.0
    _assert_derived_equal(ref, lab, ref.programmes)
    # the stored labels are read-only; the caller's array is not aliased
    with pytest.raises(ValueError):
        ref.programme_of_feature[0] = "p1"
    mine = np.array(["p0", "p1", "Shared", "Shared"], dtype=object)
    ProgrammeReference(mine, ["p0", "p1"], np.zeros(4))
    mine[0] = "p1"  # still writeable
    # changing the programme list in place or assigning new labels recomputes
    ref.programmes.remove("p2")
    _assert_derived_equal(ref, lab, ["p0", "p1"])
    new_lab = np.array(["p1"] * 4 + ["p0"] * 2 + ["Shared"] * 4, dtype=object)
    ref.programme_of_feature = new_lab
    _assert_derived_equal(ref, new_lab, ["p0", "p1"])
    new_lab[0] = "p0"  # an assigned (writeable) array is never cached
    _assert_derived_equal(ref, new_lab, ["p0", "p1"])
    # pickling and copying keep the labels read-only and the results equal
    for other in (pickle.loads(pickle.dumps(_toy_reference())), copy.deepcopy(_toy_reference())):
        assert not other.programme_of_feature.flags.writeable
        _assert_derived_equal(other, lab, other.programmes)


def test_background_rates_use_cells_outside_the_defining_group():
    lab = np.array(["A", "B", "Shared", "Shared"])
    B = np.array([[1, 0, 1, 1],    # group A: D = 2
                  [1, 1, 1, 0],    # group B: D = 1
                  [0, 1, 1, 1]], float)  # group B: D = 2
    g = np.array(["A", "B", "B"])
    bt = reference_background_rates(sparse.csr_matrix(B), g, lab, ["A", "B"])
    assert bt[0] == pytest.approx((1 + 0) / (1 + 2))   # peak of A, estimated on the B cells
    assert bt[1] == pytest.approx(0 / 2)               # peak of B, estimated on the A cell
    np.testing.assert_array_equal(bt[2:], 0.0)


# ============================================================================= evidence
def _reference_background_rates_v010(B_ref, ref_groups, programme_of_feature, programmes, programme_groups=None):
    """The v0.1.0 implementation, kept as the reference."""
    g = np.asarray(ref_groups).astype(str)
    lab = np.asarray(programme_of_feature).astype(str)
    bgm = ~np.isin(lab, list(programmes))
    D_ref = np.asarray(B_ref[:, bgm].sum(axis=1)).ravel()
    beta_t = np.zeros(B_ref.shape[1])
    Bc = sparse.csc_matrix(B_ref)
    for p in programmes:
        cols = np.flatnonzero(lab == p)
        if cols.size == 0:
            continue
        out = g != (programme_groups or {}).get(p, p)
        beta_t[cols] = np.asarray(Bc[out][:, cols].sum(axis=0)).ravel() / max(D_ref[out].sum(), 1e-12)
    return beta_t


@pytest.mark.parametrize("kind", ["binary", "float32_counts", "uint32_counts", "int64_csc", "nonint", "dense"])
def test_reference_background_rates_match_v010(kind):
    X, g = _synthetic_programme_data(seed=24)
    B = binarize(X)
    rng = np.random.default_rng(1)
    M = {"binary": B, "float32_counts": (B * 3).astype(np.float32), "uint32_counts": (B * 2).astype(np.uint32),
         "int64_csc": sparse.csc_matrix((B * 4).astype(np.int64)),
         "nonint": B.multiply(rng.random(B.shape)).tocsr(), "dense": B.toarray()}[kind]
    lab = EXTERNAL
    progs = ["g0", "g1", "g2", "absent"]
    for pg in (None, {"g0": "g1", "g2": "nobody"}):
        got = reference_background_rates(M, g, lab, progs, programme_groups=pg)
        want = _reference_background_rates_v010(M, g, lab, progs, programme_groups=pg)
        assert got.dtype == want.dtype and got.tobytes() == want.tobytes()


def _group_detection_rates_v010(B, labels, groups):
    """The v0.1.0 implementation, kept as the reference."""
    labels = np.asarray(labels).astype(str)
    X = sparse.csr_matrix(B, copy=True)
    X.data = np.ones_like(X.data, dtype=np.float32)
    index = {g: i for i, g in enumerate(groups)}
    rows = np.flatnonzero(np.isin(labels, groups))
    cols = np.array([index[l] for l in labels[rows]], dtype=np.int64)
    ind = sparse.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(labels.size, len(groups)))
    group_sum = np.asarray((ind.T @ X).todense(), dtype=np.float64)
    total_sum = np.asarray(X.sum(axis=0), dtype=np.float64).ravel()
    n_in = np.asarray(ind.sum(axis=0)).ravel()
    n_out = labels.size - n_in
    inside = group_sum / np.maximum(n_in, 1)[:, None]
    outside = (total_sum[None, :] - group_sum) / np.maximum(n_out, 1)[:, None]
    return inside.astype(np.float32), outside.astype(np.float32)


@pytest.mark.parametrize("kind", ["binary", "counts"])
def test_group_detection_rates_and_programmes_match_v010(kind):
    from scchart.programme_entropy import _group_detection_rates
    X, g = _synthetic_programme_data(seed=25)
    B = binarize(X) if kind == "binary" else (X * 3).tocsr()
    groups = ["g0", "g2", "g1"]
    for M in (B, sparse.csr_array(B)):
        for got, want in zip(_group_detection_rates(M, g, groups), _group_detection_rates_v010(M, g, groups)):
            assert got.dtype == want.dtype and got.tobytes() == want.tobytes()


def test_programme_evidence_explicit():
    ref = _toy_reference(0.05)
    B = np.array([[1, 1, 0, 0, 0, 0, 1, 1, 0, 0],       # D=2, only p0 open
                  [1, 0, 1, 0, 0, 1, 1, 1, 1, 1]], float)  # D=4, one peak of each
    a = programme_evidence(B, ref)
    # a_ik = max(0, (O_ik - D_i * sum_{R_k} beta) / |R_k|)
    np.testing.assert_allclose(a[0], [(2 - 2 * 0.1) / 2, 0.0, 0.0])
    np.testing.assert_allclose(a[1], [(1 - 4 * 0.1) / 2] * 3)
    r = programme_membership(a)
    np.testing.assert_allclose(r[0], [1, 0, 0])
    np.testing.assert_allclose(r[1], [1 / 3] * 3)
    P = programme_mixing(r)
    assert P[0] == pytest.approx(0.0) and P[1] == pytest.approx(1.0)


@pytest.mark.parametrize("counts", [False, True])
def test_background_depth_and_evidence_match_v010(counts):
    X, g = _synthetic_programme_data(seed=23)
    if counts:
        X = (X * 3).astype(np.float32)
    for ref in crossfit_programme_references(X, g, n_folds=3, rule=RULE) + [
            external_programme_reference(EXTERNAL, X, g)]:
        B = binarize(X)
        D_v010 = np.asarray(B[:, ref.background_mask].sum(axis=1)).ravel()
        M = ref.membership_matrix()
        O = (B @ M).toarray()
        a_v010 = np.maximum(0.0, (O - D_v010[:, None] * np.asarray(M.T @ ref.beta_tilde).ravel()[None, :])
                            / np.asarray(M.sum(axis=0)).ravel())
        assert background_depth(X, ref).tobytes() == D_v010.tobytes()
        assert programme_evidence(X, ref).tobytes() == a_v010.tobytes()
        s = score_programmes(X, ref)
        assert s.D.tobytes() == D_v010.tobytes() and s.evidence.tobytes() == a_v010.tobytes()


def test_evidence_clips_the_programme_mean_not_each_peak():
    ref = _toy_reference(0.3)
    B = np.array([[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]], float)  # D=2, E_j = 0.6 per peak
    a = programme_evidence(B, ref)
    # max(0, ((1 - 0.6) + (0 - 0.6)) / 2) = 0, not (max(0, 0.4) + max(0, -0.6)) / 2 = 0.2
    assert a[0, 0] == pytest.approx(0.0)
    assert np.isnan(programme_membership(a)).all()


def test_evidence_uses_unit_weights_and_rejects_other_universes():
    ref = _toy_reference(0.0)
    B = np.array([[1, 0, 1, 1, 0, 0, 0, 0, 0, 0]], float)
    np.testing.assert_allclose(programme_evidence(B, ref), [[0.5, 1.0, 0.0]])
    with pytest.raises(ValueError, match="feature universes"):
        programme_evidence(B[:, :5], ref)
    with pytest.raises(ValueError, match="non-negative"):
        programme_membership(np.array([[0.2, -0.1]]))


def test_zero_evidence_is_unscorable_never_imputed():
    ref = _toy_reference(0.05)
    B = np.zeros((2, 10))
    B[0, 6:] = 1   # background only
    B[1, 0] = 1
    r = programme_membership(programme_evidence(B, ref))
    assert np.isnan(r[0]).all()
    P = programme_mixing(r)
    assert np.isnan(P[0]) and P[1] == pytest.approx(0.0)
    tab = programme_population_table(r, ["G", "G"])
    assert tab.loc["G", "n_scored"] == 1 and tab.loc["G", "scored_fraction"] == 0.5
    assert np.isnan(programme_divergence(r, ["G", "G"])[0])
    s = score_programmes(B, ref)
    np.testing.assert_array_equal(s.scored, [False, True])
    np.testing.assert_array_equal(s.D, [4, 0])


# ============================================================================= P_i, d_i^(P), J_G^(P)
def test_P_range_and_J_identity_and_decomposition():
    rng = np.random.default_rng(9)
    a = rng.gamma(0.5, size=(300, 5))
    a[rng.random(300) < 0.1] = 0.0
    r = programme_membership(a)
    P = programme_mixing(r)
    ok = ~np.isnan(P)
    assert np.all((P[ok] >= -1e-12) & (P[ok] <= 1 + 1e-12))
    np.testing.assert_allclose(P[ok], entropy_bits(r[ok]) / np.log2(5))
    g = rng.choice(["A", "B", "C"], 300)
    tab = programme_population_table(r, g)
    d = programme_divergence(r, g)
    for G in "ABC":
        m = (g == G) & ok
        rbar = r[m].mean(axis=0)
        J_id = entropy_bits(rbar[None])[0] - entropy_bits(r[m]).mean()
        assert tab.loc[G, "J_P_bits"] == pytest.approx(J_id, abs=1e-12)
        assert np.mean(d[m]) == pytest.approx(tab.loc[G, "J_P_bits"], abs=1e-12)
        np.testing.assert_allclose(d[m], kl_bits(r[m], rbar))
        assert np.isnan(d[(g == G) & ~ok]).all()
        assert tab.loc[G, "scored_fraction"] == pytest.approx(m.sum() / (g == G).sum())


def test_equally_mixed_cells_have_high_P_and_zero_J():
    r = programme_membership(np.full((10, 4), 0.25))
    assert np.allclose(programme_mixing(r), 1.0)
    assert programme_population_table(r, ["G"] * 10).loc["G", "J_P_bits"] == pytest.approx(0.0, abs=1e-15)


# ============================================================================= cross-fitting
def test_crossfit_no_leakage():
    X, g = _synthetic_programme_data()
    refs = crossfit_programme_references(X, g, n_folds=4, rule=RULE)
    assert np.array_equal(np.sort(np.concatenate([r.score_index for r in refs])), np.arange(X.shape[0]))
    for ref in refs:
        assert np.intersect1d(ref.train_index, ref.score_index).size == 0
        assert ref.source == "endogenous_crossfit"
    # R and beta~ of a fold are unchanged when its held-out cells are replaced.
    ref0 = refs[0]
    X2 = X.tolil()
    for i in ref0.score_index:
        X2[i, :] = 0
        X2[i, 0] = 1
    refs2 = crossfit_programme_references(X2.tocsr(), g, n_folds=4, rule=RULE)
    np.testing.assert_array_equal(refs2[0].programme_of_feature, ref0.programme_of_feature)
    np.testing.assert_array_equal(refs2[0].beta_tilde, ref0.beta_tilde)


def test_crossfit_scores_every_cell_once_and_recovers_own_programme():
    X, g = _synthetic_programme_data()
    refs = crossfit_programme_references(X, g, n_folds=4, rule=RULE)
    s = score_crossfit(X, refs)
    assert s.r.shape == (X.shape[0], 3)
    dom = np.argmax(np.nan_to_num(s.r, nan=-1), axis=1)
    own = np.array([s.programmes.index(x) for x in g])
    assert (dom == own).mean() > 0.9


def test_score_crossfit_rejects_incomplete_or_overlapping_folds():
    X, g = _synthetic_programme_data()
    refs = crossfit_programme_references(X, g, n_folds=4, rule=RULE)
    with pytest.raises(ValueError, match="not held out"):
        score_crossfit(X, refs[:3])
    with pytest.raises(ValueError, match="more than one fold"):
        score_crossfit(X, refs + refs[:1])


def test_external_reference_supported():
    X, g = _synthetic_programme_data(seed=13)
    refs = crossfit_programme_references(X, g, n_folds=3, external=EXTERNAL)
    assert all(r.source == "external" for r in refs)
    assert all(np.array_equal(r.programme_of_feature.astype(str), EXTERNAL) for r in refs)
    ref = external_programme_reference(EXTERNAL, beta_tilde=np.where(EXTERNAL == "Shared", 0, 0.01))
    assert ref.K == 3 and ref.score_index is None
    ref2 = external_programme_reference(EXTERNAL, X, g)
    np.testing.assert_allclose(ref2.beta_tilde, reference_background_rates(binarize(X), g, EXTERNAL,
                                                                           ["g0", "g1", "g2"]))
    with pytest.raises(ValueError, match="beta_tilde"):
        external_programme_reference(EXTERNAL)


# ============================================================================= technical null
def test_programme_null_preserves_block_margins_and_background_row_sums():
    X, g = _synthetic_programme_data(seed=11)
    ref = crossfit_programme_references(X, g, n_folds=4, rule=RULE)[0]
    B = binarize(X[ref.score_index[g[ref.score_index] == "g1"]]).toarray()
    prog = ~ref.background_mask
    D_prog, _ = swap_randomise(B[:, prog], seed=5)
    np.testing.assert_array_equal(D_prog.sum(axis=1), B[:, prog].sum(axis=1))   # programme burden
    np.testing.assert_array_equal(D_prog.sum(axis=0), B[:, prog].sum(axis=0))   # peak frequency
    D_bg, _ = swap_randomise(B[:, ~prog], seed=6)
    np.testing.assert_array_equal(D_bg.sum(axis=1), B[:, ~prog].sum(axis=1))    # D_i preserved
    r1, _ = programme_null_replicates(B, ref, n_replicates=3, seed=40)
    r2, _ = programme_null_replicates(B, ref, n_replicates=3, seed=40, randomise_background=True)
    for x, y in zip(r1, r2):
        np.testing.assert_array_equal(np.isnan(x), np.isnan(y))
        np.testing.assert_array_equal(np.nan_to_num(x), np.nan_to_num(y))  # bg enters only via D_i


def test_programme_null_deterministic_and_delta_definition():
    X, g = _synthetic_programme_data(seed=12)
    refs = crossfit_programme_references(X, g, n_folds=4, rule=RULE)
    t1, o1 = crossfit_population_calibration(X, g, refs, n_replicates=4, seed=77)
    t2, _ = crossfit_population_calibration(X, g, refs, n_replicates=4, seed=77)
    pd.testing.assert_frame_equal(t1, t2)
    for cal in o1.values():
        assert cal.Delta == pytest.approx(cal.J_obs - np.nanmean(cal.J_null))
        assert cal.margins_exact_all
    J = programme_population_table(score_crossfit(X, refs).r, g)
    np.testing.assert_allclose(t1["J_P_obs"].to_numpy(), J["J_P_bits"].to_numpy(), atol=1e-12)
    np.testing.assert_allclose(t1["scored_fraction"].to_numpy(), J["scored_fraction"].to_numpy())


def test_programme_null_with_a_single_fixed_reference():
    X, g = _synthetic_programme_data(seed=16)
    ref = external_programme_reference(EXTERNAL, X, g)
    tab, _ = crossfit_population_calibration(X, g, [ref], n_replicates=3, seed=5)
    J = programme_population_table(score_programmes(X, ref).r, g)
    np.testing.assert_allclose(tab["J_P_obs"].to_numpy(), J["J_P_bits"].to_numpy(), atol=1e-12)


def test_null_cells_option():
    X, g = _synthetic_programme_data(seed=17)
    ref = crossfit_programme_references(X, g, n_folds=3, rule=RULE)[0]
    part = [(X[ref.score_index[:40]], ref, 3)]
    with pytest.raises(ValueError, match="null_cells"):
        programme_null_calibration(part, n_replicates=2, null_cells="none")
    cal = programme_null_calibration(part, n_replicates=2, null_cells="all")
    assert cal.J_null.size == 2


def _programme_null_replicates_v010(B, ref, n_replicates, sampler, seed, randomise_background):
    """The v0.1.0 serial loop (one swap_randomise call per chain), kept as the reference."""
    Bb = binarize(B)
    bgm = ref.background_mask
    M = ref.membership_matrix()[np.flatnonzero(~bgm)]
    Bp, Bg = Bb[:, np.flatnonzero(~bgm)], Bb[:, np.flatnonzero(bgm)]
    D0 = np.asarray(Bg.sum(axis=1)).ravel()
    size = np.asarray(M.sum(axis=0)).ravel()
    rs, diags = [], []
    for b in range(n_replicates):
        Np, dp = swap_randomise(Bp, seed + 2 * b, sampler=sampler)
        diags.append(("programme", dp))
        D = D0
        if randomise_background:
            Nb, db = swap_randomise(Bg, seed + 2 * b + 1, sampler=sampler)
            diags.append(("background", db))
            D = Nb.sum(axis=1).astype(np.float64)
        O = (sparse.csr_matrix(Np, dtype=np.float64) @ M).toarray()
        a = np.maximum(0.0, (O - D[:, None] * np.asarray(M.T @ ref.beta_tilde[~bgm]).ravel()[None, :]) / size)
        rs.append(programme_membership(a))
    return rs, diags


@pytest.mark.parametrize("sampler", ["proposal_clock", "accepted_swap"])
@pytest.mark.parametrize("randomise_background", [False, True])
def test_programme_null_replicates_match_v010_serial_loop_for_any_n_jobs(sampler, randomise_background):
    X, g = _synthetic_programme_data(seed=21)
    ref = crossfit_programme_references(X, g, n_folds=3, rule=RULE)[0]
    B = X[ref.score_index[g[ref.score_index] == "g2"]]
    want_r, want_d = _programme_null_replicates_v010(B, ref, 5, sampler, 300, randomise_background)
    for n_jobs in (1, 2, 3, 8):
        r, d = programme_null_replicates(B, ref, n_replicates=5, sampler=sampler, seed=300,
                                         randomise_background=randomise_background, n_jobs=n_jobs)
        assert [x.tobytes() for x in r] == [x.tobytes() for x in want_r]
        assert d == want_d


@pytest.mark.parametrize("null_cells", ["scored", "all"])
def test_programme_calibration_does_not_depend_on_n_jobs(null_cells):
    X, g = _synthetic_programme_data(seed=22)
    refs = crossfit_programme_references(X, g, n_folds=3, rule=RULE)
    t1, c1 = crossfit_population_calibration(X, g, refs, n_replicates=5, seed=9, null_cells=null_cells)
    t4, c4 = crossfit_population_calibration(X, g, refs, n_replicates=5, seed=9, null_cells=null_cells, n_jobs=4)
    pd.testing.assert_frame_equal(t1, t4, check_exact=True)
    for G in c1:
        assert c1[G].J_null.tobytes() == c4[G].J_null.tobytes()
    d1 = depth_standardized_programme_comparison(X * 2, g, refs, 10.0, n_replicates=2, n_null=3)
    d4 = depth_standardized_programme_comparison(X * 2, g, refs, 10.0, n_replicates=2, n_null=3, n_jobs=3)
    pd.testing.assert_frame_equal(d1.replicates, d4.replicates, check_exact=True)


# ============================================================================= matched depth
def test_depth_standardized_programme_runs_and_never_upsamples():
    X, g = _synthetic_programme_data(seed=18)
    X = sparse.csr_matrix(X.toarray() * np.where(g == "g0", 2.0, 1.0)[:, None])   # g0 is deeper
    refs = crossfit_programme_references(X, g, n_folds=3, rule=RULE)
    D_star = float(np.median(np.asarray(X[g == "g1"].sum(axis=1)).ravel()))
    res = depth_standardized_programme_comparison(X, g, refs, D_star, n_replicates=2, n_null=2)
    assert res.axis == "programme" and res.D_star == D_star
    assert res.summary.loc["g0", "q"] < 1 and res.summary.loc["g1", "q"] == 1
    assert res.summary.loc["g0", "n_replicates"] == 2
    np.testing.assert_allclose(res.replicates["Delta_P"], res.replicates["J_P_obs"] - res.replicates["E0_J_P"])


# ============================================================================= AnnData class
def test_constructor_has_no_removed_estimator_options():
    params = set(inspect.signature(ProgrammeHeterogeneity.__init__).parameters)
    for removed in ("dirichlet_alpha", "weight_mode", "balance_features", "idf_smoothing",
                    "heterogeneity_mode", "information_scale"):
        assert removed not in params
    params_fit = list(inspect.signature(ProgrammeHeterogeneity.fit).parameters)
    assert params_fit[:3] == ["self", "adata", "groupby"]


def test_fit_endogenous_crossfit_matches_functional_api():
    a = _adata()
    model = ProgrammeHeterogeneity(n_folds=4, rule=RULE, verbose=False)
    scores = model.fit(a, groupby="group")
    g = a.obs["group"].astype(str).to_numpy()
    refs = crossfit_programme_references(a.X, g, a.var_names.to_numpy(), n_folds=4, rule=RULE)
    s = score_crossfit(a.X, refs)
    np.testing.assert_allclose(scores["scchart_programme_P"], s.P)
    np.testing.assert_allclose(a.obsm["scchart_programme_composition"], s.r)
    np.testing.assert_allclose(scores["scchart_programme_d"], programme_divergence(s.r, g))
    tab = programme_population_table(s.r, g)
    np.testing.assert_allclose(model.population_summary_.loc[tab.index, "J_P_obs"], tab["J_P_bits"])
    assert model.reference_mode_ == "endogenous_crossfit"
    assert all(r.source == "endogenous_crossfit" for r in model.references_)
    assert model.programmes_ == ["g0", "g1", "g2"]
    for G in ("g0", "g1", "g2"):
        m = g == G
        assert scores["scchart_programme_population_J_P_obs"][m].iloc[0] == pytest.approx(tab.loc[G, "J_P_bits"])
    assert (scores["scchart_programme_dominant"].astype(str).to_numpy() == g).mean() > 0.9


def test_fit_writes_outputs_and_metadata():
    a = _adata()
    ProgrammeHeterogeneity(n_folds=3, rule=RULE, verbose=False).fit(a, groupby="group")
    for col in ("scchart_programme_P", "scchart_programme_d", "scchart_programme_population_J_P_obs",
                "scchart_programme_scored", "scchart_programme_dominant", "scchart_programme_dominant_weight",
                "scchart_programme_total_evidence", "scchart_programme_background_depth"):
        assert col in a.obs
    assert a.obsm["scchart_programme_composition"].shape == (a.n_obs, 3)
    assert a.obsm["scchart_programme_evidence"].shape == (a.n_obs, 3)
    meta = a.uns["scchart_programme"]
    assert meta["programmes"] == ["g0", "g1", "g2"] and meta["K"] == 3
    assert meta["reference_mode"] == "endogenous_crossfit"
    assert meta["n_folds"] == 3
    assert meta["log_base"] == 2 and meta["units"] == "bits"
    assert meta["calibration"] == "none"
    assert "max(0, (1/|R_k|)" in meta["evidence"]


def test_fit_external_programme_key_and_fixed_reference():
    a = _adata(seed=13)
    a.var["programme"] = EXTERNAL
    a.var.loc[a.var_names[-1], "programme"] = np.nan   # missing = background
    model = ProgrammeHeterogeneity(programme_key="programme", n_folds=3, verbose=False)
    model.fit(a, groupby="group")
    assert model.reference_mode_ == "external_crossfit_background"
    assert all(r.source == "external" for r in model.references_)
    assert a.uns["scchart_programme"]["programme_key"] == "programme"

    ref = external_programme_reference(EXTERNAL, beta_tilde=np.where(EXTERNAL == "Shared", 0.0, 0.002))
    fixed = ProgrammeHeterogeneity(verbose=False)
    scores = fixed.fit(a, groupby="group", reference=ref)
    assert fixed.reference_mode_ == "fixed" and fixed.references_ == [ref]
    np.testing.assert_allclose(scores["scchart_programme_P"], score_programmes(a.X, ref).P)
    with pytest.raises(ValueError, match="feature universes"):
        fixed.fit(a[:, :50].copy(), groupby="group", reference=ref)


def test_unscorable_cells_are_na_in_adata():
    a = _adata()
    X = a.X.tolil()          # edit outside AnnData: anndata >= 0.12 accepts only CSR/CSC in .X
    X[0, :] = 0
    X[0, 200] = 1            # background only -> zero evidence
    a.X = X.tocsr()
    lab = np.array(["g0"] * 30 + ["g1"] * 30 + ["g2"] * 30 + ["Shared"] * 150)
    ref = external_programme_reference(lab, a.X, a.obs["group"].astype(str).to_numpy())
    ProgrammeHeterogeneity(verbose=False).fit(a, groupby="group", reference=ref)
    assert not a.obs["scchart_programme_scored"].iloc[0]
    assert np.isnan(a.obs["scchart_programme_P"].iloc[0])
    assert np.isnan(a.obs["scchart_programme_d"].iloc[0])
    assert pd.isna(a.obs["scchart_programme_dominant"].iloc[0])
    assert np.isnan(a.obsm["scchart_programme_composition"][0]).all()


def test_layer_is_used_when_given():
    a = _adata()
    a.layers["counts"] = a.X.copy()
    a.X = sparse.csr_matrix(a.shape)
    ProgrammeHeterogeneity(layer="counts", n_folds=3, rule=RULE, verbose=False).fit(a, groupby="group")
    assert a.obs["scchart_programme_scored"].all()
    with pytest.raises(KeyError):
        ProgrammeHeterogeneity(layer="missing", verbose=False).fit(a, groupby="group")


def test_fit_transform_result_and_h5ad_round_trip(tmp_path):
    import anndata

    a = _adata()
    model = ProgrammeHeterogeneity(n_folds=3, rule=RULE, verbose=False)
    res = model.fit_transform(a, groupby="group", inplace=False)
    assert isinstance(res, ProgrammeHeterogeneityResult)
    assert list(res.composition.columns) == ["g0", "g1", "g2"]
    pd.testing.assert_index_equal(res.scores.index, a.obs_names)
    assert model.fit_transform(a, groupby="group") is a
    path = tmp_path / "programme.h5ad"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a.write_h5ad(path)
    meta = anndata.read_h5ad(path).uns["scchart_programme"]
    assert list(meta["programmes"]) == ["g0", "g1", "g2"]
    assert meta["reference_mode"] == "endogenous_crossfit"


def test_repr_and_missing_keys():
    model = ProgrammeHeterogeneity(verbose=False)
    assert "Not Yet" in repr(model)
    a = _adata()
    with pytest.raises(KeyError):
        model.fit(a, groupby="missing")
    with pytest.raises(KeyError):
        ProgrammeHeterogeneity(programme_key="missing", verbose=False).fit(a, groupby="group")
    with pytest.raises(ValueError, match="n_folds"):
        ProgrammeHeterogeneity(n_folds=1)
    ProgrammeHeterogeneity(n_folds=3, rule=RULE, verbose=False).fit(a, groupby="group")
