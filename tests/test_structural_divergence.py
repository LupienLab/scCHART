"""Tests of the structural axis (``scchart.structural_divergence``).

Both simplex adapters (signed channel, the default, and spectral energy) share
one KL / generalized-JSD engine, so the estimand tests run on both. Also
covered: zero-vector behaviour, the fixed LSI projection, the fixed-margin
technical null and Delta_G, the same-representation requirement and the
matched-depth comparison.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from scchart._scoring import generalized_jsd
from scchart.structural_divergence import (
    LSIProjection,
    StructuralHeterogeneity,
    StructuralHeterogeneityResult,
    adapter_probability,
    depth_standardized_structural_comparison,
    signed_channel_probability,
    spectral_energy_probability,
    structural_chain_rule,
    structural_J,
    structural_null_calibration,
    structural_population_calibration,
)

ADAPTERS = ["signed_channel", "spectral_energy"]


def _adata(Z, labels, X=None):
    n = len(labels)
    if X is None:
        X = sparse.csr_matrix(np.eye(n, max(n, 4), dtype=float))
    a = AnnData(X=X)
    a.obs["group"] = pd.Categorical(labels)
    a.obsm["X_lsi"] = np.asarray(Z, dtype=float)
    a.obs_names = [f"cell_{i}" for i in range(n)]
    return a


def _random_adata(seed=0, n_per=60, n_components=6):
    rng = np.random.default_rng(seed)
    n = 2 * n_per
    X = sparse.csr_matrix(rng.binomial(1, 0.3, size=(n, 40)).astype(float))
    Z = rng.normal(size=(n, n_components))
    return _adata(Z, ["A"] * n_per + ["B"] * n_per, X=X)


def _fit(Z, labels, adapter="signed_channel"):
    a = _adata(Z, labels)
    model = StructuralHeterogeneity(adapter=adapter, verbose=False)
    scores = model.fit(a, groupby="group", use_rep="X_lsi")
    return model, scores


def _binary(rng, n, p, rate):
    B = (rng.random((n, p)) < rate).astype(np.float64)
    B[np.arange(n), rng.integers(0, p, n)] = 1.0  # every cell has >= 1 open feature
    return B


# ---------------------------------------------------------------------------
# Public signature
# ---------------------------------------------------------------------------


def test_constructor_selects_adapter_not_representation():
    sig = inspect.signature(StructuralHeterogeneity.__init__)
    assert set(sig.parameters) == {"self", "adapter", "store_probabilities", "verbose"}
    assert sig.parameters["adapter"].default == "signed_channel"
    assert StructuralHeterogeneity(verbose=False).adapter == "signed_channel"
    assert "qc_components" not in StructuralHeterogeneityResult.__dataclass_fields__
    with pytest.raises(ValueError, match="adapter"):
        StructuralHeterogeneity(adapter="softmax")


# ---------------------------------------------------------------------------
# Signed-channel adapter
# ---------------------------------------------------------------------------


def test_signed_channel_construction_explicit():
    Z = np.array([[1.0, -2.0], [-3.0, 0.0]])
    P = signed_channel_probability(Z)
    # channels: [+c1, +c2, -c1, -c2]
    np.testing.assert_allclose(P[0], [1 / 5, 0, 0, 4 / 5])
    np.testing.assert_allclose(P[1], [0, 0, 1, 0])
    np.testing.assert_allclose(P.sum(axis=1), 1.0)


def test_signed_channel_keeps_relative_sign():
    z = np.array([[1.0, 2.0, -0.5]])
    P1 = signed_channel_probability(z)
    P2 = signed_channel_probability(-z)
    assert not np.any((P1 > 0) & (P2 > 0))  # z and -z have disjoint support
    assert generalized_jsd(np.vstack([P1, P2]))["J"] == pytest.approx(1.0)  # antipodal: 1 bit
    # spectral energy collapses the same antipodal pair
    S = spectral_energy_probability(np.vstack([z, -z]))
    assert generalized_jsd(S)["J"] == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_signed_channel_invariances(seed):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(30, 5))
    J0 = structural_J(Z)["J"]
    d0 = structural_J(Z)["d"]
    flip = np.where(rng.random(5) < 0.5, -1.0, 1.0)
    assert structural_J(Z * flip)["J"] == pytest.approx(J0, rel=1e-12)          # global sign flips
    np.testing.assert_allclose(structural_J(Z * rng.uniform(0.1, 9, (30, 1)))["d"], d0, rtol=1e-10)
    assert structural_J(3.7 * Z)["J"] == pytest.approx(J0, rel=1e-12)           # global scale
    perm = rng.permutation(30)
    assert structural_J(Z[perm])["J"] == pytest.approx(J0, rel=1e-12)          # cell order


def test_signed_channel_is_not_rotation_invariant():
    rng = np.random.default_rng(4)
    Z = rng.normal(size=(40, 3))
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    assert abs(structural_J(Z @ Q)["J"] - structural_J(Z)["J"]) > 1e-3


# ---------------------------------------------------------------------------
# Spectral-energy adapter
# ---------------------------------------------------------------------------


def test_spectral_energy_probability_rows_sum_to_one():
    Z = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    P = spectral_energy_probability(Z)
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    np.testing.assert_allclose(P[0], np.array([1, 4, 9]) / 14)
    assert np.all(P >= 0)


def test_spectral_energy_is_invariant_to_component_sign_flip():
    Z = np.array([[1.0, 2.0, -3.0], [4.0, -5.0, 6.0]])
    Z2 = Z.copy()
    Z2[:, 1] *= -1
    np.testing.assert_allclose(spectral_energy_probability(Z), spectral_energy_probability(Z2))


def test_spectral_energy_is_invariant_to_positive_row_scaling():
    Z = np.array([[1.0, 2.0, 3.0], [4.0, 1.0, 2.0]])
    P1 = spectral_energy_probability(Z)
    P2 = spectral_energy_probability(Z * np.array([[7.0], [0.2]]))
    np.testing.assert_allclose(P1, P2)


def test_spectral_energy_is_selectable_without_warning():
    import warnings

    a = _random_adata()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = StructuralHeterogeneity(adapter="spectral_energy", verbose=False)
        model.fit(a, groupby="group")
    np.testing.assert_allclose(model.probabilities_, spectral_energy_probability(a.obsm["X_lsi"]))
    assert a.uns["scchart_structural"]["simplex_adapter"] == "spectral_energy"
    assert structural_J(a.obsm["X_lsi"][:60], adapter="spectral_energy")["J"] == pytest.approx(
        model.population_summary_.loc["A", "J_obs"], rel=1e-12)


# ---------------------------------------------------------------------------
# Zero-vector behaviour (both adapters)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_zero_vector_rejected(adapter):
    with pytest.raises(ValueError, match="z_i = 0"):
        adapter_probability(np.array([[1.0, 0.0], [0.0, 0.0]]), adapter)
    with pytest.raises(ValueError, match="non-finite"):
        adapter_probability(np.array([[np.nan, 1.0]]), adapter)


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_fit_raises_on_zero_vector_cell(adapter):
    Z = np.array([[1.0, 2.0], [0.0, 0.0], [3.0, 1.0], [1.0, 1.0]])
    a = _adata(Z, ["A", "A", "B", "B"])
    with pytest.raises(ValueError, match="z_i = 0"):
        StructuralHeterogeneity(adapter=adapter, verbose=False).fit(a, groupby="group")


# ---------------------------------------------------------------------------
# Representation used exactly as supplied
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_scorer_uses_supplied_representation_unchanged(adapter):
    a = _random_adata()
    Z = a.obsm["X_lsi"].copy()
    model = StructuralHeterogeneity(adapter=adapter, verbose=False)
    model.fit(a, groupby="group")
    np.testing.assert_array_equal(a.obsm["X_lsi"], Z)
    assert model.n_dimensions_ == Z.shape[1]
    np.testing.assert_allclose(model.probabilities_, adapter_probability(Z, adapter))


# ---------------------------------------------------------------------------
# Estimand (same engine for both adapters)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_identical_cells_within_group_have_zero_divergence(adapter):
    Z = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 1.0], [3.0, 1.0]])
    model, scores = _fit(Z, ["A", "A", "B", "B"], adapter)
    np.testing.assert_allclose(scores["scchart_structural_d"], 0.0, atol=1e-14)
    np.testing.assert_allclose(model.population_summary_["J_obs"], 0.0, atol=1e-14)


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_single_cell_population_has_zero_divergence(adapter):
    Z = np.array([[2.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.4, 1.0, 2.5]])
    model, scores = _fit(Z, ["A", "A", "solo"], adapter)
    assert scores["scchart_structural_d"].iloc[2] == pytest.approx(0.0, abs=1e-15)
    assert model.population_summary_.loc["solo", "J_obs"] == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_kl_matches_direct_definition(adapter):
    a = _random_adata(seed=3)
    model = StructuralHeterogeneity(adapter=adapter, verbose=False)
    scores = model.fit(a, groupby="group")
    P = adapter_probability(a.obsm["X_lsi"], adapter)
    labels = a.obs["group"].astype(str).to_numpy()
    for g in ("A", "B"):
        Pg = P[labels == g]
        pbar = Pg.mean(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.sum(np.where(Pg > 0, Pg * np.log2(Pg / pbar), 0.0), axis=1)
        np.testing.assert_allclose(scores["scchart_structural_d"].to_numpy()[labels == g], d,
                                   rtol=1e-12)
        assert model.population_summary_.loc[g, "J_obs"] == pytest.approx(d.mean(), rel=1e-12)


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_generalized_jsd_identity_matches_cell_kl_mean(adapter):
    rng = np.random.default_rng(0)
    for _ in range(10):
        n = int(rng.integers(3, 40))
        Z = rng.normal(size=(n, int(rng.integers(2, 8))))
        model, scores = _fit(Z, ["A"] * n, adapter)
        summary = model.population_summary_
        assert summary.loc["A", "identity_error"] < 1e-12
        assert summary.loc["A", "J_obs"] == pytest.approx(
            summary.loc["A", "gjds_identity_bits"], abs=1e-12)
        assert summary.loc["A", "J_obs"] == pytest.approx(
            scores["scchart_structural_d"].mean(), abs=1e-12)


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_cell_permutation_invariance(adapter):
    a = _random_adata(seed=1)
    Z = a.obsm["X_lsi"]
    labels = a.obs["group"].astype(str).to_numpy()
    m1, s1 = _fit(Z, labels, adapter)
    perm = np.random.default_rng(7).permutation(len(labels))
    m2, s2 = _fit(Z[perm], labels[perm], adapter)
    pd.testing.assert_series_equal(
        m1.population_summary_["J_obs"].sort_index(),
        m2.population_summary_["J_obs"].sort_index(),
        rtol=1e-12,
    )
    np.testing.assert_allclose(s2["scchart_structural_d"].to_numpy(),
                               s1["scchart_structural_d"].to_numpy()[perm], rtol=1e-12)


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_component_permutation_invariance(adapter):
    a = _random_adata(seed=2)
    Z = a.obsm["X_lsi"]
    labels = a.obs["group"].astype(str).to_numpy()
    _, s1 = _fit(Z, labels, adapter)
    perm = np.random.default_rng(11).permutation(Z.shape[1])
    _, s2 = _fit(Z[:, perm], labels, adapter)
    np.testing.assert_allclose(s1["scchart_structural_d"], s2["scchart_structural_d"], rtol=1e-12)


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_non_negativity_and_bounds(adapter):
    a = _random_adata(seed=4, n_components=8)
    model = StructuralHeterogeneity(adapter=adapter, verbose=False)
    scores = model.fit(a, groupby="group")
    assert np.all(scores["scchart_structural_d"] >= 0)
    summary = model.population_summary_
    n_channels = model.probabilities_.shape[1]
    assert n_channels == (16 if adapter == "signed_channel" else 8)
    eps = 1e-12
    assert np.all(summary["J_obs"] >= 0)
    assert np.all(summary["J_obs"] <= summary["consensus_entropy_bits"] + eps)
    assert np.all(summary["consensus_entropy_bits"] <= np.log2(n_channels) + eps)
    assert np.all(summary["J_obs"] <= np.log2(summary["n_cells"]) + eps)


# ---------------------------------------------------------------------------
# Chain rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_chain_rule_within_plus_between_equals_total(adapter):
    rng = np.random.default_rng(5)
    Z = rng.normal(size=(90, 5))
    P = adapter_probability(Z, adapter)
    groups = np.repeat(["G1", "G2"], 45)
    subgroups = rng.choice(["c1", "c2", "c3"], size=90)
    table = structural_chain_rule(P, groups, subgroups)

    np.testing.assert_allclose(table["within_bits"] + table["between_bits"],
                               table["J_structural_bits"], rtol=1e-12)
    assert table["decomposition_error"].max() < 1e-12

    model, _ = _fit(Z, groups, adapter)
    np.testing.assert_allclose(table["J_structural_bits"],
                               model.population_summary_.loc[table.index, "J_obs"],
                               rtol=1e-12)


def test_chain_rule_trivial_partitions():
    rng = np.random.default_rng(6)
    P = signed_channel_probability(rng.normal(size=(20, 4)))
    groups = np.repeat(["G"], 20)
    one = structural_chain_rule(P, groups, groups)
    assert one.loc["G", "between_bits"] == pytest.approx(0.0, abs=1e-15)
    singletons = structural_chain_rule(P, groups, np.arange(20))
    assert singletons.loc["G", "within_bits"] == pytest.approx(0.0, abs=1e-15)
    assert singletons.loc["G", "between_bits"] == pytest.approx(
        singletons.loc["G", "J_structural_bits"], rel=1e-12)


def test_chain_rule_rejects_invalid_probabilities():
    with pytest.raises(ValueError, match="sum to 1"):
        structural_chain_rule(np.array([[0.5, 0.2], [0.5, 0.5]]), ["A", "A"], ["x", "y"])


# ---------------------------------------------------------------------------
# Fixed projection
# ---------------------------------------------------------------------------


def test_lsi_projection_is_fixed_and_rejects_empty_cells():
    rng = np.random.default_rng(8)
    B = _binary(rng, 40, 50, 0.2)
    proj = LSIProjection.fit(B, n_components=5, components=[1, 2, 3])
    Z = proj(B)
    assert Z.shape == (40, 3)
    np.testing.assert_allclose(proj(B[:10]), Z[:10])  # row-wise, no refit
    np.testing.assert_allclose(proj.with_components([0, 1])(B), LSIProjection.fit(B, n_components=5)(B)[:, :2])
    B2 = B.copy()
    B2[0] = 0
    with pytest.raises(ValueError, match="no open feature"):
        proj(B2)
    with pytest.raises(ValueError, match="features"):
        proj(B[:, :10])


def test_lsi_projection_binarises_counts_of_any_dtype():
    rng = np.random.default_rng(10)
    B = _binary(rng, 30, 40, 0.3)
    counts = B * rng.integers(1, 5, B.shape)
    proj = LSIProjection.fit(B, n_components=4)
    expected = proj(sparse.csr_matrix(B))
    for X in (counts, sparse.csr_matrix(counts), sparse.csr_matrix(counts.astype(np.int64)),
              sparse.csr_matrix(counts.astype(np.float32))):
        np.testing.assert_allclose(proj(X), expected, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# Technical null, Delta_G and same-representation requirement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_structural_null_calibration_deterministic_and_consistent(adapter):
    rng = np.random.default_rng(7)
    B = _binary(rng, 50, 80, 0.2)
    proj = LSIProjection.fit(B, n_components=6)
    c1 = structural_null_calibration(B, proj, n_replicates=5, seed_key=(1, 2), adapter=adapter)
    c2 = structural_null_calibration(B, proj, n_replicates=5, seed_key=(1, 2), n_jobs=3, adapter=adapter)
    np.testing.assert_array_equal(c1.J_null, c2.J_null)
    assert c1.Delta == pytest.approx(c1.J_obs - c1.J_null.mean())
    assert c1.margins_exact_all and c1.chains_complete_all
    assert c1.J_obs == pytest.approx(structural_J(proj(B), adapter=adapter)["J"])
    assert np.mean(c1.d_obs) == pytest.approx(c1.J_obs)
    assert c1.adapter == adapter


def test_structural_null_rejects_cells_without_open_features():
    rng = np.random.default_rng(9)
    B = _binary(rng, 20, 30, 0.2)
    proj = LSIProjection.fit(B, n_components=4)
    B[0] = 0
    with pytest.raises(ValueError, match="no open feature"):
        structural_null_calibration(B, proj, n_replicates=2)


def test_observed_and_null_cannot_mix_representations():
    """Z != project(X): calibration refuses; J_G and Delta_G always share one representation."""
    rng = np.random.default_rng(21)
    B = _binary(rng, 60, 80, 0.2)
    proj = LSIProjection.fit(B, n_components=6)
    Zp = proj(B)
    Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    Z_other = Zp @ Q                                   # same shape, different representation
    assert structural_J(Z_other)["J"] != pytest.approx(structural_J(Zp)["J"])

    with pytest.raises(ValueError, match="same representation"):
        structural_null_calibration(B, proj, n_replicates=3, Z_obs=Z_other)
    with pytest.raises(ValueError, match="same representation"):
        structural_null_calibration(B, proj, n_replicates=3, Z_obs=Zp[:, :5])
    c = structural_null_calibration(B, proj, n_replicates=3, Z_obs=Zp)
    assert c.J_obs == structural_null_calibration(B, proj, n_replicates=3).J_obs


def test_population_calibration_seeds_each_population_separately():
    rng = np.random.default_rng(22)
    B = _binary(rng, 60, 70, 0.2)
    g = np.repeat(["a", "b"], 30)
    proj = LSIProjection.fit(B, n_components=5)
    tab, objs = structural_population_calibration(B, g, proj, seed=3, n_replicates=3)
    assert list(tab.index) == ["a", "b"]
    solo = structural_null_calibration(B[30:], proj, n_replicates=3, seed_key=(3, 1))
    assert tab.loc["b", "Delta_G"] == pytest.approx(solo.Delta, abs=1e-15)
    np.testing.assert_allclose(tab["Delta_G"], tab["J_obs"] - tab["E0_J"])


# ---------------------------------------------------------------------------
# Matched-depth comparison
# ---------------------------------------------------------------------------


def test_depth_standardized_structural_runs_and_reports_D_star():
    rng = np.random.default_rng(14)
    X = rng.poisson(0.6, (60, 70)).astype(float)
    X[np.arange(60), rng.integers(0, 70, 60)] = 2
    X[:30] *= 2   # population a is deeper
    X = sparse.csr_matrix(X)
    g = np.repeat(["a", "b"], 30)
    proj = LSIProjection.fit(X, n_components=4)
    D_star = float(np.median(np.asarray(X[30:].sum(axis=1)).ravel()))
    res = depth_standardized_structural_comparison(X, g, proj, D_star, n_replicates=2, n_null=3)
    assert res.D_star == D_star and res.axis == "structural"
    assert res.summary.loc["a", "q"] < 1 and res.summary.loc["b", "q"] == 1
    assert res.summary.loc["b", "n_replicates"] == 1   # never upsampled or resampled
    assert res.summary.loc["a", "n_replicates"] == 2
    assert set(res.pairwise.columns) == {"first", "second", "frac_first_greater", "replicates_unanimous"}


# ---------------------------------------------------------------------------
# AnnData outputs and provenance
# ---------------------------------------------------------------------------


def test_fit_writes_canonical_outputs_and_metadata():
    Z = np.array([[1.0, 2.0, 1.0], [2.0, 1.0, 1.0], [3.0, 1.0, 2.0], [1.0, 3.0, 2.0]])
    a = _adata(Z, ["A", "A", "B", "B"])
    model = StructuralHeterogeneity(verbose=False)
    scores = model.fit(a, groupby="group")

    for col in ("scchart_structural_d", "scchart_structural_entropy", "scchart_structural_population_J_obs"):
        assert col in scores
        assert col in a.obs

    assert a.obsm["scchart_structural_probabilities"].shape == (4, 6)
    meta = a.uns["scchart_structural"]
    assert meta["method"] == "scchart_structural_information_divergence"
    assert meta["groupby"] == "group"
    assert meta["representation"] == "X_lsi"
    assert meta["n_dimensions"] == 3
    assert meta["n_channels"] == 6
    assert meta["component_selection"] == "none"
    assert meta["simplex_adapter"] == "signed_channel"
    assert meta["log_base"] == 2
    assert meta["units"] == "bits"
    assert meta["pseudocount"] == 0.0
    assert meta["calibration"] == "none"


def test_missing_group_or_representation_raises():
    a = _adata(np.ones((4, 3)), ["A", "A", "B", "B"])
    model = StructuralHeterogeneity(verbose=False)
    with pytest.raises(KeyError):
        model.fit(a, groupby="missing")
    with pytest.raises(KeyError):
        model.fit(a, groupby="group", use_rep="missing")


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_fit_transform_returns_result_container_when_not_inplace(adapter):
    a = _random_adata()
    model = StructuralHeterogeneity(adapter=adapter, verbose=False)
    result = model.fit_transform(a, groupby="group", inplace=False)
    assert isinstance(result, StructuralHeterogeneityResult)
    pd.testing.assert_index_equal(result.scores.index, a.obs_names)
    np.testing.assert_allclose(result.scores["scchart_structural_d"], a.obs["scchart_structural_d"])
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0)
    D = a.obsm["X_lsi"].shape[1]
    if adapter == "signed_channel":
        assert result.probabilities.shape[1] == 2 * D
        assert result.probabilities.columns[0] == "component_1_pos"
        assert result.probabilities.columns[D] == "component_1_neg"
    else:
        assert list(result.probabilities.columns) == [f"component_{i + 1}" for i in range(D)]
    assert list(result.population_summary.index) == ["A", "B"]
    assert model.fit_transform(a, groupby="group") is a


def test_metadata_survives_h5ad_round_trip(tmp_path):
    import anndata

    a = _random_adata()
    StructuralHeterogeneity(verbose=False).fit(a, groupby="group")
    path = tmp_path / "structural.h5ad"
    a.write_h5ad(path)
    meta = anndata.read_h5ad(path).uns["scchart_structural"]
    assert meta["pseudocount"] == 0.0
    assert meta["probability_key"] == "scchart_structural_probabilities"
    assert meta["calibration"] == "none"
    assert meta["simplex_adapter"] == "signed_channel"
    assert meta["n_dimensions"] == 6


def test_repr_reports_adapter():
    model = StructuralHeterogeneity(verbose=False)
    assert "signed_channel" in repr(model)
    model.fit(_random_adata(), groupby="group")
    assert "Fitted" in repr(model) and "Variant" not in repr(model)
    assert "spectral_energy" in repr(StructuralHeterogeneity(adapter="spectral_energy"))


def test_module_imports_only_shared_scoring_internals():
    import ast

    import scchart.structural_divergence as module

    tree = ast.parse(inspect.getsource(module))
    relative = {("." * node.level) + (node.module or "")
                for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.level}
    assert relative == {"._scoring"}
