"""scchart.Heterogeneity: equivalence with the axis functions, storage and input checks.

The numerical baseline is checked through Heterogeneity in test_baseline.py.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import scchart
from scchart.programme_entropy import (crossfit_population_calibration, crossfit_programme_references,
                                       programme_population_table, score_crossfit, score_programmes)
from scchart.structural_divergence import structural_J, structural_population_calibration
from tests.baseline_workload import synthetic_data

N_FOLDS = 3

def make_adata(X, g, features, external, proj):
    adata = ad.AnnData(X=sparse.csr_matrix((X > 0).astype(np.float32)),
                       obs=pd.DataFrame({"cell_type": pd.Categorical(g)}, index=[f"c{i}" for i in range(X.shape[0])]),
                       var=pd.DataFrame({"programme": external}, index=features))
    adata.layers["counts"] = X.copy()   # counts in a layer, as in Scanpy workflows
    adata.obsm["X_lsi"] = proj(X)
    return adata


@pytest.fixture(scope="module")
def data():
    """Reference-workload populations with cells shuffled, so labels are unsorted."""
    X_ref, X, g, features, external = synthetic_data()
    perm = np.random.default_rng(5).permutation(X.shape[0])
    X, g = X[perm].tocsr(), g[perm]
    assert list(pd.unique(g)) != sorted(set(g)), "first-appearance order must differ from sorted order"
    proj = scchart.LSIProjection.fit(X_ref, n_components=8, random_state=0)
    return X, g, features, external, proj


def fit(data, populations=None, **kw):
    adata = make_adata(*data)
    kw = {"layer": "counts", "projection": data[4], "n_folds": N_FOLDS, "verbose": False, **kw}
    model = scchart.Heterogeneity(**kw).fit(adata, groupby="cell_type", populations=populations)
    return model, adata


def assert_tables_equal(got, want):
    pd.testing.assert_frame_equal(got, want, check_exact=True, check_names=False)


# ---------------------------------------------------------------------------
# Default order: sorted labels, the calibration functions' default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sampler", ["proposal_clock", "accepted_swap"])
def test_both_axes_match_calibration_functions(data, sampler):
    X, g, features, external, proj = data
    model, adata = fit(data, null_sampler=sampler)
    order = sorted(set(g))
    assert model.population_order_ == order
    s_tab, _ = structural_population_calibration(X, g, proj, sampler=sampler)
    p_tab, _ = crossfit_population_calibration(X, g, model.references_, sampler=sampler)
    assert list(s_tab.index) == list(p_tab.index) == order
    assert list(model.structural_.index) == list(model.programme_.index) == order
    for col in ["E0_J", "Delta_G", "sd0_diagnostic", "R_G_diagnostic", "median_n_open"]:
        np.testing.assert_array_equal(model.structural_[col], s_tab[col], err_msg=col)
    np.testing.assert_allclose(model.structural_["J_obs"], s_tab["J_obs"], rtol=1e-12)   # Z vs projection(B)
    for col in ["E0_J_P", "Delta_P", "sd0_diagnostic", "null_scored_fraction"]:
        np.testing.assert_array_equal(model.programme_[col], p_tab[col], err_msg=col)
    # The scorer table and the calibration compute J_G^(P) by different summation paths (~1e-16).
    np.testing.assert_allclose(model.programme_["J_P_obs"], p_tab["J_P_obs"], rtol=1e-12)


def test_structural_axis_alone(data):
    model, adata = fit(data, programme=False)
    assert model.programme_ is None and model.references_ is None
    assert not any(c.startswith("scchart_programme") for c in adata.obs)
    assert "structural" in adata.uns["scchart"] and "programme" not in adata.uns["scchart"]
    assert set(model.seeds_) == {"structural"}


def test_programme_axis_alone(data):
    adata = make_adata(*data)
    del adata.obsm["X_lsi"]  # the programme axis needs no representation
    model = scchart.Heterogeneity(structural=False, layer="counts", n_folds=N_FOLDS, verbose=False)
    model.fit(adata, groupby="cell_type")
    assert model.structural_ is None and not any(c.startswith("scchart_structural") for c in adata.obs)
    assert np.isfinite(model.programme_["Delta_P"]).all()
    assert set(model.seeds_) == {"programme"}


def test_programme_axis_with_a_single_fixed_reference(data):
    X, g, features, external, proj = data
    ref = scchart.external_programme_reference(external, X, g)
    adata = make_adata(*data)
    model = scchart.Heterogeneity(structural=False, layer="counts", verbose=False)
    model.fit(adata, groupby="cell_type", reference=ref)
    s = score_programmes(X, ref)
    np.testing.assert_array_equal(adata.obs["scchart_programme_P"].to_numpy(float), s.P)
    tab = programme_population_table(s.r, g)
    np.testing.assert_allclose(model.programme_["J_P_obs"], tab["J_P_bits"])
    p_tab, _ = crossfit_population_calibration(X, g, [ref])
    np.testing.assert_array_equal(model.programme_["Delta_P"], p_tab["Delta_P"])
    assert model.programme_model_.reference_mode_ == "fixed" and model.references_ == [ref]


def test_observed_and_null_cannot_mix_representations(data):
    X, g, features, external, proj = data
    adata = make_adata(*data)
    Q, _ = np.linalg.qr(np.random.default_rng(21).normal(size=(8, 8)))
    adata.obsm["X_rotated"] = adata.obsm["X_lsi"] @ Q          # a different Z than projection(B)
    model = scchart.Heterogeneity(programme=False, use_rep="X_rotated", layer="counts", projection=proj,
                                  verbose=False)
    with pytest.raises(ValueError, match="same representation"):
        model.fit(adata, groupby="cell_type")
    raw = scchart.Heterogeneity(programme=False, use_rep="X_rotated", calibrate=False, verbose=False)
    assert "Delta_G" not in raw.fit(adata, groupby="cell_type").structural_


def test_spectral_energy_adapter(data):
    X, g, features, external, proj = data
    model, adata = fit(data, programme=False, adapter="spectral_energy")
    assert model.declarations_["structural_adapter"] == "spectral_energy"
    for p in model.population_order_:
        want = structural_J(adata.obsm["X_lsi"][g == p], adapter="spectral_energy")["J"]
        assert model.structural_.loc[p, "J_obs"] == pytest.approx(want, rel=1e-12)
    s_tab, _ = structural_population_calibration(X, g, proj, adapter="spectral_energy")
    np.testing.assert_array_equal(model.structural_["Delta_G"], s_tab["Delta_G"])


def test_external_programmes_match_axis_functions(data):
    X, g, features, external, proj = data
    model, adata = fit(data, structural=False, programme_key="programme")
    refs = crossfit_programme_references(X, g, n_folds=N_FOLDS, external=external)
    np.testing.assert_array_equal(adata.obsm["scchart_programme_composition"], score_crossfit(X, refs).r)
    p_tab, _ = crossfit_population_calibration(X, g, refs)
    np.testing.assert_array_equal(model.programme_["Delta_P"], p_tab["Delta_P"])


def test_without_projection_reports_raw_structural_only(data):
    model, adata = fit(data, programme=False, projection=None)
    assert model.structural_calibrations_ is None
    assert (model.structural_["calibration"] == "not run: needs X and a fixed projection of B into Z").all()
    assert "E0_J" not in model.structural_ and model.seeds_ == {}
    assert adata.uns["scchart"]["structural"]["calibrated"] is False


def test_uncalibrated_tables_have_raw_values_only(data):
    model, adata = fit(data, calibrate=False)
    assert "Delta_G" not in model.structural_ and "Delta_P" not in model.programme_
    assert model.seeds_ == {}


# ---------------------------------------------------------------------------
# Declared order and seed records
# ---------------------------------------------------------------------------


def test_declared_order_sets_seeds_for_both_axes(data):
    X, g, features, external, proj = data
    order = ["c_mix", "a_homog", "b_two"]
    model, adata = fit(data, populations=order)
    s_tab, _ = structural_population_calibration(X, g, proj, populations=order)
    p_tab, _ = crossfit_population_calibration(X, g, model.references_, populations=order)
    assert model.population_order_ == order == list(model.structural_.index) == list(model.programme_.index)
    for col in ["E0_J", "Delta_G"]:
        np.testing.assert_array_equal(model.structural_[col], s_tab[col])
    for col in ["E0_J_P", "Delta_P"]:
        np.testing.assert_array_equal(model.programme_[col], p_tab[col])
    assert adata.uns["scchart"]["population_order_rule"] == "declared with populations="


def test_seed_records_match_the_seeds_used(data):
    model, adata = fit(data, structural_seed=(7, 3), programme_seed=500)
    order = model.population_order_
    for k, p in enumerate(order):
        cal = model.structural_calibrations_[p]
        assert model.seeds_["structural"][p] == [7, 3, k]
        np.testing.assert_array_equal(cal.seeds, scchart._scoring.seeds_from_key((7, 3, k), len(cal.seeds)))
        assert model.seeds_["programme"][p] == {f: 500 + 10_000 * f + 100 * k for f in range(N_FOLDS)}
    uns = adata.uns["scchart"]
    np.testing.assert_array_equal(uns["structural"]["seed_keys"], [[7, 3, k] for k in range(len(order))])
    np.testing.assert_array_equal(uns["programme"]["base_seeds"],
                                  [[500 + 10_000 * f + 100 * k for f in range(N_FOLDS)] for k in range(len(order))])


# ---------------------------------------------------------------------------
# AnnData storage, reporting and input checks
# ---------------------------------------------------------------------------


def test_outputs_use_scchart_prefix(data):
    model, adata = fit(data)
    new = [c for c in adata.obs if c != "cell_type"]
    assert new and all(c.startswith("scchart_") for c in new)
    assert {"scchart_structural_d", "scchart_programme_P", "scchart_programme_d"} <= set(new)
    assert all(k.startswith("scchart") for k in adata.uns)
    assert all(k.startswith("scchart_") for k in adata.obsm if k != "X_lsi")


def test_results_survive_h5ad(data, tmp_path):
    model, adata = fit(data)
    path = tmp_path / "fit.h5ad"
    adata.write_h5ad(path)
    back = ad.read_h5ad(path)
    for col in adata.obs:
        if pd.api.types.is_numeric_dtype(adata.obs[col]):   # includes bool; NaN compares equal
            np.testing.assert_array_equal(back.obs[col].to_numpy(float), adata.obs[col].to_numpy(float), err_msg=col)
        else:
            want = adata.obs[col].astype(object).where(adata.obs[col].notna(), None)
            got = back.obs[col].astype(object).where(back.obs[col].notna(), None)
            assert got.tolist() == want.tolist(), col
    uns = back.uns["scchart"]
    assert uns["groupby"] == "cell_type"
    assert list(uns["population_order"]) == model.population_order_
    np.testing.assert_array_equal(uns["structural"]["table"]["Delta_G"], model.structural_["Delta_G"])
    np.testing.assert_array_equal(uns["structural"]["table"]["J_obs"], model.structural_["J_obs"])
    np.testing.assert_array_equal(uns["programme"]["table"]["Delta_P"], model.programme_["Delta_P"])
    np.testing.assert_array_equal(uns["programme"]["table"]["J_P_obs"], model.programme_["J_P_obs"])
    assert uns["structural"]["null_J"].shape == (3, 20)
    assert back.uns["scchart_structural"]["calibration"] == "fixed-margin null"
    np.testing.assert_array_equal(back.obsm["scchart_programme_composition"],
                                  adata.obsm["scchart_programme_composition"])


def test_verbose_reports_configuration_stages_and_outputs(data, capsys):
    adata = make_adata(*data)
    model = scchart.Heterogeneity(layer="counts", projection=data[4], n_folds=N_FOLDS)
    text = repr(model)
    assert "Planned analyses" in text and "Not yet" in text
    model.fit(adata, groupby="cell_type")
    out = capsys.readouterr().out
    assert "[scCHART 01/06] Validate inputs" in out and "[scCHART 06/06] Store results" in out
    assert "sorted population labels: a_homog, b_two, c_mix" in out
    assert "adata.uns['scchart']" in out and "Fit complete" in out
    assert "Fitted                            Yes" in repr(model)


def test_scorer_classes_take_groupby(data):
    adata = make_adata(*data)
    s = scchart.StructuralHeterogeneity(verbose=False)
    s.fit(adata, groupby="cell_type")
    assert s.groupby_ == "cell_type" and list(s.population_summary_.index) == sorted(set(data[1]))
    p = scchart.ProgrammeHeterogeneity(n_folds=N_FOLDS, layer="counts", verbose=False)
    p.fit(adata, groupby="cell_type")
    assert p.groupby_ == "cell_type" and list(p.population_summary_.index) == sorted(set(data[1]))
    assert "J_obs" in s.population_summary_ and "J_P_obs" in p.population_summary_


def test_input_errors(data):
    adata = make_adata(*data)
    with pytest.raises(ValueError, match="at least one"):
        scchart.Heterogeneity(structural=False, programme=False)
    with pytest.raises(ValueError, match="sampler"):
        scchart.Heterogeneity(null_sampler="bogus")
    model = scchart.Heterogeneity(layer="counts", verbose=False)
    with pytest.raises(ValueError, match="every label once"):
        model.fit(adata, groupby="cell_type", populations=["a_homog", "b_two"])
    with pytest.raises(ValueError, match="duplicates"):
        model.fit(adata, groupby="cell_type", populations=["a_homog", "b_two", "c_mix", "c_mix"])
    with pytest.raises(KeyError, match="groupby"):
        model.fit(adata, groupby="missing")
    with pytest.raises(TypeError, match="AnnData"):
        model.fit(adata.X, groupby="cell_type")
    with pytest.raises(KeyError, match="layer"):
        scchart.Heterogeneity(layer="nope", verbose=False).fit(adata, groupby="cell_type")
