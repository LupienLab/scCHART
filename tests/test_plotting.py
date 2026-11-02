"""Tests of scchart.pl on the outputs stored by scchart.Heterogeneity and the axis scorers.

Figures are not compared pixel-wise: the tests check that each function reads
the right stored quantities, returns an Axes, writes no files and fails with an
informative error when its input is missing.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import scchart
from tests.baseline_workload import synthetic_data

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

pl = scchart.pl


def _adata():
    X_ref, X, g, features, external = synthetic_data()
    perm = np.random.default_rng(3).permutation(X.shape[0])   # unsorted labels
    X, g = X[perm].tocsr(), g[perm]
    proj = scchart.LSIProjection.fit(X_ref, n_components=8, random_state=0)
    a = ad.AnnData(X=sparse.csr_matrix(X), obs=pd.DataFrame({"cell_type": pd.Categorical(g)},
                                                             index=[f"c{i}" for i in range(X.shape[0])]),
                   var=pd.DataFrame(index=features))
    a.obsm["X_lsi"] = proj(X)
    a.obsm["X_umap"] = np.random.default_rng(1).normal(size=(a.n_obs, 2))
    return a, proj


@pytest.fixture(scope="module")
def fitted():
    a, proj = _adata()
    model = scchart.Heterogeneity(projection=proj, n_folds=3, verbose=False).fit(a, groupby="cell_type")
    return a, model


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Per-cell plots
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("color", ["scchart_structural_d", "scchart_programme_d", "scchart_programme_P"])
def test_embedding_plots_per_cell_outputs(fitted, color):
    a, _ = fitted
    ax = pl.embedding(a, color)
    assert isinstance(ax, Axes)
    values = a.obs[color].to_numpy(float)
    missing, points = ax.collections[0], ax.collections[1]
    assert len(missing.get_offsets()) == int(np.isnan(values).sum())
    np.testing.assert_array_equal(np.sort(points.get_array()), np.sort(values[np.isfinite(values)]))
    assert np.all(np.diff(points.get_array()) >= 0)   # largest values drawn last
    colorbar_label = ax.figure.axes[-1].get_ylabel()
    assert colorbar_label == pl.CELL_LABELS[color]
    stem = pl.CELL_LABELS[color].split(" (")[0]
    assert ax.get_title(loc="left") == stem[0].upper() + stem[1:]


def test_embedding_marks_unscorable_cells_and_options(fitted):
    a, _ = fitted
    n_unscored = int((~a.obs["scchart_programme_scored"]).sum())
    assert n_unscored > 0
    ax = pl.embedding(a, "scchart_programme_d", basis="umap", groupby="cell_type", log=True, quantiles=(0, 1))
    assert any(f"{n_unscored:,} cells without a value" in t.get_text() for t in ax.texts)
    assert {t.get_text() for t in ax.texts} >= {"a_homog", "b_two", "c_mix"}
    assert ax.figure.axes[-1].get_ylabel().endswith("log scale")
    ax = pl.embedding(a, "scchart_structural_d", title="custom")
    assert ax.get_title(loc="left") == "custom" and not ax.texts


def test_embedding_errors(fitted):
    a, _ = fitted
    with pytest.raises(KeyError, match="scchart_structural_d"):
        pl.embedding(a, "structural_heterogeneity")   # old name: no alias
    with pytest.raises(KeyError, match="X_tsne"):
        pl.embedding(a, "scchart_structural_d", basis="X_tsne")
    with pytest.raises(ValueError, match="not numeric"):
        pl.embedding(a, "cell_type")
    with pytest.raises(ValueError, match="not numeric"):
        pl.embedding(a, "scchart_programme_scored")
    with pytest.raises(KeyError, match="groupby"):
        pl.embedding(a, "scchart_structural_d", groupby="missing")
    with pytest.raises(TypeError, match="AnnData"):
        pl.embedding(a.obs, "scchart_structural_d")


def test_embedding_without_colour_shows_populations(fitted):
    a, _ = fitted
    ax = pl.embedding(a, groupby="cell_type")
    assert len(ax.collections) == 1 and len(ax.collections[0].get_offsets()) == a.n_obs
    assert len(ax.figure.axes) == 1                       # no colour bar
    assert {t.get_text() for t in ax.texts} == {"a_homog", "b_two", "c_mix"}
    assert ax.get_title(loc="left") == "Populations"


def _boxes(ax, texts):
    renderer = ax.figure.canvas.get_renderer()
    return [t.get_window_extent(renderer) for t in texts]


def _overlapping(boxes):
    return [(i, j) for i in range(len(boxes)) for j in range(i + 1, len(boxes)) if boxes[i].overlaps(boxes[j])]


def test_embedding_population_labels_do_not_overlap(fitted):
    a, _ = fitted
    ax = pl.embedding(a, "scchart_structural_d", groupby="cell_type")
    ax.figure.canvas.draw()
    labels = [t for t in ax.texts if t.get_text() in {"a_homog", "b_two", "c_mix"}]
    assert len(labels) == 3 and not _overlapping(_boxes(ax, labels))


def test_programme_composition_shows_population_means_of_scored_cells(fitted):
    a, model = fitted
    ax = pl.programme_composition(a)
    assert isinstance(ax, Axes)
    programmes = list(a.uns["scchart_programme"]["programmes"])
    order = model.population_order_
    # endogenous programmes are named after the populations: columns follow the rows
    assert sorted(programmes) == sorted(order)
    assert [t.get_text() for t in ax.get_xticklabels()] == order
    assert [t.get_text() for t in ax.get_yticklabels()] == order
    r = pd.DataFrame(a.obsm["scchart_programme_composition"], columns=programmes)
    scored = a.obs["scchart_programme_scored"].to_numpy()
    want = r[scored].groupby(a.obs["cell_type"].astype(str).to_numpy()[scored]).mean().loc[order, order]
    np.testing.assert_allclose(np.asarray(ax.collections[0].get_array()).ravel(), want.to_numpy().ravel())
    assert len(ax.texts) == int((want.to_numpy() >= 0.05).sum())
    assert not pl.programme_composition(a, annotate=False).texts


def test_programme_composition_orders_external_programmes_by_main_user():
    a, _ = _adata()
    a.var["programme"] = np.repeat(["P2", "P0", "P1", "Shared"], a.n_vars // 4)
    scchart.ProgrammeHeterogeneity(programme_key="programme", n_folds=3, verbose=False).fit(a, groupby="cell_type")
    ax = pl.programme_composition(a)
    rows = [t.get_text() for t in ax.get_yticklabels()]
    cols = [t.get_text() for t in ax.get_xticklabels()]
    assert sorted(cols) == ["P0", "P1", "P2"]
    values = np.asarray(ax.collections[0].get_array()).reshape(len(rows), len(cols))
    top_row = values.argmax(axis=0)
    assert list(top_row) == sorted(top_row)              # programmes ordered by the row that uses them most


def test_programme_composition_works_with_the_axis_scorer_alone():
    a, _ = _adata()
    scchart.ProgrammeHeterogeneity(n_folds=3, verbose=False).fit(a, groupby="cell_type")
    ax = pl.programme_composition(a)
    assert [t.get_text() for t in ax.get_yticklabels()] == ["a_homog", "b_two", "c_mix"]


def test_programme_composition_errors():
    a, _ = _adata()
    with pytest.raises(KeyError, match="programme outputs not found"):
        pl.programme_composition(a)
    scchart.ProgrammeHeterogeneity(n_folds=3, verbose=False).fit(a, groupby="cell_type")
    with pytest.raises(KeyError, match="groupby"):
        pl.programme_composition(a, groupby="missing")


# ---------------------------------------------------------------------------
# Population-level plots
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", ["structural", "programme"])
def test_calibration_plots_observed_null_and_excess(fitted, axis):
    a, model = fitted
    table = model.structural_ if axis == "structural" else model.programme_
    cols = pl.COLUMNS[axis]
    ax = pl.calibration(a, axis)
    assert isinstance(ax, Axes)
    assert [t.get_text() for t in ax.get_yticklabels()] == model.population_order_
    null_points, observed = ax.collections[-2], ax.collections[-1]
    np.testing.assert_allclose(null_points.get_offsets()[:, 0], table[cols["E0"]])
    np.testing.assert_allclose(observed.get_offsets()[:, 0], table[cols["J"]])
    header, *values = [t.get_text() for t in ax.texts]
    assert header == pl.SYMBOLS[axis]["Delta"]
    assert values == [f"{d:.3f}" for d in table[cols["Delta"]]]
    assert ax.get_legend() is not None


def test_calibration_population_subset(fitted):
    a, _ = fitted
    ax = pl.calibration(a, "programme", populations=["c_mix", "a_homog"])
    assert [t.get_text() for t in ax.get_yticklabels()] == ["c_mix", "a_homog"]
    with pytest.raises(KeyError, match="unknown populations"):
        pl.calibration(a, populations=["nope"])


@pytest.mark.parametrize("axis", ["structural", "programme"])
def test_null_distribution_shows_every_replicate(fitted, axis):
    a, model = fitted
    cals = model.structural_calibrations_ if axis == "structural" else model.programme_calibrations_
    table = model.structural_ if axis == "structural" else model.programme_
    ax = pl.null_distribution(a, axis)
    assert isinstance(ax, Axes)
    order = model.population_order_
    for i, p in enumerate(order):
        reps = np.asarray(cals[p].J_null)
        np.testing.assert_allclose(np.sort(ax.collections[i].get_offsets()[:, 0]), np.sort(reps[np.isfinite(reps)]))
    e0 = [seg[0][0] for seg in ax.collections[len(order)].get_segments()]   # vertical marks at E0
    np.testing.assert_allclose(e0, table[pl.COLUMNS[axis]["E0"]])
    np.testing.assert_allclose(ax.collections[len(order) + 1].get_offsets()[:, 0], table[pl.COLUMNS[axis]["J"]])
    sub = pl.null_distribution(a, axis, populations=[order[-1]])
    np.testing.assert_allclose(np.sort(sub.collections[0].get_offsets()[:, 0]), np.sort(cals[order[-1]].J_null))


def test_structural_vs_programme_plots_calibrated_excess(fitted):
    a, model = fitted
    ax = pl.structural_vs_programme(a)
    assert isinstance(ax, Axes)
    xy = ax.collections[0].get_offsets()
    np.testing.assert_allclose(xy[:, 0], model.structural_["Delta_G"])
    np.testing.assert_allclose(xy[:, 1], model.programme_["Delta_P"].reindex(model.structural_.index))
    labels = [t for t in ax.texts if t.get_text()]          # leader lines are empty annotations
    assert [t.get_text() for t in labels] == model.population_order_
    ax.figure.canvas.draw()
    assert not _overlapping(_boxes(ax, labels))
    assert "excess" in ax.get_xlabel() and "excess" in ax.get_ylabel()
    assert not pl.structural_vs_programme(a, label=False).texts


def test_structural_vs_programme_colours_groups(fitted):
    a, model = fitted
    groups = {"a_homog": "one", "b_two": "two", "c_mix": "two"}
    ax = pl.structural_vs_programme(a, groups=groups)
    colours = ax.collections[0].get_facecolors()
    order = model.population_order_
    assert np.allclose(colours[order.index("b_two")], colours[order.index("c_mix")])
    assert not np.allclose(colours[order.index("a_homog")], colours[order.index("b_two")])
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["one", "two"]
    with pytest.raises(KeyError, match="missing from groups"):
        pl.structural_vs_programme(a, groups={"a_homog": "one"})


def test_set_style_sets_publication_defaults():
    with matplotlib.rc_context():
        pl.set_style()
        rc = matplotlib.rcParams
        assert rc["font.sans-serif"][0] == "Arial"
        assert rc["pdf.fonttype"] == 42 and rc["svg.fonttype"] == "none"
        assert rc["savefig.dpi"] == 600 and rc["axes.linewidth"] == 0.5


def test_population_plots_need_a_calibrated_fit():
    a, proj = _adata()
    with pytest.raises(KeyError, match="fit scchart.Heterogeneity"):
        pl.calibration(a)
    scchart.Heterogeneity(programme=False, verbose=False).fit(a, groupby="cell_type")   # no projection
    with pytest.raises(ValueError, match="no fixed projection"):
        pl.calibration(a, "structural")
    with pytest.raises(ValueError, match="no fixed projection"):
        pl.null_distribution(a, "structural")
    with pytest.raises(ValueError, match="no fixed projection"):
        pl.structural_vs_programme(a)
    with pytest.raises(ValueError, match="axis must be"):
        pl.calibration(a, "both")
    b, proj = _adata()
    scchart.Heterogeneity(projection=proj, n_folds=3, calibrate=False, verbose=False).fit(b, groupby="cell_type")
    with pytest.raises(ValueError, match="calibrate=False"):
        pl.calibration(b, "programme")
    c, proj = _adata()
    scchart.Heterogeneity(projection=proj, programme=False, verbose=False).fit(c, groupby="cell_type")
    with pytest.raises(KeyError, match="programme axis was not fitted"):
        pl.structural_vs_programme(c)


# ---------------------------------------------------------------------------
# Storage and side effects
# ---------------------------------------------------------------------------


def test_plots_work_after_h5ad_round_trip(fitted, tmp_path):
    a, model = fitted
    path = tmp_path / "fit.h5ad"
    a.write_h5ad(path)
    b = ad.read_h5ad(path)
    for ax in (pl.calibration(b, "structural"), pl.null_distribution(b, "programme"),
               pl.structural_vs_programme(b), pl.programme_composition(b),
               pl.embedding(b, "scchart_structural_d")):
        assert isinstance(ax, Axes)
    np.testing.assert_allclose(pl.structural_vs_programme(b).collections[0].get_offsets()[:, 0],
                               model.structural_["Delta_G"])


def test_plots_write_no_files_and_draw_on_given_axes(fitted, tmp_path, monkeypatch):
    a, _ = fitted
    monkeypatch.chdir(tmp_path)
    fig, axes = plt.subplots(2, 3)
    calls = [lambda ax: pl.embedding(a, "scchart_programme_P", ax=ax),
             lambda ax: pl.programme_composition(a, ax=ax),
             lambda ax: pl.calibration(a, "structural", ax=ax),
             lambda ax: pl.null_distribution(a, "structural", ax=ax),
             lambda ax: pl.structural_vs_programme(a, ax=ax)]
    for call, ax in zip(calls, axes.flat):
        assert call(ax) is ax
    assert list(tmp_path.iterdir()) == []
