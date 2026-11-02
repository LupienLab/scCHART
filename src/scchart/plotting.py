"""Plotting for scCHART outputs, available as ``scchart.pl``.

Every function reads the outputs that scCHART stores on an ``AnnData``
object, draws with Matplotlib and returns the ``Axes``. Nothing is written
to disk; save with ``ax.figure.savefig(...)``.

Per-cell (any fit of the axis scorers or of :class:`scchart.Heterogeneity`)
    :func:`embedding`
        A per-cell output (``scchart_structural_d``, ``scchart_programme_d``,
        ``scchart_programme_P``, ...) on an existing embedding.
    :func:`programme_composition`
        Mean programme composition ``r`` of each population.

Population level (needs ``scchart.Heterogeneity(calibrate=True)``)
    :func:`calibration`
        Observed ``J``, null expectation ``E0[J]`` and calibrated excess
        ``Delta`` of each population, for one axis.
    :func:`null_distribution`
        Observed ``J`` against the null replicates of each population.
    :func:`structural_vs_programme`
        Calibrated excess on the structural axis against the programme axis.

Style
    :func:`set_style` sets publication defaults (Arial, 7 pt text, thin lines,
    editable text in PDF/SVG, 600 dpi raster export). The plotting functions
    set their own sizes, colours and line widths either way.

Per-cell divergences ``d_i`` are uncalibrated decomposition terms of their
own population's ``J``; they show where heterogeneity comes from within a
population and are not compared between populations. Only the population
quantities are null-calibrated.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

AXES = ("structural", "programme")
# Population-table columns of each axis (see scchart.Heterogeneity).
COLUMNS = {
    "structural": {"J": "J_obs", "E0": "E0_J", "Delta": "Delta_G"},
    "programme": {"J": "J_P_obs", "E0": "E0_J_P", "Delta": "Delta_P"},
}
SYMBOLS = {
    "structural": {"J": r"$J_G$", "E0": r"$E_0[J_G]$", "Delta": r"$\Delta_G$"},
    "programme": {"J": r"$J_G^{(P)}$", "E0": r"$E_0[J_G^{(P)}]$", "Delta": r"$\Delta_G^{(P)}$"},
}
# Colour-bar labels of the per-cell outputs.
CELL_LABELS = {
    "scchart_structural_d": r"structural divergence $d_i(G)$ (bits)",
    "scchart_structural_entropy": r"structural entropy $H(p_i)$ (bits)",
    "scchart_programme_d": r"programme divergence $d_i^{(P)}(G)$ (bits)",
    "scchart_programme_P": r"programme mixing $P_i$ (0–1)",
    "scchart_programme_dominant_weight": r"dominant programme weight $\max_k r_{ik}$",
    "scchart_programme_total_evidence": r"total programme evidence $\sum_k a_{ik}$",
    "scchart_programme_background_depth": r"open background features $D_i$",
}

# Ink and chrome (recessive axes and grid; data carries the colour).
INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, MISSING_COLOUR, SURFACE = "#e1e0d9", "#d9d8d4", "#ffffff"
# Categorical slots in fixed order (the first three pass all-pairs colour-vision checks).
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
AXIS_COLOUR = {"structural": CATEGORICAL[0], "programme": CATEGORICAL[1]}
# Single-hue sequential ramps, light -> dark: blue for structural, orange for programme quantities.
RAMPS = {
    "structural": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    "programme": ["#fde3d6", "#f8c0a4", "#f19a6e", "#eb6834", "#c4501f", "#9a3d16", "#7a2c0c"],
}
# Text sizes (pt) and line widths (pt).
TITLE, LABEL, TICK, NOTE = 8.0, 7.0, 6.5, 6.0
HAIRLINE, THIN = 0.4, 0.5
MARKER = 30.0   # scatter area (pt^2) of the population markers


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


def set_style() -> None:
    """Set Matplotlib defaults for publication figures.

    Arial (Helvetica, Liberation Sans or DejaVu Sans when Arial is not
    installed) at 7 pt, 0.5 pt axes and ticks, no top or right spines,
    editable text in PDF, PS and SVG output, and 600 dpi raster export with
    a tight bounding box. Changes ``matplotlib.rcParams`` for the session.

    Examples
    --------
    >>> scchart.pl.set_style()
    >>> ax = scchart.pl.calibration(adata, "structural")
    >>> ax.figure.savefig("calibration.pdf")      # vector, editable text
    >>> ax.figure.savefig("calibration.png")      # 600 dpi
    """
    import matplotlib as mpl
    from matplotlib import font_manager

    params = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": LABEL, "axes.titlesize": TITLE, "axes.labelsize": LABEL,
        "xtick.labelsize": TICK, "ytick.labelsize": TICK, "legend.fontsize": TICK,
        "axes.linewidth": THIN, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
        "xtick.major.width": THIN, "ytick.major.width": THIN, "xtick.minor.width": HAIRLINE,
        "ytick.minor.width": HAIRLINE, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": SECONDARY, "ytick.labelcolor": SECONDARY,
        "grid.color": GRID, "grid.linewidth": HAIRLINE, "legend.frameon": False,
        "lines.linewidth": 1.0, "patch.linewidth": THIN,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "figure.dpi": 150,
        "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    }
    try:
        font_manager.findfont(font_manager.FontProperties(family="Arial"), fallback_to_default=False)
        params.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                       "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial", "mathtext.cal": "Arial:italic"})
    except ValueError:   # Arial not installed: keep Matplotlib's mathtext fonts
        pass
    mpl.rcParams.update(params)


# ---------------------------------------------------------------------------
# Shared input handling
# ---------------------------------------------------------------------------


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError("scchart.pl needs matplotlib: pip install matplotlib") from err
    return plt


def _new_axes(ax, figsize):
    if ax is None:
        _, ax = _pyplot().subplots(figsize=figsize)
    return ax


def _check_adata(adata) -> None:
    if not all(hasattr(adata, name) for name in ("obs", "obsm", "uns")):
        raise TypeError(f"expected an AnnData object, got {type(adata).__name__}")


def _check_axis(axis: str) -> None:
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES}, got {axis!r}")


def _population_table(adata, axis: str, *, calibrated: bool = True) -> pd.DataFrame:
    """Population table of one axis from ``adata.uns['scchart']``."""
    _check_adata(adata)
    _check_axis(axis)
    fit = adata.uns.get("scchart")
    if fit is None:
        raise KeyError("adata.uns['scchart'] not found; fit scchart.Heterogeneity on adata first")
    entry = fit.get(axis)
    if entry is None:
        raise KeyError(f"the {axis} axis was not fitted (scchart.Heterogeneity({axis}=True))")
    if calibrated and not bool(entry.get("calibrated", False)):
        reason = ("no fixed projection was given" if axis == "structural" and fit.get("config", {}).get("calibrate")
                  else "calibrate=False")
        raise ValueError(f"the {axis} axis was not null-calibrated ({reason}); this plot needs E0 and Delta")
    table = pd.DataFrame({k: np.asarray(v) for k, v in entry["table"].items()})
    return table.set_index("population")


def _null_replicates(adata, axis: str) -> np.ndarray:
    return np.asarray(adata.uns["scchart"][axis]["null_J"], dtype=float)


def _select(table: pd.DataFrame, populations: Optional[Sequence[str]]) -> pd.DataFrame:
    if populations is None:
        return table
    populations = [str(p) for p in populations]
    unknown = [p for p in populations if p not in table.index]
    if unknown:
        raise KeyError(f"unknown populations {unknown}; available: {list(table.index)}")
    return table.loc[populations]


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------


def _chrome(ax, *, grid: Optional[str] = None) -> None:
    """Thin recessive axes: no top/right spines, 0.5 pt ticks, optional hairline grid."""
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(THIN)
        ax.spines[side].set_color(MUTED)
    ax.tick_params(width=THIN, length=2.5, color=MUTED, labelsize=TICK, labelcolor=SECONDARY)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=HAIRLINE)
        ax.set_axisbelow(True)


def _title(ax, text: str, pad: float = 4) -> None:
    ax.set_title(text, loc="left", fontsize=TITLE, color=INK, pad=pad)


def _colorbar(ax, mappable, label: str):
    cbar = ax.figure.colorbar(mappable, ax=ax, fraction=0.04, pad=0.03, aspect=26)
    cbar.outline.set_linewidth(HAIRLINE)
    cbar.outline.set_edgecolor(MUTED)
    cbar.ax.tick_params(which="both", labelsize=NOTE, width=HAIRLINE, color=MUTED, labelcolor=SECONDARY)
    cbar.ax.tick_params(which="major", length=2)
    cbar.ax.tick_params(which="minor", length=1.2)
    cbar.set_label(label, fontsize=TICK, color=SECONDARY)
    return cbar


def _legend_above(ax, handles):
    """Legend between the title and the plot, so it never covers data.

    One row when it fits within the axes width, otherwise fewer columns.
    Returns the legend height in points, for the title padding.
    """
    renderer = _renderer(ax.figure)
    frame = ax.get_window_extent(renderer)
    for ncol in range(len(handles), 0, -1):
        legend = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=ncol,
                           frameon=False, fontsize=TICK, handletextpad=0.4, columnspacing=1.4,
                           borderaxespad=0.3, handlelength=1.4, borderpad=0.2, labelcolor=SECONDARY)
        box = legend.get_window_extent(renderer)
        if box.x1 <= frame.x1 + 1 or ncol == 1:
            break
    return box.height * 72.0 / ax.figure.dpi


def _population_rows(ax, labels) -> np.ndarray:
    y = np.arange(len(labels))
    ax.set_yticks(y, labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)   # first population at the top
    return y


def _renderer(fig):
    try:
        return fig.canvas.get_renderer()
    except AttributeError:  # pragma: no cover - non-Agg canvases
        from matplotlib.backend_bases import _get_renderer
        return _get_renderer(fig)


_CANDIDATES = [(5, 3, "left", "bottom"), (5, -3, "left", "top"), (-5, 3, "right", "bottom"),
               (-5, -3, "right", "top"), (0, 6, "center", "bottom"), (0, -6, "center", "top"),
               (7, 0, "left", "center"), (-7, 0, "right", "center")]


def _place_labels(ax, x, y, names, *, marker_pt: float = 0.0, halo: bool = False, weight: str = "normal",
                  centred: bool = False):
    """Place one label per point without overlaps; a label moved away gets a thin leader line.

    Candidate positions around each point are tried at increasing distance; a
    candidate is accepted when its box overlaps no placed label and no marker
    and stays inside the axes. Returns the text artists.
    """
    from matplotlib import patheffects

    fig = ax.figure
    ax.autoscale_view()        # final data limits before measuring
    ax.apply_aspect()
    renderer = _renderer(fig)
    px = fig.dpi / 72.0
    xs, ys = np.asarray(x, float), np.asarray(y, float)
    pts = ax.transData.transform(np.c_[xs, ys])
    half = marker_pt / 2 * px
    markers = [(a - half, b - half, a + half, b + half) for a, b in pts] if marker_pt else []
    frame = ax.get_window_extent(renderer)
    effects = [patheffects.withStroke(linewidth=2.0, foreground=SURFACE)] if halo else None
    first = [(0, 0, "center", "center")] if centred else []
    placed, artists = [], []

    def fits(bb) -> bool:
        if bb.x0 < frame.x0 or bb.x1 > frame.x1 or bb.y0 < frame.y0 or bb.y1 > frame.y1:
            return False
        return not any(bb.x0 < b[2] and bb.x1 > b[0] and bb.y0 < b[3] and bb.y1 > b[1] for b in placed + markers)

    def label(name, xi, yi, dx, dy, ha, va):
        return ax.annotate(str(name), (xi, yi), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                           fontsize=TICK, color=INK, weight=weight, path_effects=effects, annotation_clip=False)

    for name, xi, yi in zip(names, xs, ys):
        chosen = None
        for scale in (1.0, 2.0, 3.0, 4.2, 5.6):
            options = (first if scale == 1.0 else []) + _CANDIDATES
            for dx, dy, ha, va in options:
                t = label(name, xi, yi, dx * scale, dy * scale, ha, va)
                bb = t.get_window_extent(renderer).expanded(1.14, 1.25)
                if fits(bb):
                    chosen = (t, bb, scale > 1.0 and (dx, dy) != (0, 0))
                    break
                t.remove()
            if chosen:
                break
        if chosen is None:   # no free position: keep the nearest one
            dx, dy, ha, va = (first or _CANDIDATES)[0]
            t = label(name, xi, yi, dx, dy, ha, va)
            chosen = (t, t.get_window_extent(renderer), False)
        t, bb, leader = chosen
        if leader:   # thin connector from the point to the nearest edge of the label
            cx = min(max(pts[len(artists)][0], bb.x0), bb.x1)
            cy = min(max(pts[len(artists)][1], bb.y0), bb.y1)
            end = ax.transData.inverted().transform((cx, cy))
            ax.annotate("", (xi, yi), xytext=end, textcoords="data", annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=HAIRLINE, shrinkA=0,
                                        shrinkB=marker_pt / 2 + 0.5))
        placed.append((bb.x0, bb.y0, bb.x1, bb.y1))
        artists.append(t)
    return artists


def _basis_name(key: str) -> str:
    name = key[2:] if key.startswith("X_") else key
    return {"umap": "UMAP", "tsne": "t-SNE", "pca": "PC", "spectral": "Spectral"}.get(name.lower(), name)


# ---------------------------------------------------------------------------
# Per-cell plots
# ---------------------------------------------------------------------------


def embedding(adata, color: Optional[str] = None, *, basis: str = "X_umap", groupby: Optional[str] = None,
              log: bool = False, quantiles: tuple = (0.02, 0.98), cmap=None, point_size: Optional[float] = None,
              title: Optional[str] = None, ax=None):
    """Plot a per-cell scCHART output on an existing embedding.

    Cells with a missing value (for example programme-unscorable cells) are
    drawn in light grey underneath; the other cells are drawn in increasing
    order of the value, so the largest values are on top. Structural outputs
    use a single-hue blue scale and programme outputs a single-hue orange
    scale.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with the per-cell output in ``adata.obs`` and the embedding
        in ``adata.obsm``.
    color : str, optional
        Numeric ``adata.obs`` column, typically ``"scchart_structural_d"``,
        ``"scchart_programme_d"`` or ``"scchart_programme_P"``. Default None
        draws all cells in grey (use with ``groupby`` to show the
        populations).
    basis : str, optional
        ``adata.obsm`` key of the embedding; ``"umap"`` is read as
        ``"X_umap"``. The first two columns are plotted. Default
        ``"X_umap"``.
    groupby : str, optional
        ``adata.obs`` column whose population names are written at the
        population medians, moved apart where they would overlap. Default:
        no labels.
    log : bool, optional
        Logarithmic colour scale (useful for the heavy-tailed ``d_i``).
        Non-positive values are then drawn in grey. Default False.
    quantiles : tuple of float, optional
        Colour-scale limits as quantiles of the plotted values. Default
        ``(0.02, 0.98)``; ``(0, 1)`` uses the full range.
    cmap : str or matplotlib.colors.Colormap, optional
        Colour map. Default: blue for structural outputs, orange for
        programme outputs.
    point_size : float, optional
        Marker area. Default: scaled to the number of cells.
    title : str, optional
        Axes title. Default: the quantity plotted.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Default: a new figure.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    KeyError
        If ``color``, ``basis`` or ``groupby`` is missing.
    ValueError
        If ``color`` is not numeric or has no plottable value.

    Notes
    -----
    ``d_i(G)`` and ``d_i^(P)(G)`` are uncalibrated contributions to the cell's
    own population divergence; they are not comparable between populations.

    Examples
    --------
    >>> scchart.pl.embedding(adata, "scchart_structural_d", basis="X_umap", groupby="cell_type", log=True)
    >>> scchart.pl.embedding(adata, "scchart_programme_P")
    >>> scchart.pl.embedding(adata, groupby="cell_type")      # populations only
    """
    _check_adata(adata)
    _pyplot()
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

    key = basis if basis in adata.obsm else f"X_{basis}"
    if key not in adata.obsm:
        raise KeyError(f"embedding {basis!r} not found in adata.obsm; available: {list(adata.obsm)}")
    xy = np.asarray(adata.obsm[key])[:, :2]
    if groupby is not None and groupby not in adata.obs:
        raise KeyError(f"groupby={groupby!r} not found in adata.obs")
    size = point_size if point_size is not None else float(np.clip(5000.0 / max(adata.n_obs, 1), 0.3, 8.0))

    label, n_missing = None, 0
    if color is not None:
        if color not in adata.obs:
            outputs = sorted(c for c in adata.obs if c.startswith("scchart_"))
            raise KeyError(f"{color!r} not found in adata.obs; scCHART per-cell outputs present: {outputs}")
        column = adata.obs[color]
        if not pd.api.types.is_numeric_dtype(column) or pd.api.types.is_bool_dtype(column):
            raise ValueError(f"adata.obs[{color!r}] is not numeric")
        values = column.to_numpy(dtype=float)
        ok = np.isfinite(values) & ((values > 0) if log else True)
        if not ok.any():
            raise ValueError(f"adata.obs[{color!r}] has no {'positive ' if log else ''}finite values to plot")

    ax = _new_axes(ax, (3.4, 2.9) if color is not None else (2.9, 2.9))
    if color is None:
        ax.scatter(*xy.T, s=size, c=MISSING_COLOUR, linewidths=0, rasterized=True)
    else:
        lo, hi = np.quantile(values[ok], quantiles)
        if hi <= lo:
            hi = lo + (abs(lo) * 1e-6 or 1e-12)
        norm = LogNorm(lo, hi) if log else Normalize(lo, hi)
        if cmap is None:
            ramp = RAMPS["programme" if color.startswith("scchart_programme") else "structural"]
            cmap = LinearSegmentedColormap.from_list(f"scchart_{color}", ramp)
        ax.scatter(*xy[~ok].T, s=size, c=MISSING_COLOUR, linewidths=0, rasterized=True)
        order = np.flatnonzero(ok)[np.argsort(values[ok], kind="stable")]
        points = ax.scatter(*xy[order].T, c=values[order], s=size, cmap=cmap, norm=norm,
                            linewidths=0, rasterized=True)
        label = CELL_LABELS.get(color, color)
        _colorbar(ax, points, label + (", log scale" if log else ""))
        n_missing = int((~ok).sum())

    ax.set(xticks=[], yticks=[])
    ax.margins(0.04)
    ax.spines[:].set_visible(False)
    name = _basis_name(key)
    ax.set_xlabel(f"{name} 1", fontsize=NOTE, color=MUTED, loc="left")
    ax.set_ylabel(f"{name} 2", fontsize=NOTE, color=MUTED, loc="bottom")
    stem = None if label is None else label.split(" (")[0]
    default_title = "Populations" if stem is None else stem[0].upper() + stem[1:]
    _title(ax, title if title is not None else default_title)
    if n_missing:
        ax.text(1.0, -0.01, f"{n_missing:,} cells without a value (grey)", transform=ax.transAxes,
                ha="right", va="top", fontsize=NOTE, color=MUTED)
    if groupby is not None:
        groups = adata.obs[groupby].astype(str).to_numpy()
        names = list(pd.unique(groups))
        centres = np.array([np.median(xy[groups == n], axis=0) for n in names])
        _place_labels(ax, centres[:, 0], centres[:, 1], names, halo=True, weight="bold", centred=True)
    return ax


def programme_composition(adata, *, groupby: Optional[str] = None, annotate: bool = True,
                          cmap=None, ax=None):
    """Plot the mean programme composition of each population.

    Each row is the mean of ``r_i`` over the scored cells of a population;
    unscorable cells are skipped. Rows follow the fitted population order.
    Columns are aligned with the rows: when the programmes are named after
    the populations (endogenous programmes), each population's own programme
    sits in the same position as its row, so the diagonal shows each
    population's weight on its own programme; otherwise programmes are
    ordered by the population that uses them most.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData fitted with :class:`scchart.ProgrammeHeterogeneity` (or
        :class:`scchart.Heterogeneity` with ``programme=True``).
    groupby : str, optional
        ``adata.obs`` column defining the rows. Default: the grouping used
        for the fit.
    annotate : bool, optional
        Write the mean weight in every cell of at least 0.05. Default True.
    cmap : str or matplotlib.colors.Colormap, optional
        Colour map. Default: a single-hue orange scale from white.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Default: a new figure.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    KeyError
        If the programme outputs or ``groupby`` are missing.

    Notes
    -----
    With endogenous programmes each programme is defined by one
    population, so most of a population's weight is expected on its own
    programme; off-diagonal weight shows use of related populations'
    programmes.

    Examples
    --------
    >>> ax = scchart.pl.programme_composition(adata)
    >>> ax.figure.savefig("composition.pdf")
    """
    _check_adata(adata)
    _pyplot()
    from matplotlib.colors import LinearSegmentedColormap

    meta = adata.uns.get("scchart_programme")
    if meta is None or "scchart_programme_composition" not in adata.obsm:
        raise KeyError("programme outputs not found; fit the programme axis on adata first")
    groupby = groupby or meta["groupby"]
    if groupby not in adata.obs:
        raise KeyError(f"groupby={groupby!r} not found in adata.obs")
    programmes = [str(p) for p in np.asarray(meta["programmes"])]
    r = pd.DataFrame(np.asarray(adata.obsm["scchart_programme_composition"], dtype=float),
                     index=adata.obs_names, columns=programmes)
    labels = adata.obs[groupby].astype(str).to_numpy()
    scored = r.notna().all(axis=1).to_numpy()
    mean = r[scored].groupby(labels[scored]).mean()
    fit = adata.uns.get("scchart", {})
    order = ([str(p) for p in np.asarray(fit["population_order"])]
             if fit.get("groupby") == groupby and "population_order" in fit else sorted(set(labels)))
    mean = mean.reindex([p for p in order if p in mean.index])
    rows = list(mean.index)
    if set(programmes) <= set(rows):
        columns = [p for p in rows if p in programmes]
    else:   # order programmes by the row that uses them most
        top = mean.to_numpy().argmax(axis=0)
        columns = [programmes[j] for j in sorted(range(len(programmes)),
                                                 key=lambda j: (top[j], -mean.iloc[top[j], j]))]
    mean = mean[columns]

    values = mean.to_numpy()
    vmax = float(np.nanmax(values)) if values.size else 1.0
    if cmap is None:
        cmap = LinearSegmentedColormap.from_list("scchart_composition", [SURFACE] + RAMPS["programme"][1:])
    cell = 0.24
    ax = _new_axes(ax, (cell * len(columns) + 1.9, cell * len(rows) + 1.2))
    mesh = ax.pcolormesh(values, cmap=cmap, vmin=0, vmax=vmax, edgecolors=SURFACE, linewidth=0.8)
    ax.set_xlim(0, len(columns))
    ax.set_ylim(len(rows), 0)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(len(columns)) + 0.5, columns, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(rows)) + 0.5, rows)
    ax.tick_params(length=0, labelsize=TICK, labelcolor=INK, pad=2)
    ax.spines[:].set_visible(False)
    if annotate:
        for (i, j), value in np.ndenumerate(values):
            if np.isfinite(value) and value >= 0.05:
                ax.text(j + 0.5, i + 0.5, f"{value:.2f}".lstrip("0"), ha="center", va="center", fontsize=5.0,
                        color=SURFACE if value > 0.55 * vmax else INK)
    _colorbar(ax, mesh, r"mean weight $r_{ik}$ (scored cells)")
    ax.set_xlabel("programme", fontsize=LABEL, color=INK)
    ax.set_ylabel(groupby, fontsize=LABEL, color=INK)
    _title(ax, "Programme composition by population")
    return ax


# ---------------------------------------------------------------------------
# Population-level plots
# ---------------------------------------------------------------------------


def _delta_column(ax, y, values, header: str) -> None:
    """Calibrated excess as a right-hand column of numbers, aligned with the rows."""
    from matplotlib.transforms import blended_transform_factory

    tr = blended_transform_factory(ax.transAxes, ax.transData)
    ax.annotate(header, (1.03, 1.0), xycoords="axes fraction", xytext=(0, 2), textcoords="offset points",
                ha="left", va="bottom", fontsize=TICK, color=SECONDARY, annotation_clip=False)
    for yi, d in zip(y, values):
        ax.text(1.03, yi, f"{d:.3f}", transform=tr, ha="left", va="center", fontsize=TICK, color=INK,
                clip_on=False)


def _row_axes(ax) -> None:
    """Population rows: no left spine or y ticks, names in ink."""
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=LABEL, labelcolor=INK, pad=3)


def calibration(adata, axis: str = "structural", *, populations: Optional[Sequence[str]] = None, ax=None):
    """Plot observed ``J``, null expectation ``E0[J]`` and excess ``Delta`` per population.

    The null expectation (open circle, bar = null standard deviation) and the
    observed divergence (filled circle) are joined by a line whose length is
    the calibrated excess ``Delta = J - E0[J]``, also listed on the right.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData fitted with ``scchart.Heterogeneity(calibrate=True)``.
    axis : {"structural", "programme"}, optional
        Heterogeneity axis. Default ``"structural"``.
    populations : sequence of str, optional
        Populations to show, in order. Default: all, in the fitted
        population order.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Default: a new figure.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    KeyError
        If the axis was not fitted or a population is unknown.
    ValueError
        If the axis was not null-calibrated.

    Notes
    -----
    ``Delta`` is not depth-invariant; rank populations at a common depth
    (``depth_standardized_*_comparison``).

    Examples
    --------
    >>> scchart.pl.calibration(adata, "structural")
    >>> scchart.pl.calibration(adata, "programme", populations=["CD4 Naive", "CD8 Naive"])
    """
    table = _select(_population_table(adata, axis), populations)
    cols, sym, colour = COLUMNS[axis], SYMBOLS[axis], AXIS_COLOUR[axis]
    J, E0, D = (table[cols[k]].to_numpy(dtype=float) for k in ("J", "E0", "Delta"))
    sd0 = table["sd0_diagnostic"].to_numpy(dtype=float)
    plt = _pyplot()
    ax = _new_axes(ax, (3.8, 0.24 * len(table) + 1.0))
    y = _population_rows(ax, list(table.index))
    ax.hlines(y, E0, J, color=colour, lw=1.6, zorder=2)
    ax.errorbar(E0, y, xerr=sd0, fmt="none", ecolor=SECONDARY, elinewidth=THIN, capsize=1.5, capthick=THIN,
                zorder=3)
    ax.scatter(E0, y, s=MARKER, facecolor=SURFACE, edgecolor=colour, lw=1.0, zorder=4)
    ax.scatter(J, y, s=MARKER, color=colour, edgecolor=SURFACE, lw=0.8, zorder=5)
    right = float(np.nanmax(np.r_[J, E0 + sd0]))
    ax.set_xlim(min(0.0, float(np.nanmin(E0 - sd0))), 1.06 * right if right > 0 else 1.0)
    _chrome(ax, grid="x")
    _row_axes(ax)
    ax.set_xlabel("divergence (bits)", fontsize=LABEL)
    _delta_column(ax, y, D, sym["Delta"])
    handles = [plt.Line2D([], [], marker="o", ls="", ms=5, mfc=SURFACE, mec=colour, mew=1.0,
                          label=f"null {sym['E0']} ± sd"),
               plt.Line2D([], [], marker="o", ls="", ms=5, color=colour, label=f"observed {sym['J']}"),
               plt.Line2D([], [], color=colour, lw=1.6, label=f"excess {sym['Delta']}")]
    _title(ax, f"{axis.capitalize()} heterogeneity: observed vs null", pad=_legend_above(ax, handles) + 3)
    return ax


def null_distribution(adata, axis: str = "structural", *, populations: Optional[Sequence[str]] = None,
                      ax=None):
    """Plot each population's observed ``J`` against its null replicates.

    Grey points are the null replicates of the fixed-margin null, the short
    vertical line is their mean ``E0[J]`` and the filled circle the observed
    ``J``.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData fitted with ``scchart.Heterogeneity(calibrate=True)``.
    axis : {"structural", "programme"}, optional
        Heterogeneity axis. Default ``"structural"``.
    populations : sequence of str, optional
        Populations to show, in order. Default: all, in the fitted
        population order.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Default: a new figure.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    KeyError
        If the axis was not fitted or a population is unknown.
    ValueError
        If the axis was not null-calibrated.

    Notes
    -----
    The replicates estimate ``E0[J]``; their spread measures Monte Carlo
    precision and is not a test of heterogeneity. Programme null replicates
    in which every cell is unscorable are NaN and are not drawn.

    Examples
    --------
    >>> scchart.pl.null_distribution(adata, "programme")
    """
    full = _population_table(adata, axis)
    table = _select(full, populations)
    rows = [list(full.index).index(p) for p in table.index]
    nulls = _null_replicates(adata, axis)[rows]
    cols, sym, colour = COLUMNS[axis], SYMBOLS[axis], AXIS_COLOUR[axis]
    plt = _pyplot()
    ax = _new_axes(ax, (3.8, 0.24 * len(table) + 1.0))
    y = _population_rows(ax, list(table.index))
    jitter = np.random.default_rng(0).uniform(-0.16, 0.16, size=nulls.shape)   # display only
    for i, reps in enumerate(nulls):
        keep = np.isfinite(reps)
        ax.scatter(reps[keep], i + jitter[i, keep], s=5, color=MUTED, alpha=0.75, linewidths=0, zorder=2)
    E0 = table[cols["E0"]].to_numpy(dtype=float)
    ax.vlines(E0, y - 0.3, y + 0.3, color=INK, lw=0.8, zorder=3)
    ax.scatter(table[cols["J"]], y, s=MARKER, color=colour, edgecolor=SURFACE, lw=0.8, zorder=4)
    _chrome(ax, grid="x")
    _row_axes(ax)
    ax.margins(x=0.06)
    ax.set_xlabel(f"{sym['J']} (bits)", fontsize=LABEL)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=3, color=MUTED,
                          label=f"null replicates (n = {nulls.shape[1]})"),
               plt.Line2D([], [], marker="|", ls="", ms=7, mew=0.8, color=INK, label=f"null mean {sym['E0']}"),
               plt.Line2D([], [], marker="o", ls="", ms=5, color=colour, label=f"observed {sym['J']}")]
    _title(ax, f"{axis.capitalize()} heterogeneity: observed vs null replicates",
           pad=_legend_above(ax, handles) + 3)
    return ax


def structural_vs_programme(adata, *, populations: Optional[Sequence[str]] = None,
                            groups: Optional[Mapping[str, str]] = None, label: bool = True, ax=None):
    """Plot the calibrated structural excess against the programme excess.

    One point per population: ``Delta_G`` (structural) on the x axis and
    ``Delta_G^(P)`` (programme) on the y axis. The two axes are shown
    together but never combined into one score. Population names are placed
    next to the points without overlaps, with a thin leader line when a name
    has to move away from its point.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData fitted with ``scchart.Heterogeneity(calibrate=True)`` on both
        axes.
    populations : sequence of str, optional
        Populations to show. Default: all.
    groups : mapping, optional
        Population -> group (for example a lineage) used to colour the points,
        with a legend. Colours follow the order in which groups first appear;
        at most three groups keep every pair distinguishable under colour-vision
        deficiency. Default: one colour.
    label : bool, optional
        Write the population names next to the points. Default True.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Default: a new figure.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    KeyError
        If either axis was not fitted, a population is unknown, or a
        population is missing from ``groups``.
    ValueError
        If either axis was not null-calibrated, or ``groups`` has more than
        eight groups.

    Notes
    -----
    Values are at the observed depth of each population. ``Delta`` depends
    on depth, so populations of different depth are compared at a common
    ``D*`` with the ``depth_standardized_*_comparison`` functions.

    Examples
    --------
    >>> scchart.pl.structural_vs_programme(adata)
    >>> scchart.pl.structural_vs_programme(adata, groups={"CD4 Naive": "T", "Naive B": "B", "cDC": "Myeloid"})
    """
    s = _select(_population_table(adata, "structural"), populations)
    p = _population_table(adata, "programme").reindex(s.index)
    x = s[COLUMNS["structural"]["Delta"]].to_numpy(dtype=float)
    y = p[COLUMNS["programme"]["Delta"]].to_numpy(dtype=float)
    palette = None
    if groups is not None:
        missing = [g for g in s.index if g not in groups]
        if missing:
            raise KeyError(f"populations missing from groups: {missing}")
        names = list(dict.fromkeys(str(groups[g]) for g in s.index))
        if len(names) > len(CATEGORICAL):
            raise ValueError(f"at most {len(CATEGORICAL)} groups can be coloured, got {len(names)}")
        palette = dict(zip(names, CATEGORICAL))
        colours = [palette[str(groups[g])] for g in s.index]
    else:
        colours = SECONDARY
    plt = _pyplot()
    ax = _new_axes(ax, (3.6, 3.0))
    ax.axhline(0, color=MUTED, lw=THIN, zorder=1)
    ax.axvline(0, color=MUTED, lw=THIN, zorder=1)
    ax.scatter(x, y, s=MARKER + 6, c=colours, edgecolor=SURFACE, lw=0.8, zorder=3)
    lo_x, hi_x = min(0.0, float(np.nanmin(x))), max(0.0, float(np.nanmax(x)))
    lo_y, hi_y = min(0.0, float(np.nanmin(y))), max(0.0, float(np.nanmax(y)))
    pad_x, pad_y = 0.1 * (hi_x - lo_x or 1.0), 0.1 * (hi_y - lo_y or 1.0)
    ax.set_xlim(lo_x - pad_x, hi_x + 2.2 * pad_x)
    ax.set_ylim(lo_y - pad_y, hi_y + 1.6 * pad_y)
    _chrome(ax, grid="both")
    ax.set_xlabel(f"structural excess {SYMBOLS['structural']['Delta']} (bits)", fontsize=LABEL)
    ax.set_ylabel(f"programme excess {SYMBOLS['programme']['Delta']} (bits)", fontsize=LABEL)
    pad = 4.0
    if palette is not None:
        pad = _legend_above(ax, [plt.Line2D([], [], marker="o", ls="", ms=5, color=c, mec=SURFACE, label=n)
                                 for n, c in palette.items()]) + 3
    _title(ax, "Calibrated excess at observed depth", pad=pad)
    if label:
        _place_labels(ax, x, y, list(s.index), marker_pt=float(np.sqrt(MARKER + 6)))
    return ax


__all__ = ["embedding", "programme_composition", "calibration", "null_distribution", "structural_vs_programme",
           "set_style"]
