"""scCHART: single-cell Chromatin Heterogeneity Assessment and Ranking Tool.

scCHART measures two complementary kinds of heterogeneity in single-cell
chromatin-accessibility data. They are reported separately and are never
combined into a single score.

Model (:class:`Heterogeneity`)
    ``model = scchart.Heterogeneity(...)`` and ``model.fit(adata,
    groupby=...)`` fit either axis or both, run the technical null and store
    per-cell outputs, population tables and seeds on the AnnData.

Structural axis (:mod:`scchart.structural_divergence`)
    How dispersed a population's cells are in a declared representation Z:
    per-cell ``d_i(G) = KL(p_i || pbar_G)`` and population ``J_G``.
    ``LSIProjection`` is the fixed projection B -> Z used by the null.

Programme axis (:mod:`scchart.programme_entropy`)
    How cells differ in their usage of regulatory programmes: within-cell
    mixing ``P_i``, per-cell ``d_i^(P)`` and population ``J_G^(P)``, from
    endogenous cross-fitted, external or fixed (``ProgrammeReference``)
    programmes.

Matched depth
    ``depth_standardized_structural_comparison`` and
    ``depth_standardized_programme_comparison`` compare populations at a
    declared common depth ``D*``.

Plotting (``scchart.pl``, module :mod:`scchart.plotting`)
    Matplotlib figures of the stored outputs: ``embedding``,
    ``programme_composition``, ``calibration``, ``null_distribution`` and
    ``structural_vs_programme``.

Technical null
    Every calibration uses a fixed-margin 2 x 2 swap chain on the binary
    cells x features block. The default sampler is the proposal clock
    (``null_sampler="proposal_clock"``): a fixed number of proposals, with
    rejected proposals counted as self-loops; its stationary distribution is
    uniform over the matrices with the observed margins. The accepted-swap
    sampler (``"accepted_swap"``) is available as an alternative; its
    stationary distribution is generally not uniform.

Lower-level functions (adapters, evidence, membership, swap null, seeds,
downsampling) are available from their modules.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("scchart")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0+unknown"
del PackageNotFoundError, _version


from .model import Heterogeneity
from .structural_divergence import (
    LSIProjection,
    StructuralHeterogeneity,
    StructuralHeterogeneityResult,
    depth_standardized_structural_comparison,
)
from .programme_entropy import (
    ProgrammeHeterogeneity,
    ProgrammeHeterogeneityResult,
    ProgrammeReference,
    depth_standardized_programme_comparison,
    external_programme_reference,
)
from . import plotting as pl


__all__ = [
    "Heterogeneity",
    # Axis scorers
    "StructuralHeterogeneity",
    "StructuralHeterogeneityResult",
    "ProgrammeHeterogeneity",
    "ProgrammeHeterogeneityResult",
    # Inputs: fixed projection and fixed programme references
    "LSIProjection",
    "ProgrammeReference",
    "external_programme_reference",
    # Matched-depth comparison
    "depth_standardized_structural_comparison",
    "depth_standardized_programme_comparison",
]
