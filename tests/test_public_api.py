"""Tests of the top-level scchart API surface."""

import inspect
import types

import scchart

CANONICAL = [
    "Heterogeneity",
    "StructuralHeterogeneity",
    "StructuralHeterogeneityResult",
    "ProgrammeHeterogeneity",
    "ProgrammeHeterogeneityResult",
    "LSIProjection",
    "ProgrammeReference",
    "external_programme_reference",
    "depth_standardized_structural_comparison",
    "depth_standardized_programme_comparison",
]

LOW_LEVEL = [
    # calibration functions and sampler names used by Heterogeneity
    "structural_null_calibration",
    "structural_population_calibration",
    "crossfit_programme_references",
    "crossfit_population_calibration",
    "DEFAULT_SAMPLER",
    "SAMPLER_PROPOSAL_CLOCK",
    "SAMPLER_ACCEPTED_SWAP",
    "entropy_bits",
    "kl_bits",
    "generalized_jsd",
    "binarize",
    "swap_randomise",
    "swap_replicates",
    "seeds_from_key",
    "binomial_downsample",
    "reference_background_rates",
    "programme_evidence",
    "programme_membership",
    "programme_mixing",
    "programme_divergence",
    "signed_channel_probability",
    "spectral_energy_probability",
]


def test_all_is_exactly_the_canonical_api():
    assert list(scchart.__all__) == CANONICAL


def test_all_exports_resolve():
    for name in scchart.__all__:
        assert hasattr(scchart, name), f"scchart.__all__ lists {name!r} but it is not importable"


def test_no_stray_public_names_missing_from_all():
    # Every public (non-underscore), non-submodule top-level attribute is in
    # __all__. Submodules register themselves on import and are excluded.
    public_names = {
        name
        for name, value in vars(scchart).items()
        if not name.startswith("_")
        and name != "__version__"
        and not isinstance(value, types.ModuleType)
    }
    assert public_names == set(scchart.__all__)


def test_low_level_helpers_live_in_modules_not_top_level():
    from scchart import _scoring, programme_entropy, structural_divergence

    modules = (_scoring, programme_entropy, structural_divergence)
    for name in LOW_LEVEL:
        assert name not in scchart.__all__
        assert any(hasattr(m, name) for m in modules), name


def test_scorer_classes_share_fit_signature():
    for cls in (scchart.StructuralHeterogeneity, scchart.ProgrammeHeterogeneity):
        params = list(inspect.signature(cls.fit).parameters)
        assert params[:3] == ["self", "adata", "groupby"], f"{cls.__name__}.fit: {params}"


def test_canonical_objects_are_the_module_objects():
    from scchart import model, programme_entropy, structural_divergence

    assert scchart.Heterogeneity is model.Heterogeneity
    assert scchart.StructuralHeterogeneity is structural_divergence.StructuralHeterogeneity
    assert scchart.LSIProjection is structural_divergence.LSIProjection
    assert scchart.ProgrammeHeterogeneity is programme_entropy.ProgrammeHeterogeneity
    assert scchart.ProgrammeReference is programme_entropy.ProgrammeReference


def test_structural_default_adapter_is_signed_channel():
    assert scchart.StructuralHeterogeneity(verbose=False).adapter == "signed_channel"
    assert scchart.Heterogeneity(verbose=False).adapter == "signed_channel"


def test_version_is_a_string():
    assert isinstance(scchart.__version__, str)
    assert scchart.__version__


def test_plotting_namespace_is_pl():
    from scchart import plotting

    assert scchart.pl is plotting
    assert plotting.__all__ == ["embedding", "programme_composition", "calibration", "null_distribution",
                                "structural_vs_programme", "set_style"]
    public = {n for n in vars(plotting) if not n.startswith("_") and callable(getattr(plotting, n))
              and getattr(getattr(plotting, n), "__module__", None) == plotting.__name__}
    assert public == set(plotting.__all__)
