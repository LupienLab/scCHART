"""Worked example with pinned values for both axes.

Two kinds of expected value appear here:

* Independently derived. ``J`` and ``J^(P)`` of the small populations follow
  from the defining formulas by hand (0.5 and 1.5 bits), and the null
  expectations marked "exact" enumerate every binary matrix with the observed
  margins, independently of any sampler.
* Previously established. The calibration values marked "established" are
  20-replicate swap-chain estimates for the pinned seeds (proposal-clock
  sampler, the default, and accepted-swap sampler). They were recorded before
  scCHART 0.1.0 and are reproduced exactly by the current implementation; they
  guard against change but are Monte Carlo estimates, not derived values.
"""

import itertools

import numpy as np
import pytest

from scchart.programme_entropy import (
    ProgrammeReference,
    programme_evidence,
    programme_membership,
    programme_mixing,
    programme_null_calibration,
    programme_population_table,
)
from scchart.structural_divergence import LSIProjection, structural_J, structural_null_calibration

U = np.array([[1, 1, .5], [1, 1, -.5], [1, 1, 0], [1, -1, .5], [1, -1, -.5], [1, -1, 0]], float)
PROJ = LSIProjection(idf=np.ones(6), loadings=U, components=[0, 1, 2])
LAB = np.array(["p0", "p0", "p1", "p1", "p2", "p2", "Shared", "Shared", "Shared", "Shared"])
REF = ProgrammeReference(LAB, ["p0", "p1", "p2"], np.where(LAB == "Shared", 0.0, 0.05))


def _rows(sets, p):
    B = np.zeros((len(sets), p))
    for i, s in enumerate(sets):
        B[i, [f - 1 for f in s]] = 1
    return B


def _margin_matrices(B):
    r, c = B.sum(1).astype(int), B.sum(0).astype(int)
    for choice in itertools.product(*[itertools.combinations(range(B.shape[1]), k) for k in r]):
        M = np.zeros(B.shape)
        for i, ch in enumerate(choice):
            M[i, list(ch)] = 1
        if (M.sum(0) == c).all():
            yield M


S_A = _rows([(1, 2, 4), (1, 3, 5), (2, 4, 6), (3, 5, 6)], 6)
S_B = _rows([(1, 2, 3), (1, 2, 3), (4, 5, 6), (4, 5, 6)], 6)
P_A = _rows([(1, 3, 7, 8), (1, 5, 8, 9), (2, 4, 9, 10), (2, 6, 7, 10)], 10)
P_B = _rows([(1, 2, 7, 8), (3, 4, 8, 9), (5, 6, 9, 10), (1, 2, 7, 10)], 10)


def test_structural_worked_example():
    ZB = PROJ(S_B)
    np.testing.assert_allclose(ZB, [[1, 1, 0], [1, 1, 0], [1, -1, 0], [1, -1, 0]])
    outB = structural_J(ZB)
    np.testing.assert_allclose(outB["pbar"], [0.5, 0.25, 0, 0, 0.25, 0])
    np.testing.assert_allclose(outB["d"], 0.5)
    assert outB["J"] == pytest.approx(0.5, abs=1e-12)
    assert structural_J(PROJ(S_A))["J"] == pytest.approx(0.123279, abs=5e-7)
    E0 = np.mean([structural_J(PROJ(M))["J"] for M in _margin_matrices(S_B)])   # exact
    assert E0 == pytest.approx(0.216284, abs=5e-7)
    E0A = np.mean([structural_J(PROJ(M))["J"] for M in _margin_matrices(S_A)])
    assert E0A == pytest.approx(E0)   # same margins -> same null
    leg = structural_null_calibration(S_B, PROJ, seed_key=(20260928, 1), sampler="accepted_swap")
    assert leg.E0_J == pytest.approx(0.189829, abs=5e-7)   # established, 20-replicate estimate
    assert leg.Delta == pytest.approx(0.310171, abs=5e-7)
    cal = structural_null_calibration(S_B, PROJ, seed_key=(20260928, 1))   # proposal clock; established
    assert cal.E0_J == pytest.approx(0.278202, abs=5e-7)
    assert cal.Delta == pytest.approx(0.221798, abs=5e-7)


def test_structural_worked_example_spectral_energy_collapses_antipodal_cells():
    # S_B has two antipodal states along component 2; spectral energy maps them together.
    assert structural_J(PROJ(S_B), adapter="spectral_energy")["J"] == pytest.approx(0.0, abs=1e-12)


def test_programme_worked_example():
    aA = programme_evidence(P_A, REF)
    np.testing.assert_allclose(aA, [[0.4, 0.4, 0], [0.4, 0, 0.4], [0.4, 0.4, 0], [0.4, 0, 0.4]])
    rA = programme_membership(aA)
    np.testing.assert_allclose(programme_mixing(rA), 1 / np.log2(3))
    rB = programme_membership(programme_evidence(P_B, REF))
    np.testing.assert_allclose(programme_mixing(rB), 0.0, atol=1e-15)
    tA = programme_population_table(rA, ["A"] * 4)
    tB = programme_population_table(rB, ["B"] * 4)
    assert tA.loc["A", "J_P_bits"] == pytest.approx(0.5, abs=1e-12)
    assert tB.loc["B", "J_P_bits"] == pytest.approx(1.5, abs=1e-12)
    null = []
    for M in _margin_matrices(P_B[:, :6]):   # exact
        Bn = P_B.copy()
        Bn[:, :6] = M
        rn = programme_membership(programme_evidence(Bn, REF))
        null.append(programme_population_table(rn, ["B"] * 4).loc["B", "J_P_bits"])
    assert np.mean(null) == pytest.approx(0.756410, abs=5e-7)
    leg = programme_null_calibration([(P_B, REF, 20260928)], n_replicates=20, sampler="accepted_swap")  # established
    assert leg.E0_J == pytest.approx(0.75, abs=1e-9) and leg.Delta == pytest.approx(0.75, abs=1e-9)
    cal = programme_null_calibration([(P_B, REF, 20260928)], n_replicates=20)   # proposal clock; established
    assert cal.E0_J == pytest.approx(0.7625, abs=1e-9) and cal.Delta == pytest.approx(0.7375, abs=1e-9)
