"""Validation of the fixed-margin null samplers (proposal clock and accepted swap).

* proposal clock (default): fixed number of proposals, rejections are self-loops;
* accepted-swap clock: stops after a fixed number of accepted swaps.

Covers margins, determinism, input immutability, degenerate and frozen
matrices and, as independently derived checks, the exact stationary and
finite-budget laws of both samplers on small fixed-margin state spaces
(enumerated in ``scchart._null_enumeration``). The accepted-swap digest test
at the end is a previously established regression value.
"""

import hashlib

import numpy as np
import pytest
from scipy import sparse

from scchart._null_enumeration import (
    finite_law,
    jump_kernel,
    jump_stationary,
    margin_states,
    proposal_kernel,
    state_index,
    total_variation,
)
from scchart._scoring import (
    DEFAULT_SAMPLER,
    PROPOSAL_MULTIPLIER,
    SWAP_MULTIPLIER,
    default_multiplier,
    swap_randomise,
    swap_replicates,
)

SAMPLERS = ["proposal_clock", "accepted_swap"]

# Row sums (3, 1, 3), column sums (2, 2, 2, 1): 7 states, acceptance varies, so the
# accepted-swap jump chain's stationary law is not uniform (TV 0.0571 from uniform, the
# largest among the 735 non-trivial 3x3 / 3x4 spaces).
BIASED = np.array([[1, 1, 1, 0],
                   [0, 0, 0, 1],
                   [1, 1, 1, 0]], np.uint8)


def _random_binary(seed, n, f, p):
    return (np.random.default_rng(seed).random((n, f)) < p).astype(np.uint8)


# ----------------------------------------------------------------------------- defaults
def test_default_sampler_is_proposal_clock():
    assert DEFAULT_SAMPLER == "proposal_clock"
    assert default_multiplier() == PROPOSAL_MULTIPLIER == 100
    assert default_multiplier("accepted_swap") == SWAP_MULTIPLIER == 10
    with pytest.raises(ValueError):
        swap_randomise(np.eye(3), 1, sampler="accepted")


# ----------------------------------------------------------------------------- margins
@pytest.mark.parametrize("sampler", SAMPLERS)
@pytest.mark.parametrize("shape,p", [((40, 60), 0.2), ((7, 300), 0.05), ((200, 9), 0.4), ((15, 15), 0.9)])
def test_margins_exact(sampler, shape, p):
    B = _random_binary(3, *shape, p)
    for D, d in swap_replicates(B, [1, 2, 3], sampler=sampler):
        np.testing.assert_array_equal(D.sum(axis=1), B.sum(axis=1))
        np.testing.assert_array_equal(D.sum(axis=0), B.sum(axis=0))
        assert d.margins_exact and set(np.unique(D)) <= {0, 1}


def test_proposal_clock_budget_and_diagnostics():
    B = _random_binary(4, 30, 40, 0.2)
    nnz = int(B.sum())
    D, d = swap_randomise(B, 9)
    assert d.sampler == "proposal_clock" and d.target == d.attempts == d.proposals == 100 * nnz
    assert d.complete and 0 < d.accepted < d.attempts
    D, d = swap_randomise(B, 9, n_mult=3.5)
    assert d.attempts == int(3.5 * nnz)


# ----------------------------------------------------------------------------- determinism
@pytest.mark.parametrize("sampler", SAMPLERS)
def test_fixed_seed_is_reproducible_and_seeds_differ(sampler):
    B = _random_binary(5, 25, 35, 0.2)
    a, _ = swap_randomise(B, 77, sampler=sampler)
    b, _ = swap_randomise(B, 77, sampler=sampler)
    c, _ = swap_randomise(B, 78, sampler=sampler)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_parallel_equals_serial(sampler):
    B = _random_binary(6, 30, 45, 0.15)
    seeds = [11, 22, 33, 44, 55]
    par = [D.copy() for D, _ in swap_replicates(B, seeds, sampler=sampler, n_jobs=3)]
    ser = [swap_randomise(B, s, sampler=sampler)[0] for s in seeds]
    for x, y in zip(par, ser):
        np.testing.assert_array_equal(x, y)


def test_proposal_clock_prefix_property():
    """A longer budget extends the same proposal stream (nested budgets)."""
    B = _random_binary(7, 10, 12, 0.3)
    _, d1 = swap_randomise(B, 5, n_mult=1)
    _, d2 = swap_randomise(B, 5, n_mult=2)
    assert d2.accepted >= d1.accepted


# ----------------------------------------------------------------------------- inputs untouched
@pytest.mark.parametrize("sampler", SAMPLERS)
@pytest.mark.parametrize("kind", ["uint8", "float_counts", "csr_counts", "bool"])
def test_input_is_not_mutated(sampler, kind):
    rng = np.random.default_rng(8)
    C = rng.poisson(0.4, size=(20, 25))
    X = {"uint8": (C > 0).astype(np.uint8), "float_counts": C.astype(float),
         "csr_counts": sparse.csr_matrix(C.astype(float)), "bool": C > 0}[kind]
    before = X.toarray().copy() if sparse.issparse(X) else X.copy()
    D, _ = swap_randomise(X, 3, sampler=sampler)
    after = X.toarray() if sparse.issparse(X) else X
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(D.sum(axis=1), (C > 0).sum(axis=1))   # counts are binarised


# ----------------------------------------------------------------------------- degenerate inputs
@pytest.mark.parametrize("sampler", SAMPLERS)
@pytest.mark.parametrize("B", [np.zeros((4, 5)), np.ones((4, 5)), np.eye(1), np.zeros((0, 3)),
                               np.array([[0, 1, 0]]), np.array([[1], [1], [0]]),
                               np.array([[1, 1, 1, 0, 0]]), np.array([[1], [0], [1]])],
                         ids=["zeros", "ones", "1x1", "no_rows", "one_nnz", "one_column", "one_row",
                              "single_column_two"])
def test_degenerate_matrices_are_returned_unchanged(sampler, B):
    D, d = swap_randomise(B, 1, sampler=sampler)
    np.testing.assert_array_equal(D, (B > 0).astype(np.uint8))
    assert d.margins_exact and d.accepted == 0


def test_frozen_margins_no_valid_swap():
    """Nested (Ferrers) margins admit exactly one matrix: no swap is ever valid."""
    B = np.array([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]], np.uint8)
    assert len(margin_states(B)) == 1
    D, d = swap_randomise(B, 2)
    np.testing.assert_array_equal(D, B)
    # proposal clock: every proposal is a self-loop; the budget is still spent
    assert d.accepted == 0 and d.attempts == 100 * int(B.sum()) and d.complete
    D, d = swap_randomise(B, 2, sampler="accepted_swap", max_attempt_factor=2)
    np.testing.assert_array_equal(D, B)
    # accepted-swap clock: cannot reach its accepted target and stops at the attempt cap
    assert d.accepted == 0 and not d.complete and d.attempts == 2 * 10 * int(B.sum())


def test_few_valid_swaps_two_state_space():
    """A 2x2 permutation matrix has two states; the proposal clock visits both equally."""
    B = np.array([[1, 0], [0, 1]], np.uint8)
    S = margin_states(B)
    assert len(S) == 2
    idx = state_index(S)
    counts = np.zeros(2)
    for s in range(4000):
        D, _ = swap_randomise(B, s)
        counts[idx[D.tobytes()]] += 1
    assert abs(counts[0] / 4000 - 0.5) < 0.04


# ----------------------------------------------------------------------------- exact enumeration
def test_enumeration_helper_counts_states():
    # 3x3 permutation matrices: 6 states; 2x2 all-ones-row-sum-1: 2 states
    assert len(margin_states(np.eye(3))) == 6
    assert len(margin_states(BIASED)) == 7


def test_proposal_kernel_is_symmetric_and_uniform_is_stationary():
    S = margin_states(BIASED)
    P = proposal_kernel(S)
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    np.testing.assert_allclose(P, P.T, atol=1e-15)
    u = np.full(len(S), 1 / len(S))
    np.testing.assert_allclose(u @ P, u, atol=1e-15)
    assert np.all(np.diag(P) > 0)   # self-loops: aperiodic


def test_accepted_swap_stationary_law_is_not_uniform():
    S = margin_states(BIASED)
    Q, a = jump_kernel(proposal_kernel(S))
    pi = jump_stationary(a)
    np.testing.assert_allclose(pi @ Q, pi, atol=1e-12)
    u = np.full(len(S), 1 / len(S))
    assert total_variation(pi, u) == pytest.approx(0.0571428571, abs=1e-9)


def test_exact_finite_laws_from_fixed_start():
    S = margin_states(BIASED)
    P = proposal_kernel(S)
    Q, _ = jump_kernel(P)
    E = int(BIASED.sum())
    start = state_index(S)[BIASED.tobytes()]
    u = np.full(len(S), 1 / len(S))
    assert total_variation(finite_law(P, start, 100 * E), u) < 1e-10       # proposal clock: uniform
    assert total_variation(finite_law(Q, start, 10 * E), u) > 0.05         # accepted swap: biased


def test_empirical_sampler_laws_match_exact_laws():
    """Run the numba implementation from a fixed start over many seeds.

    The proposal clock's empirical law is consistent with uniform; the accepted-swap
    sampler's empirical law is consistent with its (non-uniform) exact law and
    clearly separated from uniform.
    """
    S = margin_states(BIASED)
    idx = state_index(S)
    P = proposal_kernel(S)
    Q, _ = jump_kernel(P)
    E = int(BIASED.sum())
    start = idx[BIASED.tobytes()]
    u = np.full(len(S), 1 / len(S))
    exact_accepted = finite_law(Q, start, 10 * E)
    n = 30000
    freq = {}
    for sampler in SAMPLERS:
        c = np.zeros(len(S))
        for s in range(n):
            D, _ = swap_randomise(BIASED, 1_000_003 + s, sampler=sampler)
            c[idx[D.tobytes()]] += 1
        freq[sampler] = c / n
    # Expected TV of an n-sample empirical law with 7 cells is ~0.01; bounds are ~3x that.
    assert total_variation(freq["proposal_clock"], u) < 0.015
    assert total_variation(freq["accepted_swap"], exact_accepted) < 0.015
    assert total_variation(freq["accepted_swap"], u) > 0.04


# ----------------------------------------------------------------------------- established digest
def test_accepted_swap_sampler_matches_established_digest():
    """SHA-256 of accepted-swap outputs over a fixed set of matrices and seeds.

    Previously established: the digest was recorded before scCHART 0.1.0 and is
    reproduced exactly by the current implementation (a regression value).
    """
    h = hashlib.sha256()
    for k, (n, f, p) in enumerate([(40, 60, .2), (30, 45, .15), (100, 300, .05), (12, 9, .5), (5, 5, .8)]):
        B = (np.random.default_rng(k).random((n, f)) < p).astype(np.uint8)
        for s in (1, 123, 99991):
            D, d = swap_randomise(B, s, sampler="accepted_swap")
            h.update(D.tobytes())
            h.update(str((d.accepted, d.attempts)).encode())
        for D, d in swap_replicates(B, [5, 6, 7, 8], n_jobs=4, sampler="accepted_swap"):
            h.update(D.tobytes())
    assert h.hexdigest() == "9e8fc1e90eea574203f5fe6af84a444b833ed865bf9528a79341ad35699d1ef4"
