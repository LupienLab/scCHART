"""Strict tests of the bit-packed fixed-margin chain.

The v0.1.0 chain (dense ``uint8`` state, int64 entry arrays) is kept here as the
reference. The bit-packed chain must give byte-identical matrices and identical
diagnostics for every seed, sampler, budget and block shape, including feature
counts around the 64-bit word size.
"""

import numpy as np
import pytest
from numba import njit
from scipy import sparse

from scchart import _scoring as S
from scchart._scoring import SAMPLERS, swap_randomise, swap_replicates


@njit
def _chain_v010(rows, cols, D, target, max_attempts, seed, accepted_clock):  # pragma: no cover - numba
    np.random.seed(seed)
    E = rows.size
    acc = 0
    att = 0
    while att < max_attempts:
        if accepted_clock and acc >= target:
            break
        e1 = np.random.randint(E)
        e2 = np.random.randint(E)
        att += 1
        i = rows[e1]
        a = cols[e1]
        j = rows[e2]
        b = cols[e2]
        if i == j or a == b:
            continue
        if D[i, b] != 0 or D[j, a] != 0:
            continue
        D[i, a] = 0
        D[j, b] = 0
        D[i, b] = 1
        D[j, a] = 1
        cols[e1] = b
        cols[e2] = a
        acc += 1
    return acc, att


def _swap_randomise_v010(B, seed, sampler="proposal_clock", n_mult=None, max_attempt_factor=S.MAX_ATTEMPT_FACTOR):
    """The v0.1.0 swap_randomise (dense state), kept as the reference."""
    D0 = (B.toarray() > 0) if sparse.issparse(B) else (np.asarray(B) > 0)
    D0 = np.ascontiguousarray(D0.astype(np.uint8))
    D = D0.copy()
    rows, cols = np.nonzero(D)
    rows, cols = rows.astype(np.int64), cols.astype(np.int64)
    nnz = rows.size
    m = S.default_multiplier(sampler) if n_mult is None else float(n_mult)
    target = int(m * nnz)
    accepted_clock = sampler == "accepted_swap"
    max_att = int(max_attempt_factor) * max(target, 1) if accepted_clock else target
    if nnz < 2:
        return D, {"nnz": nnz, "accepted": 0, "attempts": 0, "target": 0}
    acc, att = _chain_v010(rows, cols, D, target, max_att, int(seed), accepted_clock)
    rr, cc = np.nonzero(D0)
    return D, {"nnz": nnz, "accepted": int(acc), "attempts": int(att), "target": target,
               "retained": float(D[rr, cc].sum()) / nnz,
               "margins_exact": bool((D.sum(1) == D0.sum(1)).all() and (D.sum(0) == D0.sum(0)).all())}


def _block(n, F, rate, seed):
    rng = np.random.default_rng(seed)
    M = (rng.random((n, F)) < rate).astype(np.float64)
    M[0, :] = 1.0                     # full row
    M[:, F - 1] = 1.0                 # full last column (last bit of the last word)
    return M


def _check(D, d, D_ref, d_ref):
    assert D.dtype == np.uint8 and D.flags.c_contiguous and D.shape == D_ref.shape
    assert D.tobytes() == D_ref.tobytes()
    for k, v in d_ref.items():
        assert getattr(d, k) == v, k


@pytest.mark.parametrize("F", [1, 2, 63, 64, 65, 127, 128, 129, 300])
@pytest.mark.parametrize("sampler", SAMPLERS)
def test_bitpacked_chain_matches_v010_dense_chain(F, sampler):
    M = _block(17, F, 0.3, F)
    for seed in (1, 2, 12345):
        D, d = swap_randomise(sparse.csr_matrix(M), seed, sampler=sampler)
        _check(D, d, *_swap_randomise_v010(M, seed, sampler))
    D, d = swap_randomise(M, 9, sampler=sampler, n_mult=3.5, max_attempt_factor=2)
    _check(D, d, *_swap_randomise_v010(M, 9, sampler, n_mult=3.5, max_attempt_factor=2))


@pytest.mark.parametrize("sampler", SAMPLERS)
@pytest.mark.parametrize("n_jobs", [1, 3])
def test_bitpacked_replicates_match_v010(sampler, n_jobs):
    M = _block(40, 200, 0.15, 4)
    seeds = [5, 6, 7, 8, 9]
    got = list(swap_replicates(M, seeds, sampler=sampler, n_jobs=n_jobs))
    for (D, d), s in zip(got, seeds):
        _check(D, d, *_swap_randomise_v010(M, s, sampler))


def test_degenerate_blocks_match_v010():
    for M in (np.zeros((3, 70)), np.eye(1, 70), np.eye(1, 9) + np.eye(1, 9, 3), np.eye(5), np.ones((4, 66))):
        for sampler in SAMPLERS:
            D, d = swap_randomise(M, 3, sampler=sampler)
            _check(D, d, *_swap_randomise_v010(M, 3, sampler))


def test_pack_unpack_roundtrip_and_padding():
    rng = np.random.default_rng(0)
    for F in (1, 63, 64, 65, 200):
        D = (rng.random((9, F)) < 0.4).astype(np.uint8)
        entries, bits = S._entries(D)
        rr, cc = np.nonzero(D)
        assert entries.dtype == np.int32 and entries.shape == (rr.size, 2) and entries.flags.c_contiguous
        assert np.array_equal(entries[:, 0], rr) and np.array_equal(entries[:, 1], cc)   # np.nonzero order
        assert bits.dtype == np.uint64 and bits.shape == (9, (F + 63) // 64)
        assert S._unpack_bits(bits, F).tobytes() == D.tobytes()
        if F % 64:                                     # bits past the last column stay zero
            assert not np.any(bits[:, -1] >> np.uint64(F % 64))


def test_int64_indices_give_the_same_chain():
    M = _block(30, 130, 0.2, 7)
    entries, bits = S._entries((M > 0).astype(np.uint8))
    target = 50 * entries.shape[0]
    b32, b64 = bits.copy(), bits.copy()
    e32, e64 = entries.copy(), entries.astype(np.int64)
    out32 = S._swap_chain(e32, b32, target, target, 77, False)
    out64 = S._swap_chain(e64, b64, target, target, 77, False)
    assert out32[:2] == out64[:2]
    assert b32.tobytes() == b64.tobytes() and np.array_equal(e32, e64)
    assert np.array_equal(e32[:, 0], entries[:, 0])    # a swap never moves an entry to another row
    assert S._index_dtype(10, 2 ** 31) is np.int64 and S._index_dtype(2 ** 31 - 1, 5) is np.int32


def test_parallel_chains_match_serial_and_leave_inputs_unchanged():
    M = _block(25, 140, 0.25, 8)
    entries, bits = S._entries((M > 0).astype(np.uint8))
    e_copy, b_copy = entries.copy(), bits.copy()
    seeds = np.array([3, 4, 5, 6], np.int64)
    target = 40 * entries.shape[0]
    par, acc, att, _ = S._run_chains(bits, entries, target, target, seeds, False)
    assert np.array_equal(entries, e_copy) and np.array_equal(bits, b_copy)
    for k, s in enumerate(seeds):
        b = bits.copy()
        a_, t_, _ = S._swap_chain(entries.copy(), b, target, target, int(s), False)
        assert (acc[k], att[k]) == (a_, t_) and par[k].tobytes() == b.tobytes()


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_budgets_shorter_than_the_lookahead_match_v010(sampler):
    # budgets of 0 to a few proposals, below and around the read-ahead depth
    M = _block(12, 70, 0.1, 11)
    nnz = int((M > 0).sum())
    for target in (0, 1, 2, S._LOOKAHEAD - 1, S._LOOKAHEAD, S._LOOKAHEAD + 1, 3 * S._LOOKAHEAD):
        n_mult = target / nnz + 1e-9
        for maf in (1, 2, 100):
            D, d = swap_randomise(M, 21, sampler=sampler, n_mult=n_mult, max_attempt_factor=maf)
            _check(D, d, *_swap_randomise_v010(M, 21, sampler, n_mult=n_mult, max_attempt_factor=maf))
