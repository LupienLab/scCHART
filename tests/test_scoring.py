"""Tests of the shared scoring primitives (``scchart._scoring``).

Information theory (entropy, KL, generalized JSD identity), binarisation, the
binary fixed-margin swap null, deterministic seeds, depth downsampling (never
upsample) and the pairwise replicate agreement.
"""

import numpy as np
import pytest
from scipy import sparse

from scchart._scoring import (
    UpsamplingWarning,
    _as_binary,
    as_binary_dense,
    binarize,
    binomial_downsample,
    downsampling_fractions,
    entropy_bits,
    generalized_jsd,
    kl_bits,
    pairwise_replicate_agreement,
    seeds_from_key,
    swap_randomise,
    swap_replicates,
)


def _binary(rng, n, p, rate):
    B = (rng.random((n, p)) < rate).astype(np.float64)
    B[np.arange(n), rng.integers(0, p, n)] = 1.0  # every cell has >= 1 open feature
    return B


def _simplex(rng, n, k):
    P = rng.gamma(0.7, size=(n, k))
    return P / P.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------- information
def test_entropy_and_kl_known_values():
    assert entropy_bits(np.array([[0.5, 0.5]]))[0] == pytest.approx(1.0)
    assert entropy_bits(np.array([[1.0, 0.0]]))[0] == pytest.approx(0.0)       # 0 log 0 = 0
    assert kl_bits(np.array([[1.0, 0.0]]), np.array([0.5, 0.5]))[0] == pytest.approx(1.0)
    assert kl_bits(np.array([[0.5, 0.5]]), np.array([0.5, 0.5]))[0] == pytest.approx(0.0)


def test_nan_rows_are_unscorable():
    P = np.array([[0.5, 0.5], [np.nan, np.nan]])
    assert np.isnan(entropy_bits(P)[1]) and not np.isnan(entropy_bits(P)[0])
    assert np.isnan(kl_bits(P, np.array([0.5, 0.5]))[1])
    with pytest.raises(ValueError, match="NaN"):
        generalized_jsd(P)


def test_generalized_jsd_identity_and_non_negativity():
    rng = np.random.default_rng(0)
    for _ in range(20):
        P = _simplex(rng, int(rng.integers(2, 40)), int(rng.integers(2, 12)))
        out = generalized_jsd(P)
        assert out["identity_error"] < 1e-12
        assert np.all(out["d"] >= -1e-15)
        assert out["J"] == pytest.approx(np.mean(out["d"]))
        assert out["J"] == pytest.approx(out["H_consensus"] - out["H_cells_mean"], abs=1e-12)


def test_generalized_jsd_zero_iff_identical_rows():
    P = np.tile([[0.2, 0.3, 0.5]], (6, 1))
    assert generalized_jsd(P)["J"] == pytest.approx(0.0, abs=1e-14)
    assert generalized_jsd(np.vstack([P, [[1.0, 0.0, 0.0]]]))["J"] > 0


def test_binarize():
    X = sparse.csr_matrix(np.array([[0, 2, 0], [3, 0, 1]], float))
    np.testing.assert_array_equal(binarize(X).toarray(), [[0, 1, 0], [1, 0, 1]])


# ----------------------------------------------------------------------------- fixed-margin null
def _binarize_v010(X):
    """The v0.1.0 implementation, kept as the reference."""
    B = sparse.csr_matrix(X, dtype=np.float64, copy=True)
    B.data = (B.data > 0).astype(np.float64)
    B.eliminate_zeros()
    return B


def _same_csr(a, b):
    return (type(a) is type(b) and a.shape == b.shape and a.dtype == b.dtype
            and all(x.dtype == y.dtype and x.tobytes() == y.tobytes()
                    for x, y in ((a.data, b.data), (a.indices, b.indices), (a.indptr, b.indptr))))


@pytest.mark.parametrize("fmt", ["csr", "csc", "coo", "dense"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.uint32, np.int64])
def test_binarize_and_as_binary_match_v010(fmt, dtype):
    rng = np.random.default_rng(4)
    X = (rng.integers(-2 if np.issubdtype(dtype, np.signedinteger) or dtype in (np.float64, np.float32) else 0, 4,
                      (30, 50)) * (rng.random((30, 50)) < 0.3)).astype(dtype)
    M = X if fmt == "dense" else getattr(sparse, f"{fmt}_matrix")(X)
    if fmt == "csr":
        M.data[:3] = 0  # explicit zeros
    before = M.copy()
    want = _binarize_v010(M)
    assert _same_csr(binarize(M), want)
    assert _same_csr(_as_binary(M), want)
    assert _same_csr(_as_binary(want), want) and _as_binary(want) is want  # already binary: not copied
    if fmt == "dense":
        np.testing.assert_array_equal(M, before)
    else:
        assert (M != before).nnz == 0  # input not modified


def test_binarize_nan_and_duplicates_match_v010():
    csr = sparse.csr_matrix((np.array([np.nan, 2.0, -1.0, 1.0, 1.0]), np.array([0, 1, 2, 0, 0]),
                             np.array([0, 3, 5])), shape=(2, 3))
    assert _same_csr(binarize(csr), _binarize_v010(csr))
    assert _same_csr(_as_binary(csr), _binarize_v010(csr))
    assert _as_binary(sparse.csr_array(np.eye(3))) is not None  # csr_array input is converted, not returned
    assert type(_as_binary(sparse.csr_array(np.eye(3)))) is sparse.csr_matrix


def _as_binary_dense_v010(B):
    """The v0.1.0 implementation (dense float copy of the input), kept as the reference."""
    D = (B.toarray() > 0) if sparse.issparse(B) else (np.asarray(B) > 0)
    return np.ascontiguousarray(D.astype(np.uint8))


@pytest.mark.parametrize("fmt", ["csr", "csc", "coo", "dense"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64])
def test_as_binary_dense_matches_v010(fmt, dtype):
    rng = np.random.default_rng(3)
    X = rng.integers(-1, 3, (40, 70)).astype(dtype) * (rng.random((40, 70)) < 0.2)
    M = X if fmt == "dense" else getattr(sparse, f"{fmt}_matrix")(X)
    got, want = as_binary_dense(M), _as_binary_dense_v010(M)
    assert got.dtype == np.uint8 and got.flags.c_contiguous
    assert got.tobytes() == want.tobytes()


def test_as_binary_dense_non_canonical_sparse_matches_v010():
    # duplicate entries are summed (as toarray() does); explicit zeros, negatives and NaN are closed
    coo = sparse.coo_matrix((np.array([1.0, -1.0, 2.0, 0.0, np.nan, 3.0, -4.0, 5.0]),
                             (np.array([0, 0, 1, 1, 2, 2, 3, 3]), np.array([1, 1, 2, 3, 0, 0, 2, 2]))),
                            shape=(4, 4))
    csr = sparse.csr_matrix((np.array([2.0, -2.0, 1.0, 0.0, 1.0]), np.array([1, 1, 0, 2, 0]),
                             np.array([0, 2, 5])), shape=(2, 3))
    assert not csr.has_canonical_format
    for M in (coo, csr):
        assert as_binary_dense(M).tobytes() == _as_binary_dense_v010(M).tobytes()
    assert csr.data.tolist() == [2.0, -2.0, 1.0, 0.0, 1.0]  # input not modified


def test_swap_null_preserves_margins_and_is_deterministic():
    rng = np.random.default_rng(5)
    B = _binary(rng, 40, 60, 0.2)
    D1, d1 = swap_randomise(B, 123)
    D2, _ = swap_randomise(B, 123)
    np.testing.assert_array_equal(D1, D2)
    np.testing.assert_array_equal(D1.sum(axis=1), B.sum(axis=1))
    np.testing.assert_array_equal(D1.sum(axis=0), B.sum(axis=0))
    assert d1.margins_exact and d1.complete and d1.sampler == "proposal_clock"
    assert d1.attempts == 100 * int(B.sum())   # proposal clock counts proposals
    assert d1.retained < 0.9  # the chain actually moved


def test_accepted_swap_sampler_counts_accepted_swaps():
    rng = np.random.default_rng(5)
    B = _binary(rng, 40, 60, 0.2)
    D1, d1 = swap_randomise(B, 123, sampler="accepted_swap")
    np.testing.assert_array_equal(D1.sum(axis=1), B.sum(axis=1))
    np.testing.assert_array_equal(D1.sum(axis=0), B.sum(axis=0))
    assert d1.margins_exact and d1.complete and d1.sampler == "accepted_swap"
    assert d1.accepted == 10 * int(B.sum())


def test_parallel_replicates_bit_identical_to_serial():
    rng = np.random.default_rng(6)
    B = _binary(rng, 30, 45, 0.15)
    seeds = [11, 22, 33, 44]
    par = [D.copy() for D, _ in swap_replicates(B, seeds, n_jobs=4)]
    ser = [swap_randomise(B, s)[0] for s in seeds]
    for a, b in zip(par, ser):
        np.testing.assert_array_equal(a, b)


def test_degenerate_block_is_returned_unchanged():
    B = np.zeros((3, 4))
    B[0, 1] = 1
    D, diag = swap_randomise(B, 1)
    np.testing.assert_array_equal(D, B)
    assert diag.nnz == 1 and diag.accepted == 0


def test_seeds_from_key_is_deterministic():
    a = seeds_from_key((1, 2), 5)
    np.testing.assert_array_equal(a, seeds_from_key((1, 2), 5))
    assert len(set(a.tolist())) == 5
    assert not np.array_equal(a, seeds_from_key((1, 3), 5))
    np.testing.assert_array_equal(seeds_from_key(7, 3), seeds_from_key((7,), 3))


# ----------------------------------------------------------------------------- depth
def test_downsampling_never_upsamples():
    depth = np.r_[np.full(10, 100.0), np.full(10, 40.0)]
    g = np.repeat(["hi", "lo"], 10)
    with pytest.warns(UpsamplingWarning):
        fr = downsampling_fractions(depth, g, 60.0)
    assert fr.loc["hi", "q"] == pytest.approx(0.6)
    assert fr.loc["lo", "q"] == 1.0 and not fr.loc["lo", "reaches_D_star"]
    X = sparse.csr_matrix(np.random.default_rng(0).poisson(3, (20, 30)).astype(float))
    Xd = binomial_downsample(X, 1.7, np.random.default_rng(1))  # q clipped to 1
    np.testing.assert_array_equal(Xd.toarray(), X.toarray())
    Xh = binomial_downsample(X, 0.5, np.random.default_rng(1))
    assert np.all(Xh.toarray() <= X.toarray())


def test_downsampling_requires_integer_counts():
    with pytest.raises(ValueError, match="integer"):
        binomial_downsample(sparse.csr_matrix([[0.5, 1.0]]), 0.5, np.random.default_rng(0))


def test_pairwise_replicate_agreement_is_descriptive():
    import pandas as pd

    tab = pairwise_replicate_agreement(pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.0, 3.0, 1.0]}))
    assert tab.loc[0, "frac_first_greater"] == pytest.approx(2 / 3)
    assert not tab.loc[0, "replicates_unanimous"]


def test_downsampling_fractions_label_length():
    with pytest.raises(ValueError, match="one label per cell"):
        downsampling_fractions(np.ones(5), ["a"] * 4, 1.0)
