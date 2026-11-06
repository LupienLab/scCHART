"""Shared computational utilities for scCHART.

This module provides the mathematical and computational functions used by
the structural heterogeneity and regulatory programme heterogeneity axes.

The shared utilities include:

- Shannon entropy, Kullback–Leibler divergence, and generalized
  Jensen–Shannon divergence, expressed in bits.
- Conversion of accessibility counts to binary presence/absence values.
- Fixed-margin randomization of binary accessibility matrices to
  construct technical null distributions.
- Count-depth standardization for cross-population comparisons.

Information-theoretic calculations
----------------------------------
For a population containing n probability distributions p_i, the
population consensus is their arithmetic mean:

    pbar = (1/n) * sum_i p_i

The equal-weight generalized Jensen–Shannon divergence is:

    J = (1/n) * sum_i KL(p_i || pbar)
      = H(pbar) - (1/n) * sum_i H(p_i)

Here, H denotes Shannon entropy and KL denotes Kullback–Leibler
divergence. All logarithms use base 2, and results are expressed in bits.

Zero-probability contributions follow the convention 0 log 0 = 0.
Rows containing NaN values are considered unscorable.

Fixed-margin technical null
---------------------------
The technical null randomizes a binary cell-by-feature accessibility
matrix using degree-preserving 2 x 2 swaps.

Each valid swap preserves both the number of accessible features per
cell and the number of cells accessible at each feature.

Two sampling procedures are available:

- "proposal_clock" (default): performs a fixed number of proposed
  swaps, including rejected proposals. Its symmetric transition
  kernel has a uniform stationary distribution over matrices with
  the observed row and column sums.

- "accepted_swap": performs a fixed target number of accepted swaps,
  subject to an attempt limit. Its stationary distribution generally
  differs from the uniform fixed-margin distribution.

The default randomization budgets are 100 proposals or 10 accepted
swaps per occupied entry, respectively, with 20 null replicates.

These budgets define computational effort; they do not guarantee
adequate Markov-chain mixing.

Depth standardization
---------------------
For comparisons across populations with different sequencing depths,
the module supports binomial downsampling toward a declared reference
depth D*.

For each population G, the retention fraction is:

    q_G = min(1, D* / median_depth_G)

Populations below the reference depth are not upsampled.
"""

from __future__ import annotations

import operator
import warnings
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd
import numba
from numba import njit, prange
from scipy import sparse


# ---------------------------------------------------------------------------
# Information theory (base 2)
# ---------------------------------------------------------------------------


def entropy_bits(P: np.ndarray) -> np.ndarray:
    """Compute the row-wise Shannon entropy in bits.

    For each probability distribution p_i, calculate
    H(p_i) = -sum_k p_ik * log2(p_ik). A point mass has zero entropy;
    a uniform distribution over K categories has entropy log2(K).

    Parameters
    ----------
    P : numpy.ndarray
        An (n, K) matrix of probability distributions. A one-dimensional
        input is treated as a single row.

    Returns
    -------
    numpy.ndarray
        One entropy value per row, in bits.

    Notes
    -----
    Zero probabilities contribute zero (0 log2 0 = 0). Rows containing
    NaN return NaN. This function assumes non-negative, normalized
    probabilities; it does not validate or normalize the input.
    """
    P = np.atleast_2d(np.asarray(P, dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(P > 0, P * np.log2(np.where(P > 0, P, 1.0)), 0.0)
    h = -t.sum(axis=1)
    h[np.isnan(P).any(axis=1)] = np.nan
    return h


def kl_bits(P: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute each row's Kullback-Leibler divergence from q, in bits.

    For each row p_i, calculate

        KL(p_i || q) = sum_k p_ik * log2(p_ik / q_k).

    For valid distributions the result is non-negative and is zero
    when p_i equals q.

    Parameters
    ----------
    P : numpy.ndarray
        An (n, K) matrix of probability distributions. A one-dimensional
        input is treated as a single row.
    q : numpy.ndarray
        Reference distribution of length K. It must have positive
        probability wherever any row in P has positive probability.

    Returns
    -------
    numpy.ndarray
        One KL divergence per row, in bits.

    Notes
    -----
    Zero entries in P contribute zero; rows containing NaN return NaN.
    Inputs are not normalized or validated. In particular, if P has
    positive mass where q is zero, the true KL divergence is infinite,
    but this implementation returns a finite value instead. The support
    requirement holds when q is the mean of the valid rows in P.
    """
    P = np.atleast_2d(np.asarray(P, dtype=np.float64))
    q = np.asarray(q, dtype=np.float64).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(P > 0, P * np.log2(np.where(P > 0, P, 1.0) / np.where(q > 0, q, 1.0)), 0.0)
    d = t.sum(axis=1)
    d[np.isnan(P).any(axis=1)] = np.nan
    return d


def generalized_jsd(P: np.ndarray) -> dict:
    """Compute the equal-weight generalized Jensen-Shannon divergence.

    Given n probability distributions p_i, define their consensus as
    pbar = mean_i p_i. The per-row contribution is
    d_i = KL(p_i || pbar), and the population divergence is

        J = mean_i d_i = H(pbar) - mean_i H(p_i).

    All logarithms use base 2, so divergences are in bits.

    Parameters
    ----------
    P : numpy.ndarray
        An (n, K) matrix of valid, non-negative, normalized
        probability distributions without NaN values.

    Returns
    -------
    dict
        Dictionary with the following entries:

        - ``d`` : row-wise KL divergences from the consensus, in bits.
        - ``J`` : mean of ``d``, in bits.
        - ``pbar`` : consensus (mean) probability distribution.
        - ``H_consensus`` : entropy of ``pbar``, in bits.
        - ``H_cells_mean`` : mean entropy of the input rows, in bits.
        - ``identity_error`` : absolute difference between the KL and
          entropy-difference expressions for J (a numerical check).

    Raises
    ------
    ValueError
        If P has no rows or contains NaN values.

    Notes
    -----
    Each row receives equal weight. Non-negativity and normalization
    are assumed rather than explicitly validated.
    """
    P = np.atleast_2d(np.asarray(P, dtype=np.float64))
    if P.shape[0] == 0:
        raise ValueError("generalized_jsd needs at least one row")
    if np.isnan(P).any():
        raise ValueError("generalized_jsd received NaN rows; drop unscorable rows first")
    pbar = P.mean(axis=0)
    d = kl_bits(P, pbar)
    J = float(d.mean())
    H_bar = float(entropy_bits(pbar[None, :])[0])
    H_mean = float(entropy_bits(P).mean())
    return {"d": d, "J": J, "pbar": pbar, "H_consensus": H_bar, "H_cells_mean": H_mean,
            "identity_error": abs(J - (H_bar - H_mean))}


# ---------------------------------------------------------------------------
# Binary accessibility
# ---------------------------------------------------------------------------


def binarize(X) -> sparse.csr_matrix:
    """Convert a cell-by-feature matrix to binary accessibility.

    Each entry is transformed as B_ij = 1 if X_ij > 0, and 0 otherwise.

    Parameters
    ----------
    X : array-like or scipy.sparse matrix
        Accessibility counts or binary presence/absence values.

    Returns
    -------
    scipy.sparse.csr_matrix
        A float64 binary CSR matrix with the same shape as X and no
        explicitly stored zeros.

    Notes
    -----
    Negative and zero values become zero. The input is copied rather
    than modified; no filtering or normalization is performed.
    """
    B = sparse.csr_matrix(X, dtype=np.float64, copy=True)
    np.greater(B.data, 0, out=B.data, casting="unsafe")  # 1.0 where X > 0, else 0.0, in place
    B.eliminate_zeros()
    return B


def _as_binary(X) -> sparse.csr_matrix:
    """Return ``binarize(X)``, or ``X`` itself when it is already that matrix.

    ``X`` is returned unchanged (not copied) when it is exactly a float64
    ``csr_matrix`` (not a subclass such as SciPy 1.10's ``csr_array``) whose
    stored values are all 1.0, which ``binarize`` would reproduce exactly.
    For internal use where the result is only read.
    """
    if type(X) is sparse.csr_matrix and X.dtype == np.float64 and bool(np.all(X.data == 1.0)):
        return X
    return binarize(X)


def check_labels(labels, n_cells: int, *, name: str = "groups") -> np.ndarray:
    """Convert population labels to a one-dimensional string array.

    Parameters
    ----------
    labels : array-like
        One population label per cell.
    n_cells : int
        Number of cells (rows) in the corresponding data matrix.
    name : str, optional
        Argument name to use in the error message. Default ``"groups"``.

    Returns
    -------
    numpy.ndarray
        String labels of shape (n_cells,).

    Raises
    ------
    ValueError
        If labels are not one-dimensional or their length differs
        from n_cells.

    Notes
    -----
    Missing values are not rejected; they are converted to strings
    such as ``"nan"`` or ``"None"``.
    """
    g = np.asarray(labels).astype(str)
    if g.ndim != 1 or g.size != int(n_cells):
        raise ValueError(f"{name} must have one label per cell: got shape {g.shape}, "
                         f"expected ({int(n_cells)},)")
    return g


# ---------------------------------------------------------------------------
# Binary fixed-margin null
# ---------------------------------------------------------------------------

#: Fixed number of proposals; rejected proposals are self-loops.
SAMPLER_PROPOSAL_CLOCK = "proposal_clock"
#: Chain stops after a fixed number of accepted swaps.
SAMPLER_ACCEPTED_SWAP = "accepted_swap"
#: Default null sampler.
DEFAULT_SAMPLER = SAMPLER_PROPOSAL_CLOCK
SAMPLERS = (SAMPLER_PROPOSAL_CLOCK, SAMPLER_ACCEPTED_SWAP)
#: Proposal-clock budget per replicate = PROPOSAL_MULTIPLIER * nnz.
PROPOSAL_MULTIPLIER = 100
#: Accepted-swap target per replicate = SWAP_MULTIPLIER * nnz.
SWAP_MULTIPLIER = 10
#: Default number of null replicates (both samplers).
N_NULL_REPLICATES = 20
#: Accepted-swap attempt cap relative to target accepted swaps.
MAX_ATTEMPT_FACTOR = 1000


def check_sampler(sampler: str) -> str:
    """Validate a fixed-margin null sampler name.

    Parameters
    ----------
    sampler : str
        Either ``"proposal_clock"`` or ``"accepted_swap"``.

    Returns
    -------
    str
        The sampler name, unchanged.

    Raises
    ------
    ValueError
        If the name is not supported.
    """
    if sampler not in SAMPLERS:
        raise ValueError(f"unknown null sampler {sampler!r}; choose one of {SAMPLERS}")
    return sampler


def default_multiplier(sampler: str = DEFAULT_SAMPLER) -> float:
    """Return the default randomization budget per occupied entry.

    ``"proposal_clock"`` uses 100 proposals per occupied entry;
    ``"accepted_swap"`` targets 10 accepted swaps per occupied entry.
    These budgets do not guarantee Markov-chain mixing.

    Parameters
    ----------
    sampler : str, optional
        Fixed-margin sampler. Default ``DEFAULT_SAMPLER``.

    Returns
    -------
    int
        Budget multiplier for the selected sampler
        (``PROPOSAL_MULTIPLIER`` or ``SWAP_MULTIPLIER``).

    Raises
    ------
    ValueError
        If the sampler name is unsupported.
    """
    return PROPOSAL_MULTIPLIER if check_sampler(sampler) == SAMPLER_PROPOSAL_CLOCK else SWAP_MULTIPLIER


def check_n_jobs(n_jobs) -> int:
    """Validate ``n_jobs`` and return the number of chains run at once.

    Parameters
    ----------
    n_jobs : int
        A positive integer, or ``-1`` for all numba threads
        (``numba.get_num_threads()``).

    Returns
    -------
    int
        Chains run at once: ``n_jobs`` capped at ``numba.get_num_threads()``
        (more cannot run concurrently; the cap only saves memory). Results
        do not depend on it.

    Raises
    ------
    TypeError
        If ``n_jobs`` is not an integer (``bool`` is rejected).
    ValueError
        If ``n_jobs`` is 0 or below -1.
    """
    if isinstance(n_jobs, (bool, np.bool_)):
        raise TypeError(f"n_jobs must be an integer, got {n_jobs!r}")
    try:
        n = operator.index(n_jobs)
    except TypeError:
        raise TypeError(f"n_jobs must be an integer, got {n_jobs!r}") from None
    threads = max(1, int(numba.get_num_threads()))
    if n == -1:
        return threads
    if n < 1:
        raise ValueError(f"n_jobs must be a positive integer or -1 (all numba threads), got {n}")
    return min(n, threads)


def _index_dtype(n_rows: int, n_cols: int):
    """Smallest integer type used for entry indices: int32 when it fits, else int64."""
    return np.int32 if max(int(n_rows), int(n_cols)) < 2 ** 31 else np.int64


#: Proposals drawn ahead by the chain (a power of two; see :func:`_chain`).
_LOOKAHEAD = 16


@njit(cache=True, inline="always")
def _bit(c):  # pragma: no cover - numba
    """Mask of column ``c`` within its 64-bit word."""
    return np.uint64(1) << np.uint64(c & 63)


@njit(cache=True)
def _pack_bits(entries, n_rows, n_cols):  # pragma: no cover - numba
    """Bit-packed binary matrix with ones at the ``(row, col)`` rows of ``entries``.

    Row ``i`` is ``ceil(n_cols / 64)`` ``uint64`` words; column ``c`` is bit
    ``c % 64`` of word ``c // 64``. Bits past ``n_cols`` stay zero.
    """
    bits = np.zeros((n_rows, (n_cols + 63) >> 6), np.uint64)
    for e in range(entries.shape[0]):
        c = entries[e, 1]
        bits[entries[e, 0], c >> 6] |= _bit(c)
    return bits


@njit(cache=True)
def _unpack_bits(bits, n_cols):  # pragma: no cover - numba
    """Dense ``uint8`` matrix of a bit-packed matrix (see :func:`_pack_bits`)."""
    n_rows = bits.shape[0]
    D = np.zeros((n_rows, n_cols), np.uint8)
    for i in range(n_rows):
        for w in range(bits.shape[1]):
            word = bits[i, w]
            if word == 0:
                continue
            base = w << 6
            for k in range(64):
                if (word >> np.uint64(k)) & np.uint64(1):
                    D[i, base + k] = 1
    return D


@njit(cache=True)
def _chain(entries, bits, target, max_attempts, accepted_clock):  # pragma: no cover - numba
    """Run one fixed-margin swap chain on a bit-packed matrix in place.

    Each proposal draws two occupied entries e1 = (i, a) and e2 = (j, b)
    uniformly with replacement (E**2 ordered pairs). If i != j, a != b and
    entries (i, b) and (j, a) are both 0, the 2 x 2 swap is applied;
    otherwise the state is unchanged (a self-loop). ``entries`` is updated
    so that it always lists the occupied entries.

    Parameters
    ----------
    entries : numpy.ndarray
        ``(E, 2)`` array of the ``(row, col)`` of each occupied entry (int32,
        or int64 for indices of ``2**31`` or more; modified in place). Row and
        column are stored side by side so that one memory access fetches
        both.
    bits : numpy.ndarray
        Bit-packed binary matrix (see :func:`_pack_bits`; modified in place).
    target : int
        Number of accepted swaps at which the accepted-swap clock stops.
        Ignored when ``accepted_clock`` is False.
    max_attempts : int
        Maximum number of proposals.
    accepted_clock : bool
        False for the proposal clock (stop after ``max_attempts``
        proposals); True for the accepted-swap clock (stop after
        ``target`` accepted swaps or ``max_attempts`` proposals,
        whichever comes first).

    Returns
    -------
    accepted : int
        Number of accepted swaps.
    attempts : int
        Number of proposals made.
    touched : int
        Checksum of the entries read ahead (see Notes). It has no meaning;
        it is returned only so that the compiler keeps those reads.

    Notes
    -----
    The numba random state must already be seeded. The draws, tests and
    updates are those of the v0.1.0 dense-matrix chain, so the result is
    the same matrix for the same seed.

    The chain is limited by memory latency: the entries drawn are random
    positions in arrays far larger than the caches. The two draws of
    proposal ``t + _LOOKAHEAD`` are therefore made at proposal ``t`` and
    their entries read early, so that they are cached when that proposal
    is evaluated. Proposals are still evaluated one at a time, in order,
    with their own draws. The accepted-swap clock can stop with up to
    ``_LOOKAHEAD`` drawn proposals unused; they are discarded (every chain
    seeds the random state afresh, so this is not observable).
    """
    E = entries.shape[0]
    mask = _LOOKAHEAD - 1
    ahead1 = np.empty(_LOOKAHEAD, np.int64)
    ahead2 = np.empty(_LOOKAHEAD, np.int64)
    drawn = 0
    while drawn < _LOOKAHEAD and drawn < max_attempts:
        ahead1[drawn] = np.random.randint(E)
        ahead2[drawn] = np.random.randint(E)
        drawn += 1
    touched = np.uint64(0)
    acc = 0
    att = 0
    while att < max_attempts:
        if accepted_clock and acc >= target:
            break
        slot = att & mask
        e1 = ahead1[slot]
        e2 = ahead2[slot]
        if drawn < max_attempts:          # draw proposal att + _LOOKAHEAD, read its entries early
            f1 = np.random.randint(E)
            f2 = np.random.randint(E)
            ahead1[slot] = f1
            ahead2[slot] = f2
            drawn += 1
            touched += np.uint64(entries[f1, 0] + entries[f1, 1] + entries[f2, 0] + entries[f2, 1])
        att += 1
        i = entries[e1, 0]
        a = entries[e1, 1]
        j = entries[e2, 0]
        b = entries[e2, 1]
        if i == j or a == b:
            continue
        wa = a >> 6
        wb = b >> 6
        ma = _bit(a)
        mb = _bit(b)
        if (bits[i, wb] & mb) != 0 or (bits[j, wa] & ma) != 0:
            continue
        bits[i, wa] ^= ma
        bits[j, wb] ^= mb
        bits[i, wb] |= mb
        bits[j, wa] |= ma
        entries[e1, 1] = b
        entries[e2, 1] = a
        acc += 1
    return acc, att, touched


@njit(cache=True)
def _swap_chain(entries, bits, target, max_attempts, seed, accepted_clock):  # pragma: no cover - numba
    """Seed the numba random state and run :func:`_chain` once.

    Arguments and return values are as in :func:`_chain`, with the
    additional integer ``seed``.
    """
    np.random.seed(seed)
    return _chain(entries, bits, target, max_attempts, accepted_clock)


@njit(parallel=True, cache=True)
def _run_chains(bits0, entries0, target, max_attempts, seeds, accepted_clock):  # pragma: no cover - numba
    """Run one independent chain per seed in parallel.

    Each chain starts from a copy of ``bits0`` and ``entries0`` and is
    seeded with its own entry of ``seeds``, so its result matches the
    serial chain for the same seed.

    Returns
    -------
    bits : numpy.ndarray
        Randomized bit-packed matrices, shape (n_seeds, *bits0.shape).
    acc : numpy.ndarray
        Accepted swaps per chain.
    att : numpy.ndarray
        Proposals per chain.
    touched : numpy.ndarray
        Read-ahead checksums (see :func:`_chain`); meaningless.
    """
    Bn = seeds.size
    out = np.empty((Bn, bits0.shape[0], bits0.shape[1]), np.uint64)
    acc = np.zeros(Bn, np.int64)
    att = np.zeros(Bn, np.int64)
    touched = np.zeros(Bn, np.uint64)
    for b in prange(Bn):
        bits = out[b]
        bits[:, :] = bits0
        entries = entries0.copy()
        np.random.seed(seeds[b])
        a_, t_, k_ = _chain(entries, bits, target, max_attempts, accepted_clock)
        acc[b] = a_
        att[b] = t_
        touched[b] = k_
    return out, acc, att, touched


@dataclass(frozen=True)
class SwapDiagnostics:
    """Per-replicate diagnostics of one fixed-margin chain.

    Attributes
    ----------
    seed : int
        Chain seed.
    nnz : int
        Number of occupied entries in the input block.
    target : int
        Budget in the sampler's own clock: proposals for the proposal
        clock, accepted swaps for the accepted-swap sampler.
    accepted : int
        Number of accepted swaps.
    attempts : int
        Number of proposals made.
    complete : bool
        True when the chain used its full budget. Always True for the
        proposal clock; accepted-swap chains can stop at the attempt
        cap first.
    margins_exact : bool
        True when every row and column sum equals that of the input.
    retained : float
        Fraction of the input's occupied entries that are still occupied.
    sampler : str
        ``"proposal_clock"`` (default) or ``"accepted_swap"``.
    """

    seed: int
    nnz: int
    target: int
    accepted: int
    attempts: int
    complete: bool
    margins_exact: bool
    retained: float
    sampler: str = DEFAULT_SAMPLER

    @property
    def acceptance(self) -> float:
        """Fraction of proposals that were accepted swaps."""
        return self.accepted / max(self.attempts, 1)

    @property
    def proposals(self) -> int:
        """Number of proposals made (alias of ``attempts``)."""
        return self.attempts


def as_binary_dense(B) -> np.ndarray:
    """Return a dense binary copy of a matrix.

    Parameters
    ----------
    B : array-like or scipy.sparse matrix
        Binary or count matrix.

    Returns
    -------
    numpy.ndarray
        C-contiguous ``uint8`` array with entries ``1[B > 0]``.

    Notes
    -----
    A sparse input is written straight into the ``uint8`` result from its
    stored entries, without an intermediate dense array of the input's
    dtype. Duplicate entries are summed first, as ``toarray()`` does.
    """
    if sparse.issparse(B):
        B = sparse.csr_matrix(B)
        if not B.has_canonical_format:
            B = B.copy()
            B.sum_duplicates()
        D = np.zeros(B.shape, dtype=np.uint8)
        rows = np.repeat(np.arange(B.shape[0], dtype=np.intp), np.diff(B.indptr))
        pos = B.data > 0
        if pos.all():
            D[rows, B.indices] = 1
        else:
            D[rows[pos], B.indices[pos]] = 1
        return D
    D = np.asarray(B) > 0
    return np.ascontiguousarray(D.astype(np.uint8))



def _entries(D0):
    """Chain inputs for a dense binary block: ``(E, 2)`` entry array and bit-packed copy.

    The entries are listed in ``np.nonzero`` order (row-major), as in v0.1.0,
    because the chain draws entries by position.
    """
    rows, cols = np.nonzero(D0)
    entries = np.empty((rows.size, 2), _index_dtype(*D0.shape))
    entries[:, 0] = rows
    entries[:, 1] = cols
    return entries, _pack_bits(entries, D0.shape[0], D0.shape[1])

def seed_key(seed) -> tuple:
    """Normalize an integer or a sequence of integers to a seed-key tuple.

    Parameters
    ----------
    seed : int or sequence of int
        Seed or seed key.

    Returns
    -------
    tuple of int
        ``(seed,)`` for a scalar, otherwise the elements as integers.
    """
    return (int(seed),) if np.isscalar(seed) else tuple(int(x) for x in seed)


def seeds_from_key(seed_key: int | Sequence[int], n: int) -> np.ndarray:
    """Derive deterministic replicate seeds from a seed key.

    Replicate b receives the seed
    ``numpy.random.default_rng((*seed_key, b)).integers(2**31 - 1)``.
    This scheme is used by the structural null calibration.

    Parameters
    ----------
    seed_key : int or sequence of int
        Seed key identifying the calibration.
    n : int
        Number of replicate seeds.

    Returns
    -------
    numpy.ndarray
        ``int64`` array of ``n`` seeds.
    """
    key = (seed_key,) if np.isscalar(seed_key) else tuple(seed_key)
    return np.array([int(np.random.default_rng((*key, b)).integers(2 ** 31 - 1)) for b in range(n)],
                    dtype=np.int64)


def _diagnostics(D0, D, seed, nnz, target, acc, att, r0, c0, sampler) -> SwapDiagnostics:
    """Assemble the :class:`SwapDiagnostics` of one finished chain.

    ``D0`` is the input block, ``D`` the randomized block, and ``r0`` and
    ``c0`` the input row and column sums.
    """
    exact = bool((D.sum(axis=1) == r0).all() and (D.sum(axis=0) == c0).all())
    if nnz:
        rr, cc = np.nonzero(D0)
        retained = float(D[rr, cc].sum()) / nnz
    else:
        retained = 1.0
    done = (att == target) if sampler == SAMPLER_PROPOSAL_CLOCK else (acc == target)
    return SwapDiagnostics(int(seed), int(nnz), int(target), int(acc), int(att),
                           bool(done), exact, retained, sampler)


def _budget(nnz: int, sampler: str, n_mult, max_attempt_factor) -> tuple[int, int, bool]:
    """Return ``(target, max_attempts, accepted_clock)`` for one chain.

    The target is ``int(m * nnz)`` with ``m`` the budget multiplier. The
    proposal clock uses ``target`` proposals; the accepted-swap clock
    targets ``target`` accepted swaps with at most
    ``max_attempt_factor * max(target, 1)`` proposals.
    """
    m = default_multiplier(sampler) if n_mult is None else float(n_mult)
    target = int(m * nnz)
    if sampler == SAMPLER_PROPOSAL_CLOCK:
        return target, target, False
    return target, int(max_attempt_factor) * max(target, 1), True


def swap_randomise(B, seed: int, *, sampler: str = DEFAULT_SAMPLER, n_mult: Optional[float] = None,
                   max_attempt_factor: int = MAX_ATTEMPT_FACTOR) -> tuple[np.ndarray, SwapDiagnostics]:
    """Generate one fixed-margin replicate of a binary block (serial chain).

    Parameters
    ----------
    B : array-like or scipy.sparse matrix
        Binary (or count; binarized as ``B > 0``) cells x features block.
        It is copied; the input is never modified.
    seed : int
        Chain seed (numba ``np.random.seed``).
    sampler : {"proposal_clock", "accepted_swap"}, optional
        ``"proposal_clock"`` (default): run exactly ``int(n_mult * nnz)``
        proposals; rejected proposals are self-loops. Its stationary
        distribution is uniform over the matrices with the input's margins.
        ``"accepted_swap"``: stop after ``int(n_mult * nnz)`` accepted
        swaps, with at most ``max_attempt_factor`` times that many
        proposals. Its stationary distribution is generally not uniform.
    n_mult : float, optional
        Budget per occupied entry. Default 100 proposals (proposal clock)
        or 10 accepted swaps (accepted-swap sampler).
    max_attempt_factor : int, optional
        Proposal cap relative to the accepted-swap target. Used only by
        the accepted-swap sampler. Default ``MAX_ATTEMPT_FACTOR``.

    Returns
    -------
    D : numpy.ndarray
        Randomized dense ``uint8`` block with the same row and column sums.
    diagnostics : SwapDiagnostics
        Chain diagnostics.

    Raises
    ------
    ValueError
        If the sampler name is unsupported.
    AssertionError
        If the margins are not preserved. This is a hard invariant of the
        swap move and is not expected to occur.

    Notes
    -----
    A block with fewer than two occupied entries is returned unchanged,
    with a budget of zero. The budget fixes computational effort; it does
    not guarantee that the chain has mixed.
    """
    check_sampler(sampler)
    D0 = as_binary_dense(B)
    entries, bits = _entries(D0)
    nnz = entries.shape[0]
    r0, c0 = D0.sum(axis=1), D0.sum(axis=0)
    target, max_att, accepted_clock = _budget(nnz, sampler, n_mult, max_attempt_factor)
    if nnz < 2:
        D = D0.copy()
        return D, _diagnostics(D0, D, seed, nnz, 0, 0, 0, r0, c0, sampler)
    acc, att, _ = _swap_chain(entries, bits, target, max_att, int(seed), accepted_clock)
    D = _unpack_bits(bits, D0.shape[1])
    diag = _diagnostics(D0, D, seed, nnz, target, acc, att, r0, c0, sampler)
    assert diag.margins_exact, "fixed-margin swap chain changed a margin"
    return D, diag


def swap_replicates(B, seeds: Sequence[int], *, sampler: str = DEFAULT_SAMPLER,
                    n_mult: Optional[float] = None, max_attempt_factor: int = MAX_ATTEMPT_FACTOR,
                    n_jobs: int = 1) -> Iterator[tuple[np.ndarray, SwapDiagnostics]]:
    """Yield fixed-margin replicates of ``B``, one per seed.

    Parameters
    ----------
    B : array-like or scipy.sparse matrix
        Binary (or count; binarized as ``B > 0``) cells x features block.
    seeds : sequence of int
        One chain seed per replicate.
    sampler : {"proposal_clock", "accepted_swap"}, optional
        Fixed-margin sampler, as in :func:`swap_randomise`.
    n_mult : float, optional
        Budget per occupied entry, as in :func:`swap_randomise`.
    max_attempt_factor : int, optional
        Accepted-swap proposal cap, as in :func:`swap_randomise`.
    n_jobs : int, optional
        Number of chains run in parallel per batch (numba ``prange``):
        a positive integer, or -1 for all numba threads. Values above
        ``numba.get_num_threads()`` are capped at it. Default 1 (serial).

    Yields
    ------
    D : numpy.ndarray
        Randomized dense ``uint8`` block.
    diagnostics : SwapDiagnostics
        Chain diagnostics.

    Raises
    ------
    TypeError, ValueError
        If ``n_jobs`` is invalid (see :func:`check_n_jobs`). The sampler and
        ``n_jobs`` are checked when the function is called, before the first
        replicate is requested.
    ValueError
        If the sampler name is unsupported.
    AssertionError
        If a chain does not preserve the margins (not expected to occur).

    Notes
    -----
    Each replicate equals the output of :func:`swap_randomise` for the
    same seed, whatever the value of ``n_jobs``. The chains keep the matrix
    bit-packed (``n_cells * n_features / 8`` bytes each) with 8 bytes per
    occupied entry per chain (16 when an index exceeds ``2**31``), plus the
    dense ``uint8`` input and each yielded block
    (``n_cells * n_features`` bytes each).
    """
    check_sampler(sampler)
    step = check_n_jobs(n_jobs)
    return _swap_replicates(B, seeds, sampler, n_mult, max_attempt_factor, step)


def _swap_replicates(B, seeds, sampler, n_mult, max_attempt_factor, step):
    """Generator behind :func:`swap_replicates` (arguments already validated)."""
    D0 = as_binary_dense(B)
    entries0, bits0 = _entries(D0)
    nnz = entries0.shape[0]
    r0, c0 = D0.sum(axis=1), D0.sum(axis=0)
    target, max_att, accepted_clock = _budget(nnz, sampler, n_mult, max_attempt_factor)
    seeds = np.asarray(seeds, dtype=np.int64)
    if nnz < 2:
        for s in seeds:
            yield D0.copy(), _diagnostics(D0, D0, s, nnz, 0, 0, 0, r0, c0, sampler)
        return
    for start in range(0, seeds.size, step):
        batch = seeds[start:start + step]
        if batch.size == 1:
            bits = bits0.copy()
            acc, att, _ = _swap_chain(entries0.copy(), bits, target, max_att, int(batch[0]), accepted_clock)
            out = [(bits, acc, att)]
        else:
            bitsets, accs, atts, _ = _run_chains(bits0, entries0, target, max_att, batch, accepted_clock)
            out = [(bitsets[k], accs[k], atts[k]) for k in range(batch.size)]
            del bitsets
        for s, (bits, acc, att) in zip(batch, out):
            D = _unpack_bits(bits, D0.shape[1])
            diag = _diagnostics(D0, D, s, nnz, target, acc, att, r0, c0, sampler)
            assert diag.margins_exact, "fixed-margin swap chain changed a margin"
            yield D, diag


# ---------------------------------------------------------------------------
# Depth standardisation
# ---------------------------------------------------------------------------

class UpsamplingWarning(UserWarning):
    """Warning issued when a population's median depth is below ``D*``.

    Such a population is left unchanged rather than upsampled, so it is
    not matched to ``D*``.
    """


def count_depth(X) -> np.ndarray:
    """Return the per-cell count depth.

    Parameters
    ----------
    X : array-like or scipy.sparse matrix
        Cells x features counts.

    Returns
    -------
    numpy.ndarray
        Row sums of ``X`` as float64.
    """
    return np.asarray(X.sum(axis=1)).ravel().astype(np.float64)


def downsampling_fractions(depth, groups, D_star: float, *, populations=None) -> pd.DataFrame:
    """Compute the per-population retention fraction for a target depth.

    For each population G,

        q_G = min(1, D* / median_depth_G).

    Parameters
    ----------
    depth : array-like
        Count depth of each cell.
    groups : array-like
        Population label of each cell.
    D_star : float
        Declared common effective depth, in the units of ``depth``.
    populations : sequence, optional
        Populations to report, in order. Default: all labels, sorted.

    Returns
    -------
    pandas.DataFrame
        Indexed by population, with columns ``median_depth``, ``q`` and
        ``reaches_D_star`` (False when the median depth is below ``D*``).

    Raises
    ------
    ValueError
        If ``D_star`` is not a positive finite number, or the labels do not
        match ``depth``.

    Warns
    -----
    UpsamplingWarning
        If any population's median depth is below ``D*``. Such populations
        are not upsampled, so their comparison is not matched to ``D*``.
    """
    if not np.isfinite(D_star) or D_star <= 0:
        raise ValueError("D_star must be a positive finite depth")
    depth = np.asarray(depth, dtype=np.float64).ravel()
    g = check_labels(groups, depth.size)
    pops = list(populations) if populations is not None else sorted(set(g))
    med = pd.Series({p: float(np.median(depth[g == p])) for p in pops}, name="median_depth")
    out = med.to_frame()
    out["q"] = np.minimum(1.0, D_star / out["median_depth"])
    out["reaches_D_star"] = out["median_depth"] >= D_star
    out.index.name = "population"
    below = out.index[~out.reaches_D_star].tolist()
    if below:
        warnings.warn(f"median depth below D*={D_star:g} for {below}: these populations are not "
                      "upsampled, so they are not matched to D*", UpsamplingWarning, stacklevel=2)
    return out


def binomial_downsample(X, q, rng: np.random.Generator) -> sparse.csr_matrix:
    """Thin integer counts binomially.

    Each count X_ij is replaced by a draw from Binomial(X_ij, q_i).
    On binary data this is independent retention of each open feature.

    Parameters
    ----------
    X : array-like or scipy.sparse matrix
        Cells x features integer counts. It is copied, not modified.
    q : float or array-like
        Retention probability: a scalar or one value per cell. Values are
        clipped to [0, 1], so counts are never upsampled.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    scipy.sparse.csr_matrix
        Thinned float64 counts with explicit zeros removed.

    Raises
    ------
    ValueError
        If ``X`` contains non-integer values, or ``q`` is neither a scalar
        nor of length ``n_cells``.
    """
    X = sparse.csr_matrix(X, copy=True)
    if X.nnz and not np.allclose(X.data, np.round(X.data)):
        raise ValueError("binomial downsampling needs integer counts")
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 0:
        qq = np.full(X.data.size, float(q))
    else:
        if q.size != X.shape[0]:
            raise ValueError("q must be a scalar or have one entry per cell")
        qq = np.repeat(q, np.diff(X.indptr))
    qq = np.clip(qq, 0.0, 1.0)
    X.data = rng.binomial(X.data.astype(np.int64), qq).astype(np.float64)
    X.eliminate_zeros()
    return X


@dataclass
class DepthStandardizedResult:
    """Result of a matched-depth cross-population comparison (either axis).

    Attributes
    ----------
    D_star : float
        Declared common effective depth.
    replicates : pandas.DataFrame
        One row per (population, downsampling replicate).
    summary : pandas.DataFrame
        One row per population.
    pairwise : pandas.DataFrame
        Descriptive agreement of population orderings across replicates
        (:func:`pairwise_replicate_agreement`).
    axis : str
        ``"structural"`` or ``"programme"``.
    """

    D_star: float
    replicates: pd.DataFrame      # one row per (population, replicate)
    summary: pd.DataFrame         # one row per population
    pairwise: pd.DataFrame        # descriptive replicate agreement
    axis: str


def pairwise_replicate_agreement(values: pd.DataFrame) -> pd.DataFrame:
    """Summarize pairwise ordering agreement across downsampling replicates.

    Parameters
    ----------
    values : pandas.DataFrame
        Scores with one row per replicate and one column per population.

    Returns
    -------
    pandas.DataFrame
        One row per population pair with columns ``first``, ``second``,
        ``frac_first_greater`` (fraction of replicates with
        first > second) and ``replicates_unanimous`` (True when every
        replicate gives the same outcome of first > second).

    Notes
    -----
    This is a descriptive summary of the downsampling replicates. Ties
    count as "not greater".
    """
    cols = list(values.columns)
    rows = []
    for a_i in range(len(cols)):
        for b_i in range(a_i + 1, len(cols)):
            a, b = cols[a_i], cols[b_i]
            gt = (values[a] > values[b]).to_numpy()
            rows.append({"first": a, "second": b, "frac_first_greater": float(gt.mean()),
                         "replicates_unanimous": bool(gt.all() or (~gt).all())})
    return pd.DataFrame(rows)
