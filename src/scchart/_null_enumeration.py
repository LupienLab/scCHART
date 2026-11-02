"""Exact enumeration and validation of fixed-margin null samplers.

This internal module provides reference calculations for validating the
fixed-margin randomization procedures implemented in scchart._scoring.

For a small binary matrix, it enumerates all possible binary matrices
with the same row and column sums. These matrices form the fixed-margin
state space.

The module constructs the exact transition probabilities for the two
sampling procedures:

1. Proposal-clock sampler
   Two occupied entries are selected uniformly with replacement.
   If they define a valid 2 x 2 swap, the swap is performed.
   Otherwise, the matrix remains unchanged (a self-loop).

   The resulting transition matrix P describes the probability of
   moving between states in a single proposal.

2. Accepted-swap sampler
   Only successful swaps advance the sampling clock.

   The corresponding embedded transition matrix Q is derived from P:

       Q(x, y) = P(x, y) / a(x), for y != x

   where a(x) = 1 - P(x, x) is the probability that a proposal
   produces an accepted swap from state x.

The module also calculates exact finite-budget distributions from a
specified starting matrix, allowing the behaviour of the samplers
to be compared against their theoretical distributions.

These calculations are intended for small binary matrices, typically
containing no more than approximately 16 entries, because the number
of possible fixed-margin matrices grows rapidly with matrix size.

This module is used for mathematical validation and is not part of
the routine scCHART heterogeneity scoring workflow.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import numpy as np


def margin_states(B) -> List[np.ndarray]:
    """Enumerate all binary matrices with the margins of ``B``.

    Each row with sum r_i is drawn from the binary vectors of length f
    with r_i ones; combinations whose column sums equal those of ``B`` are
    kept.

    Parameters
    ----------
    B : array-like
        Two-dimensional matrix. It is binarized as ``B > 0`` before the
        row and column sums are taken.

    Returns
    -------
    list of numpy.ndarray
        Every ``uint8`` matrix with the row and column sums of ``B``,
        including the binarized ``B`` itself.

    Notes
    -----
    The search visits the product of all admissible rows, so the cost
    grows combinatorially with matrix size. Use only for small matrices.
    """
    B = (np.asarray(B) > 0).astype(np.uint8)
    n, f = B.shape
    r, c = B.sum(axis=1), B.sum(axis=0)
    rows = [[np.array(x, np.uint8) for x in itertools.product((0, 1), repeat=f) if sum(x) == k] for k in r]
    out = []
    for combo in itertools.product(*rows):
        M = np.vstack(combo) if combo else np.zeros((0, f), np.uint8)
        if np.array_equal(M.sum(axis=0), c):
            out.append(M)
    return out


def proposal_kernel(states: List[np.ndarray]) -> np.ndarray:
    """Construct the exact one-proposal transition matrix.

    Each proposal selects two occupied entries uniformly with replacement.
    Valid 2 x 2 swaps change the matrix; rejected proposals are self-loops.

    Parameters
    ----------
    states : list of numpy.ndarray
        Complete fixed-margin state space, containing one binary matrix
        per state.

    Returns
    -------
    numpy.ndarray
        Row-stochastic transition matrix P, where P[x, y] is the
        probability of transitioning from state x to state y.

    Notes
    -----
    The proposal kernel is symmetric over the complete fixed-margin
    state space.
    """
    idx: Dict[bytes, int] = {s.tobytes(): k for k, s in enumerate(states)}
    N = len(states)
    P = np.zeros((N, N))
    for a, B in enumerate(states):
        edges = np.argwhere(B)
        E = len(edges)
        if E == 0:
            P[a, a] = 1.0
            continue
        for (i, c), (j, d) in itertools.product(edges, edges):
            if i != j and c != d and not B[i, d] and not B[j, c]:
                C = B.copy()
                C[i, c] = C[j, d] = 0
                C[i, d] = C[j, c] = 1
                b = idx[C.tobytes()]
            else:
                b = a
            P[a, b] += 1.0 / E ** 2
    return P


def jump_kernel(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Construct the embedded jump kernel of the accepted-swap sampler.

    For each state x with acceptance probability a(x) = 1 - P(x, x) > 0,

        Q(x, y) = P(x, y) / a(x),  for y != x,  and  Q(x, x) = 0.

    A state with a(x) = 0 (no valid swap) is absorbing: Q(x, x) = 1.

    Parameters
    ----------
    P : numpy.ndarray
        One-proposal transition matrix, as returned by
        :func:`proposal_kernel`.

    Returns
    -------
    Q : numpy.ndarray
        Row-stochastic transition matrix of one accepted swap.
    a : numpy.ndarray
        Acceptance probability a(x) of each state.

    Notes
    -----
    Acceptance probabilities at or below 1e-15 are treated as zero.
    """
    a = 1.0 - np.diag(P)
    Q = P.copy()
    np.fill_diagonal(Q, 0.0)
    move = a > 1e-15
    Q[move] /= a[move, None]
    Q[~move, :] = 0.0
    Q[~move, ~move] = 0.0
    for k in np.flatnonzero(~move):
        Q[k, k] = 1.0
    return Q, a


def jump_stationary(a: np.ndarray) -> np.ndarray:
    """Return the stationary distribution of the accepted-swap jump chain.

    When the proposal kernel P is symmetric, the embedded jump chain has
    stationary distribution pi(x) = a(x) / sum_y a(y).

    Parameters
    ----------
    a : numpy.ndarray
        Acceptance probability of each state, as returned by
        :func:`jump_kernel`.

    Returns
    -------
    numpy.ndarray
        Stationary probabilities, proportional to ``a``.
    """
    return a / a.sum()


def finite_law(K: np.ndarray, start: int, steps: int) -> np.ndarray:
    """Compute the exact state distribution after a fixed number of steps.

    Starting from a point mass at ``start``, the distribution after t
    steps is e_start^T K^t.

    Parameters
    ----------
    K : numpy.ndarray
        Row-stochastic transition matrix.
    start : int
        Index of the starting state.
    steps : int
        Number of transitions.

    Returns
    -------
    numpy.ndarray
        Probability of each state after ``steps`` transitions.
    """
    v = np.zeros(K.shape[0])
    v[start] = 1.0
    return v @ np.linalg.matrix_power(K, int(steps))


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """Compute the total-variation distance between two distributions.

    TV(p, q) = (1/2) * sum_x |p(x) - q(x)|.

    Parameters
    ----------
    p, q : array-like
        Probability vectors over the same states.

    Returns
    -------
    float
        Total-variation distance, between 0 and 1 for valid distributions.
    """
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def state_index(states: List[np.ndarray]) -> Dict[bytes, int]:
    """Map each state to its position in ``states``.

    Parameters
    ----------
    states : list of numpy.ndarray
        Fixed-margin state space.

    Returns
    -------
    dict
        Mapping from the ``uint8`` byte representation of each matrix to
        its index in ``states``.
    """
    return {s.astype(np.uint8).tobytes(): k for k, s in enumerate(states)}
