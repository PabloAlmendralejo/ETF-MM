"""Paired-bootstrap confidence intervals.

Implements the paired *percentile* bootstrap on per-path paired-difference
arrays (typically AS minus Symmetric on terminal P&L or max drawdown) with
a deterministic RNG seeded from the analytics-reserved
:class:`numpy.random.SeedSequence` (see
:func:`etf_mm_sim.seeding.analytics_seed`). See ``design.md`` §Bootstrap.

Methodology
-----------
Given a 1-D array ``diff`` of paired per-path differences of length
``n``, the procedure is:

1. Resample with replacement ``B`` times: ``idx = rng.integers(0, n,
   size=(B, n))`` produces a ``(B, n)`` matrix of indices.
2. For each bootstrap sample ``b`` compute the resampled mean
   ``means[b] = mean(diff[idx[b]])``.
3. Report ``(diff.mean(), quantile(means, alpha/2),
   quantile(means, 1 - alpha/2))`` as ``(point, lo, hi)``.

Pairing is enforced by the *caller*: the input ``diff`` is already a
per-path difference, so resampling rows of ``diff`` resamples whole
``(AS, Symmetric)`` pairs and removes between-path variance driven by
mid-price seed variation.

Determinism
-----------
The RNG is built once via ``np.random.default_rng(ss)`` from the
caller-supplied :class:`SeedSequence`. Two invocations with
*independently constructed* seed sequences carrying the same entropy /
spawn key produce byte-identical ``(point, lo, hi)`` tuples
(Property 21).

The resampling matrix is drawn with ``rng.integers(0, n, size=(B, n))``
in a single call so the draw order does not depend on Python iteration;
this is also the cheapest path through NumPy for moderate ``B``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["paired_bootstrap_ci"]


def paired_bootstrap_ci(
    diff: np.ndarray,
    B: int,
    alpha: float,
    ss: np.random.SeedSequence,
) -> tuple[float, float, float]:
    """Return ``(point, lo, hi)`` for the paired percentile bootstrap.

    Parameters
    ----------
    diff:
        1-D ``np.ndarray`` of per-path paired differences. Must be
        non-empty.
    B:
        Number of bootstrap resamples. Must be ``>= 1``.
    alpha:
        Two-sided significance level, e.g. ``0.05`` for a 95% CI. Must
        satisfy ``0 < alpha < 1``.
    ss:
        Seed sequence dedicated to bootstrap randomization. In normal
        use this comes from :func:`etf_mm_sim.seeding.analytics_seed`,
        possibly after a ``spawn`` to keep multiple independent
        bootstraps from sharing state.

    Returns
    -------
    tuple[float, float, float]
        ``(point, lo, hi)`` where:

        * ``point`` equals ``float(diff.mean())`` exactly.
        * ``lo`` is the ``alpha/2`` quantile of the bootstrap mean
          distribution.
        * ``hi`` is the ``1 - alpha/2`` quantile of the bootstrap mean
          distribution.

        With ``B >= 1`` and ``alpha < 1`` we always have
        ``lo <= hi``; with sufficiently many bootstrap iterations and
        non-degenerate ``diff`` we typically also have
        ``lo <= point <= hi``, but that ordering is not enforced here
        because it can fail in degenerate cases (e.g. ``B = 1``) where
        the test is responsible for restricting inputs.

    Raises
    ------
    ValueError
        If ``diff`` is not a non-empty 1-D ndarray, ``B < 1``, or
        ``alpha`` is not strictly between ``0`` and ``1``.
    """
    if not isinstance(diff, np.ndarray):
        raise ValueError(
            f"diff: must be a numpy ndarray; got {type(diff).__name__}"
        )
    if diff.ndim != 1:
        raise ValueError(
            f"diff: must be 1-D; got shape {diff.shape!r}"
        )
    if diff.shape[0] < 1:
        raise ValueError(
            f"diff: must be non-empty; got shape {diff.shape!r}"
        )
    if not isinstance(B, int) or isinstance(B, bool):
        raise ValueError(f"B: must be an int; got {B!r}")
    if B < 1:
        raise ValueError(f"B: must be >= 1; got {B!r}")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(
            f"alpha: must satisfy 0 < alpha < 1; got {alpha!r}"
        )

    rng = np.random.default_rng(ss)
    n = int(diff.shape[0])
    diff_f = np.ascontiguousarray(diff, dtype=np.float64)

    # Single vectorized resample. integers(0, n, size=(B, n)) draws B*n
    # uniform discrete samples from {0, ..., n-1}; indexing into diff
    # produces the (B, n) bootstrap matrix and a row-wise mean gives the
    # bootstrap mean distribution.
    idx = rng.integers(0, n, size=(B, n))
    means = diff_f[idx].mean(axis=1)

    point = float(diff_f.mean())
    lo = float(np.quantile(means, float(alpha) / 2.0))
    hi = float(np.quantile(means, 1.0 - float(alpha) / 2.0))
    return point, lo, hi
