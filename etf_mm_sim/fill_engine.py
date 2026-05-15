"""Fill engine.

Implements the per-side Poisson fill model from ``design.md`` §Fill_Engine.

Mathematical model
------------------
The instantaneous arrival intensity of a fill at quote distance ``delta``
from the mid-price is

.. math::

    \\lambda(\\delta) = A \\cdot \\exp(-k \\cdot \\delta)

with ``A > 0`` setting the at-the-mid arrival rate and ``k > 0`` setting
how quickly arrivals fall off as the quote moves further from the mid.
Across one simulation step of length ``dt`` the per-step fill probability
is the Bernoulli approximation of a homogeneous Poisson process,

.. math::

    p(\\delta, dt) = 1 - \\exp(-\\lambda(\\delta) \\cdot dt)
                   = -\\operatorname{expm1}(-\\lambda(\\delta) \\cdot dt).

The ``-expm1(-x)`` form is used because it is numerically clean in the
small-``lambda*dt`` regime, where the naive ``1 - exp(-x)`` loses precision
to catastrophic cancellation.

Crossed-quote regime
--------------------
When ``delta <= 0`` the quote is at or through the mid-price (a "crossed"
quote) and we deem the fill guaranteed within the step (Req 5.6). The
probability functions short-circuit to exactly ``1.0`` rather than passing
a potentially-huge ``lambda`` through the exponential, which avoids
``inf * 0`` artefacts at the float64 boundary and keeps the result an
exact Bernoulli-1.

RNG conventions
---------------
:func:`draw_fill_uniforms` materializes the per-step Bernoulli inputs from
a single ``SeedSequence`` so the path runner can consume pre-drawn arrays
without re-entering the RNG inside its inner loop. The draw order is
fixed::

    rng = np.random.default_rng(ss)
    u_b = rng.random(n)   # bid side first
    u_a = rng.random(n)   # ask side second

This order is part of the reproducibility contract (Req 5.7) and must not
be reshuffled even if the two arrays are eventually concatenated.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "fill_intensity",
    "fill_intensity_arr",
    "fill_probability",
    "fill_probability_arr",
    "draw_fill_uniforms",
]


# --------------------------------------------------------------------------- #
# Validation helpers                                                          #
# --------------------------------------------------------------------------- #


def _validate_A_k(A: float, k: float) -> None:
    """Validate the intensity parameters shared by every entry point."""
    if not (A > 0):
        raise ValueError(f"A: must be > 0; got {A!r}")
    if not (k > 0):
        raise ValueError(f"k: must be > 0; got {k!r}")


def _validate_dt(dt: float) -> None:
    """Validate the time-step parameter."""
    if not (dt > 0):
        raise ValueError(f"dt: must be > 0; got {dt!r}")


# --------------------------------------------------------------------------- #
# Scalar entry points                                                         #
# --------------------------------------------------------------------------- #


def fill_intensity(delta: float, A: float, k: float) -> float:
    """Return the Poisson arrival intensity at quote distance ``delta``.

    Computes the closed form ``lambda(delta) = A * exp(-k * delta)`` for any
    real ``delta``. The result is non-negative; for ``delta <= 0`` it is at
    least ``A`` (and may overflow to ``+inf`` for extreme negative ``delta``
    paired with a large ``k``). Callers that need a Bernoulli probability
    should use :func:`fill_probability`, which short-circuits the
    crossed-quote regime to an exact ``1.0``.

    ``np.exp`` is used (rather than :func:`math.exp`) so that the overflow
    boundary degrades gracefully to ``+inf`` instead of raising
    :class:`OverflowError`.

    Parameters
    ----------
    delta:
        Signed distance from the mid to the posted quote.
    A:
        At-the-mid arrival rate. Must be ``> 0``.
    k:
        Decay rate of the arrival intensity in ``delta``. Must be ``> 0``.

    Returns
    -------
    float
        ``A * exp(-k * delta)``.

    Raises
    ------
    ValueError
        If ``A <= 0`` or ``k <= 0``.
    """
    _validate_A_k(A, k)
    return float(A * np.exp(-k * delta))


def fill_probability(delta: float, A: float, k: float, dt: float) -> float:
    """Return the per-step Bernoulli fill probability.

    For ``delta > 0``::

        p = 1 - exp(-A * exp(-k * delta) * dt)
          = -expm1(-A * exp(-k * delta) * dt)

    For ``delta <= 0`` the quote crosses the mid-price and the fill is
    deemed guaranteed within the step (Req 5.6); the function returns
    exactly ``1.0`` without computing ``lambda``.

    Parameters
    ----------
    delta:
        Signed distance from the mid to the posted quote.
    A, k:
        Intensity parameters. See :func:`fill_intensity`.
    dt:
        Step length. Must be ``> 0``.

    Returns
    -------
    float
        Bernoulli probability in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``A <= 0``, ``k <= 0``, or ``dt <= 0``.
    """
    _validate_A_k(A, k)
    _validate_dt(dt)
    if delta <= 0.0:
        return 1.0
    lam = A * math.exp(-k * delta)
    # ``-expm1(-x) == 1 - exp(-x)`` analytically; the ``expm1`` form keeps
    # precision in the small-``lam*dt`` regime where ``1 - exp(-x)`` loses
    # leading digits to cancellation.
    return -math.expm1(-lam * dt)


# --------------------------------------------------------------------------- #
# Vectorized entry points                                                     #
# --------------------------------------------------------------------------- #


def fill_intensity_arr(
    delta: np.ndarray, A: float, k: float
) -> np.ndarray:
    """Vectorized :func:`fill_intensity` over an array of distances.

    Parameters
    ----------
    delta:
        Array of signed distances of any shape.
    A, k:
        Intensity parameters. See :func:`fill_intensity`.

    Returns
    -------
    np.ndarray
        ``A * exp(-k * delta)`` evaluated element-wise as ``float64``.

    Raises
    ------
    ValueError
        If ``A <= 0`` or ``k <= 0``.
    """
    _validate_A_k(A, k)
    delta_arr = np.asarray(delta, dtype=np.float64)
    return A * np.exp(-k * delta_arr)


def fill_probability_arr(
    delta: np.ndarray, A: float, k: float, dt: float
) -> np.ndarray:
    """Vectorized :func:`fill_probability` over an array of distances.

    Equivalent to::

        prob[i] = 1.0                                  if delta[i] <= 0
        prob[i] = 1 - exp(-A * exp(-k * delta[i]) * dt) otherwise

    The non-crossed branch is computed via :func:`numpy.expm1` to preserve
    precision when ``lambda * dt`` is small.

    Parameters
    ----------
    delta:
        Array of signed distances of any shape.
    A, k:
        Intensity parameters. See :func:`fill_intensity`.
    dt:
        Step length. Must be ``> 0``.

    Returns
    -------
    np.ndarray
        Bernoulli probabilities in ``[0, 1]`` with the same shape as
        ``delta`` and dtype ``float64``.

    Raises
    ------
    ValueError
        If ``A <= 0``, ``k <= 0``, or ``dt <= 0``.
    """
    _validate_A_k(A, k)
    _validate_dt(dt)
    delta_arr = np.asarray(delta, dtype=np.float64)
    lam = A * np.exp(-k * delta_arr)
    prob = -np.expm1(-lam * dt)
    # Force the crossed-quote branch to exactly 1.0; this also guards against
    # any ``inf * 0``-style artefact when ``lam`` overflows for very negative
    # ``delta`` with large ``k``.
    return np.where(delta_arr <= 0.0, 1.0, prob)


# --------------------------------------------------------------------------- #
# RNG draws                                                                   #
# --------------------------------------------------------------------------- #


def draw_fill_uniforms(
    ss: np.random.SeedSequence, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n`` uniform variates per side from a single ``SeedSequence``.

    The draw order is **bid-first, ask-second** and is part of the
    reproducibility contract (Req 5.7). Two invocations with the same
    ``SeedSequence`` produce bit-identical ``(u_b, u_a)`` pairs.

    Parameters
    ----------
    ss:
        Per-path fill seed sequence (typically the ``fill`` slot of a
        :class:`etf_mm_sim.seeding.PathSeeds`).
    n:
        Number of step-level uniforms to draw per side. Must be ``>= 0``;
        ``n == 0`` returns two empty arrays (a degenerate but legal case).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(u_b, u_a)``, each of shape ``(n,)`` with dtype ``float64``.

    Raises
    ------
    ValueError
        If ``n < 0`` or is not an ``int``.
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError(f"n: must be an int; got {n!r}")
    if n < 0:
        raise ValueError(f"n: must be >= 0; got {n!r}")
    rng = np.random.default_rng(ss)
    u_b = rng.random(n)
    u_a = rng.random(n)
    return u_b, u_a
