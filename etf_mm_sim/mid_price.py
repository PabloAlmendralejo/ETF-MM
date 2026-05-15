"""Mid-price path generators.

Implements the vectorized GBM and regime-switching mid-price simulators that
feed the per-path runner. See ``design.md`` §Mid_Price_Simulator.

Both generators follow these conventions:

* The number of inner steps is ``n = ceil(T / dt)`` so that ``n * dt >= T``.
* The returned price array has length ``n + 1`` (the initial price plus one
  observation after each inner step), per Requirement 2.4.
* All randomness is derived from a ``numpy.random.SeedSequence`` argument.
  Two invocations with two freshly-constructed ``SeedSequence(seed)`` objects
  produce bit-identical paths (Property 2 / Requirement 2.3).
* GBM uses an exact log-Euler discretization::

      log S_{i+1} = log S_i + (mu - 0.5 * sigma**2) * dt + sigma * dW_i

  which keeps prices strictly positive for any finite normal draw and any
  ``sigma >= 0`` (Property 4 / Requirement 2.5). When ``sigma == 0`` the
  stochastic term vanishes identically and the path collapses to the
  deterministic drift trajectory ``s0 * exp(mu * t_i)`` regardless of seed
  (Property 5 / Requirement 2.6).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .config import RegimeParams

__all__ = [
    "simulate_gbm",
    "simulate_regime_switching",
]


# --------------------------------------------------------------------------- #
# Input validation helpers                                                    #
# --------------------------------------------------------------------------- #


def _validate_horizon(T: float, dt: float) -> None:
    if not (T > 0):
        raise ValueError(f"T: must be > 0; got {T!r}")
    if not (dt > 0):
        raise ValueError(f"dt: must be > 0; got {dt!r}")
    if dt > T:
        raise ValueError(f"dt: must be <= T; got dt={dt!r}, T={T!r}")


def _validate_gbm_params(s0: float, sigma: float) -> None:
    if not (s0 > 0):
        raise ValueError(f"s0: must be > 0; got {s0!r}")
    if sigma < 0:
        raise ValueError(f"sigma: must be >= 0; got {sigma!r}")


# --------------------------------------------------------------------------- #
# GBM                                                                         #
# --------------------------------------------------------------------------- #


def simulate_gbm(
    s0: float,
    mu: float,
    sigma: float,
    T: float,
    dt: float,
    ss: np.random.SeedSequence,
) -> np.ndarray:
    """Simulate a single GBM path using exact log-Euler discretization.

    Parameters
    ----------
    s0:
        Initial mid-price. Must be ``> 0``.
    mu:
        Drift parameter. Any finite real.
    sigma:
        Volatility parameter. Must be ``>= 0``.
    T:
        Simulation horizon. Must be ``> 0``.
    dt:
        Time step. Must satisfy ``0 < dt <= T``.
    ss:
        ``numpy.random.SeedSequence`` from which the path RNG is derived.

    Returns
    -------
    np.ndarray
        Float64 array of length ``ceil(T/dt) + 1``. Element 0 is exactly
        ``s0``; subsequent elements are strictly positive.

    Raises
    ------
    ValueError
        If any input violates its constraint.
    """
    _validate_gbm_params(s0, sigma)
    _validate_horizon(T, dt)

    n = math.ceil(T / dt)
    rng = np.random.default_rng(ss)

    # Exact log-Euler: log S_{i+1} - log S_i = (mu - 0.5*sigma^2)*dt + sigma*dW.
    # When sigma == 0, the second term is identically zero (regardless of dW),
    # so the path collapses to s0 * exp(mu * t_i) bit-for-bit (Property 5).
    dW = rng.standard_normal(n) * math.sqrt(dt)
    log_increments = (mu - 0.5 * sigma * sigma) * dt + sigma * dW
    log_s = math.log(s0) + np.cumsum(log_increments)
    return np.concatenate(([s0], np.exp(log_s)))


# --------------------------------------------------------------------------- #
# Regime-switching GBM                                                        #
# --------------------------------------------------------------------------- #


def _validate_regime_switching_inputs(
    s0: float,
    regimes: Sequence[RegimeParams],
    P: np.ndarray,
    T: float,
    dt: float,
    initial_regime: int,
) -> None:
    if not (s0 > 0):
        raise ValueError(f"s0: must be > 0; got {s0!r}")
    if len(regimes) == 0:
        raise ValueError("regimes: must be non-empty")
    R = len(regimes)
    for i, r in enumerate(regimes):
        if r.sigma < 0:
            raise ValueError(
                f"regimes[{i}].sigma: must be >= 0; got {r.sigma!r}"
            )
    _validate_horizon(T, dt)

    if not isinstance(P, np.ndarray):
        raise ValueError(f"P: must be a numpy.ndarray; got {type(P).__name__}")
    if P.shape != (R, R):
        raise ValueError(
            f"P: must have shape ({R}, {R}) to match number of regimes; "
            f"got {P.shape!r}"
        )
    if np.any(P < 0.0) or np.any(P > 1.0):
        raise ValueError("P: every entry must be in [0, 1]")
    row_sums = P.sum(axis=1)
    if not np.all(np.isclose(row_sums, 1.0, atol=1e-9)):
        raise ValueError(
            f"P: every row must sum to 1 within 1e-9; got row sums {row_sums!r}"
        )
    if not isinstance(initial_regime, (int, np.integer)) or isinstance(
        initial_regime, bool
    ):
        raise ValueError(
            f"initial_regime: must be an int; got {initial_regime!r}"
        )
    if not (0 <= int(initial_regime) < R):
        raise ValueError(
            f"initial_regime: must satisfy 0 <= initial_regime < {R}; "
            f"got {initial_regime!r}"
        )


def simulate_regime_switching(
    s0: float,
    regimes: tuple[RegimeParams, ...],
    P: np.ndarray,
    T: float,
    dt: float,
    ss: np.random.SeedSequence,
    initial_regime: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a Markov regime-switching GBM path.

    The regime active during step ``i`` (i.e. the regime that produces the
    log-increment from ``s_path[i]`` to ``s_path[i+1]``) is recorded in
    ``regime_indices[i]``. After the increment is applied, the regime is
    transitioned by inverse-CDF sampling on the row ``P[r, :]`` of the
    transition matrix, with ``r`` being the active regime at step ``i``.

    Parameters
    ----------
    s0:
        Initial mid-price. Must be ``> 0``.
    regimes:
        Tuple of :class:`RegimeParams`. Each entry contributes one regime's
        ``(mu, sigma)``. Must be non-empty.
    P:
        Row-stochastic transition matrix of shape ``(R, R)`` where
        ``R == len(regimes)``. Every row must sum to 1 within 1e-9 and every
        entry must lie in ``[0, 1]``.
    T:
        Simulation horizon. Must be ``> 0``.
    dt:
        Time step. Must satisfy ``0 < dt <= T``.
    ss:
        ``numpy.random.SeedSequence`` from which both the regime-walk
        uniforms and the per-step normal innovations are derived.
    initial_regime:
        Index of the regime active at step 0. Defaults to 0.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(s_path, regime_indices)`` where ``s_path`` has length
        ``ceil(T/dt) + 1`` and ``regime_indices`` has length ``ceil(T/dt)``
        (one entry per inner step). All entries of ``s_path`` are strictly
        positive whenever ``s0 > 0`` and every regime has ``sigma >= 0``.

    Raises
    ------
    ValueError
        If any input violates its constraint.
    """
    _validate_regime_switching_inputs(s0, regimes, P, T, dt, initial_regime)

    n = math.ceil(T / dt)
    R = len(regimes)
    mu_per_regime = np.array([r.mu for r in regimes], dtype=np.float64)
    sigma_per_regime = np.array([r.sigma for r in regimes], dtype=np.float64)

    rng = np.random.default_rng(ss)
    # Two pre-drawn streams keeps the regime walk and the GBM innovations
    # independent (and reproducible). Order matters for byte-identity: regime
    # uniforms first, then normals.
    u_reg = rng.random(n)
    dW_unscaled = rng.standard_normal(n)

    regime_indices = np.empty(n, dtype=np.int64)
    mu_arr = np.empty(n, dtype=np.float64)
    sigma_arr = np.empty(n, dtype=np.float64)

    # Cumulative-row cache: cumsum of P[r] is precomputed once per row to
    # amortize the inverse-CDF lookup across all visits to that row.
    P_cum = np.cumsum(P, axis=1)
    # Numerical clean-up: force the last column to be exactly 1.0 so that
    # u_reg[i] in [0, 1) is always covered by some bucket. fp accumulation in
    # cumsum can leave a tiny shortfall.
    P_cum[:, -1] = 1.0

    r = int(initial_regime)
    for i in range(n):
        regime_indices[i] = r
        mu_arr[i] = mu_per_regime[r]
        sigma_arr[i] = sigma_per_regime[r]
        # Inverse-CDF: smallest j with u_reg[i] < P_cum[r, j]. searchsorted
        # with side='right' gives the index where the value would be inserted
        # to keep order, which is exactly what we want for "first bucket
        # whose cumulative mass exceeds u_reg[i]".
        r = int(np.searchsorted(P_cum[r], u_reg[i], side="right"))
        # Defensive clamp in case of edge cases at u_reg[i] == 1.0 - eps.
        if r >= R:
            r = R - 1

    # Vectorized GBM step: increments depend on the per-step (mu, sigma) and
    # the pre-drawn unscaled normals.
    log_increments = (
        (mu_arr - 0.5 * sigma_arr * sigma_arr) * dt
        + sigma_arr * dW_unscaled * math.sqrt(dt)
    )
    log_s = math.log(s0) + np.cumsum(log_increments)
    s_path = np.concatenate(([s0], np.exp(log_s)))
    return s_path, regime_indices
