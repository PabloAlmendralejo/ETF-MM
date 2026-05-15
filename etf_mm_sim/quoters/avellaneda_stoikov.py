"""Avellaneda-Stoikov quoter.

Implements the closed-form reservation price and optimal half-spread from
Avellaneda-Stoikov (2008) (see ``design.md`` §Quoters)::

    r(s, q, t)        = s - q * gamma * sigma**2 * (T - t)
    delta*(t)         = 0.5 * gamma * sigma**2 * (T - t)
                        + (1 / gamma) * log(1 + gamma / k)
    p_b(s, q, t)      = r(s, q, t) - delta*(t)
    p_a(s, q, t)      = r(s, q, t) + delta*(t)

The horizon factor ``(T - t)`` is non-negative across any realistic
simulation; once ``t == T`` the quoter is no longer well-defined (the
inventory-skew term collapses to zero and the spread-collapse limit of the
solution does not specify a unique terminal quote). At and beyond ``t == T``
we therefore *suppress* the quote rather than synthesizing one. Callers
must respect the suppression mask returned by :func:`quote_arrays` (or, for
the scalar entry-point :func:`as_quote`, they must catch the
:class:`ValueError` raised when ``t >= T``).

Vectorization
-------------
:func:`precompute` is the bulk-evaluation entry point used by the path
runner. It produces two ``length-(n+1)`` arrays:

* ``delta_star`` — the time-varying optimal half-spread on the full step
  grid ``t_i = i * dt`` for ``i in {0, ..., n}``.
* ``skew_coef`` — the inventory-skew coefficient ``gamma * sigma**2 *
  (T - t_i)``, again on the full step grid. Multiplying by ``q[i]`` and
  subtracting from ``s[i]`` reproduces the reservation price ``r(s, q, t_i)``.

At ``t_i == T`` (the terminal step ``i == n``), ``T - t_i`` is ``0`` so the
``skew_coef`` is zero and ``delta_star`` collapses to the constant tail term
``(1 / gamma) * log1p(gamma / k)``. The terminal quote is suppressed by
:func:`quote_arrays` regardless of the closed-form values, so consumers do
not need to reason about the degenerate end-of-horizon case.

Numerical conventions
---------------------
* The ``log(1 + gamma / k)`` term is computed via :func:`numpy.log1p` /
  :func:`math.log1p` to keep the small-``gamma/k`` regime free of
  catastrophic cancellation.
* Validation is consistent across the scalar and vector entry points:
  ``gamma > 0``, ``k > 0``, ``sigma >= 0``. Violations raise
  :class:`ValueError`.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "as_quote",
    "precompute",
    "quote_arrays",
]


# --------------------------------------------------------------------------- #
# Validation helpers                                                          #
# --------------------------------------------------------------------------- #


def _validate_params(gamma: float, sigma: float, k: float) -> None:
    """Validate the AS parameters shared by all entry points."""
    if not (gamma > 0):
        raise ValueError(f"gamma: must be > 0; got {gamma!r}")
    if not (k > 0):
        raise ValueError(f"k: must be > 0; got {k!r}")
    if sigma < 0:
        raise ValueError(f"sigma: must be >= 0; got {sigma!r}")


# --------------------------------------------------------------------------- #
# Scalar entry point                                                          #
# --------------------------------------------------------------------------- #


def as_quote(
    s: float,
    q: float,
    t: float,
    T: float,
    gamma: float,
    sigma: float,
    k: float,
) -> tuple[float, float]:
    """Return ``(p_b, p_a)`` for a single ``(s, q, t)`` per the AS closed form.

    Parameters
    ----------
    s:
        Current mid-price.
    q:
        Current inventory (signed). Integer-valued in production but
        accepted as ``float`` so callers can pass the raw inventory without
        casting.
    t:
        Current time. Must satisfy ``t < T``; at or past the horizon the
        AS solution is not defined and the caller is expected to suppress
        the quote.
    T:
        Simulation horizon.
    gamma:
        Risk-aversion coefficient. Must be ``> 0``.
    sigma:
        Reference volatility used by the AS schedule. Must be ``>= 0``.
    k:
        Order-arrival decay rate. Must be ``> 0``.

    Returns
    -------
    tuple[float, float]
        ``(bid, ask)`` quote pair.

    Raises
    ------
    ValueError
        If ``gamma <= 0``, ``k <= 0``, ``sigma < 0``, or ``t >= T``.
    """
    _validate_params(gamma, sigma, k)
    if t >= T:
        raise ValueError(
            "AS quoter cannot produce a quote at t >= T (caller should suppress)"
        )

    horizon = T - t
    skew_coef = gamma * sigma * sigma * horizon
    reservation = s - q * skew_coef
    delta_star = 0.5 * skew_coef + (1.0 / gamma) * math.log1p(gamma / k)
    return reservation - delta_star, reservation + delta_star


# --------------------------------------------------------------------------- #
# Vectorized entry points                                                     #
# --------------------------------------------------------------------------- #


def precompute(
    s_path: np.ndarray,
    dt: float,
    T: float,
    gamma: float,
    sigma: float,
    k: float,
) -> dict[str, np.ndarray]:
    """Precompute the AS time schedules on the full step grid.

    Parameters
    ----------
    s_path:
        Mid-price path of length ``n + 1``. Only its length is used; values
        do not enter the closed-form schedules.
    dt:
        Time step. Must be ``> 0``.
    T:
        Simulation horizon. Must be ``> 0``.
    gamma, sigma, k:
        AS parameters. See :func:`as_quote`.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping with keys

        * ``"delta_star"`` — ``(n + 1,)`` float64 array. The optimal
          half-spread evaluated at ``t_i = i * dt`` for
          ``i in {0, ..., n}``. The terminal entry collapses to the
          constant tail term ``(1 / gamma) * log1p(gamma / k)`` because
          ``T - t_n == 0``; callers must still suppress that step (see
          :func:`quote_arrays`).
        * ``"skew_coef"`` — ``(n + 1,)`` float64 array. The inventory-skew
          coefficient ``gamma * sigma**2 * (T - t_i)``. The terminal entry
          is exactly zero. Multiplying by ``q[i]`` and subtracting from
          ``s_path[i]`` yields the reservation price at step ``i``.

    Raises
    ------
    ValueError
        If ``dt <= 0``, ``T <= 0``, ``gamma <= 0``, ``k <= 0``, or
        ``sigma < 0``.
    """
    _validate_params(gamma, sigma, k)
    if not (dt > 0):
        raise ValueError(f"dt: must be > 0; got {dt!r}")
    if not (T > 0):
        raise ValueError(f"T: must be > 0; got {T!r}")

    n_plus_1 = int(s_path.shape[0])
    t_grid = np.arange(n_plus_1, dtype=np.float64) * dt
    horizon = np.maximum(T - t_grid, 0.0)
    skew_coef = gamma * sigma * sigma * horizon
    delta_star = 0.5 * skew_coef + (1.0 / gamma) * np.log1p(gamma / k)
    return {"delta_star": delta_star, "skew_coef": skew_coef}


def quote_arrays(
    s_path: np.ndarray,
    dt: float,
    T: float,
    gamma: float,
    sigma: float,
    k: float,
    q_path: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized AS quote evaluation across an inventory path.

    For each step ``i in {0, ..., n}`` this computes the closed-form
    ``(bid, ask)`` quote pair from ``s_path[i]`` and ``q_path[i]`` using
    the schedules returned by :func:`precompute`. The terminal step ``i ==
    n`` (and any other step where ``T - t_i == 0``) is flagged in the
    suppression mask; the corresponding ``bid`` / ``ask`` entries are set
    to ``NaN`` to make accidental use loud.

    Parameters
    ----------
    s_path:
        Mid-price path of length ``n + 1``.
    dt:
        Time step. Must be ``> 0``.
    T:
        Simulation horizon. Must be ``> 0``.
    gamma, sigma, k:
        AS parameters. See :func:`as_quote`.
    q_path:
        Inventory path of length ``n + 1``. Entry ``q_path[i]`` is the
        inventory carried into step ``i``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(bid_quote, ask_quote, suppress_mask)``. All three have length
        ``n + 1``. ``bid_quote`` and ``ask_quote`` are float64; entries at
        suppressed steps are ``NaN``. ``suppress_mask`` is a bool array;
        ``True`` means "do not post a quote at this step".

    Raises
    ------
    ValueError
        If ``s_path`` and ``q_path`` have mismatched lengths or any of the
        parameter validations fail.
    """
    if s_path.shape != q_path.shape:
        raise ValueError(
            "s_path and q_path must have the same shape; "
            f"got {s_path.shape!r} and {q_path.shape!r}"
        )

    schedules = precompute(s_path, dt, T, gamma, sigma, k)
    delta_star = schedules["delta_star"]
    skew_coef = schedules["skew_coef"]

    s_arr = np.asarray(s_path, dtype=np.float64)
    q_arr = np.asarray(q_path, dtype=np.float64)

    reservation = s_arr - q_arr * skew_coef
    bid_quote = reservation - delta_star
    ask_quote = reservation + delta_star

    # Suppress whenever the AS horizon factor (T - t_i) is non-positive. This
    # catches the terminal step (and any pathological earlier step that lands
    # at or beyond T). The check is on the time grid directly so that a
    # zero-volatility regime — where ``skew_coef`` is identically zero —
    # does not accidentally suppress every step.
    n_plus_1 = int(s_arr.shape[0])
    t_grid = np.arange(n_plus_1, dtype=np.float64) * dt
    suppress_mask = t_grid >= T
    bid_quote = np.where(suppress_mask, np.nan, bid_quote)
    ask_quote = np.where(suppress_mask, np.nan, ask_quote)

    return bid_quote, ask_quote, suppress_mask
