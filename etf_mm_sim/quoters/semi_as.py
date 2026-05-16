"""Semi-Avellaneda-Stoikov quoter.

Posts ``(s - delta*, s + delta*)`` using the *AS dynamic half-spread schedule*
but with **no inventory skew**. Quotes are always centered on the mid-price.
Serves as a third strategy alongside Avellaneda-Stoikov and Symmetric so a
paired AS-vs-Semi-AS comparison cleanly isolates the effect of inventory
skewing alone (the dynamic spread schedule is the same in both arms).

Closed form
-----------
The half-spread schedule mirrors the AS one (see ``design.md`` §Quoters and
:mod:`etf_mm_sim.quoters.avellaneda_stoikov`)::

    delta*(t) = 0.5 * gamma * sigma**2 * max(T - t, 0)
              + (1 / gamma) * log(1 + gamma / k)

The reservation price collapses to ``s`` (no inventory term)::

    p_b(s, t) = s - delta*(t)
    p_a(s, t) = s + delta*(t)

Numerical conventions
---------------------
* The ``log(1 + gamma / k)`` term is computed via :func:`numpy.log1p` /
  :func:`math.log1p` for clean small-``gamma/k`` behaviour, matching the
  AS module.
* Validation is identical to AS: ``gamma > 0``, ``k > 0``, ``sigma >= 0``,
  ``dt > 0``, ``T > 0``. Violations raise :class:`ValueError`.
* Terminal handling mirrors AS: at ``t == T`` the dynamic part of
  ``delta*`` collapses to zero and the closed form is undefined as a
  *quoting* signal. The scalar :func:`semi_as_quote` raises
  ``ValueError`` and :func:`quote_arrays` flags the terminal step in
  ``suppress_mask`` with NaN bid/ask.

Note: this module reproduces the closed form directly rather than
importing from :mod:`avellaneda_stoikov`. The two modules are kept
decoupled so changes in one do not silently propagate to the other; the
shared formula is small enough to re-state without fear of drift.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "semi_as_quote",
    "precompute",
    "quote_arrays",
]


# --------------------------------------------------------------------------- #
# Validation helpers                                                          #
# --------------------------------------------------------------------------- #


def _validate_params(gamma: float, sigma: float, k: float) -> None:
    """Validate the Semi-AS parameters shared by all entry points."""
    if not (gamma > 0):
        raise ValueError(f"gamma: must be > 0; got {gamma!r}")
    if not (k > 0):
        raise ValueError(f"k: must be > 0; got {k!r}")
    if sigma < 0:
        raise ValueError(f"sigma: must be >= 0; got {sigma!r}")


# --------------------------------------------------------------------------- #
# Scalar entry point                                                          #
# --------------------------------------------------------------------------- #


def semi_as_quote(
    s: float,
    t: float,
    T: float,
    gamma: float,
    sigma: float,
    k: float,
) -> tuple[float, float]:
    """Return ``(bid, ask) = (s - delta*, s + delta*)``.

    No inventory skew: the quote midpoint is always ``s`` exactly.

    Parameters
    ----------
    s:
        Current mid-price.
    t:
        Current time. Must satisfy ``t < T``.
    T:
        Simulation horizon. Must be ``> 0``.
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
            "Semi-AS quoter cannot produce a quote at t >= T (caller should suppress)"
        )

    horizon = T - t
    delta_star = 0.5 * gamma * sigma * sigma * horizon + (1.0 / gamma) * math.log1p(
        gamma / k
    )
    return s - delta_star, s + delta_star


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
    """Precompute the Semi-AS half-spread schedule on the full step grid.

    Parameters
    ----------
    s_path:
        Mid-price path of length ``n + 1``. Only its length is used.
    dt:
        Time step. Must be ``> 0``.
    T:
        Simulation horizon. Must be ``> 0``.
    gamma, sigma, k:
        Semi-AS parameters. See :func:`semi_as_quote`.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping with a single key ``"delta_star"`` -- ``(n + 1,)``
        float64 array. The optimal half-spread evaluated at
        ``t_i = i * dt`` for ``i in {0, ..., n}``. The terminal entry
        collapses to the constant tail term
        ``(1 / gamma) * log1p(gamma / k)`` because ``T - t_n == 0``;
        callers must still suppress that step (see :func:`quote_arrays`).

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
    delta_star = 0.5 * gamma * sigma * sigma * horizon + (1.0 / gamma) * np.log1p(
        gamma / k
    )
    return {"delta_star": delta_star}


def quote_arrays(
    s_path: np.ndarray,
    dt: float,
    T: float,
    gamma: float,
    sigma: float,
    k: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized Semi-AS quote evaluation across a mid-price path.

    Parameters
    ----------
    s_path:
        Mid-price path of length ``n + 1``.
    dt:
        Time step. Must be ``> 0``.
    T:
        Simulation horizon. Must be ``> 0``.
    gamma, sigma, k:
        Semi-AS parameters. See :func:`semi_as_quote`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(bid_quote, ask_quote, suppress_mask)``. All three have length
        ``n + 1``. ``bid_quote`` and ``ask_quote`` are float64; entries
        at suppressed steps are ``NaN``. ``suppress_mask`` is a bool
        array; ``True`` means "do not post a quote at this step" and is
        set wherever ``t_i >= T`` (mirrors AS terminal handling).

    Raises
    ------
    ValueError
        If any of the parameter validations fail.
    """
    schedule = precompute(s_path, dt, T, gamma, sigma, k)
    delta_star = schedule["delta_star"]

    s_arr = np.asarray(s_path, dtype=np.float64)
    bid_quote = s_arr - delta_star
    ask_quote = s_arr + delta_star

    # Suppress whenever t_i >= T (terminal step and beyond), matching the
    # AS terminal handling.
    n_plus_1 = int(s_arr.shape[0])
    t_grid = np.arange(n_plus_1, dtype=np.float64) * dt
    suppress_mask = t_grid >= T
    bid_quote = np.where(suppress_mask, np.nan, bid_quote)
    ask_quote = np.where(suppress_mask, np.nan, ask_quote)

    return bid_quote, ask_quote, suppress_mask
