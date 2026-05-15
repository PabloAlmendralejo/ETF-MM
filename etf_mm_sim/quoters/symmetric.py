"""Symmetric constant-spread baseline quoter.

Posts ``(s - delta_base, s + delta_base)`` regardless of inventory. Serves
as the no-skew, no-time-decay baseline for paired comparison against the
Avellaneda-Stoikov quoter (see ``design.md`` §Quoters).

Design notes
------------
* The quoter is *inventory-independent* and *time-independent*: the bid
  and ask depend only on the current mid-price and the configured
  half-spread ``delta_base``. This is exactly the behaviour required by
  Requirement 4.2 (ignores inventory) and 4.3 (mid-symmetric).
* The quoter never suppresses on its own. The risk manager handles
  inventory-bound and kill-switch suppression at run time; the quoter's
  own ``suppress_mask`` is therefore identically ``False``.
* :func:`precompute` exists so the quoter conforms to the same interface
  as :mod:`etf_mm_sim.quoters.avellaneda_stoikov`. There is nothing to
  precompute for a constant-spread quoter, so it returns an empty mapping.

Numerical conventions
---------------------
* ``delta_base`` must be ``>= 0`` (a zero half-spread quotes both sides at
  the mid; negative would invert bid and ask). Validation is consistent
  across the scalar and vector entry points.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "symmetric_quote",
    "precompute",
    "quote_arrays",
]


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #


def _validate_delta_base(delta_base: float) -> None:
    if delta_base < 0:
        raise ValueError(f"delta_base: must be >= 0; got {delta_base!r}")


# --------------------------------------------------------------------------- #
# Scalar entry point                                                          #
# --------------------------------------------------------------------------- #


def symmetric_quote(
    s: float,
    q: int,
    t: float,
    delta_base: float,
) -> tuple[float, float]:
    """Return ``(p_b, p_a) = (s - delta_base, s + delta_base)``.

    Parameters
    ----------
    s:
        Current mid-price.
    q:
        Current inventory. Ignored; present for interface compatibility
        with the AS quoter.
    t:
        Current time. Ignored; present for interface compatibility.
    delta_base:
        Constant half-spread. Must be ``>= 0``.

    Returns
    -------
    tuple[float, float]
        ``(bid, ask)`` quote pair.

    Raises
    ------
    ValueError
        If ``delta_base < 0``.
    """
    _validate_delta_base(delta_base)
    return s - delta_base, s + delta_base


# --------------------------------------------------------------------------- #
# Vectorized entry points                                                     #
# --------------------------------------------------------------------------- #


def precompute(
    s_path: np.ndarray,
    dt: float,
    T: float,
    delta_base: float,
) -> dict[str, np.ndarray]:
    """Return an empty schedule mapping.

    The symmetric quoter has no time-varying state. Validation is still
    performed so that misconfigured ``delta_base`` is caught at the same
    point as for AS.

    Raises
    ------
    ValueError
        If ``delta_base < 0``.
    """
    _validate_delta_base(delta_base)
    return {}


def quote_arrays(
    s_path: np.ndarray,
    delta_base: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized symmetric quote evaluation across a mid-price path.

    Parameters
    ----------
    s_path:
        Mid-price path of length ``n + 1``.
    delta_base:
        Constant half-spread. Must be ``>= 0``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(bid_quote, ask_quote, suppress_mask)``. All three have length
        ``n + 1``. ``suppress_mask`` is identically ``False`` because the
        symmetric quoter never suppresses on its own.

    Raises
    ------
    ValueError
        If ``delta_base < 0``.
    """
    _validate_delta_base(delta_base)
    s_arr = np.asarray(s_path, dtype=np.float64)
    bid_quote = s_arr - delta_base
    ask_quote = s_arr + delta_base
    suppress_mask = np.zeros(s_arr.shape, dtype=bool)
    return bid_quote, ask_quote, suppress_mask
