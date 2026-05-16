"""Property tests for the Semi-Avellaneda-Stoikov quoter.

# Feature: etf-mm-arbitrage-simulator
# Property A: Semi-AS quotes are mid-symmetric (centered on s)
# Property B: Semi-AS half-spread equals AS delta_star (closed form)
# Property C: Semi-AS suppresses at t >= T (same as AS terminal handling)

Each test function realizes exactly one of the three correctness
properties for the Semi-AS quoter. The closed-form identities are
checked with ``math.isclose`` at ``rel_tol=1e-12`` because the
implementation and the test share the same arithmetic; we want to catch
*algebraic* drift, not the fp noise of intermediate operations. The
spread schedule is additionally cross-checked against the AS module's
``precompute`` to confirm pointwise agreement.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from etf_mm_sim.quoters.avellaneda_stoikov import precompute as as_precompute
from etf_mm_sim.quoters.semi_as import (
    precompute as semi_precompute,
    quote_arrays,
    semi_as_quote,
)


# --------------------------------------------------------------------------- #
# Shared strategies                                                           #
# --------------------------------------------------------------------------- #


_s_strategy = st.floats(
    min_value=1e-2,
    max_value=1e4,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
_t_strategy = st.floats(
    min_value=0.0,
    max_value=10.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
_T_minus_t_strategy = st.floats(
    min_value=1e-6,
    max_value=10.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
_gamma_strategy = st.floats(
    min_value=1e-2,
    max_value=5.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
_sigma_strategy = st.floats(
    min_value=0.0,
    max_value=5.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
_k_strategy = st.floats(
    min_value=1e-1,
    max_value=10.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)


# --------------------------------------------------------------------------- #
# Property A: Mid symmetry                                                    #
# --------------------------------------------------------------------------- #


@given(
    s=_s_strategy,
    t=_t_strategy,
    T_minus_t=_T_minus_t_strategy,
    gamma=_gamma_strategy,
    sigma=_sigma_strategy,
    k=_k_strategy,
)
@settings(max_examples=50, deadline=None)
def test_property_a_mid_symmetry(
    s: float,
    t: float,
    T_minus_t: float,
    gamma: float,
    sigma: float,
    k: float,
) -> None:
    """``(bid + ask) / 2 == s`` (Semi-AS is centered on the mid).

    Independent of any inventory: the quoter is constructed with no
    inventory argument so this is structurally guaranteed. The half-sum
    can drift by one ULP for large ``s`` so we compare via
    :func:`math.isclose`.
    """
    T = t + T_minus_t
    bid, ask = semi_as_quote(s, t, T, gamma, sigma, k)
    midpoint = 0.5 * (bid + ask)
    assert math.isclose(midpoint, s, rel_tol=1e-12, abs_tol=1e-12)


# --------------------------------------------------------------------------- #
# Property B: Spread matches AS delta_star                                    #
# --------------------------------------------------------------------------- #


@given(
    s=_s_strategy,
    t=_t_strategy,
    T_minus_t=_T_minus_t_strategy,
    gamma=_gamma_strategy,
    sigma=_sigma_strategy,
    k=_k_strategy,
)
@settings(max_examples=50, deadline=None)
def test_property_b_spread_matches_as_delta_star_scalar(
    s: float,
    t: float,
    T_minus_t: float,
    gamma: float,
    sigma: float,
    k: float,
) -> None:
    """``(ask - bid) / 2`` equals the closed-form AS half-spread.

        delta*(t) = 0.5 * gamma * sigma**2 * (T - t) + (1/gamma) * log1p(gamma/k)
    """
    T = t + T_minus_t
    bid, ask = semi_as_quote(s, t, T, gamma, sigma, k)
    half_spread = 0.5 * (ask - bid)

    expected = 0.5 * gamma * sigma * sigma * T_minus_t + (1.0 / gamma) * math.log1p(
        gamma / k
    )
    assert math.isclose(half_spread, expected, rel_tol=1e-12, abs_tol=1e-12)


@given(
    s0=st.floats(
        min_value=1e-2,
        max_value=1e4,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    ),
    n=st.integers(min_value=2, max_value=50),
    dt=st.floats(
        min_value=1e-3,
        max_value=1e-1,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    ),
    gamma=_gamma_strategy,
    sigma=_sigma_strategy,
    k=_k_strategy,
)
@settings(max_examples=50, deadline=None)
def test_property_b_spread_matches_as_delta_star_vector(
    s0: float,
    n: int,
    dt: float,
    gamma: float,
    sigma: float,
    k: float,
) -> None:
    """Semi-AS schedule agrees pointwise with AS ``delta_star`` schedule.

    Cross-check Semi-AS ``precompute`` against AS ``precompute`` using
    the same ``(s_path, dt, T, gamma, sigma, k)`` inputs. Both modules
    reproduce the same closed form so the resulting arrays must be
    bit-identical (``np.array_equal``).
    """
    T = n * dt
    s_path = np.full(n + 1, s0, dtype=np.float64)
    semi = semi_precompute(s_path, dt, T, gamma, sigma, k)["delta_star"]
    asd = as_precompute(s_path, dt, T, gamma, sigma, k)["delta_star"]

    assert semi.shape == asd.shape
    # Both modules build the same array via identical arithmetic ops on
    # the same inputs, so equality is exact.
    assert np.array_equal(semi, asd)


# --------------------------------------------------------------------------- #
# Property C: Terminal suppression                                            #
# --------------------------------------------------------------------------- #


@given(
    s=_s_strategy,
    T=st.floats(
        min_value=1e-3,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    ),
    gamma=_gamma_strategy,
    sigma=_sigma_strategy,
    k=_k_strategy,
)
@settings(max_examples=50, deadline=None)
def test_property_c_terminal_suppression_scalar(
    s: float,
    T: float,
    gamma: float,
    sigma: float,
    k: float,
) -> None:
    """``semi_as_quote`` raises ``ValueError`` at and past the horizon."""
    with pytest.raises(ValueError):
        semi_as_quote(s, T, T, gamma, sigma, k)
    with pytest.raises(ValueError):
        semi_as_quote(s, T + 1e-9, T, gamma, sigma, k)


@given(
    s0=st.floats(
        min_value=1e-2,
        max_value=1e4,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    ),
    n=st.integers(min_value=2, max_value=50),
    dt=st.floats(
        min_value=1e-3,
        max_value=1e-1,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    ),
    gamma=_gamma_strategy,
    sigma=_sigma_strategy,
    k=_k_strategy,
)
@settings(max_examples=50, deadline=None)
def test_property_c_terminal_suppression_vector(
    s0: float,
    n: int,
    dt: float,
    gamma: float,
    sigma: float,
    k: float,
) -> None:
    """``quote_arrays`` flags the terminal step in ``suppress_mask``.

    The suppression mask must be ``True`` at index ``n`` (where ``t_n ==
    T``) and the corresponding ``bid``/``ask`` entries must be NaN.
    Non-terminal entries must be unsuppressed and finite.
    """
    T = n * dt
    s_path = np.full(n + 1, s0, dtype=np.float64)
    bid, ask, mask = quote_arrays(s_path, dt, T, gamma, sigma, k)

    assert mask.shape == (n + 1,)
    assert bool(mask[-1]) is True
    assert math.isnan(bid[-1])
    assert math.isnan(ask[-1])
    assert not mask[:-1].any()
    assert np.isfinite(bid[:-1]).all()
    assert np.isfinite(ask[:-1]).all()
