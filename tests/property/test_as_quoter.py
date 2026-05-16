"""Property tests for the Avellaneda-Stoikov quoter.

# Feature: etf-mm-arbitrage-simulator, Property 6: AS quote construction matches closed form
# Feature: etf-mm-arbitrage-simulator, Property 7: AS skew sign convention
# Feature: etf-mm-arbitrage-simulator, Property 8: AS no-quote at terminal

Each test function realizes exactly one of the three properties from
``design.md`` §Correctness Properties that targets the AS quoter. The
closed-form identities (Property 6) are checked with ``math.isclose`` at
``rel_tol=1e-12, abs_tol=1e-12`` because the implementation and the test
share the same arithmetic; we want to catch *algebraic* drift, not the
fp noise of intermediate operations.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, given, settings, strategies as st

from etf_mm_sim.quoters.avellaneda_stoikov import as_quote, quote_arrays


# --------------------------------------------------------------------------- #
# Shared strategies                                                           #
# --------------------------------------------------------------------------- #


# Mid-price: bounded so the closed-form arithmetic stays well within float64
# precision and the magnitude does not dominate the abs_tol.
_s_strategy = st.floats(
    min_value=1e-2,
    max_value=1e4,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
# Inventory: integer-valued in production but accepted as a float by the
# quoter; we generate ints in a realistic risk-bounded range.
_q_strategy = st.integers(min_value=-100, max_value=100)
# Current time and remaining horizon. T = t + T_minus_t with T_minus_t > 0.
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
# AS parameters bounded to keep gamma / k well away from numerical edge cases.
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
# Property 6: AS quote construction matches closed form                        #
# --------------------------------------------------------------------------- #


@given(
    s=_s_strategy,
    q=_q_strategy,
    t=_t_strategy,
    T_minus_t=_T_minus_t_strategy,
    gamma=_gamma_strategy,
    sigma=_sigma_strategy,
    k=_k_strategy,
)
@settings(max_examples=50, deadline=None)
def test_property_6_closed_form(
    s: float,
    q: int,
    t: float,
    T_minus_t: float,
    gamma: float,
    sigma: float,
    k: float,
) -> None:
    """``(bid, ask)`` decompose into the AS midpoint and half-spread.

    The quoter is required to satisfy::

        midpoint     = s - q * gamma * sigma**2 * (T - t)
        half_spread  = 0.5 * gamma * sigma**2 * (T - t) + (1/gamma) * log1p(gamma/k)

    so reading off ``(midpoint, half_spread)`` from the returned
    ``(bid, ask)`` and comparing to the formulas catches any algebraic drift.

    **Validates: Requirements 3.1, 3.2, 3.3**
    """
    T = t + T_minus_t
    bid, ask = as_quote(s, q, t, T, gamma, sigma, k)
    midpoint = 0.5 * (bid + ask)
    half_spread = 0.5 * (ask - bid)

    # Use the same arithmetic the implementation uses (``T - t``) so the
    # test does not penalize the ULP drift between ``(t + T_minus_t) - t``
    # and the original ``T_minus_t``.
    horizon = T - t
    expected_midpoint = s - q * gamma * sigma * sigma * horizon
    expected_half_spread = (
        0.5 * gamma * sigma * sigma * horizon
        + (1.0 / gamma) * math.log1p(gamma / k)
    )

    assert math.isclose(
        midpoint, expected_midpoint, rel_tol=1e-12, abs_tol=1e-12
    )
    assert math.isclose(
        half_spread, expected_half_spread, rel_tol=1e-12, abs_tol=1e-12
    )


# --------------------------------------------------------------------------- #
# Property 7: AS skew sign convention                                          #
# --------------------------------------------------------------------------- #


@given(
    s=_s_strategy,
    q=st.integers(min_value=1, max_value=100),
    t=_t_strategy,
    T_minus_t=_T_minus_t_strategy,
    gamma=_gamma_strategy,
    # sigma > 0: at sigma == 0 the inventory-skew term is zero and the
    # midpoint equals s regardless of q, so the sign relation is trivially
    # satisfied with equality but does not exercise the skew. We assert the
    # strict inequality on the active-skew regime.
    sigma=st.floats(
        min_value=1e-2,
        max_value=5.0,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    ),
    k=_k_strategy,
)
@settings(max_examples=50, deadline=None)
def test_property_7_skew_sign_positive_q(
    s: float,
    q: int,
    t: float,
    T_minus_t: float,
    gamma: float,
    sigma: float,
    k: float,
) -> None:
    """``q > 0`` shifts the quote midpoint *below* the mid-price.

    **Validates: Requirements 3.4**
    """
    # Skip pathological inputs where the algebraic skew term
    # ``q * gamma * sigma**2 * T_minus_t`` is smaller than the float64
    # resolution at ``s`` and the subtraction collapses back to ``s``.
    # The closed-form identity is unaffected; this is a fp resolution
    # limit, not a sign-convention bug.
    skew = abs(q) * gamma * sigma * sigma * T_minus_t
    assume(skew > abs(s) * 1e-15)
    T = t + T_minus_t
    bid, ask = as_quote(s, q, t, T, gamma, sigma, k)
    midpoint = 0.5 * (bid + ask)
    assert midpoint < s


@given(
    s=_s_strategy,
    q=st.integers(min_value=-100, max_value=-1),
    t=_t_strategy,
    T_minus_t=_T_minus_t_strategy,
    gamma=_gamma_strategy,
    sigma=st.floats(
        min_value=1e-2,
        max_value=5.0,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    ),
    k=_k_strategy,
)
@settings(max_examples=50, deadline=None)
def test_property_7_skew_sign_negative_q(
    s: float,
    q: int,
    t: float,
    T_minus_t: float,
    gamma: float,
    sigma: float,
    k: float,
) -> None:
    """``q < 0`` shifts the quote midpoint *above* the mid-price.

    **Validates: Requirements 3.5**
    """
    skew = abs(q) * gamma * sigma * sigma * T_minus_t
    assume(skew > abs(s) * 1e-15)
    T = t + T_minus_t
    bid, ask = as_quote(s, q, t, T, gamma, sigma, k)
    midpoint = 0.5 * (bid + ask)
    assert midpoint > s


@given(
    s=_s_strategy,
    t=_t_strategy,
    T_minus_t=_T_minus_t_strategy,
    gamma=_gamma_strategy,
    sigma=_sigma_strategy,
    k=_k_strategy,
)
@settings(max_examples=50, deadline=None)
def test_property_7_skew_sign_zero_q(
    s: float,
    t: float,
    T_minus_t: float,
    gamma: float,
    sigma: float,
    k: float,
) -> None:
    """``q == 0`` yields a midpoint exactly equal to the mid-price.

    **Validates: Requirements 3.6**
    """
    T = t + T_minus_t
    bid, ask = as_quote(s, 0, t, T, gamma, sigma, k)
    midpoint = 0.5 * (bid + ask)
    assert math.isclose(midpoint, s, rel_tol=1e-12, abs_tol=1e-12)


# --------------------------------------------------------------------------- #
# Property 8: AS no-quote at terminal                                          #
# --------------------------------------------------------------------------- #


@given(
    s=_s_strategy,
    q=_q_strategy,
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
@settings(max_examples=25, deadline=None)
def test_property_8_no_quote_at_terminal_scalar(
    s: float,
    q: int,
    T: float,
    gamma: float,
    sigma: float,
    k: float,
) -> None:
    """``as_quote`` raises ``ValueError`` when ``t >= T``.

    **Validates: Requirements 3.7**
    """
    with pytest.raises(ValueError):
        as_quote(s, q, T, T, gamma, sigma, k)
    # Strictly past the horizon as well.
    with pytest.raises(ValueError):
        as_quote(s, q, T + 1e-9, T, gamma, sigma, k)


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
@settings(max_examples=25, deadline=None)
def test_property_8_no_quote_at_terminal_vector(
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

    **Validates: Requirements 3.7**
    """
    T = n * dt
    s_path = np.full(n + 1, s0, dtype=np.float64)
    q_path = np.zeros(n + 1, dtype=np.float64)
    bid, ask, mask = quote_arrays(s_path, dt, T, gamma, sigma, k, q_path)

    assert mask.shape == (n + 1,)
    assert mask[-1] is np.True_ or bool(mask[-1]) is True
    assert math.isnan(bid[-1])
    assert math.isnan(ask[-1])
    # Non-terminal entries must be unsuppressed and finite.
    assert not mask[:-1].any()
    assert np.isfinite(bid[:-1]).all()
    assert np.isfinite(ask[:-1]).all()
