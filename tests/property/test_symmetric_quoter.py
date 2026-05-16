"""Property tests for the symmetric constant-spread quoter.

# Feature: etf-mm-arbitrage-simulator, Property 9: Symmetric quoter is
#   mid-symmetric and inventory-independent

Realizes the single property from ``design.md`` §Correctness Properties
that targets the symmetric baseline quoter. The property combines two
requirements: quotes are independent of inventory (4.2) and the quote
midpoint equals the mid-price exactly (4.3, with 4.1 covered by the
construction itself).
"""

from __future__ import annotations

import math

from hypothesis import given, settings, strategies as st

from etf_mm_sim.quoters.symmetric import symmetric_quote


_s_strategy = st.floats(
    min_value=1e-2,
    max_value=1e4,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
_q_strategy = st.integers(min_value=-1000, max_value=1000)
_t_strategy = st.floats(
    min_value=0.0,
    max_value=10.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
_delta_base_strategy = st.floats(
    min_value=0.0,
    max_value=10.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)


@given(
    s=_s_strategy,
    q1=_q_strategy,
    q2=_q_strategy,
    t=_t_strategy,
    delta_base=_delta_base_strategy,
)
@settings(max_examples=50, deadline=None)
def test_property_9_inventory_independent(
    s: float,
    q1: int,
    q2: int,
    t: float,
    delta_base: float,
) -> None:
    """Quotes ignore inventory and the midpoint equals the mid-price.

    Two invocations with the same ``(s, t, delta_base)`` but different
    inventories must yield identical ``(bid, ask)`` pairs (Requirement
    4.2). The midpoint of those quotes equals the mid-price ``s``
    algebraically; in float64 the half-sum ``0.5 * ((s - d) + (s + d))``
    can drift by one ULP when ``d`` is comparable in magnitude to ``s``,
    so we compare with a tight ``isclose`` tolerance rather than ``==``.

    **Validates: Requirements 4.1, 4.2, 4.3**
    """
    bid_q1, ask_q1 = symmetric_quote(s, q1, t, delta_base)
    bid_q2, ask_q2 = symmetric_quote(s, q2, t, delta_base)

    # 4.2: inventory independence (exact equality holds because the quoter
    # produces identical float arithmetic for any inventory).
    assert bid_q1 == bid_q2
    assert ask_q1 == ask_q2

    # 4.1: bid = s - delta_base, ask = s + delta_base.
    assert bid_q1 == s - delta_base
    assert ask_q1 == s + delta_base

    # 4.3: midpoint equals s up to one ULP from the symmetric add.
    midpoint = (bid_q1 + ask_q1) * 0.5
    assert math.isclose(midpoint, s, rel_tol=1e-12, abs_tol=1e-12)
