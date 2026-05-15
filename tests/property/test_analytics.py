"""Property tests for analytics primitives.

# Feature: etf-mm-arbitrage-simulator, Property 19: Max drawdown non-negativity
# Feature: etf-mm-arbitrage-simulator, Property 20: Spread capture per fill

Property 19 exercises the underlying drawdown helper
:func:`etf_mm_sim.analytics._max_drawdown_curve` directly so we can
generate arbitrary float64 P&L curves without first constructing a full
:class:`~etf_mm_sim.path_runner.PathResult`. Property 20 builds tiny
synthetic ``PathResult`` instances by hand, sampling fill flags / quotes
/ mid-prices independently, and asserts the per-fill spread-capture
formula matches manual reconstruction.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings, strategies as st

from etf_mm_sim.analytics import _max_drawdown_curve, spread_capture
from etf_mm_sim.path_runner import PathResult


# --------------------------------------------------------------------------- #
# Property 19: Max drawdown non-negativity                                     #
# --------------------------------------------------------------------------- #


_FINITE_FLOAT = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)


@given(
    curve=st.lists(_FINITE_FLOAT, min_size=1, max_size=200).map(
        lambda xs: np.asarray(xs, dtype=np.float64)
    )
)
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property_19_max_drawdown_non_negative(curve: np.ndarray) -> None:
    """Max drawdown is non-negative; zero iff curve is non-decreasing.

    For any 1-D float64 P&L curve of length ``>= 1``, the drawdown is
    ``max_i (running_max(curve)[i] - curve[i])``. By construction
    ``running_max(curve)[i] >= curve[i]``, so each per-step term is
    ``>= 0`` and the maximum is ``>= 0``. Equality with zero holds iff
    every per-step term is zero, i.e. iff ``curve[i] == running_max[i]``
    at every ``i``, which is the definition of monotonic non-decrease
    (in float64 arithmetic).

    **Validates: Requirements 8.4**
    """
    dd = _max_drawdown_curve(curve)
    assert dd >= 0.0

    # Direct check for monotonic non-decrease in the raw float64 sense.
    is_monotone = bool(np.all(np.diff(curve) >= 0.0)) if curve.size >= 2 else True
    if is_monotone:
        assert dd == 0.0
    else:
        assert dd > 0.0


# --------------------------------------------------------------------------- #
# Property 20: Spread capture per fill                                         #
# --------------------------------------------------------------------------- #


def _make_synthetic_path_result(
    s_path: np.ndarray,
    bid_quote: np.ndarray,
    ask_quote: np.ndarray,
    bid_fill: np.ndarray,
    ask_fill: np.ndarray,
) -> PathResult:
    """Build a minimal :class:`PathResult` for spread-capture testing.

    Only the array fields read by :func:`spread_capture` are populated
    meaningfully; the remaining fields receive plausible zero/empty
    values so the dataclass is constructible. The runner's invariants
    (NaN at suppressed steps, fill prices equal posted quotes, etc.) are
    not relevant here -- the property under test only reads
    ``s_path``, ``bid_quote``, ``ask_quote``, ``bid_fill``, ``ask_fill``.
    """
    n_plus_1 = int(s_path.shape[0])
    n = n_plus_1 - 1
    return PathResult(
        regime="synthetic",
        strategy="synthetic",
        path_index=0,
        s_path=np.asarray(s_path, dtype=np.float64),
        bid_quote=np.asarray(bid_quote, dtype=np.float64),
        ask_quote=np.asarray(ask_quote, dtype=np.float64),
        inventory=np.zeros(n_plus_1, dtype=np.int64),
        cash=np.zeros(n_plus_1, dtype=np.float64),
        bid_fill=np.asarray(bid_fill, dtype=bool),
        ask_fill=np.asarray(ask_fill, dtype=bool),
        n_bid_fills=int(np.sum(bid_fill)),
        n_ask_fills=int(np.sum(ask_fill)),
        terminal_pnl=0.0,
        kill_switch_step=None,
    )


@st.composite
def _spread_scenarios(draw):
    """Generate a tuple of synthetic arrays for one spread-capture test."""
    n = draw(st.integers(min_value=1, max_value=20))
    n_plus_1 = n + 1
    s_path = np.asarray(
        draw(
            st.lists(
                st.floats(
                    min_value=10.0,
                    max_value=1000.0,
                    allow_nan=False,
                    allow_infinity=False,
                    allow_subnormal=False,
                ),
                min_size=n_plus_1,
                max_size=n_plus_1,
            )
        ),
        dtype=np.float64,
    )
    # Half-spread per side; bid quote = s - hb, ask quote = s + ha. Both
    # half-spreads strictly positive so quotes are well-defined and the
    # captured spread on a fill is strictly positive too.
    hb = np.asarray(
        draw(
            st.lists(
                st.floats(
                    min_value=0.01,
                    max_value=5.0,
                    allow_nan=False,
                    allow_infinity=False,
                    allow_subnormal=False,
                ),
                min_size=n_plus_1,
                max_size=n_plus_1,
            )
        ),
        dtype=np.float64,
    )
    ha = np.asarray(
        draw(
            st.lists(
                st.floats(
                    min_value=0.01,
                    max_value=5.0,
                    allow_nan=False,
                    allow_infinity=False,
                    allow_subnormal=False,
                ),
                min_size=n_plus_1,
                max_size=n_plus_1,
            )
        ),
        dtype=np.float64,
    )
    bid_quote = s_path - hb
    ask_quote = s_path + ha
    bid_fill = np.asarray(
        draw(
            st.lists(st.booleans(), min_size=n, max_size=n)
        ),
        dtype=bool,
    )
    ask_fill = np.asarray(
        draw(
            st.lists(st.booleans(), min_size=n, max_size=n)
        ),
        dtype=bool,
    )
    return s_path, bid_quote, ask_quote, bid_fill, ask_fill


@given(scenario=_spread_scenarios())
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property_20_spread_capture(scenario) -> None:
    """Per-fill spread capture matches the closed-form definition.

    For every fill at step ``i``:

    * ask fill captures ``ask_quote[i] - s_path[i]``
    * bid fill captures ``s_path[i] - bid_quote[i]``

    The :func:`spread_capture` helper concatenates bid-side captures
    (in step order) followed by ask-side captures (in step order); this
    test reconstructs the same array manually from the input arrays and
    asserts byte-identical equality.

    **Validates: Requirements 8.7**
    """
    s_path, bid_quote, ask_quote, bid_fill, ask_fill = scenario
    pr = _make_synthetic_path_result(
        s_path=s_path,
        bid_quote=bid_quote,
        ask_quote=ask_quote,
        bid_fill=bid_fill,
        ask_fill=ask_fill,
    )

    bid_idx = np.flatnonzero(bid_fill)
    ask_idx = np.flatnonzero(ask_fill)
    expected_bid = s_path[bid_idx] - bid_quote[bid_idx]
    expected_ask = ask_quote[ask_idx] - s_path[ask_idx]
    expected = np.concatenate([expected_bid, expected_ask]).astype(np.float64)

    actual = spread_capture(pr)
    assert actual.shape == expected.shape
    assert actual.dtype == np.float64
    # Pure float subtraction with the same operands; exact equality.
    assert np.array_equal(actual, expected)
