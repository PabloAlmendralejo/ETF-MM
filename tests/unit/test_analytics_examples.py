"""Unit tests for analytics primitives on hand-rolled inputs.

Validates: Requirements 8.3 (Sharpe), 8.5 (fill-rate asymmetry),
8.6 (adverse selection), and 8.7 (spread capture / no-fill edge case).

Each test constructs a minimal :class:`~etf_mm_sim.path_runner.PathResult`
in code, sized to make the expected metric value trivially derivable by
inspection or hand calculation. The synthetic ``PathResult`` populates
only the array fields each metric reads; bookkeeping fields like
``inventory`` and ``cash`` are filled in only where needed by the metric
under test (e.g. Sharpe consumes the MTM curve = ``cash + inventory *
s_path``).
"""

from __future__ import annotations

import math

import numpy as np

from etf_mm_sim.analytics import (
    adverse_selection,
    fill_rate_asymmetry,
    mean_spread_capture,
    sharpe,
    spread_capture,
)
from etf_mm_sim.path_runner import PathResult


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _make_path_result(
    *,
    s_path: np.ndarray,
    cash: np.ndarray,
    inventory: np.ndarray,
    bid_fill: np.ndarray,
    ask_fill: np.ndarray,
    bid_quote: np.ndarray | None = None,
    ask_quote: np.ndarray | None = None,
) -> PathResult:
    """Construct a synthetic :class:`PathResult` with the given arrays.

    Quotes default to ``NaN``-filled (suitable for tests that ignore
    them). The function takes care of dtype canonicalization so callers
    can pass plain Python lists.
    """
    s_path = np.asarray(s_path, dtype=np.float64)
    cash = np.asarray(cash, dtype=np.float64)
    inventory = np.asarray(inventory, dtype=np.int64)
    bid_fill = np.asarray(bid_fill, dtype=bool)
    ask_fill = np.asarray(ask_fill, dtype=bool)
    n_plus_1 = s_path.shape[0]
    if bid_quote is None:
        bid_quote = np.full(n_plus_1, np.nan, dtype=np.float64)
    else:
        bid_quote = np.asarray(bid_quote, dtype=np.float64)
    if ask_quote is None:
        ask_quote = np.full(n_plus_1, np.nan, dtype=np.float64)
    else:
        ask_quote = np.asarray(ask_quote, dtype=np.float64)
    return PathResult(
        regime="unit",
        strategy="unit",
        path_index=0,
        s_path=s_path,
        bid_quote=bid_quote,
        ask_quote=ask_quote,
        inventory=inventory,
        cash=cash,
        bid_fill=bid_fill,
        ask_fill=ask_fill,
        n_bid_fills=int(np.sum(bid_fill)),
        n_ask_fills=int(np.sum(ask_fill)),
        terminal_pnl=float(cash[-1] + inventory[-1] * s_path[-1]),
        kill_switch_step=None,
    )


# --------------------------------------------------------------------------- #
# Sharpe                                                                      #
# --------------------------------------------------------------------------- #


def test_sharpe_constant_increment() -> None:
    """A linear MTM curve has zero std, so Sharpe collapses to ``0.0``.

    MTM = cash + inventory * s_path. With ``s_path == 1.0`` everywhere
    and ``inventory == 0`` everywhere, MTM equals ``cash``. Setting
    ``cash = [0, 1, 2, 3]`` produces increments ``[1, 1, 1]`` whose
    sample std is exactly ``0`` -- per the design we must return ``0.0``
    rather than ``NaN``.
    """
    pr = _make_path_result(
        s_path=np.ones(4, dtype=np.float64),
        cash=np.array([0.0, 1.0, 2.0, 3.0]),
        inventory=np.zeros(4, dtype=np.int64),
        bid_fill=np.zeros(3, dtype=bool),
        ask_fill=np.zeros(3, dtype=bool),
    )
    # Annualization factor is irrelevant when std == 0 and the formula
    # short-circuits to 0; we still pass a representative value.
    assert sharpe(pr, annualization_factor=math.sqrt(252.0)) == 0.0


def test_sharpe_alternating() -> None:
    """Hand-computed Sharpe on an alternating MTM curve.

    Curve ``[0, 1, 0, 1]`` gives increments ``[1, -1, 1]`` with
    ``mean = 1/3`` and sample std ``sqrt(((1-1/3)^2 + (-1-1/3)^2 +
    (1-1/3)^2) / 2) = sqrt(4/3) = 2/sqrt(3)``. With annualization factor
    ``A``, the Sharpe equals ``(1/3) / (2/sqrt(3)) * A = sqrt(3)/6 * A``.
    """
    pr = _make_path_result(
        s_path=np.ones(4, dtype=np.float64),
        cash=np.array([0.0, 1.0, 0.0, 1.0]),
        inventory=np.zeros(4, dtype=np.int64),
        bid_fill=np.zeros(3, dtype=bool),
        ask_fill=np.zeros(3, dtype=bool),
    )
    A = 2.5  # arbitrary positive
    expected = (math.sqrt(3.0) / 6.0) * A
    actual = sharpe(pr, annualization_factor=A)
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


# --------------------------------------------------------------------------- #
# Fill rate asymmetry                                                         #
# --------------------------------------------------------------------------- #


def test_fill_rate_asymmetry_simple() -> None:
    """``n_bid_fills - n_ask_fills`` on hand-rolled fill flags.

    Five inner steps with three bid fills and one ask fill should yield
    ``3 - 1 == 2``.
    """
    pr = _make_path_result(
        s_path=np.ones(6, dtype=np.float64),
        cash=np.zeros(6, dtype=np.float64),
        inventory=np.zeros(6, dtype=np.int64),
        bid_fill=np.array([True, True, False, True, False], dtype=bool),
        ask_fill=np.array([False, False, True, False, False], dtype=bool),
    )
    assert fill_rate_asymmetry(pr) == 2


# --------------------------------------------------------------------------- #
# Adverse selection                                                            #
# --------------------------------------------------------------------------- #


def test_adverse_selection_bid_unfavorable() -> None:
    """Bid fill, mid drops over the horizon: positive adverse selection.

    The path has 3 inner steps and 4 grid points. A bid fill at step 0
    looks ahead to step ``min(0 + 2, 3) == 2`` where ``s == 99``. The
    mid moved from ``100`` to ``99`` (down 1); we are long, so the
    drift against us is ``-(99 - 100) == +1``.
    """
    pr = _make_path_result(
        s_path=np.array([100.0, 99.5, 99.0, 99.0]),
        cash=np.zeros(4, dtype=np.float64),
        inventory=np.zeros(4, dtype=np.int64),
        bid_fill=np.array([True, False, False], dtype=bool),
        ask_fill=np.array([False, False, False], dtype=bool),
    )
    assert adverse_selection(pr, horizon_steps=2) == 1.0


def test_adverse_selection_ask_unfavorable() -> None:
    """Ask fill, mid rises over the horizon: positive adverse selection.

    Mid runs ``[100, 101, 102, 102]`` over 3 inner steps. An ask fill at
    step 0 looks ahead to step 2 where ``s == 102``: drift against us
    (we are short) is ``+(102 - 100) == +2``.
    """
    pr = _make_path_result(
        s_path=np.array([100.0, 101.0, 102.0, 102.0]),
        cash=np.zeros(4, dtype=np.float64),
        inventory=np.zeros(4, dtype=np.int64),
        bid_fill=np.array([False, False, False], dtype=bool),
        ask_fill=np.array([True, False, False], dtype=bool),
    )
    assert adverse_selection(pr, horizon_steps=2) == 2.0


# --------------------------------------------------------------------------- #
# No-fill edge cases                                                           #
# --------------------------------------------------------------------------- #


def test_no_fills_means_zero_AS_and_zero_capture() -> None:
    """A path with no fills has zero adverse selection and zero capture."""
    pr = _make_path_result(
        s_path=np.array([100.0, 101.0, 99.0, 100.5]),
        cash=np.zeros(4, dtype=np.float64),
        inventory=np.zeros(4, dtype=np.int64),
        bid_fill=np.zeros(3, dtype=bool),
        ask_fill=np.zeros(3, dtype=bool),
    )
    assert adverse_selection(pr, horizon_steps=1) == 0.0
    captures = spread_capture(pr)
    assert captures.shape == (0,)
    assert captures.dtype == np.float64
    assert mean_spread_capture(pr) == 0.0
