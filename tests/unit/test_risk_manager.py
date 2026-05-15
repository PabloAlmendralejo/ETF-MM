"""Unit tests for risk manager primitives.

Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5 (inventory bounds, kill-switch,
terminal flatten).

Each function in :mod:`etf_mm_sim.risk_manager` is covered by example-based
tests for typical inputs, edge cases, and validation errors.
"""

from __future__ import annotations

import pytest

from etf_mm_sim.risk_manager import (
    apply_inventory_bounds,
    flatten_terminal,
    kill_switch_triggered,
)


# --- apply_inventory_bounds -------------------------------------------------


def test_no_suppression_at_zero_inventory() -> None:
    assert apply_inventory_bounds(0, 50) == (False, False)


def test_suppress_bid_at_long_cap() -> None:
    assert apply_inventory_bounds(50, 50) == (True, False)


def test_suppress_ask_at_short_cap() -> None:
    assert apply_inventory_bounds(-50, 50) == (False, True)


def test_suppress_bid_above_long_cap() -> None:
    # Defensive: should not occur in practice, but the >= predicate catches it.
    assert apply_inventory_bounds(60, 50) == (True, False)


def test_q_max_zero_suppresses_both() -> None:
    # q_max == 0 means the quoter cannot hold any inventory: both sides always
    # suppressed. q == 0 satisfies both q >= 0 and q <= -0.
    assert apply_inventory_bounds(0, 0) == (True, True)


def test_q_max_negative_raises() -> None:
    with pytest.raises(ValueError):
        apply_inventory_bounds(0, -1)


# --- kill_switch_triggered --------------------------------------------------


def test_kill_switch_disabled_when_L_kill_none() -> None:
    assert kill_switch_triggered(-9999.0, None) is False


def test_kill_switch_not_triggered_above_threshold() -> None:
    assert kill_switch_triggered(-50.0, 100.0) is False


def test_kill_switch_triggered_at_threshold() -> None:
    assert kill_switch_triggered(-100.0, 100.0) is True


def test_kill_switch_triggered_below_threshold() -> None:
    assert kill_switch_triggered(-200.0, 100.0) is True


def test_kill_switch_negative_L_kill_raises() -> None:
    with pytest.raises(ValueError):
        kill_switch_triggered(-50.0, -100.0)


# --- flatten_terminal -------------------------------------------------------


def test_flatten_terminal_long_inventory() -> None:
    # Long 10 shares at 100.5 with 5.0 cash: flatten yields 5.0 + 10*100.5 = 1010.0.
    assert flatten_terminal(10, 100.5, 5.0) == 1010.0


def test_flatten_terminal_short_inventory() -> None:
    # Short 3 shares at 50.0 with 200.0 cash: flatten yields 200.0 + (-3)*50.0 = 50.0.
    assert flatten_terminal(-3, 50.0, 200.0) == 50.0


def test_flatten_terminal_flat() -> None:
    # No inventory: cash is unchanged.
    assert flatten_terminal(0, 100.0, 50.0) == 50.0
