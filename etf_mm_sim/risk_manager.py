"""Risk manager primitives.

Implements inventory-bound suppression, the kill-switch predicate, and the
terminal-flatten cash adjustment. See `design.md` §Risk_Manager.

The functions in this module are pure and side-effect free; latch behavior
for the kill-switch (i.e. once-fired-stays-fired) is the caller's
responsibility (the path runner persists the flag across steps).
"""

from __future__ import annotations

__all__ = [
    "apply_inventory_bounds",
    "kill_switch_triggered",
    "flatten_terminal",
]


def apply_inventory_bounds(q: int, q_max: int) -> tuple[bool, bool]:
    """Return ``(suppress_bid, suppress_ask)`` flags for inventory ``q``.

    A bid is suppressed when ``q >= q_max`` (no more buying once at the long
    cap) and an ask is suppressed when ``q <= -q_max`` (no more selling once
    at the short cap). The comparisons are non-strict so that a defensive
    overshoot still flags suppression on the offending side.

    The edge case ``q_max == 0`` validly suppresses both sides at all times,
    which corresponds to a quoter that never trades.

    Parameters
    ----------
    q:
        Current signed inventory.
    q_max:
        Maximum permitted absolute inventory. Must be non-negative.

    Returns
    -------
    tuple[bool, bool]
        ``(suppress_bid, suppress_ask)``.

    Raises
    ------
    ValueError
        If ``q_max`` is negative.
    """
    if q_max < 0:
        raise ValueError(f"q_max must be non-negative, got {q_max}")
    suppress_bid = q >= q_max
    suppress_ask = q <= -q_max
    return suppress_bid, suppress_ask


def kill_switch_triggered(running_pnl: float, L_kill: float | None) -> bool:
    """Return ``True`` iff the kill-switch should fire on this step.

    The kill-switch fires when ``running_pnl <= -L_kill``. Passing
    ``L_kill=None`` disables the kill-switch entirely (always returns
    ``False``). Latching the flag across subsequent steps is the caller's
    responsibility.

    Parameters
    ----------
    running_pnl:
        Current running mark-to-market P&L.
    L_kill:
        Positive loss threshold, or ``None`` to disable.

    Returns
    -------
    bool
        ``True`` if the kill-switch should fire, else ``False``.

    Raises
    ------
    ValueError
        If ``L_kill`` is not ``None`` and is non-positive.
    """
    if L_kill is None:
        return False
    if L_kill <= 0:
        raise ValueError(f"L_kill must be positive, got {L_kill}")
    return running_pnl <= -L_kill


def flatten_terminal(q_T: int, s_T: float, cash_T: float) -> float:
    """Return the terminal cash after flattening residual inventory.

    The flatten adjustment closes any residual inventory ``q_T`` at the
    terminal mid-price ``s_T``: positive inventory is sold (cash in),
    negative inventory is bought back (cash out). The arithmetic is
    ``cash_T + q_T * s_T`` for both signs.

    Parameters
    ----------
    q_T:
        Inventory at the terminal step.
    s_T:
        Mid-price at the terminal step.
    cash_T:
        Cash balance at the terminal step prior to flatten.

    Returns
    -------
    float
        Terminal cash after the flatten adjustment.
    """
    return cash_T + q_T * s_T
