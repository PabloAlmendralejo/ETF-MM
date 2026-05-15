"""Counterfactual no-skew replay for drawdown attribution (Req 8.9).

Reuses the realized AS bid/ask fill stream against a no-skew quoter that
prices at ``s_path[i] +/- delta_star_sched[i]`` to isolate the cash
impact of inventory skew. See ``design.md`` §Counterfactual.

Methodology
-----------
The AS fill stream encoded in :pyattr:`PathResult.bid_fill` and
:pyattr:`PathResult.ask_fill` is held *fixed*; we recompute cash and the
mark-to-market P&L curve under counterfactual quotes ``s +/-
delta_star``. Because fills are held fixed, the inventory trajectory of
the counterfactual replay equals the AS inventory trajectory exactly
(Property 22). What differs is the realized cash effect of each fill:
the AS quoter buys/sells at *skewed* prices, while the counterfactual
buys/sells at the unskewed mid +/- delta_star.

The drawdown of the counterfactual P&L curve, compared to the AS
drawdown, isolates the cash effect of skew. A positive
``cf_max_dd - as_max_dd`` means the AS skew reduced drawdown vs. an
otherwise-identical no-skew quoter.

Caveats
-------
This is the *cash* effect of skew, not the effect on fill probabilities.
The AS-imposed inventory bounds are still implicit in the fill stream
(no fill ever pushed inventory past ``q_max`` because AS suppressed the
relevant side at the boundary). See ``design.md`` Open Question 10.

Suppression handling
--------------------
The AS run may suppress quotes at certain steps (terminal step,
inventory bound, kill-switch). We reflect this in the counterfactual
quote arrays by setting ``cf_bid[i]`` / ``cf_ask[i]`` to ``NaN`` whenever
the AS path's recorded quote was ``NaN`` at step ``i``. The cash
recurrence is unaffected because at suppressed steps the AS fill flags
are ``False``, so no spurious ``NaN * False`` arithmetic enters the cash
update.
"""

from __future__ import annotations

import numpy as np

from .bootstrap import paired_bootstrap_ci
from .path_runner import PathResult

__all__ = [
    "replay_no_skew",
    "skew_drawdown_attribution",
]


def replay_no_skew(
    as_path: PathResult,
    delta_star_sched: np.ndarray,
) -> PathResult:
    """Replay the AS fill stream against a counterfactual no-skew quoter.

    Parameters
    ----------
    as_path:
        Realized AS path. The fields read are ``s_path``, ``bid_quote``,
        ``ask_quote``, ``inventory``, ``bid_fill``, ``ask_fill``,
        ``n_bid_fills``, ``n_ask_fills``, ``regime``, ``path_index``.
    delta_star_sched:
        AS optimal half-spread schedule of length ``N + 1`` (the same
        schedule produced by
        :func:`etf_mm_sim.quoters.avellaneda_stoikov.precompute`). The
        counterfactual posts ``s_path[i] +/- delta_star_sched[i]`` at
        every non-suppressed step.

    Returns
    -------
    PathResult
        Counterfactual path result with ``strategy =
        "counterfactual_no_skew"``. The ``inventory``, ``bid_fill``,
        ``ask_fill``, ``n_bid_fills``, and ``n_ask_fills`` fields are
        identical to ``as_path``; ``cash`` (and therefore
        ``terminal_pnl``) differs because fill prices differ.
        ``kill_switch_step`` is ``None``: the counterfactual is a pure
        replay and does not re-evaluate live risk controls.

    Raises
    ------
    ValueError
        If ``delta_star_sched`` does not match the shape of
        ``as_path.s_path``.
    """
    s_path = np.asarray(as_path.s_path, dtype=np.float64)
    delta_arr = np.asarray(delta_star_sched, dtype=np.float64)
    if delta_arr.shape != s_path.shape:
        raise ValueError(
            "delta_star_sched: shape must match as_path.s_path; "
            f"got {delta_arr.shape!r} vs {s_path.shape!r}"
        )

    n_plus_1 = int(s_path.shape[0])
    n = n_plus_1 - 1

    # Counterfactual quotes at the unskewed mid +/- delta_star. Any step
    # where the AS path itself was suppressed (recorded as NaN in the
    # AS quote arrays) inherits NaN here so consumers can detect it.
    suppressed = np.isnan(as_path.bid_quote) | np.isnan(as_path.ask_quote)
    cf_bid = np.where(suppressed, np.nan, s_path - delta_arr)
    cf_ask = np.where(suppressed, np.nan, s_path + delta_arr)

    # Inventory and fill flags are identical to AS by construction
    # (Property 22). We copy to avoid aliasing the caller's arrays.
    inventory = np.array(as_path.inventory, dtype=np.int64, copy=True)
    bid_fill = np.array(as_path.bid_fill, dtype=bool, copy=True)
    ask_fill = np.array(as_path.ask_fill, dtype=bool, copy=True)

    # Cash recurrence: cf_cash[0] = 0; for i in {0, ..., n-1}:
    #   cf_cash[i+1] = cf_cash[i]
    #     + (ask_fill[i] * cf_ask[i]) - (bid_fill[i] * cf_bid[i])
    # We walk the loop in pure Python; n is at most ~1000 for the
    # reference workload and the loop body is trivial.
    cf_cash = np.zeros(n_plus_1, dtype=np.float64)
    cash_now = 0.0
    for i in range(n):
        if ask_fill[i]:
            cash_now += float(cf_ask[i])
        if bid_fill[i]:
            cash_now -= float(cf_bid[i])
        cf_cash[i + 1] = cash_now

    terminal_pnl = float(cf_cash[n] + inventory[n] * s_path[n])

    return PathResult(
        regime=as_path.regime,
        strategy="counterfactual_no_skew",
        path_index=as_path.path_index,
        s_path=s_path,
        bid_quote=cf_bid,
        ask_quote=cf_ask,
        inventory=inventory,
        cash=cf_cash,
        bid_fill=bid_fill,
        ask_fill=ask_fill,
        n_bid_fills=int(as_path.n_bid_fills),
        n_ask_fills=int(as_path.n_ask_fills),
        terminal_pnl=terminal_pnl,
        kill_switch_step=None,
    )


def skew_drawdown_attribution(
    as_cell: list[PathResult],
    delta_star_sched_per_path: list[np.ndarray],
    bootstrap_iterations: int,
    bootstrap_alpha: float,
    bootstrap_ss: np.random.SeedSequence,
) -> tuple[float, tuple[float, float]]:
    """Compute the AS-skew drawdown-attribution metric and its CI.

    For each AS path we replay the same fill stream against the
    counterfactual no-skew quoter and compute the per-path drawdown
    difference ``cf_max_dd - as_max_dd``. The mean of these differences
    is the skew-drawdown attribution; positive values mean AS inventory
    skew reduced drawdown vs. the same-spread no-skew counterfactual
    (Req 8.9).

    Parameters
    ----------
    as_cell:
        AS path results for one regime. Must be non-empty.
    delta_star_sched_per_path:
        AS half-spread schedule per path. In the typical use case the
        schedule is shared across all paths in a regime (it depends only
        on ``T``, ``dt``, ``gamma``, ``sigma``, ``k``); callers can pass
        ``[shared_schedule] * len(as_cell)`` to satisfy the per-path
        contract without copying. Must have the same length as
        ``as_cell``.
    bootstrap_iterations, bootstrap_alpha:
        Bootstrap parameters forwarded to
        :func:`etf_mm_sim.bootstrap.paired_bootstrap_ci`.
    bootstrap_ss:
        Seed sequence dedicated to this bootstrap. Typically a fresh
        ``spawn`` from :func:`etf_mm_sim.seeding.analytics_seed` so the
        attribution bootstrap does not share state with other CIs.

    Returns
    -------
    tuple[float, tuple[float, float]]
        ``(mean_diff, (lo, hi))`` where ``mean_diff`` equals
        ``mean(cf_max_dd - as_max_dd)`` exactly (the bootstrap point
        estimate is the sample mean by construction) and ``(lo, hi)``
        is the percentile-bootstrap CI.

    Raises
    ------
    ValueError
        If ``as_cell`` is empty or ``delta_star_sched_per_path`` has a
        mismatched length.
    """
    # Local import to avoid a circular import between analytics and
    # counterfactual (analytics imports bootstrap; we import
    # max_drawdown lazily here to keep the dependency graph clean).
    from .analytics import max_drawdown

    if len(as_cell) == 0:
        raise ValueError("as_cell: must be non-empty")
    if len(delta_star_sched_per_path) != len(as_cell):
        raise ValueError(
            "delta_star_sched_per_path: length must equal len(as_cell); "
            f"got {len(delta_star_sched_per_path)} vs {len(as_cell)}"
        )

    diffs = np.empty(len(as_cell), dtype=np.float64)
    for i, (as_path, sched) in enumerate(
        zip(as_cell, delta_star_sched_per_path)
    ):
        cf = replay_no_skew(as_path, sched)
        diffs[i] = max_drawdown(cf) - max_drawdown(as_path)

    point, lo, hi = paired_bootstrap_ci(
        diffs, bootstrap_iterations, bootstrap_alpha, bootstrap_ss
    )
    return point, (lo, hi)
