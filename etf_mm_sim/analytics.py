"""P&L and performance analytics.

Computes terminal P&L, the mark-to-market (MTM) curve, annualized Sharpe
ratio, max drawdown, fill-rate asymmetry, adverse selection, and spread
capture per :class:`~etf_mm_sim.path_runner.PathResult`, plus per-cell
aggregation into :class:`CellSummary`. See ``design.md`` §PnL_Analyzer.

Conventions
-----------
* The MTM curve has length ``N + 1``: ``mtm[i] = cash[i] + inventory[i] *
  s_path[i]`` for ``i in {0, ..., N}``. By the path runner's contract the
  terminal-flatten adjustment is *not* baked into ``cash[N]``; we apply it
  implicitly via ``inventory[N] * s_path[N]`` so the post-flatten terminal
  equals ``mtm[N]``.
* The Sharpe ratio is computed on the per-step P&L *increments*
  ``np.diff(mtm)``. We use the sample standard deviation (``ddof=1``) and
  return ``0.0`` whenever the increments are constant or the path is
  too short for ``ddof=1`` to be defined; this avoids ``NaN``-poisoning
  downstream aggregates.
* Adverse selection follows the design's resolved Q6: positive values
  mean the mid moved *against* the filled side over the lookahead
  horizon. Bid fills (we are long) are penalized when the mid drops; ask
  fills (we are short) are penalized when the mid rises.
* Spread capture per fill is the realized half-spread captured at fill
  time: ``ask_quote[i] - s_path[i]`` on ask fills and ``s_path[i] -
  bid_quote[i]`` on bid fills (Req 8.7).

Edge cases
----------
* No fills on a path: ``adverse_selection`` and ``mean_spread_capture``
  return ``0.0``; ``spread_capture`` returns an empty ``float64`` array.
* Empty cell: :func:`aggregate_cell` returns an all-zero summary with
  ``n_paths=0``.
* Single-path cell: terminal-P&L std uses ``ddof=1`` which is undefined
  for one sample, so we return ``0.0`` explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bootstrap import paired_bootstrap_ci
from .path_runner import PathResult

__all__ = [
    "terminal_pnl",
    "mtm_pnl_curve",
    "sharpe",
    "max_drawdown",
    "fill_rate_asymmetry",
    "adverse_selection",
    "spread_capture",
    "mean_spread_capture",
    "CellSummary",
    "aggregate_cell",
    "PairedComparison",
    "compute_paired_comparison",
]


# --------------------------------------------------------------------------- #
# Per-path metrics                                                            #
# --------------------------------------------------------------------------- #


def terminal_pnl(pr: PathResult) -> float:
    """Return the post-flatten terminal P&L ``cash[N] + inventory[N] * s_path[N]``.

    This duplicates :pyattr:`PathResult.terminal_pnl` for symmetry with
    the rest of this module; tests that consume only the analytics API
    can stay agnostic about the runner's internal book-keeping.
    """
    return float(pr.cash[-1] + pr.inventory[-1] * pr.s_path[-1])


def mtm_pnl_curve(pr: PathResult) -> np.ndarray:
    """Return the per-step mark-to-market P&L curve of length ``N + 1``.

    ``mtm[i] = cash[i] + inventory[i] * s_path[i]``. Computed in
    ``float64``; the runner already stores ``cash`` as ``float64`` and
    ``inventory`` as ``int64``, so the multiply-add is exact at the
    representable end and matches the terminal-flatten arithmetic.
    """
    return pr.cash + pr.inventory.astype(np.float64) * pr.s_path


def _max_drawdown_curve(curve: np.ndarray) -> float:
    """Return the maximum drawdown of an arbitrary 1-D P&L curve.

    Implemented separately from :func:`max_drawdown` so that property
    tests can exercise the underlying arithmetic on synthetic curves
    without first constructing a full :class:`PathResult`.

    The maximum drawdown is the largest gap between the running maximum
    of the curve and the curve itself: ``max_i (max(curve[:i+1]) -
    curve[i])``. By construction ``max(curve[:i+1]) >= curve[i]`` so the
    quantity is always non-negative; it is exactly zero iff the curve is
    monotonically non-decreasing.
    """
    if curve.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(curve)
    return float(np.max(running_max - curve))


def max_drawdown(pr: PathResult) -> float:
    """Maximum drawdown of the MTM curve (Req 8.4).

    Always ``>= 0``; equal to ``0`` iff the MTM curve is monotonically
    non-decreasing.
    """
    return _max_drawdown_curve(mtm_pnl_curve(pr))


def sharpe(pr: PathResult, annualization_factor: float) -> float:
    """Annualized Sharpe of the per-step MTM increments.

    Returns ``mean(increments) / std(increments, ddof=1) *
    annualization_factor``; returns ``0.0`` (rather than ``NaN``) when
    the standard deviation is zero or undefined. Per Req 8.3 the
    annualization factor is supplied by the caller (typically
    ``cfg.analytics.sharpe_annualization_factor``).
    """
    curve = mtm_pnl_curve(pr)
    increments = np.diff(curve)
    # Need at least 2 increments for ddof=1 std to be defined.
    if increments.size < 2:
        return 0.0
    mean_inc = float(increments.mean())
    std_inc = float(increments.std(ddof=1))
    if std_inc == 0.0:
        return 0.0
    return (mean_inc / std_inc) * float(annualization_factor)


def fill_rate_asymmetry(pr: PathResult) -> int:
    """Signed difference between bid- and ask-side fill counts (Req 8.5)."""
    return int(pr.n_bid_fills) - int(pr.n_ask_fills)


def adverse_selection(pr: PathResult, horizon_steps: int) -> float:
    """Mean per-fill adverse-selection drift over ``horizon_steps`` (Req 8.6).

    For each fill at step ``i`` we look ahead to step
    ``min(i + horizon_steps, N)`` and compute the *signed* mid drift
    against the filled side:

    * Bid fill (we are long): ``-(s[i + h] - s[i])``. Falling mid is
      adverse and yields a positive value.
    * Ask fill (we are short): ``+(s[i + h] - s[i])``. Rising mid is
      adverse and yields a positive value.

    Returns ``0.0`` if the path has no fills.
    """
    if horizon_steps < 0:
        raise ValueError(
            f"horizon_steps: must be >= 0; got {horizon_steps!r}"
        )
    bid_fill = np.asarray(pr.bid_fill, dtype=bool)
    ask_fill = np.asarray(pr.ask_fill, dtype=bool)
    n_inner = bid_fill.shape[0]
    n_grid = pr.s_path.shape[0]  # == n_inner + 1
    s = np.asarray(pr.s_path, dtype=np.float64)

    drifts: list[float] = []
    for i in np.flatnonzero(bid_fill):
        j = min(int(i) + int(horizon_steps), n_grid - 1)
        drifts.append(-(float(s[j]) - float(s[int(i)])))
    for i in np.flatnonzero(ask_fill):
        j = min(int(i) + int(horizon_steps), n_grid - 1)
        drifts.append(+(float(s[j]) - float(s[int(i)])))
    if not drifts:
        return 0.0
    # Suppress unused-variable warning on n_inner; the asserted equality
    # is documentation of the runner's contract.
    assert n_grid == n_inner + 1
    return float(np.mean(drifts))


def spread_capture(pr: PathResult) -> np.ndarray:
    """Per-fill spread capture as a 1-D ``float64`` array (Req 8.7).

    For an ask fill at step ``i`` the captured half-spread is
    ``ask_quote[i] - s_path[i]``. For a bid fill at step ``i`` it is
    ``s_path[i] - bid_quote[i]``. Order: bid-side fills first (in step
    order), then ask-side fills (in step order). The total length equals
    ``n_bid_fills + n_ask_fills``; an empty array is returned when there
    are no fills.
    """
    bid_fill = np.asarray(pr.bid_fill, dtype=bool)
    ask_fill = np.asarray(pr.ask_fill, dtype=bool)
    s = np.asarray(pr.s_path, dtype=np.float64)
    bid_q = np.asarray(pr.bid_quote, dtype=np.float64)
    ask_q = np.asarray(pr.ask_quote, dtype=np.float64)

    bid_idx = np.flatnonzero(bid_fill)
    ask_idx = np.flatnonzero(ask_fill)
    bid_capture = s[bid_idx] - bid_q[bid_idx]
    ask_capture = ask_q[ask_idx] - s[ask_idx]
    return np.concatenate([bid_capture, ask_capture]).astype(np.float64)


def mean_spread_capture(pr: PathResult) -> float:
    """Mean of :func:`spread_capture` across all fills, ``0.0`` if none."""
    captures = spread_capture(pr)
    if captures.size == 0:
        return 0.0
    return float(np.mean(captures))


# --------------------------------------------------------------------------- #
# Per-cell aggregation                                                        #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CellSummary:
    """Aggregate statistics for one ``(strategy, regime)`` cell.

    Attributes
    ----------
    strategy:
        Strategy identifier, e.g. ``"avellaneda_stoikov"`` or
        ``"symmetric"``.
    regime:
        Volatility-regime name from
        :pyattr:`~etf_mm_sim.config.RegimeParams.name`.
    n_paths:
        Number of :class:`PathResult` aggregated into this summary.
    mean_pnl, std_pnl, p05_pnl, p50_pnl, p95_pnl:
        Mean, sample standard deviation (``ddof=1``), and 5th / 50th /
        95th percentiles of the terminal P&L distribution (Req 8.2).
        ``std_pnl`` is ``0.0`` when ``n_paths <= 1``.
    mean_sharpe:
        Mean of :func:`sharpe` across all paths (Req 8.3).
    mean_max_dd:
        Mean of :func:`max_drawdown` across all paths (Req 8.4).
    mean_fill_asymmetry:
        Mean of :func:`fill_rate_asymmetry` across all paths (Req 8.5).
    mean_adverse_selection:
        Mean per-path adverse-selection score (Req 8.6).
    mean_spread_capture:
        Mean per-path :func:`mean_spread_capture` (Req 8.7).
    """

    strategy: str
    regime: str
    n_paths: int
    mean_pnl: float
    std_pnl: float
    p05_pnl: float
    p50_pnl: float
    p95_pnl: float
    mean_sharpe: float
    mean_max_dd: float
    mean_fill_asymmetry: float
    mean_adverse_selection: float
    mean_spread_capture: float


def aggregate_cell(
    cell: list[PathResult],
    strategy: str,
    regime: str,
    annualization_factor: float,
    adv_horizon_steps: int,
) -> CellSummary:
    """Aggregate per-path metrics into a :class:`CellSummary`.

    Parameters
    ----------
    cell:
        Path results for one ``(strategy, regime)`` cell. May be empty,
        in which case an all-zero summary is returned with ``n_paths=0``.
    strategy, regime:
        Identifying labels copied onto the returned summary.
    annualization_factor:
        Scalar passed straight to :func:`sharpe`; typically
        ``cfg.analytics.sharpe_annualization_factor``.
    adv_horizon_steps:
        Lookahead horizon (in inner steps) passed to
        :func:`adverse_selection`; typically
        ``cfg.analytics.adverse_selection_horizon_steps``.
    """
    n_paths = len(cell)
    if n_paths == 0:
        return CellSummary(
            strategy=strategy,
            regime=regime,
            n_paths=0,
            mean_pnl=0.0,
            std_pnl=0.0,
            p05_pnl=0.0,
            p50_pnl=0.0,
            p95_pnl=0.0,
            mean_sharpe=0.0,
            mean_max_dd=0.0,
            mean_fill_asymmetry=0.0,
            mean_adverse_selection=0.0,
            mean_spread_capture=0.0,
        )

    terminal_pnls = np.array(
        [float(pr.terminal_pnl) for pr in cell], dtype=np.float64
    )
    mean_pnl = float(terminal_pnls.mean())
    if n_paths == 1:
        # Sample std with ddof=1 is undefined for a single sample.
        std_pnl = 0.0
    else:
        std_pnl = float(terminal_pnls.std(ddof=1))
    p05, p50, p95 = (
        float(x) for x in np.quantile(terminal_pnls, [0.05, 0.5, 0.95])
    )

    sharpes = np.array(
        [sharpe(pr, annualization_factor) for pr in cell], dtype=np.float64
    )
    max_dds = np.array([max_drawdown(pr) for pr in cell], dtype=np.float64)
    asymm = np.array(
        [fill_rate_asymmetry(pr) for pr in cell], dtype=np.float64
    )
    adv = np.array(
        [adverse_selection(pr, adv_horizon_steps) for pr in cell],
        dtype=np.float64,
    )
    cap = np.array(
        [mean_spread_capture(pr) for pr in cell], dtype=np.float64
    )

    return CellSummary(
        strategy=strategy,
        regime=regime,
        n_paths=n_paths,
        mean_pnl=mean_pnl,
        std_pnl=std_pnl,
        p05_pnl=p05,
        p50_pnl=p50,
        p95_pnl=p95,
        mean_sharpe=float(sharpes.mean()),
        mean_max_dd=float(max_dds.mean()),
        mean_fill_asymmetry=float(asymm.mean()),
        mean_adverse_selection=float(adv.mean()),
        mean_spread_capture=float(cap.mean()),
    )


# --------------------------------------------------------------------------- #
# Paired AS-vs-Symmetric comparison                                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PairedComparison:
    """Paired AS-vs-Symmetric statistics for one volatility regime (Req 8.8).

    All fields are scalars produced from per-path paired differences
    (``AS[p] - Symmetric[p]``) using the percentile bootstrap from
    :func:`etf_mm_sim.bootstrap.paired_bootstrap_ci`. The
    ``skew_dd_attribution`` fields are populated by the counterfactual
    no-skew replay (Req 8.9) and default to zero when callers do not
    supply them.

    Attributes
    ----------
    regime:
        Volatility-regime name shared by the two strategy cells.
    n_paths:
        Number of paired paths consumed (length of each diff array).
    diff_mean_pnl, diff_mean_pnl_ci:
        Mean of ``as.terminal_pnl - sym.terminal_pnl`` across paths and
        its ``(lo, hi)`` percentile-bootstrap CI.
    diff_max_dd, diff_max_dd_ci:
        Mean of :func:`max_drawdown` ``AS - Symmetric`` differences
        across paths and its CI. Note: positive values mean AS has a
        *larger* drawdown than the baseline; the AS skew effect is
        typically *negative* on this metric, hence the dedicated
        ``skew_dd_attribution`` field below.
    skew_dd_attribution, skew_dd_attribution_ci:
        Mean of ``cf_max_dd - as_max_dd`` across paths (counterfactual
        no-skew minus AS) and its CI; positive values mean AS skew
        reduced drawdown vs. an otherwise-identical no-skew quoter
        (Req 8.9).
    """

    regime: str
    n_paths: int
    diff_mean_pnl: float
    diff_mean_pnl_ci: tuple[float, float]
    diff_max_dd: float
    diff_max_dd_ci: tuple[float, float]
    skew_dd_attribution: float
    skew_dd_attribution_ci: tuple[float, float]


def compute_paired_comparison(
    as_cell: list[PathResult],
    sym_cell: list[PathResult],
    regime: str,
    bootstrap_iterations: int,
    bootstrap_alpha: float,
    bootstrap_ss: np.random.SeedSequence,
    skew_dd_attribution: float = 0.0,
    skew_dd_attribution_ci: tuple[float, float] = (0.0, 0.0),
) -> PairedComparison:
    """Compute paired AS-vs-Symmetric statistics with bootstrap CIs.

    Pairs ``as_cell[i]`` with ``sym_cell[i]`` so the diff arrays
    correspond to the same mid-price seed (Req 7.2). The bootstrap RNG
    is split into two independent sub-sequences via
    ``bootstrap_ss.spawn(2)`` so the P&L and drawdown CIs do not share
    state.

    Parameters
    ----------
    as_cell, sym_cell:
        Path results for the AS and Symmetric strategy cells in the
        same regime. Must be non-empty and of the same length. Pairing
        is *positional* and assumes both cells were produced with the
        same per-path mid-price seed (which the backtest engine
        guarantees, Req 7.2).
    regime:
        Regime name copied onto the returned :class:`PairedComparison`.
    bootstrap_iterations:
        Number of bootstrap resamples passed to
        :func:`paired_bootstrap_ci`. Must be ``>= 1``.
    bootstrap_alpha:
        Two-sided significance level passed to
        :func:`paired_bootstrap_ci`. Must satisfy ``0 < alpha < 1``.
    bootstrap_ss:
        Seed sequence dedicated to bootstrap randomization (typically
        produced by :func:`etf_mm_sim.seeding.analytics_seed`). It is
        *not* consumed directly; we spawn two independent children so
        the P&L and drawdown bootstraps run on disjoint streams.
    skew_dd_attribution, skew_dd_attribution_ci:
        Optional pass-through values for the counterfactual-replay
        attribution fields (Req 8.9). Default to zero so callers that
        only need the AS-vs-Symmetric pairing can omit them; the full
        backtest pipeline computes these via
        :func:`etf_mm_sim.counterfactual.skew_drawdown_attribution`
        and passes them in.

    Returns
    -------
    PairedComparison
        Aggregated paired-comparison statistics for the regime.

    Raises
    ------
    ValueError
        If ``as_cell`` and ``sym_cell`` differ in length or are empty.
    """
    if len(as_cell) != len(sym_cell):
        raise ValueError(
            "as_cell and sym_cell must have the same length; "
            f"got {len(as_cell)} and {len(sym_cell)}"
        )
    n_paths = len(as_cell)
    if n_paths == 0:
        raise ValueError(
            "as_cell and sym_cell must be non-empty for paired comparison"
        )

    pnl_diff = np.array(
        [
            float(a.terminal_pnl) - float(s.terminal_pnl)
            for a, s in zip(as_cell, sym_cell)
        ],
        dtype=np.float64,
    )
    dd_diff = np.array(
        [max_drawdown(a) - max_drawdown(s) for a, s in zip(as_cell, sym_cell)],
        dtype=np.float64,
    )

    # Spawn two independent sub-sequences so the P&L and drawdown
    # bootstraps consume disjoint RNG streams.
    pnl_ss, dd_ss = bootstrap_ss.spawn(2)

    point_pnl, pnl_lo, pnl_hi = paired_bootstrap_ci(
        pnl_diff, bootstrap_iterations, bootstrap_alpha, pnl_ss
    )
    point_dd, dd_lo, dd_hi = paired_bootstrap_ci(
        dd_diff, bootstrap_iterations, bootstrap_alpha, dd_ss
    )

    return PairedComparison(
        regime=regime,
        n_paths=n_paths,
        diff_mean_pnl=point_pnl,
        diff_mean_pnl_ci=(pnl_lo, pnl_hi),
        diff_max_dd=point_dd,
        diff_max_dd_ci=(dd_lo, dd_hi),
        skew_dd_attribution=float(skew_dd_attribution),
        skew_dd_attribution_ci=(
            float(skew_dd_attribution_ci[0]),
            float(skew_dd_attribution_ci[1]),
        ),
    )
