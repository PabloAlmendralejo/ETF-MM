"""Matplotlib visualization renderers.

Produces per-regime terminal-P&L histograms, sample-path diagnostics
with aligned time axes (mid+quotes, inventory, cash, cumulative P&L),
and the per-cell summary table. See ``design.md`` §Visualization.

All renderers use ``matplotlib`` directly (no seaborn) and run under the
non-interactive ``Agg`` backend so the module is safe to import in
headless environments such as CI runners and notebook executors.

Public API
----------
* :func:`plot_terminal_pnl_hist` -- per-regime overlay of AS vs.
  Symmetric terminal-P&L distributions (Req 9.1).
* :func:`plot_sample_path` -- 4-panel diagnostic for one
  :class:`PathResult` (Req 9.2).
* :func:`render_summary_table` -- ``DataFrame`` with one row per
  ``(strategy, regime)`` cell (Req 9.3).
* :func:`save_all_figures` -- bulk render every plot for a
  :class:`BacktestResult` plus the summary table HTML.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import TYPE_CHECKING

import matplotlib

# Configure non-interactive backend before importing pyplot so the module
# is importable in headless environments (CI, notebook executors).
matplotlib.use("Agg")

import matplotlib.figure  # noqa: E402  (after `use` for pyplot init order)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .analytics import CellSummary, aggregate_cell  # noqa: E402

if TYPE_CHECKING:
    from .backtest import BacktestResult
    from .path_runner import PathResult

__all__ = [
    "plot_terminal_pnl_hist",
    "plot_sample_path",
    "render_summary_table",
    "save_all_figures",
]


# Strategy → display label and color. The AS / Symmetric / Semi-AS
# triple gets distinct, colorblind-friendly colors so the overlay
# histograms remain readable when printed in grayscale.
_STRATEGY_LABELS: dict[str, str] = {
    "avellaneda_stoikov": "Avellaneda-Stoikov",
    "symmetric": "Symmetric",
    "semi_as": "Semi-AS",
}
_STRATEGY_COLORS: dict[str, str] = {
    "avellaneda_stoikov": "#1f77b4",  # tab:blue
    "symmetric": "#d62728",           # tab:red
    "semi_as": "#2ca02c",             # tab:green
}


# --------------------------------------------------------------------------- #
# Terminal-P&L overlay histogram                                              #
# --------------------------------------------------------------------------- #


def plot_terminal_pnl_hist(
    result: "BacktestResult",
    regime: str,
    bins: int = 30,
    save_path: pathlib.Path | None = None,
) -> matplotlib.figure.Figure:
    """Overlay AS, Symmetric, and Semi-AS terminal-P&L histograms for one regime.

    All histograms are drawn at 50% alpha so the overlap region is
    visible. Strategy labels appear in the legend; the y-axis is plain
    counts (not density) because the per-strategy sample sizes match by
    construction (paired Monte Carlo, Req 7.2).

    Parameters
    ----------
    result:
        The full :class:`BacktestResult` produced by
        :func:`etf_mm_sim.backtest.run_backtest`.
    regime:
        Volatility-regime name to plot. Must be present in ``result``.
    bins:
        Number of histogram bins. Defaults to 30, which gives readable
        overlays for the reference 1000-path sweep.
    save_path:
        Optional output PNG path. If provided the figure is written at
        150 dpi with ``bbox_inches="tight"``. The figure is returned in
        either case.

    Returns
    -------
    matplotlib.figure.Figure
        The figure handle. Callers retain ownership and are responsible
        for closing it (``plt.close(fig)``) when no longer needed.

    Raises
    ------
    KeyError
        If any expected strategy cell for ``regime`` is missing from the
        backtest result.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    strategies = ("avellaneda_stoikov", "symmetric", "semi_as")
    # Use a shared bin edge set so the histograms are directly
    # comparable; deriving it from the union of all samples avoids
    # one strategy's tail drifting outside the binned range.
    samples: dict[str, np.ndarray] = {}
    for strategy in strategies:
        cell = result.paths[(strategy, regime)]
        samples[strategy] = np.array(
            [float(pr.terminal_pnl) for pr in cell], dtype=np.float64
        )

    all_values = np.concatenate(list(samples.values()))
    if all_values.size > 0:
        bin_edges = np.linspace(
            float(all_values.min()), float(all_values.max()), bins + 1
        )
    else:
        # Defensive: empty regime. Use a trivial single bin so matplotlib
        # does not raise on `bins=[]`.
        bin_edges = np.array([0.0, 1.0])

    for strategy in strategies:
        ax.hist(
            samples[strategy],
            bins=bin_edges,
            alpha=0.5,
            label=_STRATEGY_LABELS[strategy],
            color=_STRATEGY_COLORS[strategy],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_title(f"Terminal P&L distribution — regime={regime}")
    ax.set_xlabel("Terminal P&L")
    ax.set_ylabel("Count")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# Sample-path diagnostic                                                      #
# --------------------------------------------------------------------------- #


def plot_sample_path(
    pr: "PathResult",
    save_path: pathlib.Path | None = None,
) -> matplotlib.figure.Figure:
    """Render a 4-panel sample-path diagnostic.

    Panels (top to bottom) share a single time axis (step index):

    1. Mid-price ``s_path`` plus the posted bid and ask quotes.
       Suppressed quotes carry ``NaN`` and are silently skipped by
       ``ax.plot``.
    2. Inventory ``q[i]``.
    3. Cash ``cash[i]``.
    4. Cumulative mark-to-market P&L ``cash[i] + q[i] * s_path[i]``.

    Parameters
    ----------
    pr:
        The :class:`PathResult` to render.
    save_path:
        Optional output PNG path. If provided, written at 150 dpi with
        ``bbox_inches="tight"``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure handle.
    """
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 8))

    s_path = np.asarray(pr.s_path, dtype=np.float64)
    bid_quote = np.asarray(pr.bid_quote, dtype=np.float64)
    ask_quote = np.asarray(pr.ask_quote, dtype=np.float64)
    inventory = np.asarray(pr.inventory, dtype=np.int64)
    cash = np.asarray(pr.cash, dtype=np.float64)
    n_plus_1 = int(s_path.shape[0])
    steps = np.arange(n_plus_1, dtype=np.int64)

    # Panel 1: mid + quotes. NaNs in the quote arrays are simply gaps.
    ax_mid = axes[0]
    ax_mid.plot(steps, s_path, color="black", linewidth=1.0, label="mid")
    ax_mid.plot(
        steps, bid_quote, color="tab:blue", linewidth=0.8, alpha=0.8,
        label="bid",
    )
    ax_mid.plot(
        steps, ask_quote, color="tab:red", linewidth=0.8, alpha=0.8,
        label="ask",
    )
    ax_mid.set_ylabel("Price")
    ax_mid.set_title(
        f"Sample path — strategy={pr.strategy}, regime={pr.regime}, "
        f"path_index={pr.path_index}"
    )
    ax_mid.legend(loc="best", fontsize=8)
    ax_mid.grid(True, alpha=0.3)

    # Panel 2: inventory.
    ax_inv = axes[1]
    ax_inv.step(steps, inventory, color="tab:purple", where="post")
    ax_inv.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
    ax_inv.set_ylabel("Inventory")
    ax_inv.grid(True, alpha=0.3)

    # Panel 3: cash.
    ax_cash = axes[2]
    ax_cash.plot(steps, cash, color="tab:green")
    ax_cash.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
    ax_cash.set_ylabel("Cash")
    ax_cash.grid(True, alpha=0.3)

    # Panel 4: cumulative MTM P&L.
    pnl_curve = cash + inventory.astype(np.float64) * s_path
    ax_pnl = axes[3]
    ax_pnl.plot(steps, pnl_curve, color="tab:orange")
    ax_pnl.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
    ax_pnl.set_ylabel("Cumulative P&L")
    ax_pnl.set_xlabel("Step index")
    ax_pnl.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# Summary table                                                               #
# --------------------------------------------------------------------------- #


def render_summary_table(cells: list[CellSummary]) -> pd.DataFrame:
    """Render per-cell summary statistics as a :class:`pandas.DataFrame`.

    One row per :class:`CellSummary`; columns mirror the dataclass
    fields. Column order matches
    :func:`dataclasses.fields(CellSummary)`, which keeps strategy and
    regime in the leftmost positions and aligns with the docstring of
    :class:`~etf_mm_sim.analytics.CellSummary`.

    Parameters
    ----------
    cells:
        Aggregated per-cell summaries from
        :func:`etf_mm_sim.analytics.aggregate_cell`.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per cell. Empty input returns an empty
        DataFrame with the expected column schema preserved.
    """
    field_names = [f.name for f in dataclasses.fields(CellSummary)]
    rows = [
        {name: getattr(cell, name) for name in field_names} for cell in cells
    ]
    return pd.DataFrame(rows, columns=field_names)


# --------------------------------------------------------------------------- #
# Bulk render                                                                 #
# --------------------------------------------------------------------------- #


def save_all_figures(
    result: "BacktestResult",
    plots_dir: pathlib.Path,
    annualization_factor: float,
    adv_horizon_steps: int,
) -> None:
    """Render every plot for ``result`` and save them under ``plots_dir``.

    Files written
    -------------
    * ``terminal_pnl_<regime>.png`` -- one per regime.
    * ``sample_path_<strategy>_<regime>.png`` -- one per
      ``(strategy, regime)`` cell, using ``path_index == 0``.
    * ``summary_table.html`` -- HTML rendering of the per-cell summary.

    All PNGs are written at 150 dpi with ``bbox_inches="tight"`` (Req 9.1,
    9.2). The summary HTML uses
    :py:meth:`pandas.DataFrame.to_html` which is the convention used by
    the notebook deliverable (Req 9.3, 9.4).

    Parameters
    ----------
    result:
        The full :class:`BacktestResult` to render.
    plots_dir:
        Output directory. Created (with parents) if it does not exist.
    annualization_factor:
        Forwarded to :func:`aggregate_cell` for Sharpe annualization;
        typically ``cfg.analytics.sharpe_annualization_factor``.
    adv_horizon_steps:
        Forwarded to :func:`aggregate_cell` for adverse-selection
        lookahead; typically
        ``cfg.analytics.adverse_selection_horizon_steps``.
    """
    plots_dir = pathlib.Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    cfg = result.config
    regime_names = [r.name for r in cfg.mid_price.regimes]
    strategies = ("avellaneda_stoikov", "symmetric", "semi_as")

    # 1) Terminal-P&L histograms per regime.
    for regime in regime_names:
        out = plots_dir / f"terminal_pnl_{regime}.png"
        fig = plot_terminal_pnl_hist(result, regime, save_path=out)
        plt.close(fig)

    # 2) Sample-path diagnostics for path 0 of every cell.
    for strategy in strategies:
        for regime in regime_names:
            cell = result.paths[(strategy, regime)]
            if not cell:
                continue
            out = plots_dir / f"sample_path_{strategy}_{regime}.png"
            fig = plot_sample_path(cell[0], save_path=out)
            plt.close(fig)

    # 3) Summary table → HTML.
    cells: list[CellSummary] = []
    for strategy in strategies:
        for regime in regime_names:
            cells.append(
                aggregate_cell(
                    result.paths[(strategy, regime)],
                    strategy,
                    regime,
                    annualization_factor,
                    adv_horizon_steps,
                )
            )
    summary = render_summary_table(cells)
    (plots_dir / "summary_table.html").write_text(
        summary.to_html(index=False), encoding="utf-8"
    )
