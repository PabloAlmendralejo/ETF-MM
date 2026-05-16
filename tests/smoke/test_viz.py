"""Smoke tests for the visualization module.

# Feature: etf-mm-arbitrage-simulator
# Validates: Requirements 9.1, 9.2, 9.3

End-to-end exercise: run a tiny backtest, then call each public
visualization function and assert it returns a well-formed
``matplotlib.figure.Figure`` (or, for the summary table, a
``pandas.DataFrame`` with the expected schema).
"""

from __future__ import annotations

import dataclasses
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure  # noqa: E402  (after `use` for backend init order)
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from etf_mm_sim import analytics, viz  # noqa: E402
from etf_mm_sim.analytics import CellSummary  # noqa: E402
from etf_mm_sim.backtest import run_backtest  # noqa: E402
from etf_mm_sim.config import (  # noqa: E402
    ASParams,
    AnalyticsParams,
    Configuration,
    FillParams,
    HorizonConfig,
    MCParams,
    MidPriceConfig,
    OutputConfig,
    RegimeParams,
    RiskParams,
    SymmetricParams,
)


def _tiny_cfg(output_dir: pathlib.Path) -> Configuration:
    return Configuration(
        master_seed=2024,
        horizon=HorizonConfig(T=0.01, dt=0.001),
        mid_price=MidPriceConfig(
            model="regime_switching",
            s0=100.0,
            regimes=(
                RegimeParams(name="calm", mu=0.0, sigma=0.5),
                RegimeParams(name="stormy", mu=0.0, sigma=2.0),
            ),
            transition_matrix=(
                (0.9, 0.1),
                (0.1, 0.9),
            ),
        ),
        quoters_as=ASParams(gamma=0.1, k=1.5, A=140.0),
        quoters_sym=SymmetricParams(delta_base=0.05),
        fill=FillParams(A_b=140.0, k_b=1.5, A_a=140.0, k_a=1.5),
        risk=RiskParams(q_max=5, L_kill=None),
        mc=MCParams(n_paths=5),
        analytics=AnalyticsParams(
            adverse_selection_horizon_steps=1,
            bootstrap_iterations=10,
            bootstrap_alpha=0.05,
            sharpe_annualization_factor=15.87,
        ),
        output=OutputConfig(dir=output_dir, format="parquet"),
    )


@pytest.fixture(scope="module")
def tiny_result(tmp_path_factory: pytest.TempPathFactory):
    out_dir = tmp_path_factory.mktemp("viz-smoke")
    return run_backtest(_tiny_cfg(out_dir))


def test_plot_terminal_pnl_hist_returns_figure(tiny_result) -> None:
    """``plot_terminal_pnl_hist`` returns a non-empty figure with axes."""
    fig = viz.plot_terminal_pnl_hist(tiny_result, regime="calm")
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
        # At least one Axes was added.
        axes = fig.get_axes()
        assert len(axes) >= 1
        # All three strategies should contribute a legend entry.
        legend = axes[0].get_legend()
        assert legend is not None
        labels = {t.get_text() for t in legend.get_texts()}
        assert "Avellaneda-Stoikov" in labels
        assert "Symmetric" in labels
        assert "Semi-AS" in labels
    finally:
        plt.close(fig)


def test_plot_terminal_pnl_hist_save(
    tiny_result, tmp_path: pathlib.Path
) -> None:
    """``save_path`` writes a non-empty PNG."""
    out = tmp_path / "hist.png"
    fig = viz.plot_terminal_pnl_hist(tiny_result, regime="stormy", save_path=out)
    try:
        assert out.is_file()
        assert out.stat().st_size > 0
    finally:
        plt.close(fig)


def test_plot_sample_path_returns_figure(tiny_result) -> None:
    """``plot_sample_path`` returns a 4-panel figure."""
    pr = tiny_result.paths[("avellaneda_stoikov", "calm")][0]
    fig = viz.plot_sample_path(pr)
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
        axes = fig.get_axes()
        assert len(axes) == 4
        # All panels share the same x-axis (sharex=True).
        for ax in axes[1:]:
            assert ax.get_shared_x_axes().joined(ax, axes[0])
    finally:
        plt.close(fig)


def test_plot_sample_path_save(
    tiny_result, tmp_path: pathlib.Path
) -> None:
    pr = tiny_result.paths[("symmetric", "stormy")][0]
    out = tmp_path / "sample.png"
    fig = viz.plot_sample_path(pr, save_path=out)
    try:
        assert out.is_file()
        assert out.stat().st_size > 0
    finally:
        plt.close(fig)


def test_render_summary_table_schema(tiny_result) -> None:
    """``render_summary_table`` produces a DataFrame with the expected schema."""
    cfg = tiny_result.config
    cells = []
    for strategy in ("avellaneda_stoikov", "symmetric", "semi_as"):
        for regime in cfg.mid_price.regimes:
            cells.append(
                analytics.aggregate_cell(
                    tiny_result.paths[(strategy, regime.name)],
                    strategy,
                    regime.name,
                    cfg.analytics.sharpe_annualization_factor,
                    cfg.analytics.adverse_selection_horizon_steps,
                )
            )
    summary = viz.render_summary_table(cells)

    assert isinstance(summary, pd.DataFrame)
    # 3 strategies × 2 regimes = 6 rows.
    assert len(summary) == 6
    # Columns must match every CellSummary field, in declared order.
    expected_columns = [f.name for f in dataclasses.fields(CellSummary)]
    assert list(summary.columns) == expected_columns
    # Required column names are present.
    for col in (
        "strategy",
        "regime",
        "n_paths",
        "mean_pnl",
        "std_pnl",
        "p05_pnl",
        "p50_pnl",
        "p95_pnl",
        "mean_sharpe",
        "mean_max_dd",
        "mean_fill_asymmetry",
        "mean_adverse_selection",
        "mean_spread_capture",
    ):
        assert col in summary.columns


def test_render_summary_table_empty_input() -> None:
    """Empty input returns an empty DataFrame with the right column schema."""
    summary = viz.render_summary_table([])
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 0
    expected_columns = [f.name for f in dataclasses.fields(CellSummary)]
    assert list(summary.columns) == expected_columns


def test_save_all_figures_writes_files(
    tiny_result, tmp_path: pathlib.Path
) -> None:
    """``save_all_figures`` writes the documented file set under ``plots_dir``."""
    cfg = tiny_result.config
    plots_dir = tmp_path / "plots"
    viz.save_all_figures(
        tiny_result,
        plots_dir,
        cfg.analytics.sharpe_annualization_factor,
        cfg.analytics.adverse_selection_horizon_steps,
    )
    # One terminal-PnL hist per regime.
    for regime in cfg.mid_price.regimes:
        f = plots_dir / f"terminal_pnl_{regime.name}.png"
        assert f.is_file() and f.stat().st_size > 0
    # One sample-path PNG per (strategy, regime) cell.
    for strategy in ("avellaneda_stoikov", "symmetric", "semi_as"):
        for regime in cfg.mid_price.regimes:
            f = plots_dir / f"sample_path_{strategy}_{regime.name}.png"
            assert f.is_file() and f.stat().st_size > 0
    # Summary HTML.
    html = plots_dir / "summary_table.html"
    assert html.is_file()
    assert html.stat().st_size > 0
