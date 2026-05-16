"""Property test for paired mid-price seeding across strategies.

# Feature: etf-mm-arbitrage-simulator, Property 17: Paired mid-price seeding across strategies
# Validates: Requirements 7.2

The Backtest_Engine must run AS and Symmetric quoters against the
*same* set of mid-price paths per regime so that observed P&L
differences come from quoting behavior alone, not from path-noise
divergence (Req 7.2). This test asserts ``s_path`` byte-equality at
every ``(regime, path_index)`` pair across the two strategies.

We do not parameterize via Hypothesis here; a single deterministic
configuration with a fixed master seed exercises every regime and path
exhaustively, which is sufficient because the backtest engine builds
each path's mid-price array from a single ``SeedSequence`` -- if the
arrays are equal once they will be equal forever.
"""

from __future__ import annotations

import pathlib

import numpy as np

from etf_mm_sim.backtest import run_backtest
from etf_mm_sim.config import (
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


def _tiny_cfg() -> Configuration:
    """Build a minimal valid Configuration for a paired-pairing check."""
    regimes = (
        RegimeParams(name="low", mu=0.0, sigma=0.5),
        RegimeParams(name="high", mu=0.0, sigma=2.0),
    )
    return Configuration(
        master_seed=20240101,
        horizon=HorizonConfig(T=0.05, dt=0.005),
        mid_price=MidPriceConfig(
            model="regime_switching",
            s0=100.0,
            regimes=regimes,
            transition_matrix=(
                (0.99, 0.01),
                (0.02, 0.98),
            ),
        ),
        quoters_as=ASParams(gamma=0.1, k=1.5, A=140.0),
        quoters_sym=SymmetricParams(delta_base=0.05),
        fill=FillParams(A_b=140.0, k_b=1.5, A_a=140.0, k_a=1.5),
        risk=RiskParams(q_max=10, L_kill=None),
        mc=MCParams(n_paths=5),
        analytics=AnalyticsParams(
            adverse_selection_horizon_steps=2,
            bootstrap_iterations=10,
            bootstrap_alpha=0.05,
            sharpe_annualization_factor=15.87,
        ),
        output=OutputConfig(dir=pathlib.Path("results"), format="parquet"),
    )


def test_property_17_paired_mid_price_seeding() -> None:
    """AS, Symmetric, and Semi-AS quoters see byte-identical ``s_path`` per cell."""
    cfg = _tiny_cfg()
    result = run_backtest(cfg)

    strategies = ("avellaneda_stoikov", "symmetric", "semi_as")
    for regime in cfg.mid_price.regimes:
        cells = {s: result.paths[(s, regime.name)] for s in strategies}
        for s in strategies:
            assert len(cells[s]) == cfg.mc.n_paths
        for p in range(cfg.mc.n_paths):
            ref = cells["avellaneda_stoikov"][p].s_path
            for s in strategies[1:]:
                assert np.array_equal(cells[s][p].s_path, ref), (
                    f"mid-price path mismatch in regime={regime.name!r} "
                    f"strategy={s!r} path={p}"
                )
