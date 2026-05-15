"""Property test for counterfactual no-skew replay invariants.

# Feature: etf-mm-arbitrage-simulator, Property 22: Counterfactual inventory equals AS inventory
# Validates: Requirements 8.9

The counterfactual no-skew replay
:func:`etf_mm_sim.counterfactual.replay_no_skew` reuses the AS bid/ask
fill stream against quotes ``s_path +/- delta_star_sched``. Because the
fill stream is held fixed, the inventory trajectory and per-side fill
counts of the counterfactual must match the AS path exactly at every
step (Property 22). Only ``cash`` (and therefore ``terminal_pnl``)
should differ.

We exercise this via a deterministic small backtest using the same
configuration shape as
``tests/property/test_backtest_pairing.py`` (n_paths=5, T=0.05,
dt=0.005, two regimes). For every AS path we precompute the AS
half-spread schedule and replay; the test then asserts the per-step
arrays are byte-identical and the per-side fill counts match.
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
from etf_mm_sim.counterfactual import replay_no_skew
from etf_mm_sim.quoters.avellaneda_stoikov import precompute as as_precompute


def _tiny_cfg() -> Configuration:
    """Build a minimal valid Configuration for the counterfactual replay."""
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


def test_property_22_counterfactual_inventory_equality() -> None:
    """Counterfactual replay preserves inventory and fill bookkeeping.

    For every ``(regime, path_index)`` the counterfactual no-skew replay
    must reproduce the AS path's:

    * inventory trajectory (byte-identical int64 array);
    * bid-side fill flags (byte-identical bool array);
    * ask-side fill flags (byte-identical bool array);
    * per-side fill counts (``n_bid_fills``, ``n_ask_fills``).

    The mid-price array is, by construction, the same input to both
    paths (the counterfactual borrows ``as_path.s_path``). Cash and
    terminal P&L are *not* required to match -- in fact they should
    typically differ because fill prices differ.

    **Validates: Requirements 8.9**
    """
    cfg = _tiny_cfg()
    result = run_backtest(cfg)

    T = cfg.horizon.T
    dt = cfg.horizon.dt

    for regime in cfg.mid_price.regimes:
        as_paths = result.paths[("avellaneda_stoikov", regime.name)]
        # AS schedule is path-independent (depends only on T, dt, gamma,
        # sigma, k); compute once per regime to mirror the backtest
        # engine and skew-attribution helper.
        sched = as_precompute(
            s_path=as_paths[0].s_path,
            dt=dt,
            T=T,
            gamma=cfg.quoters_as.gamma,
            sigma=regime.sigma,
            k=cfg.quoters_as.k,
        )
        delta_star_sched = sched["delta_star"]

        for as_path in as_paths:
            cf = replay_no_skew(as_path, delta_star_sched)

            assert np.array_equal(cf.inventory, as_path.inventory), (
                f"inventory mismatch in regime={regime.name!r} "
                f"path={as_path.path_index}"
            )
            assert np.array_equal(cf.bid_fill, as_path.bid_fill), (
                f"bid_fill mismatch in regime={regime.name!r} "
                f"path={as_path.path_index}"
            )
            assert np.array_equal(cf.ask_fill, as_path.ask_fill), (
                f"ask_fill mismatch in regime={regime.name!r} "
                f"path={as_path.path_index}"
            )
            assert cf.n_bid_fills == as_path.n_bid_fills
            assert cf.n_ask_fills == as_path.n_ask_fills

            # The counterfactual replay reuses the AS s_path verbatim;
            # mid-price equality is therefore tautological but worth
            # asserting so a future regression in replay_no_skew that
            # mutates the array is caught immediately.
            assert np.array_equal(cf.s_path, as_path.s_path)

            # Sanity: strategy label flips, kill_switch is None.
            assert cf.strategy == "counterfactual_no_skew"
            assert cf.kill_switch_step is None
