"""Integration smoke test for the backtest engine and persistence layer.

# Feature: etf-mm-arbitrage-simulator
# Validates: Requirements 7.1, 7.4, 10.4

End-to-end exercise: run a tiny backtest, persist the result, and check
that the on-disk artifacts match the documented schema and layout. This
is the lowest-fidelity integration test; subtler invariants are covered
by the property tests (Property 17 and 18).
"""

from __future__ import annotations

import json
import math
import pathlib

import pyarrow.parquet as pq

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
from etf_mm_sim.persistence import persist


_PATHS_COLUMNS = (
    "ask_quote",
    "bid_quote",
    "cash",
    "inventory",
    "mid_price",
    "path_index",
    "step_index",
)
_FILLS_COLUMNS = ("fill_price", "path_index", "side", "step_index")


def _tiny_cfg(output_dir: pathlib.Path) -> Configuration:
    return Configuration(
        master_seed=12345,
        horizon=HorizonConfig(T=0.01, dt=0.001),
        mid_price=MidPriceConfig(
            model="regime_switching",
            s0=100.0,
            regimes=(
                RegimeParams(name="calm", mu=0.0, sigma=0.5),
                RegimeParams(name="stormy", mu=0.0, sigma=2.5),
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


def test_backtest_smoke(tmp_path: pathlib.Path) -> None:
    cfg = _tiny_cfg(tmp_path / "out")

    # --- Run the backtest. -------------------------------------------- #
    result = run_backtest(cfg)

    # Per-cell counts must equal n_paths for every (strategy, regime).
    n_paths = cfg.mc.n_paths
    expected_strategies = ("avellaneda_stoikov", "symmetric", "semi_as")
    regime_names = [r.name for r in cfg.mid_price.regimes]
    assert set(result.paths.keys()) == {
        (strategy, name)
        for strategy in expected_strategies
        for name in regime_names
    }
    for key, cell in result.paths.items():
        assert len(cell) == n_paths, (
            f"cell {key!r} has {len(cell)} paths; expected {n_paths}"
        )

    # --- Persist the result. ------------------------------------------ #
    run_dir = persist(result)
    assert run_dir.exists() and run_dir.is_dir()

    # Required top-level files.
    cfg_yaml = run_dir / "config.resolved.yaml"
    manifest_json = run_dir / "manifest.json"
    assert cfg_yaml.is_file()
    assert manifest_json.is_file()
    assert cfg_yaml.stat().st_size > 0

    # Manifest content sanity.
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    assert manifest["master_seed"] == cfg.master_seed
    assert manifest["n_paths"] == n_paths
    assert manifest["n_regimes"] == len(regime_names)
    assert manifest["regime_names"] == regime_names
    assert manifest["n_steps"] == cfg.horizon.n_steps

    # --- Per-(strategy, regime) parquet files must all exist. -------- #
    n_plus_1 = cfg.horizon.n_steps + 1
    for strategy in expected_strategies:
        for regime_name in regime_names:
            paths_file = (
                run_dir / "paths" / strategy / f"regime={regime_name}.parquet"
            )
            fills_file = (
                run_dir / "fills" / strategy / f"regime={regime_name}.parquet"
            )
            assert paths_file.is_file(), f"missing {paths_file}"
            assert fills_file.is_file(), f"missing {fills_file}"

            # Read back and verify the schema.
            paths_tbl = pq.read_table(str(paths_file))
            assert tuple(paths_tbl.column_names) == _PATHS_COLUMNS, (
                f"columns out of order in {paths_file}: "
                f"{paths_tbl.column_names}"
            )
            assert paths_tbl.num_rows == n_paths * n_plus_1

            fills_tbl = pq.read_table(str(fills_file))
            assert tuple(fills_tbl.column_names) == _FILLS_COLUMNS, (
                f"columns out of order in {fills_file}: "
                f"{fills_tbl.column_names}"
            )
            # Fills tables may be empty (no fills resolved) but the file
            # must still exist and carry the documented schema. We also
            # cross-check the count against the in-memory PathResults.
            cell = result.paths[(strategy, regime_name)]
            expected_fill_count = sum(
                int(pr.n_bid_fills + pr.n_ask_fills) for pr in cell
            )
            assert fills_tbl.num_rows == expected_fill_count, (
                f"fill count mismatch in {fills_file}: "
                f"{fills_tbl.num_rows} vs {expected_fill_count}"
            )

    # --- Spot-check one path file's contents. ------------------------- #
    sample_path_file = (
        run_dir
        / "paths"
        / "avellaneda_stoikov"
        / f"regime={regime_names[0]}.parquet"
    )
    sample_tbl = pq.read_table(str(sample_path_file))
    # Convert columns to plain Python lists / NumPy arrays so we don't
    # depend on pandas being installed in the test venv.
    path_index_col = sample_tbl.column("path_index").to_pylist()
    step_index_col = sample_tbl.column("step_index").to_pylist()
    mid_price_col = sample_tbl.column("mid_price").to_pylist()
    cash_col = sample_tbl.column("cash").to_pylist()
    inventory_col = sample_tbl.column("inventory").to_pylist()
    # Persistence writes path-by-path then step-by-step within each path
    # (long format). Verify that layout exactly.
    for p in range(n_paths):
        lo = p * n_plus_1
        hi = lo + n_plus_1
        assert path_index_col[lo:hi] == [p] * n_plus_1
        assert step_index_col[lo:hi] == list(range(n_plus_1))
        # mid_price strictly positive (GBM invariant).
        for v in mid_price_col[lo:hi]:
            assert v > 0
        # cash and inventory at step 0 are zero by construction.
        assert math.isclose(float(cash_col[lo]), 0.0)
        assert int(inventory_col[lo]) == 0
