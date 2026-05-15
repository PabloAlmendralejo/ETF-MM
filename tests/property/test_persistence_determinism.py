"""Property test for bit-identical persisted outputs.

# Feature: etf-mm-arbitrage-simulator, Property 18: Bit-identical persisted outputs
# Validates: Requirements 7.5, 10.2

When the same YAML_Config and master seed are supplied, two
``run_backtest`` + ``persist`` invocations must produce byte-identical
data files (Req 7.5). The directory name and the ``run_timestamp_utc``
field of ``manifest.json`` are run-time-dependent and excluded from the
byte-equality check.

Files compared for byte equality:

* every ``paths/<strategy>/regime=<name>.parquet``
* every ``fills/<strategy>/regime=<name>.parquet``
* ``config.resolved.yaml``
* ``manifest.json`` after stripping the ``run_timestamp_utc`` key

We use a small but non-trivial configuration (5 paths, 2 regimes, 10
inner steps) so the Parquet files contain real data while the test
remains fast (well under one second).
"""

from __future__ import annotations

import json
import pathlib
import time

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


def _tiny_cfg(output_dir: pathlib.Path) -> Configuration:
    return Configuration(
        master_seed=42,
        horizon=HorizonConfig(T=0.01, dt=0.001),
        mid_price=MidPriceConfig(
            model="regime_switching",
            s0=100.0,
            regimes=(
                RegimeParams(name="low", mu=0.0, sigma=0.5),
                RegimeParams(name="high", mu=0.0, sigma=2.0),
            ),
            transition_matrix=(
                (0.95, 0.05),
                (0.05, 0.95),
            ),
        ),
        quoters_as=ASParams(gamma=0.1, k=1.5, A=140.0),
        quoters_sym=SymmetricParams(delta_base=0.05),
        fill=FillParams(A_b=140.0, k_b=1.5, A_a=140.0, k_a=1.5),
        risk=RiskParams(q_max=5, L_kill=None),
        mc=MCParams(n_paths=10),
        analytics=AnalyticsParams(
            adverse_selection_horizon_steps=1,
            bootstrap_iterations=10,
            bootstrap_alpha=0.05,
            sharpe_annualization_factor=15.87,
        ),
        output=OutputConfig(dir=output_dir, format="parquet"),
    )


def _collect_data_files(run_dir: pathlib.Path) -> dict[str, bytes]:
    """Collect every persisted *data* file under ``run_dir`` keyed by relpath.

    Excludes ``manifest.json`` because it carries the run timestamp; the
    caller compares the manifest separately after stripping the
    timestamp field.
    """
    files: dict[str, bytes] = {}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "manifest.json":
            continue
        rel = path.relative_to(run_dir).as_posix()
        files[rel] = path.read_bytes()
    return files


def _read_manifest_minus_timestamp(run_dir: pathlib.Path) -> dict:
    payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    payload.pop("run_timestamp_utc", None)
    return payload


def test_property_18_bit_identical_persisted_outputs(
    tmp_path: pathlib.Path,
) -> None:
    """Two runs of the same config produce byte-identical data files."""
    cfg = _tiny_cfg(tmp_path / "out")

    result_a = run_backtest(cfg)
    run_dir_a = persist(result_a)

    # Sleep for a bit longer than one second so the run timestamp
    # (resolution: seconds) differs between the two invocations. This
    # forces the directory names (and the ``run_timestamp_utc`` manifest
    # field) to differ -- exactly the situation the determinism contract
    # has to tolerate.
    time.sleep(1.1)

    result_b = run_backtest(cfg)
    run_dir_b = persist(result_b)

    # Sanity: directory names should have differed (timestamp + sha8).
    # If the timestamp happens to collide we still expect the contents
    # to match; we don't assert inequality here.

    files_a = _collect_data_files(run_dir_a)
    files_b = _collect_data_files(run_dir_b)

    assert set(files_a.keys()) == set(files_b.keys())
    for rel in sorted(files_a.keys()):
        assert files_a[rel] == files_b[rel], (
            f"file {rel!r} differs between runs of the same config "
            f"({len(files_a[rel])} vs {len(files_b[rel])} bytes)"
        )

    # Manifest equality after stripping the run-timestamp key.
    manifest_a = _read_manifest_minus_timestamp(run_dir_a)
    manifest_b = _read_manifest_minus_timestamp(run_dir_b)
    assert manifest_a == manifest_b
