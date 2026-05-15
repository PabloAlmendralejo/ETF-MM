"""Integration test for the CLI end-to-end ``run`` subcommand.

# Feature: etf-mm-arbitrage-simulator
# Validates: Requirements 7.4, 10.4

Writes a minimal valid YAML configuration to ``tmp_path``, invokes
``python -m etf_mm_sim run <yaml>`` via :mod:`subprocess`, and asserts
the documented filesystem contract holds (run directory + manifest).
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import textwrap


_WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _write_minimal_yaml(yaml_path: pathlib.Path, output_dir: pathlib.Path) -> None:
    """Write a minimal valid YAML matching the schema in ``config.py``."""
    # Use forward-slash paths so YAML parses cleanly on Windows; pathlib
    # accepts forward slashes interchangeably.
    out_str = output_dir.as_posix()
    yaml_text = textwrap.dedent(
        f"""\
        master_seed: 42
        horizon:
          T: 0.01
          dt: 0.001
        mid_price:
          model: regime_switching
          s0: 100.0
          regimes:
            - name: calm
              mu: 0.0
              sigma: 0.5
            - name: stormy
              mu: 0.0
              sigma: 2.0
          transition_matrix:
            - [0.9, 0.1]
            - [0.1, 0.9]
        quoters:
          avellaneda_stoikov:
            gamma: 0.1
            k: 1.5
            A: 140.0
          symmetric:
            delta_base: 0.05
        fill:
          A_b: 140.0
          k_b: 1.5
          A_a: 140.0
          k_a: 1.5
        risk:
          q_max: 5
          L_kill: null
        mc:
          n_paths: 5
        analytics:
          adverse_selection_horizon_steps: 1
          bootstrap_iterations: 10
          bootstrap_alpha: 0.05
          sharpe_annualization_factor: 15.87
        output:
          dir: {out_str}
          format: parquet
        """
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")


def test_cli_run_creates_manifest(tmp_path: pathlib.Path) -> None:
    yaml_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "results"
    _write_minimal_yaml(yaml_path, output_dir)

    cmd = [
        sys.executable,
        "-m",
        "etf_mm_sim",
        "run",
        str(yaml_path),
        "--no-plots",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(_WORKSPACE_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr

    # The first stdout line is the run directory path.
    first_line = proc.stdout.strip().splitlines()[0]
    run_dir = pathlib.Path(first_line)
    assert run_dir.exists() and run_dir.is_dir(), (
        f"run directory not found: {run_dir!s}"
    )

    # Every run directory carries a manifest.json with the master seed.
    manifest = run_dir / "manifest.json"
    assert manifest.is_file()
    assert manifest.stat().st_size > 0

    # The run directory lives under the configured output dir.
    assert run_dir.parent == output_dir
    assert run_dir.name.startswith("run_")

    # --no-plots was passed; the plots dir should not exist.
    assert not (run_dir / "plots").exists()
