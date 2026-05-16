"""Command-line entry point.

Exposes ``python -m etf_mm_sim run <config.yaml>`` for end-to-end
backtest runs including persistence, plot generation, and a brief
summary printed to stdout. See ``design.md`` §cli.

Subcommands
-----------
* ``run <config.yaml>`` -- load the YAML, run the backtest, persist
  results, render plots (unless ``--no-plots`` is set), and print the
  output run-directory plus a short comparison table to stdout.

Exit codes
----------
* ``0`` on success.
* ``1`` on configuration validation errors
  (:class:`~etf_mm_sim.config.ConfigError`).
* ``2`` on argparse usage errors (default ``argparse`` behaviour).
* ``3`` on persistence I/O errors
  (:class:`~etf_mm_sim.persistence.PersistenceError`).
* ``4`` on any other uncaught exception. The exception traceback is
  *not* printed; the message is written to ``stderr`` and the program
  exits.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Optional

import numpy as np

from . import analytics, viz
from .backtest import run_backtest
from .config import ConfigError, load_config
from .persistence import PersistenceError, persist
from .seeding import analytics_seed

__all__ = ["main"]


# --------------------------------------------------------------------------- #
# Argparse                                                                    #
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser with a single ``run`` subcommand."""
    parser = argparse.ArgumentParser(
        prog="etf-mm-sim",
        description=(
            "ETF MM Arbitrage Simulator — Monte Carlo backtest runner."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser(
        "run",
        help="Run a backtest from a YAML configuration file.",
        description=(
            "Load a YAML configuration, run the Monte Carlo backtest, "
            "persist outputs, render plots, and print a short summary."
        ),
    )
    run_p.add_argument(
        "config",
        type=pathlib.Path,
        help="Path to the YAML configuration file.",
    )
    run_p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip rendering plots and the summary HTML.",
    )
    return parser


# --------------------------------------------------------------------------- #
# Subcommand: run                                                             #
# --------------------------------------------------------------------------- #


def _format_paired_summary(rows: list[dict[str, object]]) -> str:
    """Return a fixed-width plain-text table for the per-regime summary."""
    headers = (
        "regime",
        "diff_mean_pnl",
        "diff_mean_pnl_ci",
        "diff_max_dd",
        "diff_max_dd_ci",
    )

    def _fmt(v: object) -> str:
        if isinstance(v, tuple):
            return f"({float(v[0]):+.4f}, {float(v[1]):+.4f})"
        if isinstance(v, float):
            return f"{v:+.4f}"
        return str(v)

    cells: list[list[str]] = [list(headers)]
    for row in rows:
        cells.append([_fmt(row[h]) for h in headers])

    widths = [
        max(len(cells[r][c]) for r in range(len(cells)))
        for c in range(len(headers))
    ]
    lines: list[str] = []
    for r, line_cells in enumerate(cells):
        line = "  ".join(
            line_cells[c].ljust(widths[c]) for c in range(len(headers))
        )
        lines.append(line)
        if r == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def _run_command(config_path: pathlib.Path, no_plots: bool) -> int:
    """Execute the ``run`` subcommand. Returns the process exit code."""
    cfg = load_config(config_path)

    # --- Backtest sweep. ---------------------------------------------- #
    result = run_backtest(cfg)

    # --- Persist artifacts. ------------------------------------------- #
    run_dir = persist(result)

    # --- Analytics & paired comparison per regime. -------------------- #
    n_regimes = len(cfg.mid_price.regimes)
    # Reserved analytics seed → spawn one sub-sequence per regime so the
    # bootstraps for distinct regimes do not share state. Each regime
    # spawns a further (vs-symmetric, vs-semi-as) split.
    analytics_ss = analytics_seed(cfg.master_seed, n_regimes)
    regime_seeds = analytics_ss.spawn(n_regimes)

    vs_sym_rows: list[dict[str, object]] = []
    vs_semi_rows: list[dict[str, object]] = []
    for r_idx, regime in enumerate(cfg.mid_price.regimes):
        as_cell = result.paths[("avellaneda_stoikov", regime.name)]
        sym_cell = result.paths[("symmetric", regime.name)]
        semi_cell = result.paths[("semi_as", regime.name)]

        # Aggregate per-cell summaries (kept around so callers can later
        # reuse them; the CLI itself only needs the paired comparisons).
        for strategy, cell in (
            ("avellaneda_stoikov", as_cell),
            ("symmetric", sym_cell),
            ("semi_as", semi_cell),
        ):
            analytics.aggregate_cell(
                cell,
                strategy,
                regime.name,
                cfg.analytics.sharpe_annualization_factor,
                cfg.analytics.adverse_selection_horizon_steps,
            )

        # Two independent bootstrap streams for the two paired
        # comparisons so they do not share state.
        vs_sym_ss, vs_semi_ss = regime_seeds[r_idx].spawn(2)

        paired_vs_sym = analytics.compute_paired_comparison(
            as_cell,
            sym_cell,
            regime.name,
            cfg.analytics.bootstrap_iterations,
            cfg.analytics.bootstrap_alpha,
            vs_sym_ss,
            baseline="symmetric",
        )
        paired_vs_semi = analytics.compute_paired_comparison(
            as_cell,
            semi_cell,
            regime.name,
            cfg.analytics.bootstrap_iterations,
            cfg.analytics.bootstrap_alpha,
            vs_semi_ss,
            baseline="semi_as",
        )

        for paired, rows in (
            (paired_vs_sym, vs_sym_rows),
            (paired_vs_semi, vs_semi_rows),
        ):
            rows.append(
                {
                    "regime": paired.regime,
                    "diff_mean_pnl": paired.diff_mean_pnl,
                    "diff_mean_pnl_ci": paired.diff_mean_pnl_ci,
                    "diff_max_dd": paired.diff_max_dd,
                    "diff_max_dd_ci": paired.diff_max_dd_ci,
                }
            )

    # --- Plots. ------------------------------------------------------- #
    if not no_plots:
        plots_dir = run_dir / "plots"
        viz.save_all_figures(
            result,
            plots_dir,
            cfg.analytics.sharpe_annualization_factor,
            cfg.analytics.adverse_selection_horizon_steps,
        )

    # --- stdout reporting. ------------------------------------------- #
    # Run directory first (machine-friendly: a downstream script can
    # capture the first line). Followed by the human-readable summaries:
    # AS-vs-Symmetric (combined effect of dynamic spread + skew), then
    # AS-vs-Semi-AS (isolates the inventory-skew effect alone).
    print(str(run_dir))
    print()
    print("Per-regime AS vs. Symmetric paired comparison:")
    print(_format_paired_summary(vs_sym_rows))
    print()
    print("Per-regime AS vs. Semi-AS paired comparison (isolates skew effect):")
    print(_format_paired_summary(vs_semi_rows))
    return 0


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point. Returns the process exit code.

    Parameters
    ----------
    argv:
        Argument vector excluding the program name (the same convention
        as :pyattr:`sys.argv[1:]`). When ``None`` the function reads
        from :pyattr:`sys.argv` directly, mirroring the standard
        argparse contract.

    Returns
    -------
    int
        Exit code; ``0`` on success, non-zero on failure.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        try:
            return _run_command(args.config, args.no_plots)
        except ConfigError as exc:
            print(f"configuration error: {exc}", file=sys.stderr)
            return 1
        except PersistenceError as exc:
            print(f"persistence error: {exc}", file=sys.stderr)
            return 3
        except Exception as exc:  # noqa: BLE001 - top-level CLI guard
            print(f"error: {exc}", file=sys.stderr)
            return 4
    # argparse will already have errored out for unknown commands;
    # this branch is defensive.
    parser.error(f"unknown command: {args.command!r}")
    return 2  # pragma: no cover - parser.error raises SystemExit


# Touch numpy import so static analysis doesn't drop it; analytics_seed's
# return type carries an np.random.SeedSequence that requires the module
# to be imported in this translation unit.
_ = np
