"""Generate notebooks/default_backtest.ipynb via nbformat.

Run once during build:

    python notebooks/_build_default_notebook.py

The script is checked into the repo so the notebook can be regenerated
on demand without hand-editing JSON. The output ``.ipynb`` is the
authoritative artifact and is also checked in (Req 9.4).
"""

from __future__ import annotations

import pathlib

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def _imports_cell() -> str:
    return (
        "import sys, pathlib\n"
        "sys.path.insert(0, str(pathlib.Path('..').resolve()))\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "from etf_mm_sim.config import load_config, MCParams\n"
        "from etf_mm_sim.backtest import run_backtest\n"
        "from etf_mm_sim import analytics, viz, counterfactual\n"
        "from etf_mm_sim.seeding import analytics_seed\n"
        "from dataclasses import replace\n"
    )


def _load_cfg_cell() -> str:
    return (
        "cfg = load_config('../configs/default.yaml')\n"
        "# Reduce n_paths so the notebook executes quickly in CI.\n"
        "cfg = replace(cfg, mc=MCParams(n_paths=100))\n"
    )


def _run_backtest_cell() -> str:
    return "result = run_backtest(cfg)\n"


def _aggregate_cell() -> str:
    return (
        "cells = [\n"
        "    analytics.aggregate_cell(\n"
        "        result.paths[(s, r.name)],\n"
        "        s,\n"
        "        r.name,\n"
        "        cfg.analytics.sharpe_annualization_factor,\n"
        "        cfg.analytics.adverse_selection_horizon_steps,\n"
        "    )\n"
        "    for s in ('avellaneda_stoikov', 'symmetric')\n"
        "    for r in cfg.mid_price.regimes\n"
        "]\n"
        "summary = viz.render_summary_table(cells)\n"
        "summary\n"
    )


def _hist_cell() -> str:
    return (
        "for r in cfg.mid_price.regimes:\n"
        "    fig = viz.plot_terminal_pnl_hist(result, r.name)\n"
    )


def _sample_cell() -> str:
    return (
        "sample = result.paths[("
        "'avellaneda_stoikov', cfg.mid_price.regimes[1].name)][0]\n"
        "fig = viz.plot_sample_path(sample)\n"
    )


def main() -> None:
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(
            "# ETF MM Arbitrage Simulator — Default Backtest\n"
            "\n"
            "Reference notebook driving `configs/default.yaml`. Loads the "
            "config, runs a reduced-path Monte Carlo sweep, aggregates "
            "per-cell analytics, renders the summary table, and produces "
            "the per-regime terminal-P&L histograms plus one sample-path "
            "diagnostic."
        ),
        new_code_cell(_imports_cell()),
        new_code_cell(_load_cfg_cell()),
        new_code_cell(_run_backtest_cell()),
        new_code_cell(_aggregate_cell()),
        new_code_cell(_hist_cell()),
        new_code_cell(_sample_cell()),
    ]
    # Pin a stable kernelspec so nbclient can pick it up. Use Python 3
    # which matches the project's >=3.11 requirement.
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
        },
    }

    out = pathlib.Path(__file__).resolve().parent / "default_backtest.ipynb"
    with out.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
