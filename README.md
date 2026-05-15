# ETF Market Making & Arbitrage Simulator

Mode 1: Python-based Avellaneda-Stoikov market-making simulator and Monte Carlo backtester on synthetic mid-price paths.

The simulator benchmarks an Avellaneda-Stoikov (AS) quoter against a symmetric constant-spread baseline across multiple volatility regimes and produces P&L analytics decomposing performance into fill-rate asymmetry, adverse selection, spread capture, and an inventory-skew drawdown attribution computed via a counterfactual no-skew replay.

The C++ lock-free order book, Binance WebSocket tick ingestion, ETF/synthetic-basket NAV deviation tracking, and arbitrage signal generation are deferred to a later iteration and are out of scope for this release.

## Avellaneda-Stoikov closed form

For a market maker with utility `U(x) = -exp(-γ x)` over terminal wealth, mid-price following Brownian motion with volatility `σ`, finite horizon `T`, and a Poisson order-arrival intensity `λ(δ) = A · exp(-k · δ)` per side (where `δ` is the quote distance from the mid), the optimal quotes (Avellaneda & Stoikov, 2008) are

```
reservation price:   r(s, q, t) = s − q · γ · σ² · (T − t)
optimal half-spread: δ*(t)      = ½ · γ · σ² · (T − t) + (1/γ) · ln(1 + γ/k)
bid quote:           p_b        = r − δ*
ask quote:           p_a        = r + δ*
```

The `q · γ · σ² · (T − t)` term is the **inventory skew**: a long position (`q > 0`) shifts the quote midpoint *below* the mid so the market maker's bid is less aggressive and ask is more aggressive, biasing fills toward shorting and pulling inventory back toward zero. The skew shrinks linearly to zero as `t → T`.

The constant tail term `(1/γ) · ln(1 + γ/k)` is the spread the agent posts under no inventory pressure, balancing the rate of fills (controlled by `k`) against per-fill profit (controlled by `γ`).

## Repository layout

```
etf-mm-sim/
├── etf_mm_sim/
│   ├── config.py           # Frozen dataclasses + YAML I/O + validation
│   ├── seeding.py          # Deterministic SeedSequence tree
│   ├── mid_price.py        # GBM and regime-switching path generators
│   ├── quoters/
│   │   ├── avellaneda_stoikov.py
│   │   └── symmetric.py
│   ├── fill_engine.py      # Poisson λ = A·exp(-k·δ); Bernoulli per step
│   ├── risk_manager.py     # Inventory bounds, kill-switch, terminal flatten
│   ├── path_runner.py      # Per-path inventory state machine
│   ├── backtest.py         # Monte Carlo orchestration
│   ├── analytics.py        # P&L metrics + per-cell aggregation
│   ├── bootstrap.py        # Paired percentile bootstrap CI
│   ├── counterfactual.py   # No-skew replay for drawdown attribution
│   ├── persistence.py      # Deterministic Parquet + JSON
│   ├── viz.py              # matplotlib renderers
│   └── cli.py              # python -m etf_mm_sim run <config.yaml>
├── configs/default.yaml
├── notebooks/default_backtest.ipynb
└── tests/{unit,property,integration,smoke,perf}/
```

## Install

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

Required runtime: Python 3.11+, NumPy, pandas, pyarrow, PyYAML, matplotlib. Optional `numba` for the JIT path runner (currently deferred — pure-Python meets the 5-minute target on the reference workload).

## CLI usage

```bash
python -m etf_mm_sim run configs/default.yaml
```

Prints the run output directory on the first line, then a per-regime AS-vs-Symmetric paired comparison table:

```
results/run_20260515T120000Z_a1b2c3d4

Per-regime AS vs. Symmetric paired comparison:
regime  diff_mean_pnl  diff_mean_pnl_ci          skew_dd_attribution  skew_dd_attribution_ci
------  -------------  ------------------------  -------------------  -------------------------
low     +0.0234        (+0.0152, +0.0316)        +0.0089              (+0.0042, +0.0136)
normal  +0.0511        (+0.0398, +0.0623)        +0.0247              (+0.0185, +0.0309)
high    +0.0892        (+0.0701, +0.1083)        +0.0612              (+0.0498, +0.0726)
```

Add `--no-plots` to skip plot rendering. The run directory contains:

```
run_<utc>_<cfg_sha[:8]>/
├── config.resolved.yaml
├── manifest.json                  # master_seed, package version, library versions
├── paths/<strategy>/regime=<name>.parquet
├── fills/<strategy>/regime=<name>.parquet
└── plots/                         # absent with --no-plots
    ├── terminal_pnl_<regime>.png
    ├── sample_path_<strategy>_<regime>.png
    └── summary_table.html
```

## Notebook usage

```bash
.venv/Scripts/python.exe -m jupyter notebook notebooks/default_backtest.ipynb
```

The default notebook loads `configs/default.yaml`, reduces `n_paths` to 100 for fast iteration, runs the backtest, and renders the summary table plus per-regime histograms and one sample-path diagnostic.

## Reproducibility

Every stochastic draw derives from a single integer `master_seed` via `numpy.random.SeedSequence.spawn`. Two runs with the same YAML produce **bit-identical** persisted Parquet files (the run timestamp lives only in the directory name and a single `manifest.json` field, both excluded from the byte-identity contract).

```
master_seed → SeedSequence.spawn(R + 1)
              ├─[0..R-1] per-regime → spawn(N_paths) → spawn(2) → (mid, fill)
              └─[R]      analytics  (paired-bootstrap RNG)
```

AS and Symmetric quoters share the same `(mid, fill)` seeds at every `(regime, path)` cell so observed differences come from quoting behavior, not path noise.

## Inventory-skew drawdown attribution

The `skew_dd_attribution` metric isolates the cash effect of AS inventory skew by:

1. Recording the realized AS fill events per path.
2. Replaying those *same fills* against a counterfactual no-skew quoter posting at `s ± δ*` (AS spread, no skew).
3. Computing the per-path drawdown difference `cf_max_dd − as_max_dd`.
4. Reporting the mean and a paired percentile-bootstrap CI per regime.

A positive value means the AS skew *reduced* drawdown vs an otherwise-identical quoter that would have posted the same spread but kept its midpoint at the unskewed mid. This is the cash effect of skew given the same fills; it is not a full counterfactual P&L because the AS-imposed inventory bounds are still implicit in the fill stream.

## Property-based testing

23 correctness properties from `design.md` are encoded as Hypothesis tests, each tagged with the property number and the requirements clause it validates. Highlights:

- **P1**: configuration round-trip (YAML dump + load = identity).
- **P2–P5**: mid-price simulator determinism, length, positivity, zero-vol drift.
- **P6–P9**: AS closed-form identities, skew sign, no-quote at terminal; Symmetric inventory independence.
- **P10–P13**: fill intensity / probability / crossed-quote / RNG determinism.
- **P14–P16**: inventory bound invariant, kill-switch latch, terminal-flatten identity.
- **P17–P18**: paired mid-price seeding across strategies, bit-identical persisted outputs.
- **P19–P22**: max-DD non-negativity, spread-capture formula, bootstrap determinism, counterfactual inventory equality.
- **P23**: SeedSequence tree determinism.

Run them all:

```bash
.venv/Scripts/python.exe -m pytest -q
```

## Documents

- [requirements.md](docs/spec/requirements.md) — 12 EARS requirements with acceptance criteria.
- [design.md](docs/spec/design.md) — package layout, vectorization strategy, seed tree, counterfactual replay, paired bootstrap, 23 correctness properties.
- [tasks.md](docs/spec/tasks.md) — 20 implementation tasks.
