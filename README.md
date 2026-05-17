# ETF Market Making & Arbitrage Simulator

Mode 1: Python-based Avellaneda-Stoikov market-making simulator and Monte Carlo backtester on synthetic mid-price paths.

The simulator benchmarks an Avellaneda-Stoikov (AS) quoter against two baselines — a Symmetric constant-spread quoter and a Semi-AS quoter (AS dynamic spread, no inventory skew) — across multiple volatility regimes. Comparing AS against Semi-AS isolates the inventory-skew effect; comparing AS against Symmetric measures the combined effect of dynamic spread plus skew. The simulator produces P&L analytics decomposing performance into fill-rate asymmetry, adverse selection, and spread capture.

A C++17 subtree under [`cpp/`](cpp/README.md) (Mode 3) ships a real-time WebSocket consumer for Binance BTCUSDT spot + perpetual futures, an `std::map<Price, Level>` order book backed by a single-producer single-consumer lock-free ring on the WS → book interface, a deterministic binary tape format for capture, a replay binary, and a spot-vs-perp basis-deviation strategy with EWMA smoothing and threshold-crossing events. See [`docs/spec/mode3_design.md`](docs/spec/mode3_design.md) for the architecture and [`cpp/README.md`](cpp/README.md) for build instructions.

The Mode 1 Python simulator and the Mode 3 C++ subtree are independent — the two halves do not call each other. They live in the same repo for cohesion of the CV story.

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
│   │   ├── semi_as.py
│   │   └── symmetric.py
│   ├── fill_engine.py      # Poisson λ = A·exp(-k·δ); Bernoulli per step
│   ├── risk_manager.py     # Inventory bounds, kill-switch, terminal flatten
│   ├── path_runner.py      # Per-path inventory state machine
│   ├── backtest.py         # Monte Carlo orchestration
│   ├── analytics.py        # P&L metrics + per-cell aggregation
│   ├── bootstrap.py        # Paired percentile bootstrap CI
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

Prints the run output directory on the first line, then two stacked per-regime paired comparison tables: AS-vs-Symmetric (combined effect of dynamic spread + skew) followed by AS-vs-Semi-AS (isolates the inventory-skew effect alone).

```
results/run_20260516T151922Z_1161ca49

Per-regime AS vs. Symmetric paired comparison:
regime  diff_mean_pnl  diff_mean_pnl_ci       diff_max_dd  diff_max_dd_ci
------  -------------  ---------------------  -----------  ------------------------
low     +49.8405       (+40.5546, +58.9929)   -317.6929    (-325.8949, -309.5473)
normal  +30.6556       (+4.1128, +56.9091)    -655.3652    (-683.9176, -629.2669)
high    -49.6345       (-194.8379, +80.4846)  -1516.1365   (-1681.7892, -1372.9714)

Per-regime AS vs. Semi-AS paired comparison (isolates skew effect):
regime  diff_mean_pnl  diff_mean_pnl_ci       diff_max_dd  diff_max_dd_ci
------  -------------  ---------------------  -----------  ----------------------
low     -1.3234        (-5.2422, +2.5443)     -120.4290    (-124.2056, -116.8675)
normal  -2.6985        (-12.5222, +7.0859)    -243.6900    (-253.7183, -233.8372)
high    -25.8438       (-117.0454, +37.2593)  -562.9676    (-624.4113, -508.4732)
```

Reading the headline: AS reduces max drawdown vs. Symmetric in every regime (−318 / −655 / −1516 across low/normal/high vol), with mean P&L positive in low and normal vol and statistically tied in high vol. The AS-vs-Semi-AS comparison isolates inventory skew alone; with the spread schedule held constant, skew accounts for roughly **38% / 37% / 37%** of the AS-vs-Symmetric drawdown reduction at no statistically significant P&L cost in any regime.

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

## Three-strategy paired comparison

The simulator runs three quoting strategies on the same shared mid-price seeds for every `(regime, path)` cell:

- **Symmetric** — constant half-spread `δ_base` posted at `s ± δ_base`. No inventory skew, no time decay. The simplest baseline.
- **Semi-AS** — AS dynamic half-spread `δ*(t)` posted at `s ± δ*(t)`. The spread shrinks toward `(1/γ)·ln(1 + γ/k)` as `t → T`, but quotes are always centered on the mid: no inventory skew.
- **AS** — full Avellaneda-Stoikov: `r ± δ*(t)` with `r = s − q·γ·σ²·(T − t)`. Dynamic spread plus inventory skew.

Comparing AS against the two baselines gives complementary attributions:

| Comparison | What it measures |
| --- | --- |
| AS vs Symmetric | Combined effect of dynamic spread + inventory skew |
| AS vs Semi-AS | Inventory-skew effect alone (spread schedule held fixed) |

Because the three strategies share the per-path `(mid, fill)` seed sequences, the paired bootstrap CIs are computed on per-path differences and remove the between-path variance driven by mid-price-seed noise. The Semi-AS run produces its own independent fill stream — quotes diverge from AS once inventory diverges, so realized fills differ even though the per-path uniform draws are identical. This is the key change from a counterfactual *replay*: AS-vs-Semi-AS is a true paired Monte Carlo comparison on the same seeded inputs.

### Calibrating γ across volatility regimes

The inventory-skew coefficient is `γ · σ² · (T − t)`. With a single global `γ`, the skew grows quadratically with σ across regimes; at moderate inventory in high-vol, the skewed reservation price can fall enough that AS posts crossed quotes (`δ ≤ 0`) and the fill engine treats them as guaranteed fills at heavily haircut prices. The result is a "panic-flatten" regime that controls drawdown but bleeds expected P&L.

`RegimeParams` accepts an optional per-regime `gamma` field that overrides the global AS γ. The reference config keeps `γ · σ² = 0.025` constant across regimes:

```yaml
regimes:
  - name: low
    sigma: 0.5
    gamma: 0.1       # 0.025 / 0.5**2
  - name: normal
    sigma: 1.0
    gamma: 0.025     # 0.025 / 1.0**2
  - name: high
    sigma: 2.0
    gamma: 0.00625   # 0.025 / 2.0**2
```

Without this scaling, high-vol AS underperforms Symmetric by ~140 in mean P&L; with it, the gap shrinks to ~50 and the CI straddles zero. Drawdown reduction holds in both cases.

## Property-based testing

23 correctness properties from `design.md` are encoded as Hypothesis tests, each tagged with the property number and the requirements clause it validates. Highlights:

- **P1**: configuration round-trip (YAML dump + load = identity).
- **P2–P5**: mid-price simulator determinism, length, positivity, zero-vol drift.
- **P6–P9**: AS closed-form identities, skew sign, no-quote at terminal; Symmetric inventory independence.
- **P10–P13**: fill intensity / probability / crossed-quote / RNG determinism.
- **P14–P16**: inventory bound invariant, kill-switch latch, terminal-flatten identity.
- **P17–P18**: paired mid-price seeding across strategies, bit-identical persisted outputs.
- **P19–P21**: max-DD non-negativity, spread-capture formula, bootstrap determinism.
- **P23**: SeedSequence tree determinism.
- **Semi-AS A/B/C** (`tests/property/test_semi_as_quoter.py`): Semi-AS quotes are mid-symmetric; Semi-AS half-spread equals AS `δ*(t)` pointwise; Semi-AS suppresses at `t ≥ T` (matching AS terminal handling).

Run them all:

```bash
.venv/Scripts/python.exe -m pytest -q
```

## Documents

- [requirements.md](docs/spec/requirements.md) — 12 EARS requirements with acceptance criteria.
- [design.md](docs/spec/design.md) — package layout, vectorization strategy, seed tree, counterfactual replay, paired bootstrap, 23 correctness properties.
- [tasks.md](docs/spec/tasks.md) — 20 implementation tasks.
