# Design Document

## Overview

This design realizes Mode 1 of the ETF Market Making & Arbitrage Simulator: a single-process,
single-core, vectorized-NumPy Python package that runs Monte Carlo backtests of an
Avellaneda-Stoikov (AS) quoter against a symmetric constant-spread baseline on synthetic
mid-price paths, and emits paired P&L analytics with drawdown attribution.

The design optimizes for three constraints from the requirements document:

1. **Reproducibility** (Req 10): every stochastic draw is derived from a single master integer
   seed via a documented `numpy.random.SeedSequence.spawn` tree, so identical configs produce
   bit-identical persisted outputs.
2. **Performance** (Req 11): 1000 paths × 3 regimes × 2 strategies in under 5 minutes on one
   laptop core. The mid-price simulator and fill engine are fully vectorized; the inventory
   state machine uses a tight per-step loop with pre-computed RNG draws and an optional
   `numba` JIT for the hot path, with a pure-NumPy fallback.
3. **Statistical rigor** (Req 7, 8): paired Monte Carlo design (same mid-price seeds across
   strategies), paired-bootstrap CIs, and a counterfactual no-skew replay to attribute
   AS drawdown reduction specifically to inventory skew rather than to spread sizing.

Key design decisions and rationales:

- **Strategy as a callable, not an object hierarchy.** A quoter is a function
  `quote(s, q, t) → (p_b, p_a, suppress_b, suppress_a)`. This avoids the cost of method
  dispatch in the inner loop and keeps the AS and Symmetric implementations as ~10 lines each.
- **Two-phase per-path execution.** Phase 1 vectorizes everything that does not depend on
  inventory (mid-price path, optimal half-spread schedule `δ*(t)`, fill-uniform draws).
  Phase 2 is a sequential step loop that consumes the pre-computed arrays. This is faster
  than a fully imperative loop and far simpler than a fully vectorized fixed-point inventory
  solver (which is not even well-defined under inventory-dependent fill probabilities).
- **Counterfactual no-skew replay reuses the realized fill stream.** Per Req 8.9, the
  counterfactual quoter prices at `s ± δ*_t` (AS spread, no skew) and is fed the *same*
  bid/ask fill events as the actual AS run. This isolates the cash impact of skewed fill
  prices while holding fill timing and inventory trajectory constant.
- **Parquet for path-level outputs, JSON for run metadata.** Parquet is columnar, compresses
  the long time-series cleanly, and gives bit-identical file bytes for fixed inputs (Req 7.5)
  when written with deterministic options.

## Architecture

### Package Layout

```
etf_mm_sim/
├── __init__.py
├── config.py                # Configuration dataclasses, YAML load/dump, validation
├── seeding.py               # Master seed → SeedSequence tree
├── mid_price.py             # GBM and regime-switching path generators (vectorized)
├── quoters/
│   ├── __init__.py
│   ├── avellaneda_stoikov.py
│   └── symmetric.py
├── fill_engine.py           # λ(δ) = A·exp(-k·δ); Bernoulli fills with pre-drawn uniforms
├── risk_manager.py          # q_max, kill-switch, terminal flatten
├── path_runner.py           # Per-path step loop (numba-jit + numpy fallback)
├── backtest.py              # Monte Carlo orchestration over (regime, strategy, path)
├── analytics.py             # Terminal P&L, Sharpe, max DD, fill asymmetry, adverse sel.
├── bootstrap.py             # Paired-bootstrap CI
├── counterfactual.py        # No-skew replay for drawdown attribution
├── persistence.py           # Parquet + JSON writers, deterministic
├── viz.py                   # matplotlib plots and summary table renderer
└── cli.py                   # `python -m etf_mm_sim run <config.yaml>`

configs/
└── default.yaml             # Reference YAML with three regimes
notebooks/
└── default_backtest.ipynb   # Loads default.yaml, runs backtest, renders all plots
tests/
├── unit/                    # Example-based unit tests
└── property/                # Hypothesis property tests
```

### Component Diagram

```mermaid
flowchart LR
    YAML[YAML Config File] --> CL[Config_Loader]
    CL --> CFG[Configuration]
    CFG --> SEED[Seeding Tree<br/>SeedSequence]
    CFG --> BE[Backtest_Engine]
    SEED --> BE
    BE -->|per regime, per path| MPS[Mid_Price_Simulator]
    MPS -->|s[0..N]| PR[Path_Runner]
    BE -->|strategy| Q[Quoter<br/>AS or Symmetric]
    Q --> PR
    PR <--> FE[Fill_Engine]
    PR <--> RM[Risk_Manager]
    PR -->|PathResult| PERS[Persistence]
    PERS --> AN[PnL_Analyzer]
    AN --> CF[Counterfactual<br/>No-Skew Replay]
    AN --> BS[Paired_Bootstrap]
    AN --> VZ[Visualization]
    VZ --> NB[Notebook + PNGs]
```

### Dataflow Per Path

For one `(regime r, strategy s, path p)` cell:

1. Derive `SS_path = SS_master.spawn(R)[r].spawn(N_paths)[p]`, then split into
   `(SS_mid, SS_fill)` via `SS_path.spawn(2)`.
2. `Mid_Price_Simulator(SS_mid)` returns `s[0..N]` of length `N+1` where `N = ⌈T/dt⌉`.
3. `Fill_Engine.draw_uniforms(SS_fill, N)` returns two arrays `u_b, u_a ∈ [0,1)^N` used to
   resolve Bernoulli fill events at each step.
4. `Quoter.precompute(s, dt, T, params)` returns time-varying schedules (e.g. for AS, the
   array `delta_star[0..N]` and the AS coefficient `gamma·sigma²·(T-t)` per step).
5. `Path_Runner` runs a length-N loop: at step i it
   - reads `s[i]`, current `q`, current `cash`, current running-max-PnL;
   - calls `Quoter.quote(s[i], q, i)` for `(p_b, p_a)`;
   - applies `Risk_Manager` to suppress sides at inventory bounds and to halt on kill-switch;
   - resolves bid/ask fills using `λ(δ)·dt` and `u_b[i], u_a[i]`;
   - updates `q, cash`, records bid/ask quote arrays, fill arrays, P&L array.
6. After step N, `Risk_Manager.flatten` sets terminal cash adjustment using `s[N]`.
7. The `PathResult` is returned to `Backtest_Engine` for persistence and aggregation.

## Components and Interfaces

### Config_Loader (`config.py`)

```python
def load_config(path: Path) -> Configuration: ...
def dump_config(cfg: Configuration, path: Path) -> None: ...
def validate(cfg: Configuration) -> None:  # raises ConfigError with field name
    ...
```

YAML schema (top-level):

```yaml
master_seed: 20240101
horizon:
  T: 1.0
  dt: 0.001
mid_price:
  model: regime_switching        # or "gbm"
  s0: 100.0
  regimes:
    - name: low
      mu: 0.0
      sigma: 0.5
    - name: normal
      mu: 0.0
      sigma: 1.0
    - name: high
      mu: 0.0
      sigma: 2.0
  transition_matrix:              # required iff regime_switching
    - [0.98, 0.02, 0.0]
    - [0.01, 0.97, 0.02]
    - [0.0, 0.03, 0.97]
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
  q_max: 50
  L_kill: 1000.0    # null disables
mc:
  n_paths: 1000
analytics:
  adverse_selection_horizon_steps: 50
  bootstrap_iterations: 10000
  bootstrap_alpha: 0.05
output:
  dir: results
  format: parquet
```

### Seeding (`seeding.py`)

```python
@dataclass(frozen=True)
class PathSeeds:
    mid: np.random.SeedSequence
    fill: np.random.SeedSequence

def build_seed_tree(master_seed: int, n_regimes: int, n_paths: int) \
    -> list[list[PathSeeds]]:
    root = np.random.SeedSequence(master_seed)
    regime_ss = root.spawn(n_regimes)
    return [
        [PathSeeds(*p.spawn(2)) for p in r.spawn(n_paths)]
        for r in regime_ss
    ]
```

The `PathSeeds` are shared across strategies (AS and Symmetric) at a given `(regime, path)`
to satisfy the paired-comparison requirement (Req 7.2).

### Mid_Price_Simulator (`mid_price.py`)

```python
def simulate_gbm(s0, mu, sigma, T, dt, ss: SeedSequence) -> np.ndarray:
    rng = np.random.default_rng(ss)
    n = int(np.ceil(T / dt))
    dW = rng.standard_normal(n) * np.sqrt(dt)
    log_s = np.log(s0) + np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * dW)
    return np.concatenate(([s0], np.exp(log_s)))   # length n+1

def simulate_regime_switching(
    s0, regimes, P, T, dt, ss: SeedSequence
) -> tuple[np.ndarray, np.ndarray]:    # returns (s_path, regime_indices)
    ...
```

GBM uses exact log-Euler so prices are strictly positive (Req 2.5). When `sigma == 0`, `dW`
contributes zero and the path collapses to the deterministic drift trajectory (Req 2.6).
The regime-switching variant draws a regime trajectory via inverse-CDF on the row of `P`
indexed by the current regime, then applies the per-step `(μ_r, σ_r)`.

### Quoters (`quoters/`)

```python
def as_quote(s, q, t, T, gamma, sigma, k) -> tuple[float, float]:
    horizon = max(T - t, 0.0)
    r = s - q * gamma * sigma**2 * horizon
    delta = 0.5 * gamma * sigma**2 * horizon + (1.0 / gamma) * np.log1p(gamma / k)
    return r - delta, r + delta

def symmetric_quote(s, q, t, delta_base) -> tuple[float, float]:
    return s - delta_base, s + delta_base
```

Both quoters expose a `precompute(s_path, dt, T, params) -> dict[str, np.ndarray]` method
that returns time-varying schedules so the inner loop only does scalar arithmetic. For AS,
`precompute` returns `delta_star[i]` and `coef[i] = gamma·sigma²·(T - t_i)`. For Symmetric,
`precompute` returns nothing (constant `delta_base`).

The AS quoter consumes a per-regime `sigma` to compute the closed-form schedule.
For regime-switching paths the AS quoter is given the *configured* `sigma` of the regime
that owns the path, not the realized regime trajectory; the AS formula is parameterized by
the agent's belief, and we elect to use a single regime-conditional volatility per path
to keep the AS schedule pre-computable. This is called out as an open question below.

### Fill_Engine (`fill_engine.py`)

```python
def draw_fill_uniforms(ss: SeedSequence, n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(ss)
    return rng.random(n), rng.random(n)   # u_b, u_a

def fill_probability(delta: float, A: float, k: float, dt: float) -> float:
    if delta <= 0.0:
        return 1.0
    return -np.expm1(-A * np.exp(-k * delta) * dt)   # = 1 - exp(-λ·dt)
```

`np.expm1` and `log1p` are used to keep the small-`λ·dt` regime numerically clean.
Fill events are Bernoulli per step per side. At each step, we test `u < p` to decide a fill.

### Risk_Manager (`risk_manager.py`)

```python
def apply_inventory_bounds(q, q_max) -> tuple[bool, bool]:
    """Returns (suppress_bid, suppress_ask) flags."""
    return q >= q_max, q <= -q_max

def kill_switch_triggered(running_pnl: float, L_kill: Optional[float]) -> bool:
    return L_kill is not None and running_pnl <= -L_kill

def flatten_terminal(q_T: int, s_T: float, cash_T: float) -> float:
    return cash_T + q_T * s_T   # cash adjustment of q_T·s_T
```

Once kill-switch fires, both sides remain suppressed for the rest of the path.

### Path_Runner (`path_runner.py`)

The single sequential loop. Decorated with `@numba.njit(cache=True)` if numba is importable;
otherwise falls back to a pure-Python loop. Both implementations consume the same
pre-computed arrays and produce identical output.

```python
def run_path(
    s_path: np.ndarray,            # length N+1
    delta_b_sched: np.ndarray,     # length N+1, NaN entries → use suppress
    delta_a_sched: np.ndarray,
    quote_b_sched: np.ndarray,     # bid quote per step, ignoring suppression
    quote_a_sched: np.ndarray,
    u_b: np.ndarray,               # length N
    u_a: np.ndarray,
    A_b, k_b, A_a, k_a,
    dt: float,
    q_max: int,
    L_kill: float,                 # +inf disables
) -> PathResultArrays: ...
```

The runner emits arrays of length `N+1` for `q`, `cash`, `running_pnl`, `bid_posted`,
`ask_posted`, `bid_fill`, `ask_fill`, plus integer fill-count totals.

### Backtest_Engine (`backtest.py`)

```python
def run_backtest(cfg: Configuration) -> BacktestResult:
    seeds = build_seed_tree(cfg.master_seed, len(cfg.regimes), cfg.mc.n_paths)
    results: dict[tuple[str, str], list[PathResult]] = {}
    for r_idx, regime in enumerate(cfg.regimes):
        s_paths = [
            simulate_path(regime, cfg, seeds[r_idx][p].mid)
            for p in range(cfg.mc.n_paths)
        ]
        for strategy in ("avellaneda_stoikov", "symmetric"):
            quoter = build_quoter(strategy, cfg, regime)
            results[(strategy, regime.name)] = [
                run_path_with(quoter, s_paths[p], cfg, seeds[r_idx][p].fill)
                for p in range(cfg.mc.n_paths)
            ]
    persist(results, cfg)
    return BacktestResult(results=results, config=cfg)
```

Note that `s_paths` is computed once per regime and reused across strategies (paired design,
Req 7.2), and `seeds[r][p].fill` is also reused across strategies so that any difference in
realized fills between AS and Symmetric is purely due to differing quote distances.

### PnL_Analyzer (`analytics.py`)

Per cell `(strategy, regime)`:

- `terminal_pnl[p] = cash_T[p] + q_T[p] · s_T[p]` (Req 8.1).
- `pnl_curve[p, i] = cash[p, i] + q[p, i] · s_path[p, i]` for max-DD and Sharpe.
- Sharpe per path: `mean(Δpnl) / std(Δpnl) · sqrt(252 / (T·dt scaling))`. Specifically,
  if one path covers `T` time units, annualization factor is `sqrt(252 / T)` when `T` is
  in trading-days; we expose the convention as an explicit config field (open question).
- Max DD per path: `max over i of (running_max(pnl_curve)[i] − pnl_curve[i])`.
- Fill-rate asymmetry per path: `n_bid_fills − n_ask_fills`.
- Adverse selection per fill at step `i_fill` for filled side `s ∈ {bid, ask}`:
  `signed_drift = (s_path[min(i_fill + Δt_adv, N)] − s_path[i_fill])` with sign flipped on
  bid fills (we are long, so a downward drift hurts).
- Spread capture per fill: `(p_a − s) · 1_{ask fill} + (s − p_b) · 1_{bid fill}` evaluated
  at the fill step.

### Bootstrap (`bootstrap.py`)

```python
def paired_bootstrap_ci(
    diff: np.ndarray, B: int, alpha: float, ss: SeedSequence
) -> tuple[float, float, float]:
    rng = np.random.default_rng(ss)
    n = diff.shape[0]
    idx = rng.integers(0, n, size=(B, n))
    means = diff[idx].mean(axis=1)
    return diff.mean(), np.quantile(means, alpha/2), np.quantile(means, 1 - alpha/2)
```

Diff arrays are paired terminal-P&L differences `AS[p] − Symmetric[p]` and similarly for
max-DD. The bootstrap RNG is seeded from the master seed via a dedicated child sequence
(`SS_master.spawn(...)` reserved slot for analytics) so bootstrap CIs are also reproducible.

### Counterfactual (`counterfactual.py`)

Replays a counterfactual no-skew quoter against the *same fill stream* recorded by the
actual AS run (Req 8.9):

```python
def replay_no_skew(as_path: PathResult, s_path, delta_star_sched) -> PathResult:
    # Counterfactual quotes at s ± delta_star (no inventory skew).
    cf_p_b = s_path - delta_star_sched
    cf_p_a = s_path + delta_star_sched
    # Reuse the actual AS bid_fill / ask_fill events; recompute cash and pnl_curve.
    ...
```

The counterfactual inventory trajectory is identical to the AS one (same fills), but cash
differs because fill prices differ. The drawdown of the counterfactual P&L curve, compared
to the AS drawdown, isolates the cash effect of inventory skew.

The drawdown-attribution metric per cell is

```
skew_dd_attribution = mean_p (max_dd_counterfactual[p] - max_dd_AS[p])
```

with a paired-bootstrap CI computed analogously.

### Persistence (`persistence.py`)

```
results/
└── run_<utc_iso>_<config_sha256[:8]>/
    ├── config.resolved.yaml
    ├── manifest.json              # master_seed, code git sha, library versions
    ├── paths/
    │   ├── avellaneda_stoikov/
    │   │   ├── regime=low.parquet
    │   │   ├── regime=normal.parquet
    │   │   └── regime=high.parquet
    │   └── symmetric/...
    ├── fills/
    │   └── <strategy>/<regime>.parquet
    ├── summary/
    │   ├── per_cell.parquet
    │   └── paired.parquet
    └── plots/*.png
```

Parquet writes pin `compression="zstd"`, `row_group_size=cfg.mc.n_paths`, and stable column
order so byte-equality holds across runs (Req 7.5). Floating-point arrays are stored in
`float64`. The manifest records the resolved config SHA-256 and the master seed.

### Visualization (`viz.py`)

Pure matplotlib (no seaborn). Per-regime terminal-P&L histograms overlay AS and Symmetric;
the sample-path diagnostic uses 4 stacked subplots sharing the time axis (mid+quotes,
inventory, cash, cumulative P&L). Summary tables are rendered with `pandas.DataFrame.to_html`
and embedded in the notebook. Figures are saved to `plots/` at 150 dpi PNG.

## Data Models

```python
@dataclass(frozen=True)
class RegimeParams:
    name: str
    mu: float
    sigma: float

@dataclass(frozen=True)
class MidPriceConfig:
    model: Literal["gbm", "regime_switching"]
    s0: float
    regimes: tuple[RegimeParams, ...]
    transition_matrix: Optional[np.ndarray]   # shape (R, R), rows sum to 1

@dataclass(frozen=True)
class HorizonConfig:
    T: float
    dt: float
    @property
    def n_steps(self) -> int:
        return int(np.ceil(self.T / self.dt))

@dataclass(frozen=True)
class ASParams:
    gamma: float
    k: float
    A: float

@dataclass(frozen=True)
class SymmetricParams:
    delta_base: float

@dataclass(frozen=True)
class FillParams:
    A_b: float
    k_b: float
    A_a: float
    k_a: float

@dataclass(frozen=True)
class RiskParams:
    q_max: int
    L_kill: Optional[float]

@dataclass(frozen=True)
class MCParams:
    n_paths: int

@dataclass(frozen=True)
class AnalyticsParams:
    adverse_selection_horizon_steps: int
    bootstrap_iterations: int
    bootstrap_alpha: float
    sharpe_annualization_factor: float   # e.g. sqrt(252) when T spans 1 trading day

@dataclass(frozen=True)
class OutputConfig:
    dir: Path
    format: Literal["parquet", "csv"]

@dataclass(frozen=True)
class Configuration:
    master_seed: int
    horizon: HorizonConfig
    mid_price: MidPriceConfig
    quoters_as: ASParams
    quoters_sym: SymmetricParams
    fill: FillParams
    risk: RiskParams
    mc: MCParams
    analytics: AnalyticsParams
    output: OutputConfig

@dataclass(frozen=True)
class PathResult:
    regime: str
    strategy: str
    path_index: int
    s_path: np.ndarray            # shape (N+1,)
    bid_quote: np.ndarray         # shape (N+1,), NaN where suppressed
    ask_quote: np.ndarray
    inventory: np.ndarray         # shape (N+1,)
    cash: np.ndarray              # shape (N+1,)
    bid_fill: np.ndarray          # bool, shape (N,)
    ask_fill: np.ndarray
    n_bid_fills: int
    n_ask_fills: int
    terminal_pnl: float           # cash[N] + inventory[N]·s_path[N] after flatten
    kill_switch_step: Optional[int]

@dataclass(frozen=True)
class CellSummary:
    strategy: str
    regime: str
    mean_pnl: float
    std_pnl: float
    p05_pnl: float
    p50_pnl: float
    p95_pnl: float
    mean_sharpe: float
    mean_max_dd: float
    mean_fill_asymmetry: float
    mean_adverse_selection: float
    mean_spread_capture: float

@dataclass(frozen=True)
class PairedComparison:
    regime: str
    diff_mean_pnl: float
    diff_mean_pnl_ci: tuple[float, float]
    diff_max_dd: float
    diff_max_dd_ci: tuple[float, float]
    skew_dd_attribution: float
    skew_dd_attribution_ci: tuple[float, float]

@dataclass(frozen=True)
class BacktestResult:
    config: Configuration
    paths: dict[tuple[str, str], list[PathResult]]   # (strategy, regime) → paths
    cells: list[CellSummary]
    paired: list[PairedComparison]
    output_dir: Path
```

## Vectorization Strategy

A single MC path produces arrays of length `N+1` for prices and state, and `N` for fills.
We split work into a vectorized phase and a sequential phase.

### Vectorized phase (NumPy, length-N arrays)

1. **Mid-price path**: GBM via `cumsum` of pre-drawn normals; regime-switching via a
   pre-drawn regime trajectory plus per-segment GBM. Both are vectorized.
2. **AS schedules**: `delta_star[i] = 0.5·γ·σ²·(T - t_i) + (1/γ)·log(1 + γ/k)`. This is a
   simple expression on `t_i = i·dt`, computed once per path with one NumPy call.
3. **Fill uniforms**: `u_b, u_a = rng.random(N), rng.random(N)`.
4. **Symmetric schedules**: nothing to precompute beyond the constant `delta_base`.

### Sequential phase (per-step loop)

The inventory `q[i]` depends on fills at step `i-1`; AS quotes at step `i` depend on
`q[i]`; fill probabilities at step `i` depend on AS quotes at step `i`. This loop is
genuinely sequential.

The per-step body is:

```
delta_b = s[i] - quote_b(s[i], q, i)        # for AS uses precomputed delta_star[i] + skew
delta_a = quote_a(s[i], q, i) - s[i]
suppress_b, suppress_a = q >= q_max, q <= -q_max
if kill_switch_active: suppress_b = suppress_a = True
p_fill_b = 0.0 if suppress_b else 1 - exp(-A_b · exp(-k_b · delta_b) · dt)
p_fill_a = 0.0 if suppress_a else 1 - exp(-A_a · exp(-k_a · delta_a) · dt)
filled_b = u_b[i] < p_fill_b
filled_a = u_a[i] < p_fill_a
q     += filled_b - filled_a
cash  += filled_a · p_a - filled_b · p_b
pnl_curve[i+1] = cash + q · s[i+1]
update kill_switch_active from pnl_curve
```

### Why a tight loop, not full vectorization

A naive vectorization would require knowing `q[0..N]` upfront, but `q[i]` is the sum of
prior fills, and fill probabilities depend on `q[i]` (via AS skew) and on the inventory
bounds. Closing this dependence in NumPy requires a fixed-point or a manually unrolled
recurrence, both of which sacrifice clarity for marginal gain. The loop body is ~15 scalar
ops; with `numba.njit` it runs in tens of nanoseconds per step.

### Performance budget

- Total inner steps: `1000 paths × 3 regimes × 2 strategies × 1001 steps ≈ 6.0 × 10⁶`.
- With numba (~50 ns/step): ~0.3 s of inner-loop time. RNG draws dominate.
- Pure-Python fallback (~1–2 µs/step): ~12 s of inner-loop time. Still well inside the
  5-minute target on one core.
- RNG: `n_paths × n_steps × 3 regimes × 2 sides ≈ 6 M uniform draws + 3 M normals`. The
  PCG64 generator in NumPy easily exceeds 100 M draws/s.

We choose **numba as an optional accelerator with a pure-NumPy fallback**. Numba is not a
required dependency. The fallback meets the 5-minute target by itself; numba gives headroom
for larger sweeps and for users running in environments without a JIT.

### Memory budget

Per path: 7 arrays of `float64[N+1]` ≈ 56 KB at `N=1000`. Across 6000 paths
(3 regimes × 2 strategies × 1000 paths) ≈ 350 MB. We persist to Parquet and discard the
in-memory arrays after each regime/strategy cell to keep peak RSS bounded.

## Seeding Tree

```
master_seed (int from YAML)
    │
    ▼
SS_master = SeedSequence(master_seed)
    │
    ▼
SS_master.spawn(R + 1)
    │   ├─[0]── SS_regime_0 ─── spawn(N_paths) ─── SS_path_{0,p}
    │   │                                              │
    │   │                                              ▼ spawn(2)
    │   │                                          SS_mid_{0,p}
    │   │                                          SS_fill_{0,p}
    │   ├─[1]── SS_regime_1 ─── ... (analogous)
    │   ├─[R-1]─ SS_regime_{R-1} ─── ...
    │   └─[R]── SS_analytics       (used by paired_bootstrap)
    ▼
```

Properties of the tree:

- **Independence**: `SeedSequence.spawn` produces statistically independent child sequences
  per the NumPy docs; this is the supported way to derive sub-streams.
- **Paired comparison**: `SS_mid_{r,p}` and `SS_fill_{r,p}` are reused across strategies,
  so AS and Symmetric see the same mid-price path and the same uniform draws for fills.
  Realized fills still differ across strategies because quotes differ.
- **Determinism**: The tree depends only on `master_seed`, `R`, and `N_paths`. Adding a
  new regime appends to the tree without disturbing earlier regimes. Adding more paths
  similarly only extends the leaves.
- **Reserved analytics slot**: The `R+1`-th child is reserved for paired-bootstrap and any
  future analytics randomization, keeping their RNG independent of the simulation streams.

## File Output Format and Directory Layout

```
{output.dir}/run_{utc_iso}_{cfg_sha[:8]}/
├── config.resolved.yaml          # canonical YAML round-tripped from Configuration
├── manifest.json                 # master_seed, package version, git sha, library versions
├── paths/{strategy}/regime={name}.parquet
│       columns: path_index, step_index, s, bid, ask, q, cash, pnl_mtm
├── fills/{strategy}/regime={name}.parquet
│       columns: path_index, step_index, side, fill_price
├── summary/per_cell.parquet      # one row per (strategy, regime), columns from CellSummary
├── summary/paired.parquet        # one row per regime, columns from PairedComparison
└── plots/*.png
```

Determinism notes (Req 7.5):

- Parquet write options pinned: `compression="zstd"`, `compression_level=3`, `row_group_size`
  fixed by config, columns ordered alphabetically, `use_dictionary=False`.
- Floats are written as IEEE-754 `float64` directly from the NumPy arrays; no in-flight
  rounding.
- Run timestamp lives in the directory name and `manifest.json`, never in the data files
  themselves, so file bytes are run-time-independent.

## Counterfactual No-Skew Replay (Req 8.9)

Goal: attribute the AS quoter's drawdown reduction (vs. Symmetric) to the *inventory skew*
specifically, separating it from the (also potentially beneficial) AS choice of half-spread.

Procedure for each AS path:

1. Read the AS `bid_fill[i]`, `ask_fill[i]` event arrays and the AS `delta_star[i]` schedule.
2. Construct counterfactual quotes `cf_p_b[i] = s[i] − delta_star[i]`,
   `cf_p_a[i] = s[i] + delta_star[i]`. These are the AS quotes as they would have been with
   `q ≡ 0` (no skew) but the same closed-form spread.
3. Apply the AS fill events to these counterfactual quotes:
   - `cf_q[i+1] = cf_q[i] + bid_fill[i] − ask_fill[i]`. Because the same fill events are
     applied, `cf_q ≡ q_AS` exactly.
   - `cf_cash[i+1] = cf_cash[i] + ask_fill[i]·cf_p_a[i] − bid_fill[i]·cf_p_b[i]`.
4. Compute `cf_pnl_curve[i] = cf_cash[i] + cf_q[i]·s[i]`.
5. Flatten at terminal: `cf_terminal_pnl = cf_cash[N] + cf_q[N]·s[N]`.

Per-cell drawdown attribution:

```
skew_dd_attribution = mean_p(cf_max_dd[p] − as_max_dd[p])
```

A positive value means the AS quoter's skew reduces drawdown vs. a no-skew quoter that
quotes the same spread. The paired-bootstrap CI is computed on the per-path differences
`cf_max_dd[p] − as_max_dd[p]`.

Caveat: this measures the cash effect of skew, not its effect on fill probabilities, since
fills are held fixed. This is the right notion for "attributing realized drawdown to skew",
because the realized fill stream is what actually happened.

## Paired-Bootstrap CI Methodology

For each regime, given paired arrays `as_pnl[p]` and `sym_pnl[p]` (and analogously for
max-DD), compute per-path differences `d[p] = as_pnl[p] − sym_pnl[p]`. Bootstrap:

1. Draw `B = cfg.analytics.bootstrap_iterations` samples of size `n_paths` with replacement
   from `{0, ..., n_paths-1}`. The sampler RNG is seeded from the analytics-reserved
   `SeedSequence` slot.
2. For each sample `b`, compute `mean_b = mean(d[idx[b]])`.
3. Report `(mean(d), quantile(mean_b, α/2), quantile(mean_b, 1 − α/2))` as `(point, lo, hi)`.

This is a *paired percentile bootstrap*. We use it because:

- Pairing across `(AS, Symmetric)` removes between-path variance driven by mid-price seed
  variation, which would otherwise dominate the CI width.
- The percentile method is robust to distributional asymmetries (heavy tails are common in
  trading P&L) without requiring a normality assumption.

Reproducibility: the bootstrap RNG is a child of the master seed, so CIs are
deterministic for a fixed config.



## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions
of a system — essentially, a formal statement about what the system should do. Properties
serve as the bridge between human-readable specifications and machine-verifiable
correctness guarantees.*

PBT applies to this feature: the simulator is dominated by pure mathematical functions
(price generators, AS formulas, fill probabilities, P&L analytics) with universally
quantifiable behavior across large input spaces. Visualization, file I/O smoke checks,
and the notebook deliverable are out of the property-test scope and are covered by
example/integration/smoke tests in the Testing Strategy section.

### Property 1: Configuration round-trip

*For any* valid Configuration object, serializing to YAML and parsing back SHALL produce
a Configuration equivalent to the original.

**Validates: Requirements 1.5**

### Property 2: Mid-price simulator determinism

*For any* master seed, regime index, and path index, two independent invocations of the
mid-price simulator SHALL produce bit-identical price arrays.

**Validates: Requirements 2.3**

### Property 3: Mid-price path length

*For any* configured horizon `T > 0` and step size `0 < dt ≤ T`, the generated mid-price
path SHALL have length `⌈T/dt⌉ + 1`.

**Validates: Requirements 2.4**

### Property 4: GBM strictly positive prices

*For any* GBM parameters with `s_0 > 0`, `σ ≥ 0`, finite `μ`, and any seed, every element
of the generated path SHALL be strictly positive.

**Validates: Requirements 2.5**

### Property 5: GBM zero-volatility deterministic drift

*For any* GBM parameters with `σ = 0`, the generated path SHALL equal the deterministic
trajectory `s_0 · exp(μ · t_i)` (within float tolerance) regardless of seed.

**Validates: Requirements 2.6**

### Property 6: AS quote construction matches closed form

*For any* valid `(s, q, t < T, γ > 0, σ > 0, k > 0)`, the AS quoter SHALL emit
`(p_b, p_a)` satisfying `(p_a + p_b)/2 = s − q·γ·σ²·(T − t)` and
`(p_a − p_b)/2 = γ·σ²·(T − t)/2 + (1/γ)·ln(1 + γ/k)` within float tolerance.

**Validates: Requirements 3.1, 3.2, 3.3**

### Property 7: AS skew sign convention

*For any* valid AS inputs with `σ > 0` and `t < T`, the sign of
`(p_a + p_b)/2 − s` SHALL be the opposite of the sign of `q`, and the midpoint SHALL
equal `s` exactly when `q = 0`.

**Validates: Requirements 3.4, 3.5, 3.6**

### Property 8: AS no-quote at terminal

*For any* AS inputs with `t ≥ T`, the AS quoter SHALL emit no quote (signalled via
explicit suppression on both sides).

**Validates: Requirements 3.7**

### Property 9: Symmetric quoter is mid-symmetric and inventory-independent

*For any* `(s, δ_base ≥ 0)` and any pair of inventory values `q_1, q_2`, the symmetric
quoter SHALL emit `(s − δ_base, s + δ_base)` independent of inventory; in particular
`(p_a + p_b)/2 = s` and the quotes at `q_1` and `q_2` SHALL be equal.

**Validates: Requirements 4.1, 4.2, 4.3**

### Property 10: Fill intensity closed form

*For any* `(δ, A > 0, k > 0)`, the fill engine SHALL compute per-side intensity
`λ = A · exp(−k · δ)` within float tolerance.

**Validates: Requirements 5.1**

### Property 11: Bernoulli fill probability and crossed-quote fill

*For any* `(λ ≥ 0, dt > 0)`, the fill engine SHALL compute fill probability
`1 − exp(−λ · dt)`, and *for any* `δ ≤ 0` the fill probability SHALL equal `1`.

**Validates: Requirements 5.2, 5.6**

### Property 12: Fill bookkeeping

*For any* simulated path, the inventory at every step SHALL equal the cumulative
`(bid_fill − ask_fill)` count up to that step, and recorded bid (respectively ask) fill
prices SHALL equal the posted bid (respectively ask) quote at the step at which the fill
occurred.

**Validates: Requirements 5.4, 5.5**

### Property 13: Fill engine determinism

*For any* fill-engine seed and identical `(quote schedules, intensity parameters, dt)`
inputs, two invocations SHALL produce identical bid and ask fill arrays.

**Validates: Requirements 5.7**

### Property 14: Inventory bound invariant

*For any* simulated path under risk parameter `q_max`, every step SHALL satisfy
`|q[i]| ≤ q_max`.

**Validates: Requirements 6.1, 6.2, 6.3**

### Property 15: Kill-switch halts trading

*For any* simulated path on which running mark-to-market P&L crosses below `−L_kill` at
step `i*`, no bid or ask fills SHALL occur at any step `i > i*`.

**Validates: Requirements 6.4**

### Property 16: Terminal-flatten P&L identity

*For any* simulated path, the reported terminal P&L SHALL equal
`cash[N] + q[N] · s[N]` (the cash including the terminal-flatten adjustment), within float
tolerance.

**Validates: Requirements 6.5, 8.1**

### Property 17: Paired mid-price seeding across strategies

*For any* `(regime, path)` index pair, the mid-price array seen by AS_Quoter and
Symmetric_Quoter SHALL be byte-identical.

**Validates: Requirements 7.2**

### Property 18: Bit-identical persisted outputs

*For any* valid Configuration with fixed master seed, two independent invocations of
the backtest engine SHALL produce byte-identical Parquet and JSON output files (excluding
the run-timestamp directory name).

**Validates: Requirements 7.5, 10.2**

### Property 19: Max drawdown non-negativity

*For any* mark-to-market P&L curve, the computed maximum drawdown SHALL be non-negative,
and SHALL equal zero if and only if the curve is monotonically non-decreasing.

**Validates: Requirements 8.4**

### Property 20: Spread capture per fill

*For any* fill at step `i`, the reported spread capture SHALL equal `p_a[i] − s[i]` for
ask fills and `s[i] − p_b[i]` for bid fills.

**Validates: Requirements 8.7**

### Property 21: Paired-bootstrap CI determinism and validity

*For any* paired difference array and fixed bootstrap RNG seed, two invocations of the
paired-bootstrap CI computation SHALL produce identical `(point, lo, hi)` tuples; and
the reported point estimate SHALL equal the sample mean.

**Validates: Requirements 8.8**

### Property 22: Counterfactual inventory equals AS inventory

*For any* AS path, the counterfactual no-skew replay SHALL produce an inventory
trajectory equal to the AS inventory trajectory at every step (since fills are held
fixed), and SHALL produce the same number of bid and ask fills.

**Validates: Requirements 8.9**

### Property 23: SeedSequence tree determinism

*For any* master integer seed, regime count, and path count, two independent
constructions of the seeding tree SHALL produce `SeedSequence` instances with identical
`entropy` and `spawn_key` at every leaf.

**Validates: Requirements 10.3**

## Error Handling

### Configuration errors

`ConfigError` is raised by `Config_Loader.validate` with a single canonical message format:
`"<field_path>: <constraint violated>; got <value>"`. The field path uses dot notation
matching the YAML structure (e.g. `mid_price.regimes[1].sigma`). All numeric constraints
are checked before any simulation work begins. Unknown YAML keys are rejected to prevent
silent typos from masking misconfiguration.

### Numerical guards

- `np.expm1` and `np.log1p` are used wherever subtraction or logarithm of values close to
  zero or one would otherwise lose precision (fill probabilities at small `λ·dt`, AS
  spread term `ln(1 + γ/k)`).
- AS spread for `σ = 0` and `t < T` collapses to `(1/γ)·ln(1 + γ/k)`, which is positive,
  so quotes never invert. This is asserted as a sanity check at simulator startup.
- For `t = T` the AS schedule is forced to `δ* = (1/γ)·ln(1 + γ/k)` to avoid the singular
  inventory term `q·γ·σ²·0`; quotes at this terminal step are suppressed regardless
  per Property 8.

### Risk-manager errors

If at any step a quote would invert (`p_a < p_b`) due to misconfiguration, `Path_Runner`
raises `QuoterError` with the offending step and inputs. This indicates a programmer error
rather than a runtime condition, and we prefer fail-fast over silently posting crossed
quotes.

### Persistence errors

`PersistenceError` is raised if the output directory cannot be created or written. Partial
writes are avoided by writing each Parquet file to a temporary path and atomic-renaming on
success.

### Notebook execution

The notebook is run end-to-end by CI via `nbclient`. Cell errors propagate as test
failures so that documentation drift is caught.

## Testing Strategy

### Test layout

- `tests/unit/` — example-based unit tests, fast (<5 s total).
- `tests/property/` — Hypothesis property tests, configured for ≥100 examples per property.
- `tests/integration/` — runs a small backtest end-to-end and validates persisted artifacts.
- `tests/smoke/` — validates that visualizations render and the notebook executes.
- `tests/perf/` — single benchmark verifying the 5-minute target on a reference machine,
  marked `@pytest.mark.slow` and excluded from default CI runs.

### Property-based testing

Library: **Hypothesis** (`hypothesis>=6`). It is the de-facto standard PBT framework for
Python and integrates cleanly with `pytest`. Configuration:

- Each property test runs at least `max_examples=200` for non-trivial generators and
  `max_examples=100` for the simulator-level properties (Properties 18, 22, 23) which
  involve a full backtest invocation.
- Each property test is tagged with a comment of the form
  `# Feature: etf-mm-arbitrage-simulator, Property <N>: <property text>` immediately
  above the test function.
- Custom Hypothesis strategies live in `tests/property/strategies.py` and produce valid
  Configuration values, valid AS parameter tuples, valid GBM parameters, and small
  but realistic price arrays.
- Floating-point tolerances use `math.isclose(rel_tol=1e-12, abs_tol=1e-12)` for closed-form
  identities and looser `rel_tol=1e-9` for paths involving accumulated arithmetic.

The 23 properties enumerated above each map to exactly one Hypothesis property test file
in `tests/property/`. Properties 18 and 23 (full-backtest reproducibility and seed-tree
determinism) use a reduced backtest configuration (`n_paths=10`, `T=0.01`) so that the
property suite runs in under one minute on a laptop.

### Example-based unit tests

Mapped from the EXAMPLE-classified criteria above:

- Configuration loading happy path and per-field missing-field errors (Req 1.1, 1.3).
- Out-of-range field rejection across each constrained field (Req 1.4).
- Backtest cell counts (Req 7.1).
- Sharpe, fill-rate-asymmetry, adverse-selection, summary-stats analytics on hand-rolled
  inputs (Req 8.2, 8.3, 8.5, 8.6).
- Persistence file existence and schema (Req 7.4).
- Master-seed loading from YAML (Req 10.1) and manifest contents (Req 10.4).

### Smoke tests

- GBM and regime-switching simulator return expected shapes (Req 2.1, 2.2).
- Visualization functions render non-empty figures (Req 9.1, 9.2, 9.3).
- The notebook executes end-to-end via `nbclient` (Req 9.4).
- Out-of-scope items: there are no active tests; absence of WebSocket, C++, NAV-arb code
  is enforced by repository-structure checks in CI.

### Performance test

A single benchmark in `tests/perf/test_target.py` runs the production-shape backtest
(`n_paths=1000`, three regimes, two strategies, `dt=0.001`, `T=1`) and asserts that wall
time is under 300 seconds on the reference laptop CPU. This test is opt-in via
`pytest -m slow` and runs nightly.

### Coverage goals

- 100% line coverage on `config.py`, `mid_price.py`, `quoters/`, `fill_engine.py`,
  `risk_manager.py`, `analytics.py`, `bootstrap.py`, `counterfactual.py`.
- ≥90% on `path_runner.py` (the numba-accelerated branch and the pure-Python fallback
  are both exercised in CI).

## Open Design Questions

The following decisions have been made provisionally; flagging them so the user can
override before implementation begins.

1. **AS volatility for regime-switching paths.** The closed-form AS schedule
   `δ*(t) = γ·σ²·(T − t)/2 + (1/γ)·ln(1 + γ/k)` requires a single `σ` per path. For
   regime-switching paths the realized regime trajectory varies within the path. Three
   options:
   - **(a)** Use the regime's *initial* σ for the AS schedule (current proposal).
   - **(b)** Use the long-run stationary σ implied by the transition matrix.
   - **(c)** Recompute `δ*` step-by-step using the realized regime's σ (breaks
     pre-computation but is closer to a "regime-aware" agent).
   The choice affects how aggressively AS quotes shrink in low-σ regimes within a
   high-σ path. *Default: (a).*

2. **Sharpe annualization convention.** The acceptance criterion specifies a 252-day
   trading-year convention but `T` and `dt` are dimensionless in the spec. We expose
   `analytics.sharpe_annualization_factor` as an explicit float in the YAML so the user
   chooses the convention. *Default: `sqrt(252)` assuming `T = 1` represents one trading
   day.* Should this instead be `sqrt(252 / T)` so the factor adjusts automatically when
   `T` changes?

3. **Fill semantics at crossed quotes (Req 5.6).** We treat `δ ≤ 0` as
   `p_fill = 1` for the side, regardless of how negative `δ` is. An alternative is to
   reject the configuration at validation time if the AS or symmetric quoter could ever
   produce crossed quotes given the parameters. *Default: runtime guaranteed-fill, since
   crossed quotes are a legitimate (if uncommon) AS outcome under extreme inventory.*

4. **Numba dependency.** Optional dependency (graceful fallback to pure Python) vs.
   required dependency. *Default: optional.* The fallback meets the 5-minute target but
   does not have headroom for substantially larger sweeps.

5. **Persistence format toggle.** YAML supports `output.format: parquet | csv`. CSV is
   useful for human inspection but is roughly 10× larger and slower to write
   deterministically (line-ending normalization required). *Default: Parquet only;
   CSV deferred unless explicitly requested.*

6. **Adverse selection sign convention.** We define adverse selection as the *signed*
   mid-price drift over `Δt_adv` against the filled side: positive means the fill was
   adverse (mid moved against us). The opposite sign is also common in the literature.
   *Default: positive = adverse.* Should the opposite convention be used?

7. **Bootstrap method.** We use the paired *percentile* bootstrap. Alternatives include
   BCa (bias-corrected and accelerated) for better small-sample coverage. *Default:
   percentile.* For `n_paths = 1000` the difference is typically small, but BCa is more
   defensible for small `n_paths` configs.

8. **Strategy parity beyond mid-price seeds.** Currently both strategies share the same
   `SS_fill` so that *uniform draws* used to resolve fills are paired across strategies,
   but realized fill events differ because quote distances differ. An alternative would
   be to pair *fill events* directly (force AS and Symmetric to fill on the same steps),
   which is closer to a strict A/B test but inconsistent with intensity-driven fills.
   *Default: paired uniform draws, not paired events.*

9. **Notebook execution in CI.** Running the full default notebook in CI adds ~30 s
   per run. *Default: enabled* on PR builds; can be moved to a nightly job if it slows
   feedback.

10. **Counterfactual inventory bounds.** The counterfactual no-skew replay reuses the
    AS fill stream, which was generated under AS quoting and AS-imposed inventory bounds.
    The counterfactual inventory trajectory therefore inherits the AS bounds even though
    a real no-skew quoter would not have skewed quotes to stay within them. We document
    this as a measurement of "cash effect of skew given the same fills"; this is the
    correct decomposition for drawdown attribution but should not be interpreted as a
    full counterfactual P&L.

