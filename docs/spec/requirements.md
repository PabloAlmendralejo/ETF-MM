# Requirements Document

## Introduction

This feature delivers Mode 1 of the ETF Market Making & Arbitrage Simulator: a Python-based
Avellaneda-Stoikov (AS) market-making simulator and Monte Carlo backtester that operates on
synthetic mid-price paths. The deliverable benchmarks the AS quoter against a symmetric
constant-spread baseline across multiple simulated volatility regimes and produces P&L
analytics that decompose performance into fill-rate asymmetry, adverse selection, and
spread capture, with explicit attribution of drawdown reduction to inventory skew.

The C++ lock-free order book, Binance WebSocket tick ingestion, ETF/synthetic-basket NAV
deviation tracking, and arbitrage signal generation are deferred to a later iteration and
are out of scope for this spec.

## Glossary

- **Simulator**: The top-level Python application that runs Monte Carlo backtests of quoting strategies against synthetic mid-price paths.
- **Mid_Price_Simulator**: Component that generates synthetic mid-price paths under a configurable stochastic model (GBM or regime-switching).
- **AS_Quoter**: Quoting strategy that computes reservation price `r = s - q·γ·σ²·(T-t)` and optimal half-spread `δ* = (γ·σ²·(T-t))/2 + (1/γ)·ln(1 + γ/k)` per Avellaneda-Stoikov (2008).
- **Symmetric_Quoter**: Baseline quoting strategy that posts a configurable constant half-spread symmetrically around the mid-price with no inventory skew.
- **Fill_Engine**: Component that simulates order fills using a Poisson process with intensity `λ(δ) = A·exp(-k·δ)` per side.
- **Risk_Manager**: Component that enforces inventory bounds and kill-switch behavior on the active quoter.
- **Backtest_Engine**: Component that orchestrates Monte Carlo runs across paths, regimes, and strategies.
- **PnL_Analyzer**: Component that computes per-path and aggregate performance metrics from simulation output.
- **Visualization_Module**: Component that renders matplotlib plots and the comparison notebook.
- **Config_Loader**: Component that reads, validates, and parses the YAML simulation configuration file.
- **Volatility_Regime**: A named set of stochastic-process parameters (e.g., low, normal, high) used to characterize market conditions in the backtest sweep.
- **Reservation_Price**: The inventory-adjusted indifference price `r(s, q, t)` produced by AS_Quoter.
- **Inventory_Skew**: The shift of quotes away from the mid-price driven by current inventory `q`.
- **Spread_Capture**: Realized profit per fill measured as `(fill_price - mid_price)` for asks or `(mid_price - fill_price)` for bids.
- **Adverse_Selection**: Post-fill mid-price drift against the quoter's filled side, measured over a configurable horizon Δt_adv.
- **Fill_Rate_Asymmetry**: The signed difference between bid-side fill count and ask-side fill count over a path.
- **MC_Path**: A single Monte Carlo simulation path consisting of one mid-price trajectory and the resulting quoter behavior.
- **YAML_Config**: A user-supplied YAML file specifying all simulation parameters.

## Requirements

### Requirement 1: Configuration Loading

**User Story:** As a quant researcher, I want to configure the entire simulation via a YAML file, so that I can reproduce and version-control experiments without changing code.

#### Acceptance Criteria

1. THE Config_Loader SHALL accept a path to a YAML file as its sole input.
2. THE Config_Loader SHALL parse and validate parameters covering mid-price model selection, model parameters per regime, AS quoter parameters (γ, k, A, T, dt), baseline quoter parameters, fill model parameters, risk limits, Monte Carlo settings, and RNG seed.
3. IF a required configuration field is missing, THEN THE Config_Loader SHALL raise a descriptive error naming the missing field.
4. IF a configuration value is outside its valid range (e.g., negative volatility, non-positive `k`, negative inventory bound), THEN THE Config_Loader SHALL raise a descriptive error naming the field and the violated constraint.
5. THE Config_Loader SHALL emit Configuration objects whose round-trip through serialization and parsing produces an equivalent Configuration object (round-trip property).

### Requirement 2: Mid-Price Simulation

**User Story:** As a quant researcher, I want to simulate mid-price paths under multiple stochastic models, so that I can stress-test quoting strategies across volatility regimes.

#### Acceptance Criteria

1. THE Mid_Price_Simulator SHALL support a Geometric Brownian Motion (GBM) model parameterized by initial price `s_0`, drift `μ`, and volatility `σ`.
2. THE Mid_Price_Simulator SHALL support a regime-switching model parameterized by a set of named regimes, each with its own `(μ, σ)`, and a Markov transition matrix between regimes.
3. WHEN given an MC_Path index and a master seed, THE Mid_Price_Simulator SHALL produce a deterministic, reproducible price path.
4. THE Mid_Price_Simulator SHALL produce price paths of length `N = ⌈T/dt⌉ + 1` where `T` is the horizon and `dt` is the time step from configuration.
5. WHERE the GBM model is selected, THE Mid_Price_Simulator SHALL ensure all generated prices are strictly positive.
6. WHEN the configured volatility for a regime is `0`, THE Mid_Price_Simulator SHALL produce a price path equal to the deterministic drift trajectory.

### Requirement 3: Avellaneda-Stoikov Quoter

**User Story:** As a quant researcher, I want an Avellaneda-Stoikov quoter with closed-form reservation price and optimal spread, so that I can study inventory-aware market making.

#### Acceptance Criteria

1. WHEN given the current mid-price `s`, inventory `q`, time `t`, horizon `T`, risk-aversion `γ`, volatility `σ`, and order-arrival decay `k`, THE AS_Quoter SHALL compute the Reservation_Price as `r = s - q·γ·σ²·(T - t)`.
2. WHEN given the same inputs as in criterion 1, THE AS_Quoter SHALL compute the optimal half-spread as `δ* = (γ·σ²·(T - t))/2 + (1/γ)·ln(1 + γ/k)`.
3. THE AS_Quoter SHALL emit a bid quote `p_b = r - δ*` and an ask quote `p_a = r + δ*` at each time step.
4. WHEN inventory `q` is positive, THE AS_Quoter SHALL produce quotes whose midpoint `(p_a + p_b)/2` is below the mid-price `s`.
5. WHEN inventory `q` is negative, THE AS_Quoter SHALL produce quotes whose midpoint `(p_a + p_b)/2` is above the mid-price `s`.
6. WHEN inventory `q` equals `0`, THE AS_Quoter SHALL produce quotes symmetric around `s` such that `(p_a + p_b)/2 = s`.
7. IF `t ≥ T`, THEN THE AS_Quoter SHALL emit no quotes for that time step.

### Requirement 4: Symmetric Baseline Quoter

**User Story:** As a quant researcher, I want a symmetric constant-spread baseline quoter, so that I can attribute AS performance gains to inventory skewing.

#### Acceptance Criteria

1. WHEN given the current mid-price `s` and configured half-spread `δ_base`, THE Symmetric_Quoter SHALL emit a bid quote `s - δ_base` and an ask quote `s + δ_base`.
2. THE Symmetric_Quoter SHALL ignore current inventory when setting quotes.
3. THE Symmetric_Quoter SHALL produce quotes whose midpoint equals the mid-price `s` at every time step.

### Requirement 5: Fill Engine

**User Story:** As a quant researcher, I want fills to be simulated by a Poisson process whose intensity decays with quote distance from mid, so that filled volume reflects realistic execution dynamics.

#### Acceptance Criteria

1. WHEN given a bid distance `δ_b = s - p_b` and ask distance `δ_a = p_a - s`, THE Fill_Engine SHALL compute per-side fill intensities `λ_b = A_b·exp(-k_b·δ_b)` and `λ_a = A_a·exp(-k_a·δ_a)`.
2. WHEN per-side intensities and time step `dt` are given, THE Fill_Engine SHALL sample a Bernoulli fill event per side with probability `1 - exp(-λ·dt)`.
3. WHERE the configuration specifies a single shared `(A, k)` pair, THE Fill_Engine SHALL apply the same intensity parameters to both bid and ask sides.
4. WHEN a bid fill occurs at time `t`, THE Fill_Engine SHALL increment inventory by one unit and record the fill price as the posted bid quote at time `t`.
5. WHEN an ask fill occurs at time `t`, THE Fill_Engine SHALL decrement inventory by one unit and record the fill price as the posted ask quote at time `t`.
6. IF `δ_b ≤ 0` or `δ_a ≤ 0` for the corresponding side, THEN THE Fill_Engine SHALL treat that side as guaranteed-fill within the time step (probability `1`).
7. THE Fill_Engine SHALL produce reproducible fill sequences when given a deterministic seed and identical inputs.

### Requirement 6: Risk Management and Inventory Limits

**User Story:** As a quant researcher, I want hard inventory limits and a kill-switch, so that simulated drawdowns reflect realistic risk controls.

#### Acceptance Criteria

1. THE Risk_Manager SHALL enforce a configurable maximum absolute inventory `q_max`.
2. WHEN inventory equals `+q_max`, THE Risk_Manager SHALL suppress further bid quotes for the active quoter until inventory falls below `+q_max`.
3. WHEN inventory equals `-q_max`, THE Risk_Manager SHALL suppress further ask quotes for the active quoter until inventory rises above `-q_max`.
4. WHERE a kill-switch P&L threshold `L_kill` is configured, IF the running mark-to-market P&L on a path falls below `-L_kill`, THEN THE Risk_Manager SHALL halt all further quoting on that path for the remainder of the horizon.
5. WHEN the simulation horizon `T` is reached, THE Risk_Manager SHALL flatten any remaining inventory at the terminal mid-price and record the resulting cash adjustment.

### Requirement 7: Monte Carlo Backtest Orchestration

**User Story:** As a quant researcher, I want to sweep multiple volatility regimes and quoting strategies across many Monte Carlo paths, so that I can compare strategies statistically.

#### Acceptance Criteria

1. THE Backtest_Engine SHALL execute a configurable number of MC_Paths `N_paths` per `(strategy, Volatility_Regime)` cell.
2. THE Backtest_Engine SHALL run both AS_Quoter and Symmetric_Quoter against the same set of mid-price paths per regime, using identical seeds for the mid-price process across strategies (paired comparison).
3. THE Backtest_Engine SHALL run all configured Volatility_Regimes within a single backtest invocation.
4. THE Backtest_Engine SHALL persist per-path time series (mid-price, bid quote, ask quote, inventory, cash, fills per side) and per-path summary metrics to disk in a structured format (Parquet or CSV).
5. WHEN the same YAML_Config and master seed are supplied, THE Backtest_Engine SHALL produce bit-identical persisted outputs across runs.

### Requirement 8: P&L and Performance Analytics

**User Story:** As a quant researcher, I want detailed P&L analytics, so that I can attribute strategy performance to fill-rate asymmetry, adverse selection, spread capture, and inventory skew.

#### Acceptance Criteria

1. THE PnL_Analyzer SHALL compute terminal P&L per path as `cash_T + q_T·s_T` after Risk_Manager flattening.
2. THE PnL_Analyzer SHALL compute the empirical distribution of terminal P&L per `(strategy, Volatility_Regime)` cell, reporting mean, standard deviation, and the 5th, 50th, and 95th percentiles.
3. THE PnL_Analyzer SHALL compute the per-path Sharpe ratio of the mark-to-market P&L increments, annualized using the configured `dt` and a 252-day trading-year convention.
4. THE PnL_Analyzer SHALL compute per-path maximum drawdown of the running mark-to-market P&L curve.
5. THE PnL_Analyzer SHALL compute Fill_Rate_Asymmetry per path as `(N_bid_fills - N_ask_fills)` and report its distribution per cell.
6. WHEN given a configurable adverse-selection horizon `Δt_adv`, THE PnL_Analyzer SHALL compute, per fill, the signed mid-price move over `Δt_adv` against the filled side, and report mean Adverse_Selection per cell.
7. THE PnL_Analyzer SHALL compute Spread_Capture per fill and report the per-cell mean and distribution.
8. THE PnL_Analyzer SHALL produce a paired comparison report between AS_Quoter and Symmetric_Quoter per Volatility_Regime, including the difference in mean terminal P&L, the difference in max drawdown, and a paired-bootstrap confidence interval on each difference.
9. THE PnL_Analyzer SHALL decompose AS_Quoter drawdown reduction relative to baseline into a component attributable to Inventory_Skew, computed by replaying the same fill stream against a counterfactual no-skew quoter and comparing realized drawdowns.

### Requirement 9: Visualization and Notebook Deliverable

**User Story:** As a quant researcher, I want plots and a notebook walkthrough, so that I can present and interpret results.

#### Acceptance Criteria

1. THE Visualization_Module SHALL render, for each Volatility_Regime, a histogram of terminal P&L overlaying AS_Quoter and Symmetric_Quoter distributions.
2. THE Visualization_Module SHALL render a sample-path diagnostic plot containing mid-price, bid and ask quotes, inventory, and cumulative P&L on aligned time axes.
3. THE Visualization_Module SHALL render a summary table of per-cell metrics including mean P&L, P&L standard deviation, Sharpe ratio, max drawdown, fill-rate asymmetry, mean adverse selection, and mean spread capture.
4. THE repository SHALL include a Jupyter notebook that loads a default YAML_Config, runs a small backtest, and reproduces all required plots and the summary table.

### Requirement 10: Reproducibility

**User Story:** As a quant researcher, I want fully reproducible runs, so that results can be audited and shared.

#### Acceptance Criteria

1. THE Simulator SHALL accept a single master integer seed via the YAML_Config.
2. WHEN the master seed and YAML_Config are fixed, THE Simulator SHALL produce identical mid-price paths, fill sequences, persisted outputs, and summary metrics across runs on the same machine and Python environment.
3. THE Simulator SHALL derive independent per-path and per-component sub-seeds from the master seed using a documented deterministic scheme (e.g., `numpy.random.SeedSequence.spawn`).
4. THE Simulator SHALL log the resolved configuration and master seed alongside persisted outputs.

### Requirement 11: Performance Target

**User Story:** As a quant researcher, I want backtests to complete in a reasonable time on a laptop, so that I can iterate on parameters quickly.

#### Acceptance Criteria

1. WHEN configured with `N_paths = 1000`, horizon `T = 1` (one trading day in arbitrary time units), `dt = 0.001`, and three Volatility_Regimes, THE Backtest_Engine SHALL complete a full sweep of both AS_Quoter and Symmetric_Quoter in under five minutes on a single modern laptop CPU core.
2. THE Mid_Price_Simulator and Fill_Engine SHALL be implemented using vectorized NumPy operations across time steps within a path.

### Requirement 12: Out-of-Scope Boundaries

**User Story:** As a stakeholder reviewing this spec, I want explicit boundaries, so that scope creep into deferred work is prevented.

#### Acceptance Criteria

1. THE Simulator SHALL NOT consume live market data feeds in this iteration.
2. THE Simulator SHALL NOT include a C++ order book component in this iteration.
3. THE Simulator SHALL NOT compute ETF-versus-synthetic-basket NAV deviations in this iteration.
4. THE Simulator SHALL NOT generate cross-asset arbitrage signals in this iteration.
5. THE Simulator SHALL NOT perform historical tick replay in this iteration.
