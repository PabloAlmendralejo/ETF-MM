# Implementation Plan: ETF MM Arbitrage Simulator (Mode 1)

## Overview

Convert the feature design into a series of prompts for a code-generation LLM that will
implement each step with incremental progress. Make sure that each prompt builds on the
previous prompts, and ends with wiring things together. There should be no hanging or
orphaned code that isn't integrated into a previous step. Focus ONLY on tasks that
involve writing, modifying, or testing code.

The implementation language is Python 3.11+. Tasks are ordered for incremental,
test-driven development: where a task has both implementation and property/unit test
sub-tasks, the test sub-task should be authored first (red), then the implementation
sub-task drives it green.

Sub-tasks marked with `*` are optional (test-related). Top-level tasks must always be
implemented. The 23 correctness properties (P1–P23) from `design.md` are encoded as
Hypothesis property tests, one per property, placed close to the implementation they
validate.

## Tasks

- [x] 1. Project scaffolding and dev dependencies
  - Create `pyproject.toml` declaring package `etf_mm_sim`, Python `>=3.11`, runtime deps
    `numpy`, `pandas`, `pyarrow`, `pyyaml`, `matplotlib`, optional dep `numba`, dev deps
    `pytest`, `pytest-cov`, `hypothesis>=6`, `nbclient`, `jupyter`.
  - Create empty package skeleton: `etf_mm_sim/__init__.py`, `etf_mm_sim/quoters/__init__.py`,
    and stub modules `config.py`, `seeding.py`, `mid_price.py`, `fill_engine.py`,
    `risk_manager.py`, `path_runner.py`, `backtest.py`, `analytics.py`, `bootstrap.py`,
    `counterfactual.py`, `persistence.py`, `viz.py`, `cli.py`, plus
    `quoters/avellaneda_stoikov.py` and `quoters/symmetric.py`.
  - Create empty test tree: `tests/{unit,property,integration,smoke,perf}/` with `__init__.py`
    in each, plus `tests/property/strategies.py` placeholder.
  - Create `configs/` and `notebooks/` directories with `.gitkeep` placeholders.
  - Add `pytest.ini` configuring markers (`slow`, `perf`) and discovery roots.
  - _Requirements: 11.2, 12.1–12.5_

- [x] 2. Configuration loading, validation, and dataclasses
  - [x] 2.1 Implement Configuration dataclasses
    - In `etf_mm_sim/config.py` define frozen dataclasses `RegimeParams`, `MidPriceConfig`,
      `HorizonConfig`, `ASParams`, `SymmetricParams`, `FillParams`, `RiskParams`,
      `MCParams`, `AnalyticsParams`, `OutputConfig`, `Configuration` per design §Data Models.
    - Implement `HorizonConfig.n_steps` as `ceil(T/dt)`.
    - _Requirements: 1.2_

  - [x] 2.2 Implement YAML load, dump, and validate
    - Implement `load_config(path) -> Configuration`, `dump_config(cfg, path)`, and
      `validate(cfg)` using PyYAML. Reject unknown top-level and nested keys.
    - Raise `ConfigError` with format `"<field_path>: <constraint>; got <value>"` on
      missing required fields and out-of-range values (negative `sigma`, non-positive `k`,
      negative `q_max`, transition-matrix rows that do not sum to 1, etc.).
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [x]* 2.3 Write property test for configuration round-trip
    - **Property 1: Configuration round-trip**
    - **Validates: Requirements 1.5**
    - In `tests/property/test_config_roundtrip.py` use a Hypothesis strategy that
      generates valid `Configuration` objects and assert
      `load_config(dump_config(cfg)) == cfg`.

  - [x]* 2.4 Write unit tests for missing-field and out-of-range errors
    - In `tests/unit/test_config_errors.py` cover each constrained field with one
      example (missing key, negative `sigma`, non-positive `k`, negative `q_max`,
      malformed transition matrix, unknown YAML key).
    - _Requirements: 1.3, 1.4_

- [x] 3. Seeding tree
  - [x] 3.1 Implement `build_seed_tree`
    - In `etf_mm_sim/seeding.py` implement frozen dataclass `PathSeeds(mid, fill)` and
      `build_seed_tree(master_seed, n_regimes, n_paths) -> list[list[PathSeeds]]`
      following the design's `SeedSequence.spawn` layout, including the reserved analytics
      slot at index `n_regimes`.
    - Expose `analytics_seed(master_seed, n_regimes)` returning the reserved
      `SeedSequence` for bootstrap.
    - _Requirements: 10.1, 10.3_

  - [x]* 3.2 Write property test for seed-tree determinism
    - **Property 23: SeedSequence tree determinism**
    - **Validates: Requirements 10.3**
    - In `tests/property/test_seeding.py` assert two independent constructions with the
      same `(master_seed, n_regimes, n_paths)` produce leaves with identical `entropy`
      and `spawn_key`.

- [x] 4. Mid-price simulator (GBM then regime-switching)
  - [x] 4.1 Implement vectorized GBM path generator
    - In `etf_mm_sim/mid_price.py` implement `simulate_gbm(s0, mu, sigma, T, dt, ss)`
      using exact log-Euler with `cumsum` over pre-drawn normals. Length `N+1`.
    - _Requirements: 2.1, 2.4, 2.5, 2.6, 11.2_

  - [x]* 4.2 Write property tests for GBM
    - **Property 2: Mid-price simulator determinism** — **Validates: Requirements 2.3**
    - **Property 3: Mid-price path length** — **Validates: Requirements 2.4**
    - **Property 4: GBM strictly positive prices** — **Validates: Requirements 2.5**
    - **Property 5: GBM zero-volatility deterministic drift** — **Validates: Requirements 2.6**
    - In `tests/property/test_mid_price_gbm.py` use Hypothesis strategies for
      `(s0>0, mu, sigma>=0, T>0, 0<dt<=T, seed)` and write one test function per property.

  - [x] 4.3 Implement regime-switching path generator
    - Implement `simulate_regime_switching(s0, regimes, P, T, dt, ss)` returning
      `(s_path, regime_indices)`. Draw the regime trajectory by inverse-CDF on the
      current row of `P`, then apply per-step `(mu_r, sigma_r)` GBM increments.
    - _Requirements: 2.2, 2.3, 2.4, 11.2_

  - [x]* 4.4 Write smoke tests for regime-switching shape and determinism
    - In `tests/smoke/test_regime_switching.py` assert shape `N+1`, regime indices in
      `[0, R)`, and that two calls with the same `SeedSequence` are bit-identical.
    - _Requirements: 2.2, 2.3_

- [x] 5. Quoters (Avellaneda-Stoikov and Symmetric)
  - [x] 5.1 Implement Avellaneda-Stoikov quoter
    - In `etf_mm_sim/quoters/avellaneda_stoikov.py` implement scalar
      `as_quote(s, q, t, T, gamma, sigma, k) -> (p_b, p_a)` and a vectorized
      `precompute(s_path, dt, T, params) -> dict` that returns `delta_star[0..N]` and
      the inventory-skew coefficient `gamma * sigma**2 * (T - t_i)` per step. Suppress
      both sides at `t >= T`.
    - _Requirements: 3.1, 3.2, 3.3, 3.7_

  - [x]* 5.2 Write property tests for Avellaneda-Stoikov
    - **Property 6: AS quote construction matches closed form** — **Validates: Requirements 3.1, 3.2, 3.3**
    - **Property 7: AS skew sign convention** — **Validates: Requirements 3.4, 3.5, 3.6**
    - **Property 8: AS no-quote at terminal** — **Validates: Requirements 3.7**
    - In `tests/property/test_as_quoter.py` write one Hypothesis test per property with
      `math.isclose(rel_tol=1e-12, abs_tol=1e-12)` for the closed-form identities.

  - [x] 5.3 Implement Symmetric quoter
    - In `etf_mm_sim/quoters/symmetric.py` implement
      `symmetric_quote(s, q, t, delta_base) -> (p_b, p_a) = (s - delta_base, s + delta_base)`
      and a `precompute` that returns an empty schedule.
    - _Requirements: 4.1, 4.2, 4.3_

  - [x]* 5.4 Write property test for Symmetric quoter
    - **Property 9: Symmetric quoter is mid-symmetric and inventory-independent** —
      **Validates: Requirements 4.1, 4.2, 4.3**
    - In `tests/property/test_symmetric_quoter.py` assert quotes are independent of `q`
      and that `(p_a + p_b)/2 == s` exactly.

- [x] 6. Fill engine
  - [x] 6.1 Implement intensity, probability, and uniform draws
    - In `etf_mm_sim/fill_engine.py` implement `fill_intensity(delta, A, k)`,
      `fill_probability(delta, A, k, dt)` using `np.expm1`, with `delta <= 0` short-circuit
      to probability `1`, and `draw_fill_uniforms(ss, n) -> (u_b, u_a)`.
    - _Requirements: 5.1, 5.2, 5.3, 5.6, 5.7_

  - [x]* 6.2 Write property tests for fill engine
    - **Property 10: Fill intensity closed form** — **Validates: Requirements 5.1**
    - **Property 11: Bernoulli fill probability and crossed-quote fill** —
      **Validates: Requirements 5.2, 5.6**
    - **Property 13: Fill engine determinism** — **Validates: Requirements 5.7**
    - In `tests/property/test_fill_engine.py` write one Hypothesis test per property.

- [x] 7. Risk manager
  - [x] 7.1 Implement inventory bounds, kill-switch, terminal flatten
    - In `etf_mm_sim/risk_manager.py` implement `apply_inventory_bounds(q, q_max)`
      returning `(suppress_bid, suppress_ask)`, `kill_switch_triggered(running_pnl, L_kill)`,
      and `flatten_terminal(q_T, s_T, cash_T) -> cash_T + q_T * s_T`.
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [x]* 7.2 Write unit tests for risk manager primitives
    - In `tests/unit/test_risk_manager.py` cover bound suppression at `+q_max` and
      `-q_max`, `L_kill=None` disabling the kill-switch, and the terminal flatten
      arithmetic on hand-rolled inputs.
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 8. Path runner (pure-Python first, then optional numba JIT)
  - [x] 8.1 Implement pure-Python `run_path`
    - In `etf_mm_sim/path_runner.py` implement `run_path(s_path, quoter_schedules, u_b,
      u_a, fill_params, dt, q_max, L_kill) -> PathResult` with the per-step loop from
      design §Vectorization Strategy: compute deltas from quoter schedules, apply risk
      suppression, resolve Bernoulli fills against pre-drawn uniforms, update `q`, `cash`,
      `pnl_curve`, latch the kill-switch, and call `flatten_terminal` after step `N`.
    - Define `PathResult` dataclass per design §Data Models.
    - _Requirements: 5.4, 5.5, 6.1, 6.2, 6.3, 6.4, 6.5, 8.1_

  - [-] 8.2 Add optional numba-JIT fast path
    - Add a `numba.njit(cache=True)`-decorated implementation guarded by an
      `try/except ImportError` import so the module loads without numba. Both the JIT
      and pure-Python implementations consume the same arrays and produce identical
      output. Expose a single `run_path` entry point that dispatches to the JIT version
      when available.
    - _Requirements: 11.1, 11.2_

  - [x]* 8.3 Write property tests for path-runner invariants
    - **Property 12: Fill bookkeeping** — **Validates: Requirements 5.4, 5.5**
    - **Property 14: Inventory bound invariant** — **Validates: Requirements 6.1, 6.2, 6.3**
    - **Property 15: Kill-switch halts trading** — **Validates: Requirements 6.4**
    - **Property 16: Terminal-flatten P&L identity** — **Validates: Requirements 6.5, 8.1**
    - In `tests/property/test_path_runner.py` use Hypothesis to generate small valid
      configs and short paths; one test function per property. Run each property test
      against both the pure-Python and (when available) numba implementations.

- [x] 9. Checkpoint — primitives complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Backtest engine and persistence
  - [x] 10.1 Implement `run_backtest`
    - In `etf_mm_sim/backtest.py` implement `run_backtest(cfg) -> BacktestResult`. Per
      regime, compute mid-price paths once and reuse across both strategies; per
      `(strategy, regime, path)` invoke `run_path` with the shared `SS_fill`. Build
      quoters via `build_quoter(strategy, cfg, regime)`.
    - _Requirements: 7.1, 7.2, 7.3_

  - [x] 10.2 Implement deterministic Parquet + JSON persistence
    - In `etf_mm_sim/persistence.py` implement `persist(results, cfg) -> Path` writing
      the directory layout from design §File Output Format. Pin Parquet options
      (`compression="zstd"`, `compression_level=3`, `row_group_size`, alphabetical
      column order, `use_dictionary=False`). Write `config.resolved.yaml` and
      `manifest.json` (master seed, package version, library versions). Use
      tmp-file + atomic rename. Raise `PersistenceError` on I/O failure.
    - _Requirements: 7.4, 7.5, 10.2, 10.4_

  - [x]* 10.3 Write property test for paired mid-price seeding
    - **Property 17: Paired mid-price seeding across strategies** —
      **Validates: Requirements 7.2**
    - In `tests/property/test_backtest_pairing.py` run a tiny backtest (`n_paths=5`,
      `T=0.01`) and assert the mid-price arrays seen by AS and Symmetric are
      byte-identical for every `(regime, path)`.

  - [x]* 10.4 Write property test for bit-identical persisted outputs
    - **Property 18: Bit-identical persisted outputs** —
      **Validates: Requirements 7.5, 10.2**
    - In `tests/property/test_persistence_determinism.py` use a reduced config
      (`n_paths=10`, `T=0.01`) and assert two independent `run_backtest` invocations
      produce byte-identical Parquet and `manifest.json` files (excluding the
      run-timestamp directory name).

  - [x]* 10.5 Write integration test for backtest cell counts and persisted schema
    - In `tests/integration/test_backtest_smoke.py` assert per-cell counts equal
      `n_paths`, that all expected Parquet files exist with the documented schema, and
      that `manifest.json` records the master seed.
    - _Requirements: 7.1, 7.4, 10.4_

- [x] 11. P&L analytics
  - [x] 11.1 Implement per-path metrics
    - In `etf_mm_sim/analytics.py` implement `terminal_pnl`, `mtm_pnl_curve`,
      annualized `sharpe` (using `analytics.sharpe_annualization_factor`),
      `max_drawdown`, `fill_rate_asymmetry`, `adverse_selection` (configurable horizon
      `Δt_adv` with sign convention per design), and `spread_capture` per fill.
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_

  - [x] 11.2 Implement per-cell `CellSummary` aggregation
    - Aggregate per-path metrics into `CellSummary` with mean, std, p05/p50/p95 of
      terminal P&L, mean Sharpe, mean max-DD, mean fill asymmetry, mean adverse
      selection, mean spread capture per `(strategy, regime)`.
    - _Requirements: 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_

  - [x]* 11.3 Write property tests for analytics
    - **Property 19: Max drawdown non-negativity** — **Validates: Requirements 8.4**
    - **Property 20: Spread capture per fill** — **Validates: Requirements 8.7**
    - In `tests/property/test_analytics.py` write one Hypothesis test per property.

  - [x]* 11.4 Write unit tests for Sharpe, fill asymmetry, adverse selection
    - In `tests/unit/test_analytics_examples.py` use hand-rolled small inputs to verify
      Sharpe, fill-rate asymmetry, and adverse selection values against manually
      computed references.
    - _Requirements: 8.3, 8.5, 8.6_

- [x] 12. Paired bootstrap CIs
  - [x] 12.1 Implement `paired_bootstrap_ci`
    - In `etf_mm_sim/bootstrap.py` implement `paired_bootstrap_ci(diff, B, alpha, ss) ->
      (point, lo, hi)` using the percentile method, with the RNG seeded from the
      analytics-reserved `SeedSequence`. Wire into `analytics` to produce
      `PairedComparison(diff_mean_pnl, diff_mean_pnl_ci, diff_max_dd, diff_max_dd_ci, ...)`
      per regime.
    - _Requirements: 8.8_

  - [x]* 12.2 Write property test for bootstrap determinism and validity
    - **Property 21: Paired-bootstrap CI determinism and validity** —
      **Validates: Requirements 8.8**
    - In `tests/property/test_bootstrap.py` assert two calls with the same seed produce
      identical `(point, lo, hi)` tuples and that `point == diff.mean()`.

- [x] 13. Counterfactual no-skew replay
  - [x] 13.1 Implement `replay_no_skew`
    - In `etf_mm_sim/counterfactual.py` implement `replay_no_skew(as_path, s_path,
      delta_star_sched) -> PathResult`: counterfactual quotes `s ± delta_star`, reuse
      AS bid/ask fill events, recompute cash and `pnl_curve`, terminal-flatten. Wire
      `skew_dd_attribution` and its paired-bootstrap CI into `PairedComparison`.
    - _Requirements: 8.9_

  - [x]* 13.2 Write property test for counterfactual inventory equality
    - **Property 22: Counterfactual inventory equals AS inventory** —
      **Validates: Requirements 8.9**
    - In `tests/property/test_counterfactual.py` assert `cf_q == as_q` at every step
      and that bid/ask fill counts match.

- [x] 14. Checkpoint — full backtest pipeline complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 15. Visualization module
  - [x] 15.1 Implement matplotlib renderers
    - In `etf_mm_sim/viz.py` implement `plot_terminal_pnl_hist(results, regime)`,
      `plot_sample_path(path_result)` with 4 stacked subplots (mid+quotes, inventory,
      cash, cumulative P&L), and `render_summary_table(cells) -> pd.DataFrame` (HTML
      via `to_html` for notebook embedding). Save figures to `plots/` at 150 dpi PNG.
    - _Requirements: 9.1, 9.2, 9.3_

  - [x]* 15.2 Write smoke tests for visualization
    - In `tests/smoke/test_viz.py` assert each renderer produces a non-empty
      `matplotlib.figure.Figure` and that the summary table contains all required
      columns.
    - _Requirements: 9.1, 9.2, 9.3_

- [x] 16. CLI entry point
  - [x] 16.1 Implement `python -m etf_mm_sim run <config.yaml>`
    - In `etf_mm_sim/cli.py` implement an `argparse`-based CLI with subcommand `run`
      that calls `load_config`, `run_backtest`, `persist`, and `viz`, then prints the
      output directory path. Wire `__main__.py` so the package is invokable via
      `python -m etf_mm_sim`.
    - _Requirements: 7.4, 9.1, 9.2, 9.3, 10.1, 10.4_

  - [x]* 16.2 Write integration test for CLI end-to-end run
    - In `tests/integration/test_cli.py` invoke the CLI on a minimal config
      (`n_paths=5`, `T=0.01`) via `subprocess.run` and assert the output directory
      contains `manifest.json`, the `paths/` Parquet files, and at least one PNG.
    - _Requirements: 7.4, 10.4_

- [x] 17. Default YAML config and Jupyter notebook deliverable
  - [x] 17.1 Author `configs/default.yaml`
    - Write the reference YAML from design §Config_Loader with three regimes
      (`low`, `normal`, `high`), `n_paths=1000`, `T=1`, `dt=0.001`, master seed,
      analytics fields populated.
    - _Requirements: 1.2, 7.1, 7.3, 11.1_

  - [x] 17.2 Author `notebooks/default_backtest.ipynb`
    - Notebook cells: load `configs/default.yaml` (with `n_paths` reduced for notebook
      runtime if needed), call `run_backtest`, then `viz` to produce per-regime
      terminal-P&L histograms, a sample-path diagnostic, and the summary table.
      Use `nbclient` execution conventions (no interactive widgets).
    - _Requirements: 9.4_

  - [x]* 17.3 Write CI smoke test that executes the notebook
    - In `tests/smoke/test_notebook.py` use `nbclient.NotebookClient` to execute
      `notebooks/default_backtest.ipynb` end-to-end and assert no cell raises.
    - _Requirements: 9.4_

- [ ] 18. Performance benchmark
  - [ ]* 18.1 Write the 5-minute target benchmark
    - In `tests/perf/test_target.py` mark with `@pytest.mark.slow` and `@pytest.mark.perf`,
      run `run_backtest` with `n_paths=1000`, three regimes, both strategies, `T=1`,
      `dt=0.001` on a single core, and assert wall time `< 300 s`. Excluded from default
      `pytest` runs; opt-in via `pytest -m slow`.
    - _Requirements: 11.1, 11.2_

- [x] 19. Documentation
  - [x] 19.1 Author `README.md`
    - Sections: feature summary, the Avellaneda-Stoikov derivation (reservation price and
      optimal half-spread), package layout, install instructions, CLI usage example,
      notebook usage example, links to `requirements.md`, `design.md`, and `tasks.md`.
    - _Requirements: 9.4_

- [x] 20. Final checkpoint — full deliverable
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Sub-tasks marked with `*` are optional and can be skipped for faster MVP. Top-level tasks
  must always be implemented.
- Each task references specific requirements clauses for traceability.
- The 23 correctness properties from `design.md` are realized one-to-one as Hypothesis
  property tests, each annotated with its property number and the requirements clauses it
  validates.
- Checkpoints (tasks 9, 14, 20) ensure incremental validation at primitive,
  pipeline, and deliverable boundaries.
- TDD ordering: where a parent task contains both an implementation and a property/unit
  test sub-task, author the test sub-task first to drive the implementation.
- The `numba` JIT path (task 8.2) is optional at runtime; the pure-Python fallback (task
  8.1) must satisfy the 5-minute performance target on its own.

## Workflow Completion

This workflow produces only design and planning artifacts (`requirements.md`, `design.md`,
`tasks.md`). Implementation is a separate step. To begin executing, open `tasks.md` in the
spec view and click **Start task** next to the task you want to work on first.
