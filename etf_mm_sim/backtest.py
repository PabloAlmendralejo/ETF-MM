"""Monte Carlo backtest orchestration.

Iterates over ``(regime, strategy, path)`` cells, reusing mid-price paths
across strategies for paired comparison (Req 7.2). See ``design.md``
§Backtest_Engine.

Per design Q1 (resolved): the AS quoter is parameterized by the *regime's*
configured ``sigma`` for every path inside that regime cell. The schedule
(``delta_star``, ``skew_coef``) is therefore precomputed once per regime
and reused across every path in that regime.

Per design Q4 (resolved): the optional numba JIT path (task 8.2) is
deferred; the pure-Python :func:`etf_mm_sim.path_runner.run_path` is the
sole entry point used here.

Sweep semantics
---------------
This iteration treats each configured ``RegimeParams`` entry as an
independent "single-regime cell": for regime ``r`` we build paths under
GBM with ``(mu_r, sigma_r)`` and feed them to both strategies. This
matches the design's intent of "sweeping volatility regimes" (Req 7.3)
and keeps the AS schedule pre-computable. Markov regime-switching paths
remain available via :func:`etf_mm_sim.mid_price.simulate_regime_switching`
but are not consumed by the backtest at this iteration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Configuration, validate
from .fill_engine import draw_fill_uniforms
from .mid_price import simulate_gbm
from .path_runner import PathResult, run_path
from .quoters.avellaneda_stoikov import precompute as as_precompute
from .seeding import build_seed_tree

__all__ = ["BacktestResult", "run_backtest"]


@dataclass(frozen=True)
class BacktestResult:
    """Aggregated output of one full Monte Carlo sweep.

    Attributes
    ----------
    config:
        The :class:`Configuration` used to drive the sweep. Stored so
        downstream consumers (persistence, analytics) can resolve every
        parameter without round-tripping through YAML.
    paths:
        Mapping from ``(strategy, regime_name)`` to a list of
        :class:`PathResult` of length ``cfg.mc.n_paths``. Both strategies
        and every regime are present.
    """

    config: Configuration
    paths: dict[tuple[str, str], list[PathResult]]


# Strategy iteration order. Fixed so persistence layout and downstream
# analytics see strategies in the same canonical order across runs.
_STRATEGIES: tuple[str, ...] = ("avellaneda_stoikov", "symmetric")


def run_backtest(cfg: Configuration) -> BacktestResult:
    """Run the full Monte Carlo sweep over ``(regime, strategy, path)``.

    For each regime the mid-price paths are built once (paired across
    strategies, Req 7.2) and the AS schedule is precomputed once. For each
    strategy and path the runner consumes pre-drawn fill uniforms derived
    from the per-path ``fill`` seed sequence so that AS and Symmetric
    quoters see the same uniform draws on the same path.

    Parameters
    ----------
    cfg:
        Validated :class:`Configuration`. ``validate(cfg)`` is called
        defensively even though :func:`etf_mm_sim.config.load_config`
        already invokes it; callers that build a ``Configuration`` by
        hand still benefit from the check.

    Returns
    -------
    BacktestResult
        The full sweep output.
    """
    # Defensive validation: callers that built Configuration in code
    # may have skipped load_config(), and a malformed cfg here would
    # fail much later in run_path() with a less helpful message.
    validate(cfg)

    n_regimes = len(cfg.mid_price.regimes)
    n_paths = cfg.mc.n_paths
    T = cfg.horizon.T
    dt = cfg.horizon.dt
    s0 = cfg.mid_price.s0

    seeds = build_seed_tree(cfg.master_seed, n_regimes, n_paths)

    results: dict[tuple[str, str], list[PathResult]] = {}

    for r_idx, regime in enumerate(cfg.mid_price.regimes):
        # --- 1) Mid-price paths per regime, shared across strategies. -- #
        # Per Req 7.2 both AS and Symmetric must see the *same* mid-price
        # paths so any P&L difference is attributable to quoting behavior
        # rather than path noise.
        s_paths: list[np.ndarray] = [
            simulate_gbm(
                s0=s0,
                mu=regime.mu,
                sigma=regime.sigma,
                T=T,
                dt=dt,
                ss=seeds[r_idx][p].mid,
            )
            for p in range(n_paths)
        ]

        # All paths from simulate_gbm with the same (T, dt) share the same
        # length n+1; we use the first to derive n and to seed the AS
        # schedule.
        n_plus_1 = int(s_paths[0].shape[0])
        n = n_plus_1 - 1

        # --- 2) AS schedule per regime. -------------------------------- #
        # Per resolved design Q1 (a): AS uses the regime's configured
        # sigma. The schedule depends only on (T, dt, gamma, sigma, k) so
        # it is path-independent and computed once per regime.
        as_sched = as_precompute(
            s_path=s_paths[0],
            dt=dt,
            T=T,
            gamma=cfg.quoters_as.gamma,
            sigma=regime.sigma,
            k=cfg.quoters_as.k,
        )
        delta_star_sched = as_sched["delta_star"]
        skew_coef_sched = as_sched["skew_coef"]
        # Suppress AS quotes wherever t_i >= T (terminal step and beyond),
        # mirroring quote_arrays in the AS module.
        t_grid = np.arange(n_plus_1, dtype=np.float64) * dt
        as_suppress_mask = t_grid >= T

        # --- 3) Run each strategy across every path. ------------------- #
        for strategy in _STRATEGIES:
            cell: list[PathResult] = []
            for p in range(n_paths):
                # Same fill seed across strategies so AS and Symmetric
                # consume identical (u_b, u_a) uniforms; differences in
                # realized fills come purely from differing quote
                # distances (Req 7.2).
                u_b, u_a = draw_fill_uniforms(seeds[r_idx][p].fill, n)

                if strategy == "avellaneda_stoikov":
                    res = run_path(
                        s_paths[p],
                        strategy="avellaneda_stoikov",
                        delta_star_sched=delta_star_sched,
                        skew_coef_sched=skew_coef_sched,
                        as_suppress_mask=as_suppress_mask,
                        u_b=u_b,
                        u_a=u_a,
                        A_b=cfg.fill.A_b,
                        k_b=cfg.fill.k_b,
                        A_a=cfg.fill.A_a,
                        k_a=cfg.fill.k_a,
                        dt=dt,
                        q_max=cfg.risk.q_max,
                        L_kill=cfg.risk.L_kill,
                        regime=regime.name,
                        path_index=p,
                    )
                else:  # strategy == "symmetric"
                    res = run_path(
                        s_paths[p],
                        strategy="symmetric",
                        delta_base=cfg.quoters_sym.delta_base,
                        u_b=u_b,
                        u_a=u_a,
                        A_b=cfg.fill.A_b,
                        k_b=cfg.fill.k_b,
                        A_a=cfg.fill.A_a,
                        k_a=cfg.fill.k_a,
                        dt=dt,
                        q_max=cfg.risk.q_max,
                        L_kill=cfg.risk.L_kill,
                        regime=regime.name,
                        path_index=p,
                    )
                cell.append(res)
            results[(strategy, regime.name)] = cell

    return BacktestResult(config=cfg, paths=results)
