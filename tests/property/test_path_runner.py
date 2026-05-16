"""Property tests for the per-path runner.

# Feature: etf-mm-arbitrage-simulator, Property 12: Fill bookkeeping
# Feature: etf-mm-arbitrage-simulator, Property 14: Inventory bound invariant
# Feature: etf-mm-arbitrage-simulator, Property 15: Kill-switch halts trading
# Feature: etf-mm-arbitrage-simulator, Property 16: Terminal-flatten P&L identity

Each test function realizes exactly one of the four invariants that the
``run_path`` step loop must satisfy. The Hypothesis strategies generate
small, bounded scenarios so a single example runs a complete path; the
tests therefore use ``max_examples=25, deadline=None`` per the
"runner-level" guidance in ``design.md`` §Property-based testing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from hypothesis import HealthCheck, given, settings, strategies as st

from etf_mm_sim.fill_engine import draw_fill_uniforms
from etf_mm_sim.mid_price import simulate_gbm
from etf_mm_sim.path_runner import PathResult, run_path
from etf_mm_sim.quoters.avellaneda_stoikov import precompute as as_precompute


# --------------------------------------------------------------------------- #
# Composite scenario strategy                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Scenario:
    """One self-contained input bundle for ``run_path``.

    Holds all pre-built arrays plus the scalar parameters needed to invoke
    the runner for either strategy. Tests pick the strategy explicitly.
    """

    n: int
    s_path: np.ndarray
    delta_star_sched: np.ndarray
    skew_coef_sched: np.ndarray
    as_suppress_mask: np.ndarray
    delta_base: float
    u_b: np.ndarray
    u_a: np.ndarray
    A_b: float
    k_b: float
    A_a: float
    k_a: float
    dt: float
    T: float
    q_max: int
    L_kill: float | None
    master_seed: int


_strategy_names = st.sampled_from(["avellaneda_stoikov", "symmetric"])


@st.composite
def _scenarios(
    draw,
    *,
    min_n: int = 5,
    max_n: int = 50,
    q_max_lo: int = 1,
    q_max_hi: int = 20,
    force_kill: bool = False,
) -> _Scenario:
    """Build a self-contained ``run_path`` scenario.

    Parameters
    ----------
    min_n, max_n:
        Inclusive bounds for the number of inner steps.
    q_max_lo, q_max_hi:
        Inclusive bounds for ``q_max``. Property 15 (kill-switch) forces
        a moderate-to-large bound so inventory can build, while Property
        14 (inventory bound) keeps the default narrow range so ``q_max``
        binds frequently.
    force_kill:
        When ``True`` we set ``L_kill`` to a small positive value so the
        kill-switch is likely to fire; when ``False`` we sample
        ``L_kill in {None, large positive}`` to mostly leave it idle.
    """
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    s0 = draw(
        st.floats(
            min_value=10.0,
            max_value=1000.0,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        )
    )
    sigma = draw(
        st.floats(
            min_value=0.1,
            max_value=2.0,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        )
    )
    mu = draw(
        st.floats(
            min_value=-0.5,
            max_value=0.5,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        )
    )
    # Bind T so dt = T / n is a clean reciprocal-of-n scaling. Keeping T
    # small bounds the AS skew coefficient ``gamma * sigma^2 * (T - t)``,
    # which keeps quotes well clear of the mid in the typical case.
    T = n * 0.01
    dt = T / n

    gamma = draw(
        st.floats(
            min_value=0.1,
            max_value=2.0,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        )
    )
    k_quoter = draw(
        st.floats(
            min_value=0.5,
            max_value=5.0,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        )
    )

    delta_base = draw(
        st.floats(
            min_value=0.01,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        )
    )

    A_b = draw(
        st.floats(
            min_value=10.0,
            max_value=200.0,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        )
    )
    k_b = draw(
        st.floats(
            min_value=0.5,
            max_value=5.0,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        )
    )
    A_a = draw(
        st.floats(
            min_value=10.0,
            max_value=200.0,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        )
    )
    k_a = draw(
        st.floats(
            min_value=0.5,
            max_value=5.0,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        )
    )

    q_max = draw(st.integers(min_value=q_max_lo, max_value=q_max_hi))

    if force_kill:
        # Tiny kill threshold: even a small accumulated short P&L should
        # trip the switch within the path's horizon.
        L_kill: float | None = draw(
            st.floats(
                min_value=0.001,
                max_value=0.5,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            )
        )
    else:
        # Leave the switch idle in most cases. A ``None`` disables it
        # entirely; a very large threshold is effectively unreachable
        # under our bounded scenarios.
        L_kill = draw(
            st.one_of(
                st.none(),
                st.floats(
                    min_value=1e6,
                    max_value=1e9,
                    allow_nan=False,
                    allow_infinity=False,
                    allow_subnormal=False,
                ),
            )
        )

    master_seed = draw(st.integers(min_value=0, max_value=2**31 - 1))

    # Build mid-price path and AS schedules from the sampled parameters.
    ss = np.random.SeedSequence(master_seed)
    mid_ss, fill_ss = ss.spawn(2)
    s_path = simulate_gbm(s0=s0, mu=mu, sigma=sigma, T=T, dt=dt, ss=mid_ss)
    # ``simulate_gbm`` derives its own ``n = ceil(T/dt)`` which can land at
    # ``n + 1`` when ``T / dt`` is very slightly above the integer ``n`` due
    # to float64 rounding (e.g. ``0.07 / 0.01 == 7.000000000000001``). Treat
    # the path length as authoritative and re-derive ``n`` from it.
    n = int(s_path.shape[0]) - 1
    sched = as_precompute(s_path=s_path, dt=dt, T=T, gamma=gamma, sigma=sigma, k=k_quoter)
    delta_star = sched["delta_star"]
    skew_coef = sched["skew_coef"]
    # Match the suppression rule baked into ``avellaneda_stoikov.quote_arrays``:
    # suppress whenever ``t_i >= T``. This catches the terminal grid point
    # and is the contract the path runner expects.
    t_grid = np.arange(s_path.shape[0], dtype=np.float64) * dt
    as_suppress_mask = t_grid >= T

    u_b, u_a = draw_fill_uniforms(fill_ss, n)

    return _Scenario(
        n=n,
        s_path=s_path,
        delta_star_sched=delta_star,
        skew_coef_sched=skew_coef,
        as_suppress_mask=as_suppress_mask,
        delta_base=delta_base,
        u_b=u_b,
        u_a=u_a,
        A_b=A_b,
        k_b=k_b,
        A_a=A_a,
        k_a=k_a,
        dt=dt,
        T=T,
        q_max=q_max,
        L_kill=L_kill,
        master_seed=master_seed,
    )


def _run(scenario: _Scenario, strategy: str) -> PathResult:
    """Invoke ``run_path`` for the given strategy on a scenario."""
    if strategy == "avellaneda_stoikov":
        return run_path(
            scenario.s_path,
            strategy="avellaneda_stoikov",
            delta_star_sched=scenario.delta_star_sched,
            skew_coef_sched=scenario.skew_coef_sched,
            as_suppress_mask=scenario.as_suppress_mask,
            u_b=scenario.u_b,
            u_a=scenario.u_a,
            A_b=scenario.A_b,
            k_b=scenario.k_b,
            A_a=scenario.A_a,
            k_a=scenario.k_a,
            dt=scenario.dt,
            q_max=scenario.q_max,
            L_kill=scenario.L_kill,
            regime="test",
            path_index=0,
        )
    return run_path(
        scenario.s_path,
        strategy="symmetric",
        delta_base=scenario.delta_base,
        u_b=scenario.u_b,
        u_a=scenario.u_a,
        A_b=scenario.A_b,
        k_b=scenario.k_b,
        A_a=scenario.A_a,
        k_a=scenario.k_a,
        dt=scenario.dt,
        q_max=scenario.q_max,
        L_kill=scenario.L_kill,
        regime="test",
        path_index=0,
    )


# --------------------------------------------------------------------------- #
# Property 12: Fill bookkeeping                                                #
# --------------------------------------------------------------------------- #


@given(scenario=_scenarios(), strategy=_strategy_names)
@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property_12_fill_bookkeeping(scenario: _Scenario, strategy: str) -> None:
    """Inventory and cash recurrences exactly track the recorded fills.

    For each step ``i in {0, ..., N-1}``::

        inventory[i+1] == inventory[i] + bid_fill[i] - ask_fill[i]

    and the per-step cash delta exactly equals ``ask_fill[i] * ask_quote[i]
    - bid_fill[i] * bid_quote[i]`` (Req 5.4 / 5.5: a bid fill records the
    posted bid, an ask fill records the posted ask). Suppressed-step
    quotes are ``NaN``, but suppressed steps cannot have fills, so the
    recurrence is well-defined everywhere.

    **Validates: Requirements 5.4, 5.5**
    """
    res = _run(scenario, strategy)
    n = scenario.n

    # Inventory recurrence.
    expected_inventory = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        expected_inventory[i + 1] = (
            expected_inventory[i] + int(res.bid_fill[i]) - int(res.ask_fill[i])
        )
    assert np.array_equal(res.inventory, expected_inventory)

    # Cash recurrence: at every fill, the cash delta equals the posted
    # quote on that side (Req 5.4 / 5.5).
    cumulative_cash = 0.0
    for i in range(n):
        delta_cash = 0.0
        if bool(res.ask_fill[i]):
            posted_ask = float(res.ask_quote[i])
            assert not math.isnan(posted_ask), (
                "ask quote must be finite at any step where ask_fill is True"
            )
            delta_cash += posted_ask
        if bool(res.bid_fill[i]):
            posted_bid = float(res.bid_quote[i])
            assert not math.isnan(posted_bid), (
                "bid quote must be finite at any step where bid_fill is True"
            )
            delta_cash -= posted_bid
        cumulative_cash += delta_cash
        # Tight tolerance: the recurrence is the same float64 arithmetic
        # the implementation does.
        assert math.isclose(
            float(res.cash[i + 1]),
            cumulative_cash,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )


# --------------------------------------------------------------------------- #
# Property 14: Inventory bound invariant                                       #
# --------------------------------------------------------------------------- #


@given(scenario=_scenarios(q_max_lo=1, q_max_hi=10), strategy=_strategy_names)
@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property_14_inventory_bound_invariant(
    scenario: _Scenario, strategy: str
) -> None:
    """Every step satisfies ``|inventory[i]| <= q_max``.

    The bound is enforced by the risk manager via per-side suppression
    (Req 6.1 / 6.2 / 6.3). Inventory starts at 0 and changes by exactly
    +1 / -1 per step, so the bound holds inductively as long as the
    suppression flag is honored.

    **Validates: Requirements 6.1, 6.2, 6.3**
    """
    res = _run(scenario, strategy)
    assert int(np.max(np.abs(res.inventory))) <= scenario.q_max


# --------------------------------------------------------------------------- #
# Property 15: Kill-switch halts trading                                       #
# --------------------------------------------------------------------------- #


# Force the kill-switch to fire by combining a small ``L_kill`` with a
# wider inventory range so meaningful negative MTM can accumulate.
@given(
    scenario=_scenarios(q_max_lo=5, q_max_hi=20, force_kill=True),
    strategy=_strategy_names,
)
@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property_15_kill_switch_halts(
    scenario: _Scenario, strategy: str
) -> None:
    """Once the kill-switch fires, no further fills are recorded.

    The kill-switch is latched (Req 6.4): once ``running_pnl <= -L_kill``
    triggers at step ``i*``, both sides remain suppressed at every step
    ``i >= i*``. This test does not require the switch to actually fire
    (some scenarios may finish without crossing the threshold), but if it
    does, post-fire fills must be all-False.

    **Validates: Requirements 6.4**
    """
    res = _run(scenario, strategy)
    if res.kill_switch_step is None:
        # The switch never tripped on this scenario; nothing to assert.
        return
    fire_step = res.kill_switch_step
    assert 1 <= fire_step <= scenario.n
    # No fills can resolve at the trigger step or after. The trigger step
    # is the index *after* the fill that pushed running_pnl below the
    # threshold (i.e. ``i + 1`` in the runner), so ``bid_fill[fire_step:]``
    # / ``ask_fill[fire_step:]`` covers the post-fire region.
    assert not np.any(res.bid_fill[fire_step:])
    assert not np.any(res.ask_fill[fire_step:])


# --------------------------------------------------------------------------- #
# Property 16: Terminal-flatten P&L identity                                   #
# --------------------------------------------------------------------------- #


@given(scenario=_scenarios(), strategy=_strategy_names)
@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property_16_terminal_flatten_identity(
    scenario: _Scenario, strategy: str
) -> None:
    """``terminal_pnl == cash[N] + inventory[N] * s_path[N]`` exactly.

    The reported terminal P&L is the post-flatten mark-to-market (Req
    6.5, 8.1) computed from the un-flattened terminal cash and the
    residual inventory at terminal price. The identity holds with very
    tight tolerance because the implementation does the same arithmetic.

    **Validates: Requirements 6.5, 8.1**
    """
    res = _run(scenario, strategy)
    expected = float(res.cash[-1] + res.inventory[-1] * res.s_path[-1])
    assert math.isclose(
        res.terminal_pnl, expected, rel_tol=1e-12, abs_tol=1e-9
    )
