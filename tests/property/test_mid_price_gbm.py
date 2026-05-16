"""Property tests for the GBM mid-price simulator.

# Feature: etf-mm-arbitrage-simulator, Property 2: Mid-price simulator determinism
# Feature: etf-mm-arbitrage-simulator, Property 3: Mid-price path length
# Feature: etf-mm-arbitrage-simulator, Property 4: GBM strictly positive prices
# Feature: etf-mm-arbitrage-simulator, Property 5: GBM zero-volatility deterministic drift

Each test function realizes exactly one of the four properties from
``design.md`` §Correctness Properties that targets ``simulate_gbm``. Bounded
float strategies are used throughout to keep the underlying log-Euler arithmetic
numerically tame: ``s0`` stays well above subnormal range, ``sigma`` is capped
at 5 (the analytic GBM is well-defined for any sigma but extreme draws can
produce ``inf`` after ``exp(sigma**2 * T)`` blows up), ``T`` is bounded in
``[1e-3, 10]``, and ``dt`` is derived as ``T * f`` with ``f in (0, 1]`` so
``0 < dt <= T`` by construction.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from etf_mm_sim.mid_price import simulate_gbm


# --------------------------------------------------------------------------- #
# Shared strategies                                                           #
# --------------------------------------------------------------------------- #


_s0_strategy = st.floats(
    min_value=1e-3,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
_mu_strategy = st.floats(
    min_value=-2.0,
    max_value=2.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
_sigma_strategy = st.floats(
    min_value=0.0,
    max_value=5.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
_T_strategy = st.floats(
    min_value=1e-3,
    max_value=10.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
# dt = T * fraction, fraction in (0, 1], yields 0 < dt <= T by construction.
_dt_fraction_strategy = st.floats(
    min_value=1e-3,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
_seed_strategy = st.integers(min_value=0, max_value=(1 << 31) - 1)


# --------------------------------------------------------------------------- #
# Property 2: Mid-price simulator determinism                                 #
# --------------------------------------------------------------------------- #


@given(
    s0=_s0_strategy,
    mu=_mu_strategy,
    sigma=_sigma_strategy,
    T=_T_strategy,
    dt_frac=_dt_fraction_strategy,
    seed=_seed_strategy,
)
@settings(max_examples=50, deadline=None)
def test_property_2_determinism(
    s0: float, mu: float, sigma: float, T: float, dt_frac: float, seed: int
) -> None:
    """Two fresh ``SeedSequence(seed)`` calls produce bit-identical paths.

    **Validates: Requirements 2.3**
    """
    dt = T * dt_frac
    # IMPORTANT: each call gets a freshly-constructed SeedSequence. Sharing one
    # SeedSequence instance is unsafe because spawn/draw advances internal state.
    ss_a = np.random.SeedSequence(seed)
    ss_b = np.random.SeedSequence(seed)
    path_a = simulate_gbm(s0, mu, sigma, T, dt, ss_a)
    path_b = simulate_gbm(s0, mu, sigma, T, dt, ss_b)
    assert np.array_equal(path_a, path_b)


# --------------------------------------------------------------------------- #
# Property 3: Mid-price path length                                           #
# --------------------------------------------------------------------------- #


@given(
    s0=_s0_strategy,
    mu=_mu_strategy,
    sigma=_sigma_strategy,
    T=_T_strategy,
    dt_frac=_dt_fraction_strategy,
    seed=_seed_strategy,
)
@settings(max_examples=50, deadline=None)
def test_property_3_length(
    s0: float, mu: float, sigma: float, T: float, dt_frac: float, seed: int
) -> None:
    """Output length equals ``ceil(T / dt) + 1``.

    **Validates: Requirements 2.4**
    """
    dt = T * dt_frac
    expected_n = math.ceil(T / dt)
    path = simulate_gbm(s0, mu, sigma, T, dt, np.random.SeedSequence(seed))
    assert path.shape == (expected_n + 1,)


# --------------------------------------------------------------------------- #
# Property 4: GBM strictly positive prices                                    #
# --------------------------------------------------------------------------- #


@given(
    s0=_s0_strategy,
    mu=_mu_strategy,
    sigma=_sigma_strategy,
    T=_T_strategy,
    dt_frac=_dt_fraction_strategy,
    seed=_seed_strategy,
)
@settings(max_examples=50, deadline=None)
def test_property_4_strictly_positive(
    s0: float, mu: float, sigma: float, T: float, dt_frac: float, seed: int
) -> None:
    """Every element of the path is strictly positive.

    Exact log-Euler with finite normal draws and ``s0 > 0`` cannot produce a
    non-positive price.

    **Validates: Requirements 2.5**
    """
    dt = T * dt_frac
    path = simulate_gbm(s0, mu, sigma, T, dt, np.random.SeedSequence(seed))
    assert (path > 0.0).all()
    # Also assert finiteness; an inf would slip past the > 0 check.
    assert np.isfinite(path).all()


# --------------------------------------------------------------------------- #
# Property 5: GBM zero-volatility deterministic drift                         #
# --------------------------------------------------------------------------- #


@given(
    s0=_s0_strategy,
    mu=_mu_strategy,
    T=_T_strategy,
    dt_frac=_dt_fraction_strategy,
    seed=_seed_strategy,
)
@settings(max_examples=50, deadline=None)
def test_property_5_zero_vol_drift(
    s0: float, mu: float, T: float, dt_frac: float, seed: int
) -> None:
    """With ``sigma == 0`` the path is the deterministic drift trajectory.

    The expected element ``i`` is ``s0 * exp(mu * i * dt)``, regardless of the
    seed used for the (multiplied-by-zero) Brownian draws.

    **Validates: Requirements 2.6**
    """
    dt = T * dt_frac
    n = math.ceil(T / dt)
    path = simulate_gbm(s0, mu, 0.0, T, dt, np.random.SeedSequence(seed))
    t_grid = np.arange(n + 1, dtype=np.float64) * dt
    expected = s0 * np.exp(mu * t_grid)
    np.testing.assert_allclose(path, expected, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- #
# Input validation (sanity checks adjacent to the property suite)             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "s0, sigma, T, dt",
    [
        (-1.0, 0.1, 1.0, 0.1),  # s0 not positive
        (0.0, 0.1, 1.0, 0.1),   # s0 zero
        (1.0, -0.1, 1.0, 0.1),  # sigma negative
        (1.0, 0.1, 0.0, 0.1),   # T zero
        (1.0, 0.1, -1.0, 0.1),  # T negative
        (1.0, 0.1, 1.0, 0.0),   # dt zero
        (1.0, 0.1, 1.0, -0.1),  # dt negative
        (1.0, 0.1, 1.0, 2.0),   # dt > T
    ],
)
def test_simulate_gbm_rejects_invalid_inputs(
    s0: float, sigma: float, T: float, dt: float
) -> None:
    """``simulate_gbm`` raises ``ValueError`` on out-of-range inputs."""
    with pytest.raises(ValueError):
        simulate_gbm(s0, 0.0, sigma, T, dt, np.random.SeedSequence(0))
