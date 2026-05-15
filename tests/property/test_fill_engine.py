"""Property tests for the fill engine.

# Feature: etf-mm-arbitrage-simulator, Property 10: Fill intensity closed form
# Feature: etf-mm-arbitrage-simulator, Property 11: Bernoulli fill probability and crossed-quote fill
# Feature: etf-mm-arbitrage-simulator, Property 13: Fill engine determinism

Each test function realizes exactly one of the three properties from
``design.md`` §Correctness Properties that targets the fill engine. The
closed-form identities are checked with ``math.isclose`` at tight
tolerances; the determinism test is an exact ``np.array_equal`` byte-level
comparison.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings, strategies as st

from etf_mm_sim.fill_engine import (
    draw_fill_uniforms,
    fill_intensity,
    fill_intensity_arr,
    fill_probability,
    fill_probability_arr,
)


# --------------------------------------------------------------------------- #
# Shared strategies                                                           #
# --------------------------------------------------------------------------- #


# Intensity parameters bounded so the closed-form arithmetic stays well
# within float64 precision. The upper bound on ``k`` is kept moderate so
# that ``k * delta`` stays representable for ``delta`` near +/-10.
_A_strategy = st.floats(
    min_value=1e-3,
    max_value=1e3,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
_k_strategy = st.floats(
    min_value=1e-3,
    max_value=1e2,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
# Quote distance: signed; covers crossed-quote and far-out-of-the-money.
_delta_strategy = st.floats(
    min_value=-10.0,
    max_value=10.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
# Strictly positive distance branch (needed for the closed-form check on
# the non-crossed regime).
_delta_pos_strategy = st.floats(
    min_value=1e-6,
    max_value=10.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
# Step length covering the typical simulator regime.
_dt_strategy = st.floats(
    min_value=1e-6,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)


# --------------------------------------------------------------------------- #
# Property 10: Fill intensity closed form                                      #
# --------------------------------------------------------------------------- #


@given(
    delta=_delta_strategy,
    A=_A_strategy,
    k=_k_strategy,
)
@settings(max_examples=200, deadline=None)
def test_property_10_intensity(delta: float, A: float, k: float) -> None:
    """``fill_intensity`` matches the closed form ``A * exp(-k * delta)``.

    The vectorized variant ``fill_intensity_arr`` must reproduce the
    scalar result element-wise to within ``rel_tol=1e-12, abs_tol=1e-12``.
    The reference uses :func:`numpy.exp` rather than :func:`math.exp` so
    the bounded-but-large ``-k * delta`` regime degrades to ``inf`` instead
    of raising :class:`OverflowError`; the implementation does the same,
    so the comparison stays meaningful (``math.isclose(inf, inf)`` is
    ``True``).

    **Validates: Requirements 5.1**
    """
    expected = float(A * np.exp(-k * delta))
    scalar = fill_intensity(delta, A, k)
    assert math.isclose(scalar, expected, rel_tol=1e-12, abs_tol=1e-12)

    # Vectorized variant: build a small array containing ``delta`` and a
    # couple of other points and verify per-element equality with the
    # scalar formula.
    deltas = np.array([delta, delta * 0.5, -delta], dtype=np.float64)
    arr = fill_intensity_arr(deltas, A, k)
    assert arr.shape == deltas.shape
    for d, a in zip(deltas.tolist(), arr.tolist()):
        ref = float(A * np.exp(-k * d))
        assert math.isclose(a, ref, rel_tol=1e-12, abs_tol=1e-12)


# --------------------------------------------------------------------------- #
# Property 11: Bernoulli fill probability and crossed-quote fill              #
# --------------------------------------------------------------------------- #


@given(
    delta=_delta_pos_strategy,
    A=_A_strategy,
    k=_k_strategy,
    dt=_dt_strategy,
)
@settings(max_examples=200, deadline=None)
def test_property_11_probability_non_crossed(
    delta: float, A: float, k: float, dt: float
) -> None:
    """For ``delta > 0`` the per-step probability matches ``1 - exp(-lam*dt)``.

    Both the scalar and the vectorized entry points must match the bare
    closed form ``1 - exp(-A * exp(-k * delta) * dt)`` within tight
    tolerance. The implementation evaluates ``-expm1(-lam*dt)`` for
    numerical reasons; we compare against the bare expression here so
    the test does not become an identity in the implementation form.

    **Validates: Requirements 5.2**
    """
    lam = A * math.exp(-k * delta)
    expected = 1.0 - math.exp(-lam * dt)

    scalar = fill_probability(delta, A, k, dt)
    assert 0.0 <= scalar <= 1.0
    # The two equivalent forms can drift by a few ULPs; abs_tol keeps the
    # comparison meaningful when ``lam * dt`` is very small.
    assert math.isclose(scalar, expected, rel_tol=1e-12, abs_tol=1e-15)

    arr = fill_probability_arr(np.array([delta], dtype=np.float64), A, k, dt)
    assert arr.shape == (1,)
    assert math.isclose(
        float(arr[0]), expected, rel_tol=1e-12, abs_tol=1e-15
    )


@given(
    delta=st.floats(
        min_value=-10.0,
        max_value=0.0,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    ),
    A=_A_strategy,
    k=_k_strategy,
    dt=_dt_strategy,
)
@settings(max_examples=200, deadline=None)
def test_property_11_probability_crossed(
    delta: float, A: float, k: float, dt: float
) -> None:
    """For ``delta <= 0`` the per-step probability is exactly ``1.0``.

    The crossed-quote regime is treated as a guaranteed fill within the
    step (Req 5.6); the equality must be exact (no float tolerance) for
    both the scalar and vectorized entry points.

    **Validates: Requirements 5.6**
    """
    scalar = fill_probability(delta, A, k, dt)
    assert scalar == 1.0

    arr = fill_probability_arr(np.array([delta], dtype=np.float64), A, k, dt)
    assert arr.shape == (1,)
    assert float(arr[0]) == 1.0


# --------------------------------------------------------------------------- #
# Property 13: Fill engine determinism                                         #
# --------------------------------------------------------------------------- #


@given(
    seed=st.integers(min_value=0, max_value=2**63 - 1),
    n=st.integers(min_value=0, max_value=2048),
)
@settings(max_examples=200, deadline=None)
def test_property_13_determinism(seed: int, n: int) -> None:
    """Two fresh ``SeedSequence(seed)`` calls produce byte-identical draws.

    The reproducibility contract (Req 5.7) requires that, given the same
    seed, the fill engine emits the same ``(u_b, u_a)`` pair on every
    invocation. The two seed sequences are constructed independently here
    (rather than reused) to mirror the way the path runner re-derives them
    from the master seed on every backtest invocation.

    **Validates: Requirements 5.7**
    """
    ss_1 = np.random.SeedSequence(seed)
    ss_2 = np.random.SeedSequence(seed)
    u_b_1, u_a_1 = draw_fill_uniforms(ss_1, n)
    u_b_2, u_a_2 = draw_fill_uniforms(ss_2, n)

    assert u_b_1.shape == (n,)
    assert u_a_1.shape == (n,)
    assert np.array_equal(u_b_1, u_b_2)
    assert np.array_equal(u_a_1, u_a_2)
