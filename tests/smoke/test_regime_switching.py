"""Smoke tests for the regime-switching mid-price simulator.

These tests cover the shape, determinism, strict-positivity, and
single-regime collapse behaviour of :func:`etf_mm_sim.mid_price.simulate_regime_switching`.
They are intentionally example-based (rather than property-based) because
the regime-switching generator is exercised end-to-end at the property level
through the higher-level integration tests (Properties 17, 18) once the
backtest engine is wired up.

Validates portions of Requirements 2.2, 2.3, 2.4, 2.5.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from etf_mm_sim.config import RegimeParams
from etf_mm_sim.mid_price import simulate_gbm, simulate_regime_switching


# --------------------------------------------------------------------------- #
# Test fixtures                                                               #
# --------------------------------------------------------------------------- #


@pytest.fixture
def three_regimes() -> tuple[RegimeParams, ...]:
    return (
        RegimeParams(name="low", mu=0.0, sigma=0.1),
        RegimeParams(name="normal", mu=0.05, sigma=0.5),
        RegimeParams(name="high", mu=-0.02, sigma=2.0),
    )


@pytest.fixture
def three_regime_P() -> np.ndarray:
    return np.array(
        [
            [0.97, 0.03, 0.0],
            [0.02, 0.96, 0.02],
            [0.0, 0.05, 0.95],
        ],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------- #
# Shape                                                                       #
# --------------------------------------------------------------------------- #


def test_shape(
    three_regimes: tuple[RegimeParams, ...], three_regime_P: np.ndarray
) -> None:
    """``s_path`` has length ``ceil(T/dt)+1``; ``regime_indices`` has length ``n``."""
    T = 1.0
    dt = 0.01
    n = math.ceil(T / dt)
    s_path, regime_indices = simulate_regime_switching(
        s0=100.0,
        regimes=three_regimes,
        P=three_regime_P,
        T=T,
        dt=dt,
        ss=np.random.SeedSequence(2024),
    )
    R = len(three_regimes)
    assert s_path.shape == (n + 1,)
    assert regime_indices.shape == (n,)
    assert regime_indices.dtype.kind == "i"
    assert int(regime_indices.min()) >= 0
    assert int(regime_indices.max()) < R


# --------------------------------------------------------------------------- #
# Determinism                                                                 #
# --------------------------------------------------------------------------- #


def test_determinism(
    three_regimes: tuple[RegimeParams, ...], three_regime_P: np.ndarray
) -> None:
    """Two calls with two FRESH ``SeedSequence(seed)`` instances are byte-identical."""
    seed = 12345
    args = dict(
        s0=100.0,
        regimes=three_regimes,
        P=three_regime_P,
        T=1.0,
        dt=0.01,
    )
    s_path_a, reg_a = simulate_regime_switching(
        **args, ss=np.random.SeedSequence(seed)
    )
    s_path_b, reg_b = simulate_regime_switching(
        **args, ss=np.random.SeedSequence(seed)
    )
    assert np.array_equal(s_path_a, s_path_b)
    assert np.array_equal(reg_a, reg_b)


# --------------------------------------------------------------------------- #
# Strict positivity                                                           #
# --------------------------------------------------------------------------- #


def test_strict_positivity(
    three_regimes: tuple[RegimeParams, ...], three_regime_P: np.ndarray
) -> None:
    """All path elements are strictly positive when ``s0 > 0`` and every ``sigma >= 0``."""
    s_path, _ = simulate_regime_switching(
        s0=100.0,
        regimes=three_regimes,
        P=three_regime_P,
        T=1.0,
        dt=0.005,
        ss=np.random.SeedSequence(7),
    )
    assert (s_path > 0.0).all()
    assert np.isfinite(s_path).all()


# --------------------------------------------------------------------------- #
# Single regime collapses (in shape + indices) to GBM-like behaviour          #
# --------------------------------------------------------------------------- #


def test_single_regime_collapses_to_gbm() -> None:
    """With ``R = 1`` and ``P = [[1.0]]`` regime indices are all 0 and the
    output is well-formed.

    Note: regime-switching draws TWO RNG streams (``u_reg`` then normals) from
    the input ``SeedSequence`` while ``simulate_gbm`` draws only the normal
    stream, so the two outputs are NOT bit-identical even with the same seed.
    The shape, strict positivity, and constant regime trajectory are what we
    can assert here.
    """
    regimes = (RegimeParams(name="only", mu=0.05, sigma=0.3),)
    P = np.array([[1.0]], dtype=np.float64)
    T = 0.5
    dt = 0.01
    n = math.ceil(T / dt)

    seed = 99
    s_path, regime_indices = simulate_regime_switching(
        s0=50.0,
        regimes=regimes,
        P=P,
        T=T,
        dt=dt,
        ss=np.random.SeedSequence(seed),
    )

    assert s_path.shape == (n + 1,)
    assert regime_indices.shape == (n,)
    assert (regime_indices == 0).all()
    assert (s_path > 0.0).all()

    # Sanity: a separate GBM run with the same params and a fresh seed also
    # produces a strictly positive path of the same length. We do NOT compare
    # bit-for-bit because of the differing RNG-draw orders.
    gbm_path = simulate_gbm(
        50.0, 0.05, 0.3, T, dt, np.random.SeedSequence(seed)
    )
    assert gbm_path.shape == s_path.shape


# --------------------------------------------------------------------------- #
# Input validation                                                            #
# --------------------------------------------------------------------------- #


def test_rejects_non_stochastic_matrix(
    three_regimes: tuple[RegimeParams, ...]
) -> None:
    """A non-row-stochastic ``P`` is rejected at validation time."""
    bad_P = np.array(
        [
            [0.5, 0.3, 0.0],   # row sums to 0.8 -> not 1
            [0.1, 0.7, 0.2],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    with pytest.raises(ValueError):
        simulate_regime_switching(
            s0=100.0,
            regimes=three_regimes,
            P=bad_P,
            T=1.0,
            dt=0.01,
            ss=np.random.SeedSequence(0),
        )


def test_rejects_invalid_initial_regime(
    three_regimes: tuple[RegimeParams, ...], three_regime_P: np.ndarray
) -> None:
    """An out-of-range ``initial_regime`` is rejected."""
    with pytest.raises(ValueError):
        simulate_regime_switching(
            s0=100.0,
            regimes=three_regimes,
            P=three_regime_P,
            T=1.0,
            dt=0.01,
            ss=np.random.SeedSequence(0),
            initial_regime=3,  # only 0..2 are valid
        )
