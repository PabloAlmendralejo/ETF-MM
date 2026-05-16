"""Property test for paired-bootstrap CI determinism and validity.

# Feature: etf-mm-arbitrage-simulator, Property 21: Paired-bootstrap CI determinism and validity
# Validates: Requirements 8.8

For any paired difference array, two invocations of
:func:`etf_mm_sim.bootstrap.paired_bootstrap_ci` with the same inputs
and a freshly constructed ``SeedSequence(seed)`` must produce identical
``(point, lo, hi)`` tuples; the reported ``point`` must equal
``diff.mean()``; and the CI must bracket the point estimate.

The bracketing claim ``lo <= point <= hi`` can fail in degenerate cases
(``B = 1``, all-equal ``diff``) so we constrain ``B in [50, 500]`` and
require ``np.unique(diff).size >= 2`` for the bracketing assertion.
Determinism and ``point == diff.mean()`` hold unconditionally and are
asserted on every example.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from etf_mm_sim.bootstrap import paired_bootstrap_ci


_FINITE_FLOAT = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)


@given(
    diff_list=st.lists(_FINITE_FLOAT, min_size=2, max_size=100),
    B=st.integers(min_value=50, max_value=500),
    alpha=st.floats(
        min_value=0.01,
        max_value=0.5,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    ),
    seed=st.integers(min_value=0, max_value=(1 << 63) - 1),
)
@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_21_bootstrap_determinism(
    diff_list: list[float], B: int, alpha: float, seed: int
) -> None:
    """Bootstrap is deterministic; ``point == mean``; CI brackets point.

    For any ``(diff, B, alpha, seed)`` in the constrained sub-space:

    1. Two calls with freshly built ``SeedSequence(seed)`` produce
       byte-identical ``(point, lo, hi)`` tuples (determinism).
    2. ``point`` equals ``diff.mean()`` to within ``1e-12`` relative
       tolerance.
    3. ``lo <= point <= hi`` provided ``diff`` has at least two distinct
       values (the bracketing degenerates trivially when all entries
       are equal because every bootstrap sample mean equals the point
       estimate, which is fine but uninteresting).

    **Validates: Requirements 8.8**
    """
    diff = np.asarray(diff_list, dtype=np.float64)

    # Two independent calls with the *same* entropy. We rebuild the
    # SeedSequence each time so the determinism claim is on the
    # entropy/spawn-key contract rather than on RNG state aliasing.
    ss1 = np.random.SeedSequence(seed)
    ss2 = np.random.SeedSequence(seed)
    point1, lo1, hi1 = paired_bootstrap_ci(diff, B, alpha, ss1)
    point2, lo2, hi2 = paired_bootstrap_ci(diff, B, alpha, ss2)

    # Determinism: identical bytes, not just isclose.
    assert point1 == point2
    assert lo1 == lo2
    assert hi1 == hi2

    # Point estimate equals diff.mean() exactly modulo float roundoff.
    expected_mean = float(diff.mean())
    assert math.isclose(point1, expected_mean, rel_tol=1e-12, abs_tol=1e-12)

    # CI ordering always holds: alpha/2 <= 1 - alpha/2 since alpha < 1.
    assert lo1 <= hi1

    # Bracketing: ``lo <= point <= hi`` is the non-trivial percentile-
    # bootstrap claim. We restrict to ``np.unique(diff).size >= 2`` so we
    # exclude the degenerate constant-diff regime where every bootstrap
    # mean equals the point estimate (lo == point == hi trivially) and
    # the bracketing reduces to a tautology. With ``B in [50, 500]`` and
    # at least two distinct values the percentile bootstrap empirically
    # brackets the sample mean even for heavily skewed inputs.
    assume(np.unique(diff).size >= 2)
    assert lo1 <= point1 <= hi1
