"""Hypothesis strategies for property-based tests.

This module hosts shared composite strategies for the 23 correctness
properties enumerated in ``design.md`` (P1-P23).

Currently implemented
---------------------
* :func:`valid_configurations` — random valid :class:`Configuration` instances
  used by Property 1 (configuration round-trip).

The strategies are intentionally conservative on their numeric ranges: bounds
that are merely "valid" (e.g. ``T <= 10.0``) are picked to keep YAML-serialized
floats short and to keep round-trip tolerance trivially satisfiable. Property
tests that need looser ranges should compose new strategies rather than widen
these.
"""

from __future__ import annotations

import math
import pathlib

import hypothesis.strategies as st

from etf_mm_sim.config import (
    ASParams,
    AnalyticsParams,
    Configuration,
    FillParams,
    HorizonConfig,
    MCParams,
    MidPriceConfig,
    OutputConfig,
    RegimeParams,
    RiskParams,
    SymmetricParams,
)

__all__ = ["valid_configurations"]


# Bounded floats that survive YAML float -> str -> float round-trip exactly.
# We avoid subnormals and limit width so PyYAML's default float formatting
# preserves bit-exact equality.
_FINITE_FLOAT = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
_POS_FLOAT = st.floats(
    min_value=1e-3,
    max_value=1e3,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
_NONNEG_FLOAT = st.floats(
    min_value=0.0,
    max_value=1e3,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)


@st.composite
def _regime_names(draw, n: int) -> tuple[str, ...]:
    """Generate ``n`` distinct non-empty regime names."""
    names = draw(
        st.lists(
            st.text(
                alphabet=st.characters(
                    min_codepoint=ord("a"),
                    max_codepoint=ord("z"),
                ),
                min_size=1,
                max_size=8,
            ),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    return tuple(names)


@st.composite
def _row_stochastic_matrix(draw, R: int) -> tuple[tuple[float, ...], ...]:
    """Generate an ``R x R`` row-stochastic matrix.

    We sample each row as ``R`` non-negative weights and normalize, then
    snap the last entry to ``1 - sum(rest)`` so the row sums to 1 *exactly*
    in float arithmetic. This avoids spurious ``1e-16`` violations after
    YAML round-trip.
    """
    rows: list[tuple[float, ...]] = []
    for _ in range(R):
        weights = draw(
            st.lists(
                st.floats(
                    min_value=0.01,
                    max_value=1.0,
                    allow_nan=False,
                    allow_infinity=False,
                    allow_subnormal=False,
                ),
                min_size=R,
                max_size=R,
            )
        )
        total = math.fsum(weights)
        normed = [w / total for w in weights]
        # Snap the last entry to absorb the residual so the row sums to
        # exactly 1.0 in floating point.
        normed[-1] = 1.0 - math.fsum(normed[:-1])
        # Ensure non-negativity after snapping.
        if normed[-1] < 0.0:
            normed[-1] = 0.0
            # Rescale the head so the row still sums to 1.
            head_total = math.fsum(normed[:-1])
            if head_total > 0:
                normed[:-1] = [v / head_total for v in normed[:-1]]
                normed[-1] = 1.0 - math.fsum(normed[:-1])
        rows.append(tuple(normed))
    return tuple(rows)


@st.composite
def valid_configurations(draw) -> Configuration:
    """Generate random valid :class:`Configuration` instances."""
    master_seed = draw(st.integers(min_value=0, max_value=(1 << 63) - 1))

    T = draw(
        st.floats(
            min_value=1e-3,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        )
    )
    # dt = T * f where f in (0, 1] guarantees 0 < dt <= T.
    f = draw(
        st.floats(
            min_value=1e-3,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        )
    )
    dt = T * f
    horizon = HorizonConfig(T=T, dt=dt)

    s0 = draw(
        st.floats(
            min_value=1e-3,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        )
    )

    # Pick model and regime count consistently.
    model = draw(st.sampled_from(["gbm", "regime_switching"]))
    if model == "gbm":
        n_regimes = 1
        transition_matrix = None
    else:
        n_regimes = draw(st.integers(min_value=2, max_value=4))
        transition_matrix = draw(_row_stochastic_matrix(n_regimes))

    names = draw(_regime_names(n_regimes))
    regimes = tuple(
        RegimeParams(
            name=names[i],
            mu=draw(_FINITE_FLOAT),
            sigma=draw(_NONNEG_FLOAT),
        )
        for i in range(n_regimes)
    )
    mid_price = MidPriceConfig(
        model=model,
        s0=s0,
        regimes=regimes,
        transition_matrix=transition_matrix,
    )

    quoters_as = ASParams(
        gamma=draw(_POS_FLOAT),
        k=draw(_POS_FLOAT),
        A=draw(_POS_FLOAT),
    )
    quoters_sym = SymmetricParams(delta_base=draw(_NONNEG_FLOAT))

    fill = FillParams(
        A_b=draw(_POS_FLOAT),
        k_b=draw(_POS_FLOAT),
        A_a=draw(_POS_FLOAT),
        k_a=draw(_POS_FLOAT),
    )

    q_max = draw(st.integers(min_value=0, max_value=10_000))
    L_kill = draw(st.one_of(st.none(), _POS_FLOAT))
    risk = RiskParams(q_max=q_max, L_kill=L_kill)

    mc = MCParams(n_paths=draw(st.integers(min_value=1, max_value=100)))

    analytics = AnalyticsParams(
        adverse_selection_horizon_steps=draw(
            st.integers(min_value=0, max_value=1_000)
        ),
        bootstrap_iterations=draw(st.integers(min_value=1, max_value=10_000)),
        bootstrap_alpha=draw(
            st.floats(
                min_value=1e-3,
                max_value=1.0 - 1e-3,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            )
        ),
        sharpe_annualization_factor=draw(_POS_FLOAT),
    )

    output = OutputConfig(dir=pathlib.Path("results"), format="parquet")

    return Configuration(
        master_seed=master_seed,
        horizon=horizon,
        mid_price=mid_price,
        quoters_as=quoters_as,
        quoters_sym=quoters_sym,
        fill=fill,
        risk=risk,
        mc=mc,
        analytics=analytics,
        output=output,
    )
