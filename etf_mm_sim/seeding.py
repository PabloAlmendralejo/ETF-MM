"""Master-seed to ``numpy.random.SeedSequence`` tree.

Builds a deterministic per-``(regime, path)`` seed tree with a reserved
analytics slot for the paired-bootstrap RNG. The layout mirrors the diagram
documented in ``design.md`` §Seeding Tree::

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

Properties guaranteed by this layout
------------------------------------
* **Independence**: ``SeedSequence.spawn`` produces statistically independent
  child sequences per the NumPy docs; this is the supported way to derive
  sub-streams.
* **Paired comparison**: ``SS_mid_{r,p}`` and ``SS_fill_{r,p}`` are reused
  across strategies (AS and Symmetric) so they see the same mid-price path
  and the same uniform draws for fills (Req 7.2).
* **Determinism**: The tree depends only on ``master_seed``, ``n_regimes``,
  and ``n_paths``. To stay safe across NumPy versions we always spawn the
  full ``n_regimes + 1`` children at once, since spawning twice from the
  same ``SeedSequence`` is not guaranteed to be deterministic in all NumPy
  releases.
* **Reserved analytics slot**: The ``n_regimes``-th child is reserved for the
  paired-bootstrap RNG and any future analytics randomization, keeping
  analytics RNG independent of the simulation streams.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "PathSeeds",
    "build_seed_tree",
    "analytics_seed",
]


@dataclass(frozen=True)
class PathSeeds:
    """Pair of independent seed sequences for one ``(regime, path)`` cell.

    Attributes
    ----------
    mid:
        Seed sequence consumed by the mid-price simulator.
    fill:
        Seed sequence consumed by the fill-engine uniform draws.
    """

    mid: np.random.SeedSequence
    fill: np.random.SeedSequence


def _validate_inputs(master_seed: int, n_regimes: int, n_paths: int) -> None:
    """Validate the public arguments. Raises ``ValueError`` on bad input."""
    if not isinstance(master_seed, int) or isinstance(master_seed, bool):
        raise ValueError(
            f"master_seed: must be an int; got {master_seed!r}"
        )
    if master_seed < 0:
        raise ValueError(
            f"master_seed: must be >= 0; got {master_seed!r}"
        )
    if not isinstance(n_regimes, int) or isinstance(n_regimes, bool):
        raise ValueError(
            f"n_regimes: must be an int; got {n_regimes!r}"
        )
    if n_regimes < 1:
        raise ValueError(
            f"n_regimes: must be >= 1; got {n_regimes!r}"
        )
    if not isinstance(n_paths, int) or isinstance(n_paths, bool):
        raise ValueError(
            f"n_paths: must be an int; got {n_paths!r}"
        )
    if n_paths < 1:
        raise ValueError(
            f"n_paths: must be >= 1; got {n_paths!r}"
        )


def build_seed_tree(
    master_seed: int, n_regimes: int, n_paths: int
) -> list[list[PathSeeds]]:
    """Build the per-``(regime, path)`` seed tree.

    Parameters
    ----------
    master_seed:
        Non-negative integer master seed loaded from the YAML config.
    n_regimes:
        Number of volatility regimes (>= 1).
    n_paths:
        Number of Monte Carlo paths per regime (>= 1).

    Returns
    -------
    list[list[PathSeeds]]
        Nested list of shape ``[n_regimes][n_paths]``. Each leaf is a
        :class:`PathSeeds` with independent ``mid`` and ``fill`` seed
        sequences.

    Raises
    ------
    ValueError
        If any argument is out of range.
    """
    _validate_inputs(master_seed, n_regimes, n_paths)

    # Always spawn the full set of children at once. Spawning twice from the
    # same SeedSequence is not deterministic across all NumPy versions
    # (the spawn counter is updated on each call), so we do it exactly once.
    root = np.random.SeedSequence(master_seed)
    children = root.spawn(n_regimes + 1)
    # children[0..n_regimes-1] -> per-regime seed sequences
    # children[n_regimes]      -> reserved analytics seed (see analytics_seed)

    tree: list[list[PathSeeds]] = []
    for r in range(n_regimes):
        # Spawn one child per path under regime r.
        path_seqs = children[r].spawn(n_paths)
        regime_row: list[PathSeeds] = []
        for p in range(n_paths):
            # Final spawn(2) splits the per-path stream into mid/fill so the
            # mid-price simulator and the fill engine consume disjoint RNG
            # state (paired-comparison requirement, Req 7.2).
            mid_ss, fill_ss = path_seqs[p].spawn(2)
            regime_row.append(PathSeeds(mid=mid_ss, fill=fill_ss))
        tree.append(regime_row)
    return tree


def analytics_seed(master_seed: int, n_regimes: int) -> np.random.SeedSequence:
    """Return the reserved analytics ``SeedSequence``.

    The analytics seed lives at index ``n_regimes`` in the root ``spawn``
    layout. We rebuild the full ``n_regimes + 1`` spawn here (rather than
    calling ``spawn`` twice on the root) because spawning twice from the
    same ``SeedSequence`` is not deterministic across all NumPy releases.
    The result is always identical to ``build_seed_tree``'s reserved
    analytics child for the same ``(master_seed, n_regimes)``.

    Parameters
    ----------
    master_seed:
        Non-negative integer master seed loaded from the YAML config.
    n_regimes:
        Number of volatility regimes (>= 1).

    Returns
    -------
    np.random.SeedSequence
        Seed sequence dedicated to analytics randomization (e.g.
        paired-bootstrap CI sampling).

    Raises
    ------
    ValueError
        If any argument is out of range.
    """
    if not isinstance(master_seed, int) or isinstance(master_seed, bool):
        raise ValueError(
            f"master_seed: must be an int; got {master_seed!r}"
        )
    if master_seed < 0:
        raise ValueError(
            f"master_seed: must be >= 0; got {master_seed!r}"
        )
    if not isinstance(n_regimes, int) or isinstance(n_regimes, bool):
        raise ValueError(
            f"n_regimes: must be an int; got {n_regimes!r}"
        )
    if n_regimes < 1:
        raise ValueError(
            f"n_regimes: must be >= 1; got {n_regimes!r}"
        )

    return np.random.SeedSequence(master_seed).spawn(n_regimes + 1)[n_regimes]
