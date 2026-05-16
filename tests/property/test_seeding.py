"""Property tests for the seed-tree builder.

# Feature: etf-mm-arbitrage-simulator, Property 23: SeedSequence tree determinism
# Validates: Requirements 10.3
"""

from __future__ import annotations

from hypothesis import given, settings, strategies as st

from etf_mm_sim.seeding import PathSeeds, analytics_seed, build_seed_tree


@given(
    master_seed=st.integers(min_value=0, max_value=(1 << 31) - 1),
    n_regimes=st.integers(min_value=1, max_value=4),
    n_paths=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=25, deadline=None)
def test_property_23_seed_tree_determinism(
    master_seed: int, n_regimes: int, n_paths: int
) -> None:
    """Two identical builds yield leaves with identical entropy and spawn_key.

    **Validates: Requirements 10.3**
    """
    a = build_seed_tree(master_seed, n_regimes, n_paths)
    b = build_seed_tree(master_seed, n_regimes, n_paths)
    assert len(a) == len(b) == n_regimes
    for r in range(n_regimes):
        assert len(a[r]) == len(b[r]) == n_paths
        for p in range(n_paths):
            assert isinstance(a[r][p], PathSeeds)
            assert a[r][p].mid.entropy == b[r][p].mid.entropy
            assert a[r][p].mid.spawn_key == b[r][p].mid.spawn_key
            assert a[r][p].fill.entropy == b[r][p].fill.entropy
            assert a[r][p].fill.spawn_key == b[r][p].fill.spawn_key

    # analytics seed determinism + independence from leaves
    s1 = analytics_seed(master_seed, n_regimes)
    s2 = analytics_seed(master_seed, n_regimes)
    assert s1.entropy == s2.entropy
    assert s1.spawn_key == s2.spawn_key


def test_property_23_analytics_seed_disjoint_from_leaves(
    master_seed: int = 12345, n_regimes: int = 3, n_paths: int = 4
) -> None:
    """The analytics seed's spawn_key differs from every per-path leaf.

    Sanity check that we are not aliasing the reserved analytics slot with
    any of the per-regime/per-path leaves.

    **Validates: Requirements 10.3**
    """
    tree = build_seed_tree(master_seed, n_regimes, n_paths)
    a = analytics_seed(master_seed, n_regimes)
    leaf_keys = {
        (seeds.mid.spawn_key, seeds.fill.spawn_key)
        for row in tree
        for seeds in row
    }
    # Trivially true (a tuple of two different objects vs. (a, a)) but kept
    # for documentation; the tighter check is the per-leaf loop below.
    assert (a.spawn_key, a.spawn_key) not in leaf_keys
    for row in tree:
        for seeds in row:
            assert seeds.mid.spawn_key != a.spawn_key
            assert seeds.fill.spawn_key != a.spawn_key
