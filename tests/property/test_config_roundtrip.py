"""Property test for Configuration YAML round-trip.

# Feature: etf-mm-arbitrage-simulator, Property 1: Configuration round-trip
# Validates: Requirements 1.5
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings

from etf_mm_sim.config import dump_config, load_config

from .strategies import valid_configurations


@given(cfg=valid_configurations())
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_property_1_configuration_round_trip(cfg, tmp_path) -> None:
    """Serializing then loading a valid config yields an equal config."""
    p = tmp_path / "cfg.yaml"
    dump_config(cfg, p)
    loaded = load_config(p)
    assert loaded == cfg
