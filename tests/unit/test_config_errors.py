"""Unit tests for missing-field and out-of-range Config_Loader errors.

Validates: Requirements 1.3, 1.4 (Config_Loader raises descriptive errors
naming the field on missing or out-of-range values).

Each test loads a base valid YAML, mutates *one* field, and asserts that
``ConfigError`` is raised with a substring matching the offending field path.
"""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import pytest
import yaml

from etf_mm_sim.config import ConfigError, load_config


# Reference valid YAML payload. All error tests start from a deep copy of this
# dict, mutate a single field, and dump it to a tmp YAML file.
_BASE_VALID: dict[str, Any] = {
    "master_seed": 20240101,
    "horizon": {"T": 1.0, "dt": 0.001},
    "mid_price": {
        "model": "regime_switching",
        "s0": 100.0,
        "regimes": [
            {"name": "low", "mu": 0.0, "sigma": 0.5},
            {"name": "normal", "mu": 0.0, "sigma": 1.0},
        ],
        "transition_matrix": [
            [0.99, 0.01],
            [0.02, 0.98],
        ],
    },
    "quoters": {
        "avellaneda_stoikov": {"gamma": 0.1, "k": 1.5, "A": 140.0},
        "symmetric": {"delta_base": 0.05},
    },
    "fill": {"A_b": 140.0, "k_b": 1.5, "A_a": 140.0, "k_a": 1.5},
    "risk": {"q_max": 50, "L_kill": 1000.0},
    "mc": {"n_paths": 1000},
    "analytics": {
        "adverse_selection_horizon_steps": 50,
        "bootstrap_iterations": 10000,
        "bootstrap_alpha": 0.05,
        "sharpe_annualization_factor": 15.874507866387544,
    },
    "output": {"dir": "results", "format": "parquet"},
}


@pytest.fixture
def base_payload() -> dict[str, Any]:
    """Fresh deep copy of the reference valid YAML payload."""
    return copy.deepcopy(_BASE_VALID)


def _write(tmp_path: pathlib.Path, payload: dict[str, Any]) -> pathlib.Path:
    p = tmp_path / "cfg.yaml"
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return p


def test_baseline_loads_clean(tmp_path, base_payload) -> None:
    """Sanity check: the unmodified reference payload must load without error."""
    p = _write(tmp_path, base_payload)
    cfg = load_config(p)
    assert cfg.master_seed == 20240101


def test_missing_top_level_field_mc(tmp_path, base_payload) -> None:
    del base_payload["mc"]
    p = _write(tmp_path, base_payload)
    with pytest.raises(ConfigError, match=r"^mc:"):
        load_config(p)


def test_unknown_top_level_key(tmp_path, base_payload) -> None:
    base_payload["surprise"] = 42
    p = _write(tmp_path, base_payload)
    with pytest.raises(ConfigError, match=r"unknown key"):
        load_config(p)


def test_unknown_nested_key_under_mid_price(tmp_path, base_payload) -> None:
    base_payload["mid_price"]["typo"] = "oops"
    p = _write(tmp_path, base_payload)
    with pytest.raises(ConfigError, match=r"^mid_price:.*unknown key"):
        load_config(p)


def test_negative_sigma(tmp_path, base_payload) -> None:
    base_payload["mid_price"]["regimes"][1]["sigma"] = -0.5
    p = _write(tmp_path, base_payload)
    with pytest.raises(ConfigError, match=r"mid_price\.regimes\[1\]\.sigma"):
        load_config(p)


def test_non_positive_k_avellaneda_stoikov(tmp_path, base_payload) -> None:
    base_payload["quoters"]["avellaneda_stoikov"]["k"] = 0.0
    p = _write(tmp_path, base_payload)
    with pytest.raises(
        ConfigError, match=r"quoters\.avellaneda_stoikov\.k"
    ):
        load_config(p)


def test_negative_q_max(tmp_path, base_payload) -> None:
    base_payload["risk"]["q_max"] = -1
    p = _write(tmp_path, base_payload)
    with pytest.raises(ConfigError, match=r"risk\.q_max"):
        load_config(p)


def test_transition_matrix_row_does_not_sum_to_one(tmp_path, base_payload) -> None:
    base_payload["mid_price"]["transition_matrix"] = [
        [0.5, 0.4],  # sums to 0.9
        [0.02, 0.98],
    ]
    p = _write(tmp_path, base_payload)
    with pytest.raises(
        ConfigError, match=r"mid_price\.transition_matrix\[0\]"
    ):
        load_config(p)


def test_transition_matrix_wrong_shape(tmp_path, base_payload) -> None:
    base_payload["mid_price"]["transition_matrix"] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    p = _write(tmp_path, base_payload)
    with pytest.raises(ConfigError, match=r"mid_price\.transition_matrix"):
        load_config(p)


def test_gbm_with_transition_matrix_supplied(tmp_path, base_payload) -> None:
    base_payload["mid_price"]["model"] = "gbm"
    # gbm requires exactly 1 regime, so trim regimes to one to isolate the
    # transition_matrix violation.
    base_payload["mid_price"]["regimes"] = [
        {"name": "only", "mu": 0.0, "sigma": 1.0}
    ]
    base_payload["mid_price"]["transition_matrix"] = [[1.0]]
    p = _write(tmp_path, base_payload)
    with pytest.raises(
        ConfigError, match=r"mid_price\.transition_matrix.*gbm"
    ):
        load_config(p)


def test_regime_switching_without_transition_matrix(tmp_path, base_payload) -> None:
    base_payload["mid_price"]["transition_matrix"] = None
    p = _write(tmp_path, base_payload)
    with pytest.raises(
        ConfigError, match=r"mid_price\.transition_matrix.*regime_switching"
    ):
        load_config(p)


def test_dt_greater_than_T(tmp_path, base_payload) -> None:
    base_payload["horizon"] = {"T": 1.0, "dt": 2.0}
    p = _write(tmp_path, base_payload)
    with pytest.raises(ConfigError, match=r"horizon\.dt"):
        load_config(p)


def test_bootstrap_alpha_outside_unit_interval(tmp_path, base_payload) -> None:
    base_payload["analytics"]["bootstrap_alpha"] = 1.5
    p = _write(tmp_path, base_payload)
    with pytest.raises(ConfigError, match=r"analytics\.bootstrap_alpha"):
        load_config(p)
