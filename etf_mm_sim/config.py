"""Configuration dataclasses and YAML I/O for the ETF MM simulator.

This module defines the immutable, frozen ``Configuration`` dataclass tree that
parameterizes every component of the simulator, plus the YAML loader, dumper,
and validator. The schema mirrors the YAML contract documented in ``design.md``
§Data Models and §Config_Loader.

Validation philosophy
---------------------
* All structural checks (unknown keys, missing required fields, type-mismatched
  YAML) are performed in :func:`load_config` before a ``Configuration`` is
  built. By the time :func:`validate` sees a ``Configuration`` it is already
  structurally well-formed; ``validate`` is responsible for *semantic*
  constraints (positivity, row-stochasticity, ...).
* All errors raised through :class:`ConfigError` use a single canonical
  message format::

      "<field_path>: <constraint>; got <value>"

  where ``<field_path>`` is the dotted YAML path to the offending field
  (e.g. ``mid_price.regimes[1].sigma``).

Design notes
------------
* Every dataclass is ``@dataclass(frozen=True)`` so configurations are
  immutable once constructed. This rules out a class of bugs in which a
  Monte Carlo run mutates its own configuration mid-flight.
* Collections are stored as ``tuple`` (never ``list``) so equality and hashing
  remain well-defined. In particular ``MidPriceConfig.transition_matrix`` is a
  ``tuple[tuple[float, ...], ...]`` to keep ``Configuration`` hashable; the
  numpy view is provided lazily through
  :pymeth:`MidPriceConfig.transition_matrix_array`.
* ``OutputConfig.format`` is typed as ``Literal["parquet"]``: per resolved
  design question 5, only Parquet is supported in this iteration. The
  ``Literal`` is kept for forward compatibility so future formats only require
  widening the alias.
"""

from __future__ import annotations

import math
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional

import numpy as np
import yaml

__all__ = [
    "RegimeParams",
    "MidPriceConfig",
    "HorizonConfig",
    "ASParams",
    "SymmetricParams",
    "FillParams",
    "RiskParams",
    "MCParams",
    "AnalyticsParams",
    "OutputConfig",
    "Configuration",
    "ConfigError",
    "load_config",
    "dump_config",
    "validate",
]


# --------------------------------------------------------------------------- #
# Dataclasses                                                                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RegimeParams:
    """Parameters for a single named volatility regime."""

    name: str
    mu: float
    sigma: float


@dataclass(frozen=True)
class MidPriceConfig:
    """Mid-price model selection and parameters.

    Parameters
    ----------
    model:
        Either ``"gbm"`` for a single-regime Geometric Brownian Motion or
        ``"regime_switching"`` for a Markov regime-switching GBM.
    s0:
        Initial mid-price.
    regimes:
        Tuple of :class:`RegimeParams`. Length 1 is allowed for the GBM model;
        the regime-switching model uses every entry.
    transition_matrix:
        Required iff ``model == "regime_switching"``; row-stochastic
        ``(R, R)`` matrix stored as a nested tuple to preserve hashability.
    """

    model: Literal["gbm", "regime_switching"]
    s0: float
    regimes: tuple[RegimeParams, ...]
    transition_matrix: Optional[tuple[tuple[float, ...], ...]]

    @property
    def transition_matrix_array(self) -> Optional[np.ndarray]:
        """Lazily materialize the transition matrix as a NumPy array."""
        if self.transition_matrix is None:
            return None
        return np.array(self.transition_matrix, dtype=np.float64)


@dataclass(frozen=True)
class HorizonConfig:
    """Simulation horizon and time step."""

    T: float
    dt: float

    @property
    def n_steps(self) -> int:
        """Number of inner steps in a path: ``ceil(T / dt)``."""
        return math.ceil(self.T / self.dt)


@dataclass(frozen=True)
class ASParams:
    """Avellaneda-Stoikov quoter parameters."""

    gamma: float
    k: float
    A: float


@dataclass(frozen=True)
class SymmetricParams:
    """Symmetric constant-spread baseline quoter parameters."""

    delta_base: float


@dataclass(frozen=True)
class FillParams:
    """Per-side fill-intensity parameters for the Poisson fill engine."""

    A_b: float
    k_b: float
    A_a: float
    k_a: float


@dataclass(frozen=True)
class RiskParams:
    """Risk-management parameters.

    ``L_kill`` is optional; ``None`` disables the running-P&L kill switch.
    """

    q_max: int
    L_kill: Optional[float]


@dataclass(frozen=True)
class MCParams:
    """Monte Carlo orchestration parameters."""

    n_paths: int


@dataclass(frozen=True)
class AnalyticsParams:
    """Analytics parameters (adverse selection horizon, bootstrap, Sharpe)."""

    adverse_selection_horizon_steps: int
    bootstrap_iterations: int
    bootstrap_alpha: float
    sharpe_annualization_factor: float


@dataclass(frozen=True)
class OutputConfig:
    """Output directory and on-disk format selector."""

    dir: pathlib.Path
    format: Literal["parquet"]


@dataclass(frozen=True)
class Configuration:
    """Top-level configuration aggregating every component's parameters."""

    master_seed: int
    horizon: HorizonConfig
    mid_price: MidPriceConfig
    quoters_as: ASParams
    quoters_sym: SymmetricParams
    fill: FillParams
    risk: RiskParams
    mc: MCParams
    analytics: AnalyticsParams
    output: OutputConfig


# --------------------------------------------------------------------------- #
# Errors                                                                      #
# --------------------------------------------------------------------------- #


class ConfigError(ValueError):
    """Raised by :func:`load_config` and :func:`validate` on invalid configs.

    The canonical message format is::

        "<field_path>: <constraint>; got <value>"

    For example::

        "mid_price.regimes[1].sigma: must be >= 0; got -0.5"
    """

    def __init__(self, field_path: str, constraint: str, value: Any) -> None:
        self.field_path = field_path
        self.constraint = constraint
        self.value = value
        super().__init__(f"{field_path}: {constraint}; got {value!r}")


# --------------------------------------------------------------------------- #
# YAML key schema                                                             #
# --------------------------------------------------------------------------- #
#
# These constants document the *exact* set of keys allowed at each YAML level.
# Any other key is rejected by ``load_config`` with ``ConfigError`` so that
# typos in user configs surface as loud, dotted-path errors rather than silent
# defaulting.

_TOP_LEVEL_KEYS = {
    "master_seed",
    "horizon",
    "mid_price",
    "quoters",
    "fill",
    "risk",
    "mc",
    "analytics",
    "output",
}
_HORIZON_KEYS = {"T", "dt"}
_MID_PRICE_KEYS = {"model", "s0", "regimes", "transition_matrix"}
_REGIME_KEYS = {"name", "mu", "sigma"}
_QUOTERS_KEYS = {"avellaneda_stoikov", "symmetric"}
_AS_KEYS = {"gamma", "k", "A"}
_SYM_KEYS = {"delta_base"}
_FILL_KEYS = {"A_b", "k_b", "A_a", "k_a"}
_RISK_KEYS = {"q_max", "L_kill"}
_MC_KEYS = {"n_paths"}
_ANALYTICS_KEYS = {
    "adverse_selection_horizon_steps",
    "bootstrap_iterations",
    "bootstrap_alpha",
    "sharpe_annualization_factor",
}
_OUTPUT_KEYS = {"dir", "format"}


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _require_mapping(node: Any, path: str) -> Mapping[str, Any]:
    """Return ``node`` if it is a mapping; else raise ``ConfigError``."""
    if not isinstance(node, Mapping):
        raise ConfigError(path, "must be a mapping", node)
    return node


def _check_unknown_keys(node: Mapping[str, Any], allowed: set[str], path: str) -> None:
    """Raise ``ConfigError`` if ``node`` contains keys outside ``allowed``."""
    extra = set(node.keys()) - allowed
    if extra:
        # Sort for deterministic error messages.
        unknown = sorted(extra)
        raise ConfigError(path, f"unknown key(s) {unknown}", sorted(node.keys()))


def _require_field(node: Mapping[str, Any], key: str, path: str) -> Any:
    """Return ``node[key]`` or raise ``ConfigError`` on missing field."""
    if key not in node:
        raise ConfigError(f"{path}.{key}" if path else key, "missing required field", None)
    return node[key]


def _coerce_int(value: Any, path: str) -> int:
    """Strict-int coercion: rejects bools and floats, accepts ints."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(path, "must be an int", value)
    return value


def _coerce_float(value: Any, path: str) -> float:
    """Numeric coercion: accepts int or float, rejects bool, NaN, inf."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(path, "must be a number", value)
    f = float(value)
    if math.isnan(f) or math.isinf(f):
        raise ConfigError(path, "must be finite", value)
    return f


def _coerce_str(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(path, "must be a string", value)
    return value


# --------------------------------------------------------------------------- #
# load_config                                                                 #
# --------------------------------------------------------------------------- #


def _parse_regime(node: Any, path: str) -> RegimeParams:
    m = _require_mapping(node, path)
    _check_unknown_keys(m, _REGIME_KEYS, path)
    name = _coerce_str(_require_field(m, "name", path), f"{path}.name")
    mu = _coerce_float(_require_field(m, "mu", path), f"{path}.mu")
    sigma = _coerce_float(_require_field(m, "sigma", path), f"{path}.sigma")
    return RegimeParams(name=name, mu=mu, sigma=sigma)


def _parse_mid_price(node: Any, path: str) -> MidPriceConfig:
    m = _require_mapping(node, path)
    _check_unknown_keys(m, _MID_PRICE_KEYS, path)
    model = _coerce_str(_require_field(m, "model", path), f"{path}.model")
    if model not in ("gbm", "regime_switching"):
        raise ConfigError(f"{path}.model", "must be 'gbm' or 'regime_switching'", model)
    s0 = _coerce_float(_require_field(m, "s0", path), f"{path}.s0")
    regimes_raw = _require_field(m, "regimes", path)
    if not isinstance(regimes_raw, list):
        raise ConfigError(f"{path}.regimes", "must be a list", regimes_raw)
    regimes = tuple(
        _parse_regime(r, f"{path}.regimes[{i}]") for i, r in enumerate(regimes_raw)
    )
    tm_raw = m.get("transition_matrix", None)
    transition_matrix: Optional[tuple[tuple[float, ...], ...]]
    if tm_raw is None:
        transition_matrix = None
    else:
        if not isinstance(tm_raw, list):
            raise ConfigError(
                f"{path}.transition_matrix", "must be a list of lists", tm_raw
            )
        rows: list[tuple[float, ...]] = []
        for i, row in enumerate(tm_raw):
            if not isinstance(row, list):
                raise ConfigError(
                    f"{path}.transition_matrix[{i}]", "must be a list", row
                )
            rows.append(
                tuple(
                    _coerce_float(v, f"{path}.transition_matrix[{i}][{j}]")
                    for j, v in enumerate(row)
                )
            )
        transition_matrix = tuple(rows)
    return MidPriceConfig(
        model=model,  # type: ignore[arg-type]
        s0=s0,
        regimes=regimes,
        transition_matrix=transition_matrix,
    )


def _parse_horizon(node: Any, path: str) -> HorizonConfig:
    m = _require_mapping(node, path)
    _check_unknown_keys(m, _HORIZON_KEYS, path)
    T = _coerce_float(_require_field(m, "T", path), f"{path}.T")
    dt = _coerce_float(_require_field(m, "dt", path), f"{path}.dt")
    return HorizonConfig(T=T, dt=dt)


def _parse_as(node: Any, path: str) -> ASParams:
    m = _require_mapping(node, path)
    _check_unknown_keys(m, _AS_KEYS, path)
    return ASParams(
        gamma=_coerce_float(_require_field(m, "gamma", path), f"{path}.gamma"),
        k=_coerce_float(_require_field(m, "k", path), f"{path}.k"),
        A=_coerce_float(_require_field(m, "A", path), f"{path}.A"),
    )


def _parse_sym(node: Any, path: str) -> SymmetricParams:
    m = _require_mapping(node, path)
    _check_unknown_keys(m, _SYM_KEYS, path)
    return SymmetricParams(
        delta_base=_coerce_float(
            _require_field(m, "delta_base", path), f"{path}.delta_base"
        ),
    )


def _parse_fill(node: Any, path: str) -> FillParams:
    m = _require_mapping(node, path)
    _check_unknown_keys(m, _FILL_KEYS, path)
    return FillParams(
        A_b=_coerce_float(_require_field(m, "A_b", path), f"{path}.A_b"),
        k_b=_coerce_float(_require_field(m, "k_b", path), f"{path}.k_b"),
        A_a=_coerce_float(_require_field(m, "A_a", path), f"{path}.A_a"),
        k_a=_coerce_float(_require_field(m, "k_a", path), f"{path}.k_a"),
    )


def _parse_risk(node: Any, path: str) -> RiskParams:
    m = _require_mapping(node, path)
    _check_unknown_keys(m, _RISK_KEYS, path)
    q_max = _coerce_int(_require_field(m, "q_max", path), f"{path}.q_max")
    if "L_kill" not in m:
        raise ConfigError(f"{path}.L_kill", "missing required field", None)
    L_kill_raw = m["L_kill"]
    L_kill: Optional[float]
    if L_kill_raw is None:
        L_kill = None
    else:
        L_kill = _coerce_float(L_kill_raw, f"{path}.L_kill")
    return RiskParams(q_max=q_max, L_kill=L_kill)


def _parse_mc(node: Any, path: str) -> MCParams:
    m = _require_mapping(node, path)
    _check_unknown_keys(m, _MC_KEYS, path)
    return MCParams(
        n_paths=_coerce_int(_require_field(m, "n_paths", path), f"{path}.n_paths"),
    )


def _parse_analytics(node: Any, path: str) -> AnalyticsParams:
    m = _require_mapping(node, path)
    _check_unknown_keys(m, _ANALYTICS_KEYS, path)
    return AnalyticsParams(
        adverse_selection_horizon_steps=_coerce_int(
            _require_field(m, "adverse_selection_horizon_steps", path),
            f"{path}.adverse_selection_horizon_steps",
        ),
        bootstrap_iterations=_coerce_int(
            _require_field(m, "bootstrap_iterations", path),
            f"{path}.bootstrap_iterations",
        ),
        bootstrap_alpha=_coerce_float(
            _require_field(m, "bootstrap_alpha", path), f"{path}.bootstrap_alpha"
        ),
        sharpe_annualization_factor=_coerce_float(
            _require_field(m, "sharpe_annualization_factor", path),
            f"{path}.sharpe_annualization_factor",
        ),
    )


def _parse_output(node: Any, path: str) -> OutputConfig:
    m = _require_mapping(node, path)
    _check_unknown_keys(m, _OUTPUT_KEYS, path)
    dir_str = _coerce_str(_require_field(m, "dir", path), f"{path}.dir")
    fmt = _coerce_str(_require_field(m, "format", path), f"{path}.format")
    if fmt != "parquet":
        raise ConfigError(f"{path}.format", "must be 'parquet'", fmt)
    return OutputConfig(dir=pathlib.Path(dir_str), format=fmt)  # type: ignore[arg-type]


def load_config(path: str | os.PathLike[str]) -> Configuration:
    """Load and validate a YAML configuration file.

    Parameters
    ----------
    path:
        Filesystem path to a YAML file matching the schema documented in
        ``design.md`` §Config_Loader.

    Returns
    -------
    Configuration
        A validated, frozen configuration tree.

    Raises
    ------
    ConfigError
        On any structural or semantic violation of the schema.
    """
    fs_path = pathlib.Path(path)
    with fs_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ConfigError("", "config file is empty", None)
    if not isinstance(raw, Mapping):
        raise ConfigError("", "top-level config must be a mapping", raw)

    _check_unknown_keys(raw, _TOP_LEVEL_KEYS, "")

    master_seed = _coerce_int(
        _require_field(raw, "master_seed", ""), "master_seed"
    )
    horizon = _parse_horizon(_require_field(raw, "horizon", ""), "horizon")
    mid_price = _parse_mid_price(_require_field(raw, "mid_price", ""), "mid_price")

    quoters_node = _require_field(raw, "quoters", "")
    quoters_map = _require_mapping(quoters_node, "quoters")
    _check_unknown_keys(quoters_map, _QUOTERS_KEYS, "quoters")
    quoters_as = _parse_as(
        _require_field(quoters_map, "avellaneda_stoikov", "quoters"),
        "quoters.avellaneda_stoikov",
    )
    quoters_sym = _parse_sym(
        _require_field(quoters_map, "symmetric", "quoters"),
        "quoters.symmetric",
    )

    fill = _parse_fill(_require_field(raw, "fill", ""), "fill")
    risk = _parse_risk(_require_field(raw, "risk", ""), "risk")
    mc = _parse_mc(_require_field(raw, "mc", ""), "mc")
    analytics = _parse_analytics(_require_field(raw, "analytics", ""), "analytics")
    output = _parse_output(_require_field(raw, "output", ""), "output")

    cfg = Configuration(
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
    validate(cfg)
    return cfg


# --------------------------------------------------------------------------- #
# dump_config                                                                 #
# --------------------------------------------------------------------------- #


def dump_config(cfg: Configuration, path: str | os.PathLike[str]) -> None:
    """Serialize ``cfg`` to YAML at ``path``.

    The dumped YAML round-trips exactly through :func:`load_config`: tuples
    are converted back to lists and ``pathlib.Path`` to a string.
    """
    fs_path = pathlib.Path(path)
    payload: dict[str, Any] = {
        "master_seed": cfg.master_seed,
        "horizon": {"T": cfg.horizon.T, "dt": cfg.horizon.dt},
        "mid_price": {
            "model": cfg.mid_price.model,
            "s0": cfg.mid_price.s0,
            "regimes": [
                {"name": r.name, "mu": r.mu, "sigma": r.sigma}
                for r in cfg.mid_price.regimes
            ],
            "transition_matrix": (
                None
                if cfg.mid_price.transition_matrix is None
                else [list(row) for row in cfg.mid_price.transition_matrix]
            ),
        },
        "quoters": {
            "avellaneda_stoikov": {
                "gamma": cfg.quoters_as.gamma,
                "k": cfg.quoters_as.k,
                "A": cfg.quoters_as.A,
            },
            "symmetric": {"delta_base": cfg.quoters_sym.delta_base},
        },
        "fill": {
            "A_b": cfg.fill.A_b,
            "k_b": cfg.fill.k_b,
            "A_a": cfg.fill.A_a,
            "k_a": cfg.fill.k_a,
        },
        "risk": {"q_max": cfg.risk.q_max, "L_kill": cfg.risk.L_kill},
        "mc": {"n_paths": cfg.mc.n_paths},
        "analytics": {
            "adverse_selection_horizon_steps": cfg.analytics.adverse_selection_horizon_steps,
            "bootstrap_iterations": cfg.analytics.bootstrap_iterations,
            "bootstrap_alpha": cfg.analytics.bootstrap_alpha,
            "sharpe_annualization_factor": cfg.analytics.sharpe_annualization_factor,
        },
        "output": {"dir": str(cfg.output.dir), "format": cfg.output.format},
    }
    with fs_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)


# --------------------------------------------------------------------------- #
# validate                                                                    #
# --------------------------------------------------------------------------- #


def validate(cfg: Configuration) -> None:
    """Validate semantic constraints on a structurally well-formed config.

    Raises :class:`ConfigError` on the first violation encountered. The error
    message uses dotted YAML paths (e.g. ``mid_price.regimes[1].sigma``).
    """
    # master_seed
    if not isinstance(cfg.master_seed, int) or isinstance(cfg.master_seed, bool):
        raise ConfigError("master_seed", "must be an int", cfg.master_seed)
    if cfg.master_seed < 0:
        raise ConfigError("master_seed", "must be >= 0", cfg.master_seed)

    # horizon
    if not (cfg.horizon.T > 0):
        raise ConfigError("horizon.T", "must be > 0", cfg.horizon.T)
    if not (cfg.horizon.dt > 0):
        raise ConfigError("horizon.dt", "must be > 0", cfg.horizon.dt)
    if cfg.horizon.dt > cfg.horizon.T:
        raise ConfigError("horizon.dt", "must be <= horizon.T", cfg.horizon.dt)

    # mid_price
    if not (cfg.mid_price.s0 > 0):
        raise ConfigError("mid_price.s0", "must be > 0", cfg.mid_price.s0)
    if len(cfg.mid_price.regimes) == 0:
        raise ConfigError("mid_price.regimes", "must be non-empty", cfg.mid_price.regimes)
    seen: set[str] = set()
    for i, r in enumerate(cfg.mid_price.regimes):
        if not r.name:
            raise ConfigError(
                f"mid_price.regimes[{i}].name", "must be non-empty", r.name
            )
        if r.name in seen:
            raise ConfigError(
                f"mid_price.regimes[{i}].name", "must be unique within regimes", r.name
            )
        seen.add(r.name)
        if r.sigma < 0:
            raise ConfigError(
                f"mid_price.regimes[{i}].sigma", "must be >= 0", r.sigma
            )

    if cfg.mid_price.model == "regime_switching":
        if cfg.mid_price.transition_matrix is None:
            raise ConfigError(
                "mid_price.transition_matrix",
                "required when model == 'regime_switching'",
                None,
            )
        R = len(cfg.mid_price.regimes)
        tm = cfg.mid_price.transition_matrix
        if len(tm) != R:
            raise ConfigError(
                "mid_price.transition_matrix",
                f"must have {R} rows to match number of regimes",
                len(tm),
            )
        for i, row in enumerate(tm):
            if len(row) != R:
                raise ConfigError(
                    f"mid_price.transition_matrix[{i}]",
                    f"must have {R} entries to match number of regimes",
                    len(row),
                )
            for j, v in enumerate(row):
                if not (0.0 <= v <= 1.0):
                    raise ConfigError(
                        f"mid_price.transition_matrix[{i}][{j}]",
                        "must be in [0, 1]",
                        v,
                    )
            row_sum = math.fsum(row)
            if not math.isclose(row_sum, 1.0, abs_tol=1e-9):
                raise ConfigError(
                    f"mid_price.transition_matrix[{i}]",
                    "row must sum to 1 within tolerance 1e-9",
                    row_sum,
                )
    elif cfg.mid_price.model == "gbm":
        if cfg.mid_price.transition_matrix is not None:
            raise ConfigError(
                "mid_price.transition_matrix",
                "must be null when model == 'gbm'",
                cfg.mid_price.transition_matrix,
            )
        if len(cfg.mid_price.regimes) != 1:
            raise ConfigError(
                "mid_price.regimes",
                "must have exactly 1 regime when model == 'gbm'",
                len(cfg.mid_price.regimes),
            )
    else:  # pragma: no cover - load_config rejects this earlier
        raise ConfigError(
            "mid_price.model", "must be 'gbm' or 'regime_switching'", cfg.mid_price.model
        )

    # quoters_as
    if not (cfg.quoters_as.gamma > 0):
        raise ConfigError(
            "quoters.avellaneda_stoikov.gamma", "must be > 0", cfg.quoters_as.gamma
        )
    if not (cfg.quoters_as.k > 0):
        raise ConfigError(
            "quoters.avellaneda_stoikov.k", "must be > 0", cfg.quoters_as.k
        )
    if not (cfg.quoters_as.A > 0):
        raise ConfigError(
            "quoters.avellaneda_stoikov.A", "must be > 0", cfg.quoters_as.A
        )

    # quoters_sym
    if cfg.quoters_sym.delta_base < 0:
        raise ConfigError(
            "quoters.symmetric.delta_base",
            "must be >= 0",
            cfg.quoters_sym.delta_base,
        )

    # fill
    if not (cfg.fill.A_b > 0):
        raise ConfigError("fill.A_b", "must be > 0", cfg.fill.A_b)
    if not (cfg.fill.k_b > 0):
        raise ConfigError("fill.k_b", "must be > 0", cfg.fill.k_b)
    if not (cfg.fill.A_a > 0):
        raise ConfigError("fill.A_a", "must be > 0", cfg.fill.A_a)
    if not (cfg.fill.k_a > 0):
        raise ConfigError("fill.k_a", "must be > 0", cfg.fill.k_a)

    # risk
    if not isinstance(cfg.risk.q_max, int) or isinstance(cfg.risk.q_max, bool):
        raise ConfigError("risk.q_max", "must be an int", cfg.risk.q_max)
    if cfg.risk.q_max < 0:
        raise ConfigError("risk.q_max", "must be >= 0", cfg.risk.q_max)
    if cfg.risk.L_kill is not None and not (cfg.risk.L_kill > 0):
        raise ConfigError(
            "risk.L_kill", "must be None or > 0", cfg.risk.L_kill
        )

    # mc
    if cfg.mc.n_paths < 1:
        raise ConfigError("mc.n_paths", "must be >= 1", cfg.mc.n_paths)

    # analytics
    if cfg.analytics.adverse_selection_horizon_steps < 0:
        raise ConfigError(
            "analytics.adverse_selection_horizon_steps",
            "must be >= 0",
            cfg.analytics.adverse_selection_horizon_steps,
        )
    if cfg.analytics.bootstrap_iterations < 1:
        raise ConfigError(
            "analytics.bootstrap_iterations",
            "must be >= 1",
            cfg.analytics.bootstrap_iterations,
        )
    if not (0.0 < cfg.analytics.bootstrap_alpha < 1.0):
        raise ConfigError(
            "analytics.bootstrap_alpha",
            "must be in (0, 1)",
            cfg.analytics.bootstrap_alpha,
        )
    if not (cfg.analytics.sharpe_annualization_factor > 0):
        raise ConfigError(
            "analytics.sharpe_annualization_factor",
            "must be > 0",
            cfg.analytics.sharpe_annualization_factor,
        )

    # output
    if not isinstance(cfg.output.dir, pathlib.Path):
        raise ConfigError(
            "output.dir", "must be a pathlib.Path", cfg.output.dir
        )
