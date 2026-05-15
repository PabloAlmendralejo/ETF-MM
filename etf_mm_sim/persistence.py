"""Deterministic Parquet and JSON persistence.

Writes per-path time series, fill events, the resolved YAML config, and
a JSON run manifest under a timestamped run directory. Parquet options
are pinned so the *data* files are bit-identical across two runs of the
same config (Req 7.5, Property 18). The run timestamp lives only in the
directory name and in the ``manifest.json`` so the file *contents* of the
data files do not depend on wall-clock time.

Directory layout (mirrors ``design.md`` §File Output Format)::

    {output.dir}/run_{utc_iso}_{cfg_sha[:8]}/
    ├── config.resolved.yaml
    ├── manifest.json
    ├── paths/
    │   ├── avellaneda_stoikov/regime=<name>.parquet
    │   └── symmetric/regime=<name>.parquet
    └── fills/
        ├── avellaneda_stoikov/regime=<name>.parquet
        └── symmetric/regime=<name>.parquet

Parquet determinism contract
----------------------------
Every Parquet file is written with ``compression="zstd"``,
``compression_level=3``, ``use_dictionary=False``, ``row_group_size =
cfg.mc.n_paths``, and columns are written in alphabetical order. Writes
are atomic (write-to-tmp + ``os.replace``) so partial files never appear
on disk.

Schemas
-------
``paths/<strategy>/regime=<name>.parquet``: one row per (path_index,
step_index). Columns (alphabetical): ``ask_quote, bid_quote, cash,
inventory, mid_price, path_index, step_index``. ``ask_quote`` and
``bid_quote`` carry ``NaN`` at suppressed steps.

``fills/<strategy>/regime=<name>.parquet``: one row per realized fill
event. Columns (alphabetical): ``fill_price, path_index, side,
step_index``. ``side`` is ``"bid"`` or ``"ask"``.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import pathlib
import platform
import sys
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import __version__
from .config import Configuration, dump_config

if TYPE_CHECKING:
    from .backtest import BacktestResult
    from .path_runner import PathResult

__all__ = ["PersistenceError", "persist"]


class PersistenceError(IOError):
    """Raised when a persistence I/O operation fails.

    Wraps the underlying exception so callers can catch a single error
    type regardless of whether the failure originated in tmp-file
    creation, Parquet writing, or atomic rename.
    """


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #


_PATHS_COLUMNS = (
    "ask_quote",
    "bid_quote",
    "cash",
    "inventory",
    "mid_price",
    "path_index",
    "step_index",
)
_FILLS_COLUMNS = ("fill_price", "path_index", "side", "step_index")


def _atomic_write_parquet(
    table: pa.Table, dest: pathlib.Path, row_group_size: int
) -> None:
    """Write ``table`` to ``dest`` atomically with the determinism contract."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=dest.name + ".", suffix=".tmp", dir=str(dest.parent)
    )
    os.close(fd)
    tmp_path = pathlib.Path(tmp_name)
    try:
        pq.write_table(
            table,
            str(tmp_path),
            compression="zstd",
            compression_level=3,
            use_dictionary=False,
            # Pin row group size so the row-group boundary structure is
            # deterministic across runs of the same config.
            row_group_size=row_group_size,
        )
        os.replace(tmp_path, dest)
    except Exception as exc:  # pragma: no cover - error wrapping
        # Clean up the tmp file on any failure so we don't leave debris.
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise PersistenceError(f"failed to write {dest}: {exc}") from exc


def _paths_table(cell: list["PathResult"]) -> pa.Table:
    """Build the long-format paths table for one ``(strategy, regime)`` cell.

    Rows are grouped by ``path_index`` then ``step_index``, which keeps
    the on-disk layout aligned with the row-group boundary at
    ``cfg.mc.n_paths`` once paths are flattened.
    """
    # Concatenate per-path arrays in path_index order. Each path
    # contributes (n+1) rows.
    if not cell:
        # Empty cell would be a programming error; keep a defensive guard.
        raise PersistenceError("paths cell is empty; cannot build table")

    n_plus_1 = int(cell[0].s_path.shape[0])
    n_paths = len(cell)
    total_rows = n_plus_1 * n_paths

    ask_quote = np.empty(total_rows, dtype=np.float64)
    bid_quote = np.empty(total_rows, dtype=np.float64)
    cash = np.empty(total_rows, dtype=np.float64)
    inventory = np.empty(total_rows, dtype=np.int64)
    mid_price = np.empty(total_rows, dtype=np.float64)
    path_index = np.empty(total_rows, dtype=np.int64)
    step_index = np.empty(total_rows, dtype=np.int64)

    step_ids = np.arange(n_plus_1, dtype=np.int64)
    for k, pr in enumerate(cell):
        if int(pr.s_path.shape[0]) != n_plus_1:
            raise PersistenceError(
                "inconsistent path length across cell; "
                f"expected {n_plus_1}, got {pr.s_path.shape[0]}"
            )
        lo = k * n_plus_1
        hi = lo + n_plus_1
        ask_quote[lo:hi] = pr.ask_quote
        bid_quote[lo:hi] = pr.bid_quote
        cash[lo:hi] = pr.cash
        inventory[lo:hi] = pr.inventory
        mid_price[lo:hi] = pr.s_path
        path_index[lo:hi] = pr.path_index
        step_index[lo:hi] = step_ids

    # Columns in alphabetical order, per the determinism contract.
    return pa.table(
        {
            "ask_quote": ask_quote,
            "bid_quote": bid_quote,
            "cash": cash,
            "inventory": inventory,
            "mid_price": mid_price,
            "path_index": path_index,
            "step_index": step_index,
        }
    )


def _fills_table(cell: list["PathResult"]) -> pa.Table:
    """Build the long-format fills table for one ``(strategy, regime)`` cell.

    The schema is preserved even when the cell has zero realized fills:
    callers always get a well-formed Parquet file with the documented
    columns, so downstream consumers do not need conditional schema
    handling.
    """
    fill_prices: list[float] = []
    path_indices: list[int] = []
    sides: list[str] = []
    step_indices: list[int] = []

    for pr in cell:
        # Iterate steps where a fill resolved on each side. ``bid_fill``
        # and ``ask_fill`` are length n; the recorded quote price at the
        # same step is the fill price (Req 5.4 / 5.5).
        bid_steps = np.flatnonzero(pr.bid_fill)
        ask_steps = np.flatnonzero(pr.ask_fill)
        for i in bid_steps.tolist():
            fill_prices.append(float(pr.bid_quote[i]))
            path_indices.append(int(pr.path_index))
            sides.append("bid")
            step_indices.append(int(i))
        for i in ask_steps.tolist():
            fill_prices.append(float(pr.ask_quote[i]))
            path_indices.append(int(pr.path_index))
            sides.append("ask")
            step_indices.append(int(i))

    return pa.table(
        {
            "fill_price": pa.array(fill_prices, type=pa.float64()),
            "path_index": pa.array(path_indices, type=pa.int64()),
            "side": pa.array(sides, type=pa.string()),
            "step_index": pa.array(step_indices, type=pa.int64()),
        }
    )


def _utc_iso_timestamp() -> str:
    """Return a filesystem-safe UTC ISO timestamp.

    Format: ``YYYYMMDDTHHMMSSZ``. Colons are omitted because they are
    illegal in Windows path segments.
    """
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _config_sha256_hex(yaml_bytes: bytes) -> str:
    """Return the SHA-256 hex digest of the dumped config YAML bytes."""
    return hashlib.sha256(yaml_bytes).hexdigest()


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #


def persist(
    result: "BacktestResult",
    output_dir: pathlib.Path | str | None = None,
) -> pathlib.Path:
    """Persist a :class:`BacktestResult` to disk.

    Parameters
    ----------
    result:
        The full sweep output to persist.
    output_dir:
        Optional override for the output base directory. Defaults to
        ``result.config.output.dir`` from the configuration.

    Returns
    -------
    pathlib.Path
        The run output directory ``run_<utc_iso>_<cfg_sha[:8]>``. Parents
        are created as needed.

    Raises
    ------
    PersistenceError
        On any I/O failure (Parquet write, atomic rename, manifest, etc.).
    """
    cfg = result.config

    base_dir = (
        pathlib.Path(output_dir) if output_dir is not None else cfg.output.dir
    )
    base_dir = pathlib.Path(base_dir)

    # --- 1) Resolved-config bytes drive the directory hash. ----------- #
    # We dump_config to a temp file inside the (yet-to-be-created) base
    # dir's parent so the bytes captured for the manifest are exactly the
    # bytes written to disk.
    base_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Dump to an in-memory tmp first so we can hash *before* deciding
        # the run directory name. Using a tmp file under base_dir keeps
        # any failure contained.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".yaml",
            prefix="cfg.",
            dir=str(base_dir),
            delete=False,
        ) as tf:
            tmp_yaml_name = tf.name
        try:
            dump_config(cfg, tmp_yaml_name)
            yaml_bytes = pathlib.Path(tmp_yaml_name).read_bytes()
        finally:
            try:
                pathlib.Path(tmp_yaml_name).unlink(missing_ok=True)
            except OSError:
                pass
    except OSError as exc:
        raise PersistenceError(
            f"failed to materialize resolved config: {exc}"
        ) from exc

    cfg_sha_hex = _config_sha256_hex(yaml_bytes)
    timestamp = _utc_iso_timestamp()
    run_dirname = f"run_{timestamp}_{cfg_sha_hex[:8]}"
    run_dir = base_dir / run_dirname

    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        # Two backtests landing in the same UTC second on the same config
        # is exceedingly unlikely but handle gracefully by appending the
        # process id; this is purely a uniqueness fallback and does not
        # affect determinism (the data files are still byte-identical).
        run_dir = base_dir / f"{run_dirname}_{os.getpid()}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except OSError as exc:  # pragma: no cover - exotic
            raise PersistenceError(
                f"failed to create run directory {run_dir}: {exc}"
            ) from exc
    except OSError as exc:
        raise PersistenceError(
            f"failed to create run directory {run_dir}: {exc}"
        ) from exc

    # --- 2) Write resolved config bytes verbatim into run dir. -------- #
    cfg_yaml_path = run_dir / "config.resolved.yaml"
    try:
        cfg_yaml_path.write_bytes(yaml_bytes)
    except OSError as exc:
        raise PersistenceError(
            f"failed to write {cfg_yaml_path}: {exc}"
        ) from exc

    # --- 3) Per-(strategy, regime) parquet files. --------------------- #
    n_paths = cfg.mc.n_paths
    row_group_size = max(1, n_paths)
    regime_names: list[str] = []

    # Determinism contract: iterate strategies and regimes in fixed order.
    strategies_in_results = sorted({s for (s, _r) in result.paths.keys()})
    # Cross-check against the cfg-defined regimes so the order matches the
    # YAML configuration (which is also the insertion order in run_backtest).
    for r in cfg.mid_price.regimes:
        regime_names.append(r.name)

    for strategy in strategies_in_results:
        for regime_name in regime_names:
            key = (strategy, regime_name)
            if key not in result.paths:
                # If a (strategy, regime) cell is missing the backtest
                # contract was violated; surface a hard error rather than
                # silently dropping the file.
                raise PersistenceError(
                    f"missing cell {key!r} in BacktestResult.paths"
                )
            cell = result.paths[key]
            if len(cell) != n_paths:
                raise PersistenceError(
                    f"cell {key!r} has {len(cell)} paths; expected {n_paths}"
                )

            paths_dest = (
                run_dir / "paths" / strategy / f"regime={regime_name}.parquet"
            )
            fills_dest = (
                run_dir / "fills" / strategy / f"regime={regime_name}.parquet"
            )

            paths_tbl = _paths_table(cell)
            fills_tbl = _fills_table(cell)

            # Sanity-check column order matches the contract.
            if tuple(paths_tbl.column_names) != _PATHS_COLUMNS:
                raise PersistenceError(
                    "paths table column order violates determinism contract: "
                    f"got {paths_tbl.column_names!r}"
                )
            if tuple(fills_tbl.column_names) != _FILLS_COLUMNS:
                raise PersistenceError(
                    "fills table column order violates determinism contract: "
                    f"got {fills_tbl.column_names!r}"
                )

            _atomic_write_parquet(paths_tbl, paths_dest, row_group_size)
            # Fills tables can be smaller than n_paths; clamp the row
            # group size to the actual row count to avoid an empty
            # second row group at the boundary.
            fills_rows = max(1, fills_tbl.num_rows)
            _atomic_write_parquet(
                fills_tbl, fills_dest, min(row_group_size, fills_rows)
            )

    # --- 4) Manifest. ------------------------------------------------- #
    # Property 18 / Req 7.5 require the *data* files to be bit-identical
    # across runs. The manifest carries metadata that may legitimately
    # vary (the run timestamp). Consumers comparing two runs should
    # exclude the ``run_timestamp_utc`` key when checking byte-equality.
    n_steps = cfg.horizon.n_steps
    manifest: dict[str, object] = {
        "cfg_sha256": cfg_sha_hex,
        "master_seed": cfg.master_seed,
        "n_paths": int(n_paths),
        "n_regimes": int(len(cfg.mid_price.regimes)),
        "n_steps": int(n_steps),
        "numpy_version": np.__version__,
        "package_version": __version__,
        "pyarrow_version": pa.__version__,
        "python_version": platform.python_version(),
        "regime_names": list(regime_names),
        "run_timestamp_utc": timestamp,
    }
    manifest_path = run_dir / "manifest.json"
    try:
        manifest_bytes = (
            json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")
            + b"\n"
        )
        # Atomic-replace pattern to keep partial writes off-disk.
        fd, tmp_name = tempfile.mkstemp(
            prefix="manifest.", suffix=".json.tmp", dir=str(run_dir)
        )
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(manifest_bytes)
            os.replace(tmp_name, manifest_path)
        except Exception:  # pragma: no cover - error wrapping
            try:
                pathlib.Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass
            raise
    except OSError as exc:
        raise PersistenceError(
            f"failed to write {manifest_path}: {exc}"
        ) from exc

    # Touch sys reference to keep static analysis content (manifest uses
    # platform/sys data) — guarded so unused imports don't drift.
    _ = sys.version_info

    return run_dir
