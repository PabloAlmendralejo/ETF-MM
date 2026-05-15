"""Smoke test that executes the default backtest notebook end-to-end.

# Feature: etf-mm-arbitrage-simulator
# Validates: Requirements 9.4

Uses :class:`nbclient.NotebookClient` to run every cell of
``notebooks/default_backtest.ipynb``. The notebook is configured to
reduce ``n_paths`` so it completes inside the 60-second timeout in CI.
"""

from __future__ import annotations

import pathlib

import nbformat
from nbclient import NotebookClient


_WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parents[2]
_NOTEBOOK_PATH = _WORKSPACE_ROOT / "notebooks" / "default_backtest.ipynb"


def test_default_backtest_notebook_executes() -> None:
    """Every cell of ``default_backtest.ipynb`` runs without raising."""
    assert _NOTEBOOK_PATH.is_file(), f"missing notebook: {_NOTEBOOK_PATH}"

    nb = nbformat.read(str(_NOTEBOOK_PATH), as_version=4)
    client = NotebookClient(
        nb,
        timeout=60,
        # Notebook uses paths relative to ``notebooks/`` so we run it
        # from there.
        resources={"metadata": {"path": str(_NOTEBOOK_PATH.parent)}},
    )
    # NotebookClient.execute raises CellExecutionError if any cell
    # raises; letting it propagate gives us a useful pytest failure
    # message that pinpoints the offending cell.
    client.execute()
