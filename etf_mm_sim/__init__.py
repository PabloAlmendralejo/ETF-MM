"""ETF Market Making & Arbitrage Simulator (Mode 1).

A vectorized-NumPy Python package that runs Monte Carlo backtests of an
Avellaneda-Stoikov (AS) quoter against a symmetric constant-spread baseline on
synthetic mid-price paths, and emits paired P&L analytics with drawdown
attribution.

This module exposes the package version. Public API surfaces are added by the
individual submodules as they are implemented.
"""

__version__ = "0.1.0"

__all__: list[str] = []
