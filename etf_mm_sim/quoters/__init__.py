"""Quoter strategies.

Each quoter is a callable `quote(s, q, t) -> (p_b, p_a, suppress_b, suppress_a)`
plus a vectorized `precompute(s_path, dt, T, params)` helper. The two concrete
quoters are the Avellaneda-Stoikov inventory-aware quoter and the symmetric
constant-spread baseline. See `design.md` §Quoters.
"""

__all__: list[str] = []
