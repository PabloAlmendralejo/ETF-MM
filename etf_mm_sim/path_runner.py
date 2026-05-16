"""Per-path step loop (pure-Python implementation).

Drives the sequential phase of the simulation: consumes pre-computed quoter
schedules and pre-drawn uniform fill draws, applies risk controls, resolves
Bernoulli fills, and emits a :class:`PathResult` summarizing the path. See
``design.md`` §Path_Runner and §Vectorization Strategy.

Strategy dispatch
-----------------
``run_path`` supports three strategies:

* ``"avellaneda_stoikov"`` -- live re-skewing per step. The caller supplies
  the time-varying half-spread schedule ``delta_star_sched`` and inventory-
  skew coefficient schedule ``skew_coef_sched`` produced by
  :func:`etf_mm_sim.quoters.avellaneda_stoikov.precompute`, plus the
  AS-side suppression mask (typically ``t_i >= T``). At each step the
  posted quotes are computed from the *current* inventory ``q``::

      r_i   = s_path[i] - q * skew_coef_sched[i]
      bid_i = r_i - delta_star_sched[i]
      ask_i = r_i + delta_star_sched[i]

* ``"symmetric"`` -- inventory-independent constant-spread quotes. The
  caller supplies a single ``delta_base`` scalar; quotes are simply
  ``s_path[i] - delta_base`` and ``s_path[i] + delta_base``.

* ``"semi_as"`` -- AS dynamic half-spread schedule with **no inventory
  skew**. The caller supplies the same ``delta_star_sched`` and
  ``as_suppress_mask`` as for AS. Internally this branch is implemented
  as the AS recurrence with the inventory-skew coefficient identically
  zero (``skew_coef_sched = 0``); the quote midpoint therefore equals
  ``s_path[i]`` at every non-suppressed step regardless of ``q``. This
  gives a third strategy paired on the same mid-price seeds whose only
  difference from AS is the absence of inventory skewing -- the cleanest
  isolation of the skew effect.

In both cases the *risk-manager* layer additionally suppresses sides when
the inventory bound is hit (``q >= q_max`` for the bid, ``q <= -q_max``
for the ask) or when the kill-switch has fired. Once the kill-switch
fires it is latched for the remainder of the path; both sides remain
suppressed regardless of subsequent P&L.

Length conventions
------------------
Following the requirements (Req 2.4) and the design's data-model section:

* ``s_path`` and the per-step schedules have length ``N + 1`` -- one
  observation per step grid point ``t_i = i * dt`` for ``i in {0, ..., N}``.
* ``u_b`` and ``u_a`` have length ``N`` -- one per *inner* step
  ``i in {0, ..., N - 1}`` because the terminal grid point ``i == N``
  does not advance state.
* The emitted ``bid_quote``, ``ask_quote``, ``inventory``, ``cash`` arrays
  have length ``N + 1``; ``bid_fill`` and ``ask_fill`` have length ``N``.

Terminal cash and ``terminal_pnl``
----------------------------------
``cash[N]`` is the cash balance *before* the terminal-flatten adjustment.
The reported ``terminal_pnl`` is ``cash[N] + inventory[N] * s_path[N]``
which equals :func:`etf_mm_sim.risk_manager.flatten_terminal` applied to
the un-flattened cash. We deliberately keep ``cash[N]`` un-flattened so
the per-step recurrence ``cash[i+1] = cash[i] + dCash`` is uninterrupted
across the boundary; consumers that want the post-flatten cash should
read ``terminal_pnl`` directly.

Performance
-----------
The per-step loop is intentionally tight: we read schedules into local
floats, inline the Bernoulli probability via :func:`math.expm1`, and
avoid all numpy scalar dispatch inside the hot path. On a modern laptop
this takes ~1-2 us per step, comfortably within the 5-minute budget for
the reference workload (Req 11.1) without requiring numba.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["PathResult", "run_path"]


# Allowed strategy names. Centralized so callers can validate against the
# same set we accept here.
_VALID_STRATEGIES = ("avellaneda_stoikov", "symmetric", "semi_as")


@dataclass(frozen=True)
class PathResult:
    """Per-path simulation output.

    All array fields are NumPy ``float64`` (or ``int64`` / ``bool`` where
    noted). ``bid_quote`` and ``ask_quote`` carry ``NaN`` at any step that
    was suppressed (whether by quoter pre-suppression, inventory bound,
    or kill-switch).
    """

    regime: str
    strategy: str
    path_index: int
    s_path: np.ndarray  # shape (N+1,), float64
    bid_quote: np.ndarray  # shape (N+1,), float64, NaN where suppressed
    ask_quote: np.ndarray  # shape (N+1,), float64, NaN where suppressed
    inventory: np.ndarray  # shape (N+1,), int64
    cash: np.ndarray  # shape (N+1,), float64 (pre-flatten at index N)
    bid_fill: np.ndarray  # shape (N,), bool
    ask_fill: np.ndarray  # shape (N,), bool
    n_bid_fills: int
    n_ask_fills: int
    terminal_pnl: float  # cash[N] + inventory[N] * s_path[N], post-flatten MTM
    kill_switch_step: Optional[int]  # earliest step the kill-switch fired, else None


# --------------------------------------------------------------------------- #
# Validation helpers                                                          #
# --------------------------------------------------------------------------- #


def _require_1d_len(arr: np.ndarray, expected_len: int, name: str) -> None:
    """Validate that ``arr`` is 1-D of length ``expected_len``."""
    if arr.ndim != 1:
        raise ValueError(f"{name}: must be 1-D; got shape {arr.shape!r}")
    if arr.shape[0] != expected_len:
        raise ValueError(
            f"{name}: must have length {expected_len}; got {arr.shape[0]}"
        )


def _validate_common(
    s_path: np.ndarray,
    u_b: np.ndarray,
    u_a: np.ndarray,
    A_b: float,
    k_b: float,
    A_a: float,
    k_a: float,
    dt: float,
    q_max: int,
    L_kill: Optional[float],
) -> tuple[int, int]:
    """Validate inputs shared across both strategies.

    Returns the inferred ``(n, n_plus_1)`` so the strategy-specific code
    paths can build their schedule lengths without recomputing.
    """
    if s_path.ndim != 1:
        raise ValueError(f"s_path: must be 1-D; got shape {s_path.shape!r}")
    if s_path.shape[0] < 1:
        raise ValueError(f"s_path: must be non-empty; got shape {s_path.shape!r}")
    n_plus_1 = int(s_path.shape[0])
    n = n_plus_1 - 1
    _require_1d_len(u_b, n, "u_b")
    _require_1d_len(u_a, n, "u_a")
    if not (A_b > 0):
        raise ValueError(f"A_b: must be > 0; got {A_b!r}")
    if not (k_b > 0):
        raise ValueError(f"k_b: must be > 0; got {k_b!r}")
    if not (A_a > 0):
        raise ValueError(f"A_a: must be > 0; got {A_a!r}")
    if not (k_a > 0):
        raise ValueError(f"k_a: must be > 0; got {k_a!r}")
    if not (dt > 0):
        raise ValueError(f"dt: must be > 0; got {dt!r}")
    if not isinstance(q_max, int) or isinstance(q_max, bool):
        raise ValueError(f"q_max: must be an int; got {q_max!r}")
    if q_max < 0:
        raise ValueError(f"q_max: must be >= 0; got {q_max!r}")
    if L_kill is not None:
        if not (L_kill > 0):
            raise ValueError(f"L_kill: must be None or > 0; got {L_kill!r}")
    return n, n_plus_1


# --------------------------------------------------------------------------- #
# run_path                                                                    #
# --------------------------------------------------------------------------- #


def run_path(
    s_path: np.ndarray,
    *,
    strategy: str,
    # AS-only schedules; required when strategy == "avellaneda_stoikov".
    delta_star_sched: Optional[np.ndarray] = None,
    skew_coef_sched: Optional[np.ndarray] = None,
    as_suppress_mask: Optional[np.ndarray] = None,
    # Symmetric-only parameter; required when strategy == "symmetric".
    delta_base: Optional[float] = None,
    # Pre-drawn fill uniforms.
    u_b: np.ndarray,
    u_a: np.ndarray,
    # Fill-intensity parameters.
    A_b: float,
    k_b: float,
    A_a: float,
    k_a: float,
    # Step length.
    dt: float,
    # Risk parameters.
    q_max: int,
    L_kill: Optional[float],
    # Identifying labels copied straight onto the result.
    regime: str,
    path_index: int,
) -> PathResult:
    """Run one Monte Carlo path and return a :class:`PathResult`.

    Parameters
    ----------
    s_path:
        Mid-price path of length ``N + 1`` (float64).
    strategy:
        Either ``"avellaneda_stoikov"`` or ``"symmetric"``. Determines
        which set of schedule arguments is consumed.
    delta_star_sched:
        AS-only. Half-spread schedule of length ``N + 1`` produced by
        :func:`etf_mm_sim.quoters.avellaneda_stoikov.precompute`.
    skew_coef_sched:
        AS-only. Inventory-skew coefficient schedule of length ``N + 1``.
    as_suppress_mask:
        AS-only. Bool mask of length ``N + 1``; ``True`` at steps where
        the AS quoter pre-suppresses on its own (e.g. ``t_i >= T``).
    delta_base:
        Symmetric-only. Constant half-spread (must be ``>= 0``).
    u_b, u_a:
        Pre-drawn Bernoulli uniforms of length ``N`` per side, produced by
        :func:`etf_mm_sim.fill_engine.draw_fill_uniforms`.
    A_b, k_b, A_a, k_a:
        Per-side fill-intensity parameters. Must be strictly positive.
    dt:
        Step length. Must be ``> 0``.
    q_max:
        Maximum permitted absolute inventory. Must be ``>= 0``.
    L_kill:
        Kill-switch loss threshold. ``None`` disables the kill-switch;
        otherwise must be ``> 0``.
    regime, path_index:
        Identifying labels copied into the returned :class:`PathResult`.

    Returns
    -------
    PathResult
        Per-path output with all length-``N+1`` state arrays plus the
        terminal P&L and the kill-switch step (``None`` if it never
        triggered).

    Raises
    ------
    ValueError
        If any input violates its constraint or if the strategy-specific
        schedule arguments are missing.
    """
    n, n_plus_1 = _validate_common(
        s_path, u_b, u_a, A_b, k_b, A_a, k_a, dt, q_max, L_kill
    )

    # Cast to canonical dtypes once. Down-stream we read into Python floats
    # for the inner loop so the dtype only matters for the returned arrays
    # and for the few vectorized helpers that touch the schedules.
    s_path_arr = np.ascontiguousarray(s_path, dtype=np.float64)

    # --- Strategy-specific schedule resolution ------------------------- #
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"strategy: must be one of {list(_VALID_STRATEGIES)}; got {strategy!r}"
        )
    if strategy == "avellaneda_stoikov":
        if delta_star_sched is None:
            raise ValueError(
                "delta_star_sched: required when strategy == 'avellaneda_stoikov'"
            )
        if skew_coef_sched is None:
            raise ValueError(
                "skew_coef_sched: required when strategy == 'avellaneda_stoikov'"
            )
        if as_suppress_mask is None:
            raise ValueError(
                "as_suppress_mask: required when strategy == 'avellaneda_stoikov'"
            )
        delta_star_arr = np.ascontiguousarray(delta_star_sched, dtype=np.float64)
        skew_coef_arr = np.ascontiguousarray(skew_coef_sched, dtype=np.float64)
        suppress_arr = np.ascontiguousarray(as_suppress_mask, dtype=bool)
        _require_1d_len(delta_star_arr, n_plus_1, "delta_star_sched")
        _require_1d_len(skew_coef_arr, n_plus_1, "skew_coef_sched")
        _require_1d_len(suppress_arr, n_plus_1, "as_suppress_mask")
    elif strategy == "semi_as":
        # Semi-AS: AS dynamic half-spread schedule with zero inventory
        # skew. We model this internally as the AS recurrence with
        # ``skew_coef_sched`` identically zero so the quote midpoint is
        # always ``s_path[i]`` regardless of inventory. The terminal
        # suppression behaviour is identical to AS (``t_i >= T``); the
        # caller passes the same ``as_suppress_mask``.
        if delta_star_sched is None:
            raise ValueError(
                "delta_star_sched: required when strategy == 'semi_as'"
            )
        if as_suppress_mask is None:
            raise ValueError(
                "as_suppress_mask: required when strategy == 'semi_as'"
            )
        delta_star_arr = np.ascontiguousarray(delta_star_sched, dtype=np.float64)
        skew_coef_arr = np.zeros(n_plus_1, dtype=np.float64)
        suppress_arr = np.ascontiguousarray(as_suppress_mask, dtype=bool)
        _require_1d_len(delta_star_arr, n_plus_1, "delta_star_sched")
        _require_1d_len(suppress_arr, n_plus_1, "as_suppress_mask")
    else:  # strategy == "symmetric"
        if delta_base is None:
            raise ValueError("delta_base: required when strategy == 'symmetric'")
        if delta_base < 0:
            raise ValueError(f"delta_base: must be >= 0; got {delta_base!r}")
        # Empty arrays so the loop has a uniform interface; never read for
        # the symmetric branch but kept as locals for type stability.
        delta_star_arr = np.empty(0, dtype=np.float64)
        skew_coef_arr = np.empty(0, dtype=np.float64)
        suppress_arr = np.empty(0, dtype=bool)

    # --- Output arrays -------------------------------------------------- #
    bid_quote = np.full(n_plus_1, np.nan, dtype=np.float64)
    ask_quote = np.full(n_plus_1, np.nan, dtype=np.float64)
    inventory = np.zeros(n_plus_1, dtype=np.int64)
    cash = np.zeros(n_plus_1, dtype=np.float64)
    bid_fill = np.zeros(n, dtype=bool)
    ask_fill = np.zeros(n, dtype=bool)

    # --- Local fast-access aliases for the inner loop ------------------- #
    s_local = s_path_arr  # name kept short for readability inside loop
    u_b_local = np.ascontiguousarray(u_b, dtype=np.float64)
    u_a_local = np.ascontiguousarray(u_a, dtype=np.float64)

    is_as = strategy == "avellaneda_stoikov" or strategy == "semi_as"
    db = float(delta_base) if delta_base is not None else 0.0

    q = 0
    cash_now = 0.0
    kill_switch_active = False
    kill_switch_step: Optional[int] = None
    n_bid_fills = 0
    n_ask_fills = 0

    for i in range(n):
        s_i = float(s_local[i])

        # ----- 1) Compute live quotes for step i ---------------------- #
        if is_as:
            if suppress_arr[i]:
                live_bid = math.nan
                live_ask = math.nan
                quoter_suppress_b = True
                quoter_suppress_a = True
            else:
                skew_i = float(skew_coef_arr[i])
                ds_i = float(delta_star_arr[i])
                r_i = s_i - q * skew_i
                live_bid = r_i - ds_i
                live_ask = r_i + ds_i
                quoter_suppress_b = False
                quoter_suppress_a = False
        else:
            live_bid = s_i - db
            live_ask = s_i + db
            quoter_suppress_b = False
            quoter_suppress_a = False

        # ----- 2) Risk-manager suppression ---------------------------- #
        # Inventory-bound suppression (Req 6.2, 6.3).
        bound_suppress_b = q >= q_max
        bound_suppress_a = q <= -q_max
        if kill_switch_active:
            suppress_b = True
            suppress_a = True
        else:
            suppress_b = quoter_suppress_b or bound_suppress_b
            suppress_a = quoter_suppress_a or bound_suppress_a

        # ----- 3) Resolve Bernoulli fills ---------------------------- #
        # Bid side.
        if suppress_b:
            filled_b = False
            recorded_bid = math.nan
        else:
            recorded_bid = live_bid
            delta_b = s_i - live_bid
            if delta_b <= 0.0:
                # Crossed-quote regime: guaranteed fill within step (Req 5.6).
                prob_b = 1.0
            else:
                # Numerically clean Bernoulli probability:
                # 1 - exp(-A_b * exp(-k_b * delta_b) * dt) = -expm1(-lam*dt).
                lam_b = A_b * math.exp(-k_b * delta_b)
                prob_b = -math.expm1(-lam_b * dt)
            filled_b = u_b_local[i] < prob_b

        # Ask side.
        if suppress_a:
            filled_a = False
            recorded_ask = math.nan
        else:
            recorded_ask = live_ask
            delta_a = live_ask - s_i
            if delta_a <= 0.0:
                prob_a = 1.0
            else:
                lam_a = A_a * math.exp(-k_a * delta_a)
                prob_a = -math.expm1(-lam_a * dt)
            filled_a = u_a_local[i] < prob_a

        # ----- 4) Record per-step outputs ----------------------------- #
        bid_quote[i] = recorded_bid
        ask_quote[i] = recorded_ask
        bid_fill[i] = filled_b
        ask_fill[i] = filled_a
        if filled_b:
            n_bid_fills += 1
        if filled_a:
            n_ask_fills += 1

        # ----- 5) Update state --------------------------------------- #
        # Cash effect: a bid fill spends ``live_bid``; an ask fill receives
        # ``live_ask``. The recorded fill price equals the posted quote
        # at the step the fill resolved (Req 5.4, 5.5).
        if filled_a:
            cash_now += live_ask
        if filled_b:
            cash_now -= live_bid
        # Inventory: +1 on bid fill, -1 on ask fill (Req 5.4, 5.5).
        if filled_b:
            q += 1
        if filled_a:
            q -= 1

        cash[i + 1] = cash_now
        inventory[i + 1] = q

        # ----- 6) Kill-switch latch (Req 6.4) ------------------------ #
        # The running mark-to-market P&L is computed on the post-step
        # state at index i+1, using s_path[i+1] (the price observed after
        # the fills at step i).
        if (not kill_switch_active) and (L_kill is not None):
            running_pnl = cash_now + q * float(s_local[i + 1])
            if running_pnl <= -L_kill:
                kill_switch_active = True
                kill_switch_step = i + 1

    # --- Terminal grid step (i == n) ----------------------------------- #
    # No fills are resolved at the terminal grid point; we still record
    # the would-be quote at this step for transparency. AS suppresses by
    # construction (the schedule's ``suppress_mask`` is ``True`` at the
    # terminal step). Symmetric still posts a finite quote.
    s_terminal = float(s_local[n])
    if is_as:
        if suppress_arr[n]:
            bid_quote[n] = math.nan
            ask_quote[n] = math.nan
        else:
            skew_n = float(skew_coef_arr[n])
            ds_n = float(delta_star_arr[n])
            r_n = s_terminal - q * skew_n
            bid_quote[n] = r_n - ds_n
            ask_quote[n] = r_n + ds_n
    else:
        bid_quote[n] = s_terminal - db
        ask_quote[n] = s_terminal + db

    # cash[n] was already written by the last loop iteration; inventory[n]
    # likewise. We keep ``cash[n]`` un-flattened so the per-step recurrence
    # is uninterrupted across the boundary.
    cash[0] = 0.0
    inventory[0] = 0

    terminal_pnl = float(cash[n] + inventory[n] * s_terminal)

    return PathResult(
        regime=regime,
        strategy=strategy,
        path_index=path_index,
        s_path=s_path_arr,
        bid_quote=bid_quote,
        ask_quote=ask_quote,
        inventory=inventory,
        cash=cash,
        bid_fill=bid_fill,
        ask_fill=ask_fill,
        n_bid_fills=n_bid_fills,
        n_ask_fills=n_ask_fills,
        terminal_pnl=terminal_pnl,
        kill_switch_step=kill_switch_step,
    )
