"""
faithful_backtest.py
====================
A faithful portfolio simulation that tracks how a fixed pool of capital (A)
actually grows or shrinks over time when managed by the BnB allocator across
Morpho Blue USDC markets on Base.

Unlike run_backtest.py (which uses a static, fixed A at every optimisation),
this script:

  1. Tracks real asset growth via the per-market supply exchange rate WITH
     synthetic interest accrual up to each optimisation boundary.  Between
     AccrueInterest events the on-chain exchange rate is frozen, but interest
     is continuously accumulating.  We estimate what _accrueInterest would
     return at t_k using the Taylor 3rd-order compound formula (identical to
     Morpho.sol line 488):
         interest = B × (r·Δt + (r·Δt)²/2 + (r·Δt)³/6)
     where r = R0 × c(U) is the per-second borrow rate and Δt = t_k - t_last.
     The synthetic exchange rate is (Sm + interest×(1-φ)) / totalSupplyShares.

  2. Detects bad-debt events where the exchange rate *falls*, meaning
     totalSupplyAssets was reduced by a liquidation that left zero collateral.
     These losses are applied to your position and logged.

  3. Computes locked-fund lower bounds (x_min_i) at each optimisation: if
     your claim in market i exceeds that market's free liquidity, you cannot
     fully exit, so the optimizer receives x_min_i > 0 as the lower bound for
     market i.

  4. Includes a virtual idle market (0% yield) so the optimizer can
     explicitly choose to leave capital undeployed rather than forcing full
     allocation.

  5. Supports configurable optimisation intervals (default: 1 hour).

  6. Skips the optimiser if no AccrueInterest (or bad-debt) event occurred in
     any tracked market during (t_{k-1}, t_k].  Without new events neither
     the optimal allocation nor the locked-fund constraints change, so the
     previous allocation is kept and only the asset values are updated.

────────────────────────────────────────────────────────────────────────────
STAGES
────────────────────────────────────────────────────────────────────────────
  Stage 0  Configuration & paths
  Stage 1  Data loading
             1A  Load backtest_states.parquet (all ACCRUE events, all markets)
             1B  Build block → timestamp interpolation table
             1C  Load raw market events from market_states/{id}.parquet
             1D  Assign Unix timestamps to raw events via interpolation
             1E  Detect bad-debt events (exchange rate decrease between events)
  Stage 2  Per-market timeline index
             2A  Build a sorted, timestamp-indexed DataFrame per market
             2B  Identify each market's first-data timestamp
             2C  Extract per-market ground-truth R0 arrays (own ACCRUE rows)
  Stage 3  Optimisation schedule
             Generate the sequence of optimisation timestamps
  Stage 4  Simulation loop
             For each optimisation timestamp t_k:
               4A  Apply synthetic interest accrual since t_{k-1}
               4B  Detect and log bad-debt events in (t_{k-1}, t_k]
               4C  Compute total assets and x_min per market
               4D  Skip optimiser if no events in (t_{k-1}, t_k]
               4E  Run the faithful allocator
               4F  Update shares to match new allocation
               4G  Record results for this step
  Stage 5  Write output files
────────────────────────────────────────────────────────────────────────────
OUTPUT FILES  (all written to faithful_results/)
────────────────────────────────────────────────────────────────────────────
  optimization_log.parquet
      One row per optimisation.  Columns: timestamp, A_total, idle_assets,
      n_active, obj_value, solver_ms, then per-market
      {label}_x, {label}_x_min, {label}_assets, {label}_shares,
      {label}_B, {label}_Sm, {label}_R0, {label}_U.

  asset_path.parquet
      Portfolio value sampled at every optimisation boundary (and at bad-debt
      events).  Columns: timestamp, event_type, total_assets, idle_assets,
      then per-market {label}_assets, {label}_exchange_rate.

  adverse_events.parquet
      Flagged incidents: bad debt and locked-fund constraints.
      Columns: timestamp, block_number, market_id, label, event_type,
      amount_affected, x_min_before, x_min_after, description.
"""



'''
Human NOTE: 
-The phi=0 is hardcoded atm, this is fine for morpho but won't translate well to 
other protocols like euler since in interest accrual it plays a part
'''

import math
import sys
import time
import pathlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ── import the faithful allocator (copy of allocator_bnb with extensions) ──
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from allocator_bnb_faithful_eth import allocator, IDLE_MARKET_ID


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 0 — Configuration
# ══════════════════════════════════════════════════════════════════════════════
# Edit these values before running.

NAME = 'BEAR'

# Total capital to allocate (raw token units).
# USDC has 6 decimals: 1e11 = 100 000 USDC (100 k USDC).
A_INITIAL = 1e11

# Seconds between consecutive optimisations.
# 3600 = hourly (≈ 8 760 optimisations per year).
OPT_INTERVAL_SEC = 3600

# IRM parameters — identical for all Morpho Blue adaptive-curve markets.
PHI = 0.0               # protocol fee fraction (0 = no fee to supplier)
T   = 365 * 24 * 3600   # seconds per year (used in APY formula)
U0  = 0.9               # IRM target utilisation
KD  = 4                 # IRM steepness parameter

# IRM adjustment speed (kP = 50 / year, per ConstantsLib in AdaptiveCurveIrm.sol)
SPY    = 365 * 24 * 3600            # seconds per year
KP_SEC = 50.0 / SPY                 # ≈ 1.585e-6 per second

# Flush output every this many optimisation rows (limits memory use).
FLUSH_EVERY = 200

# Optional time window for the simulation.
# Set to None to use the full history, or a "YYYY-MM-DD" string to restrict.
# Times are interpreted as UTC midnight.
START_DATE = '2025-10-15'   # e.g. "2024-01-01"
END_DATE   = None   # e.g. "2024-12-31"

# Full 66-character Morpho Blue market IDs (Base chain, USDC loan token).
MARKETS = [
    "0x4565ac05d38b19374ccbb04c17cca60ca9353cd41824f0803d0fc7704f60eaed",
    "0x3a85e619751152991742810df6ec69ce473daef99e28a64ab2340d7b7ccfee49",
    "0x1590cb22d797e226df92ebc6e0153427e207299916e7e4e53461389ad68272fb",
    "0xb323495f7e4148be5643a4ea4a8221eef163e4bccfdedc2a6f4696baacbc86cc",
    "0xbbf7ce1b40d32d3e3048f5cf27eeaa6de8cb27b80194690aab191a63381d8c99",
    "0xe83d72fa5b00dcd46d9e0e860d95aa540d5ec106da5833108a9f826f21f36f52",
    "0x729badf297ee9f2f6b3f717b96fd355fc6ec00422284ce1968e76647b258cf44",
    "0xeb17955ea422baeddbfb0b8d8c9086c5be7a9cfdefb292119a102e981a30062e",
    "0xd570c19c0dc0fbe4ab7faf4a37c4150e1c141c8aada8ca3e1b4b6c1b712af93d",
    "0x32e253d33f1594a67fc6ef51bf7a39cc4bf2d14904998dee769706fcde489ed9",
    "0x8924445a76b678c536df977ed9222fb0b23ee5311497dd0223fe6270bb20b4e6",
    "0xf4e9fb49e95a34320aea8b5e0ef515391a72368c39bdcf8ad8910645fd6eab97",
    "0xbf02d6c6852fa0b8247d5514d0c91e6c1fbde9a168ac3fd2033028b5ee5ce6d0",
    "0x4565ac05d38b19374ccbb04c17cca60ca9353cd41824f0803d0fc7704f60eaed"
]

LABELS        = [m[2:10] for m in MARKETS]
LABEL_TO_ID   = dict(zip(LABELS, MARKETS))
ID_TO_LABEL   = dict(zip(MARKETS, LABELS))

SCRIPT_DIR        = pathlib.Path(__file__).parent
STATES_FILE       = SCRIPT_DIR / '..' / 'build_states' / "backtest_states.parquet"
MARKET_STATES_DIR = SCRIPT_DIR / '..' / "market_timestamps"

if NAME:
    OUT_NAME = f"{NAME}_compound_backtest_Assets_{A_INITIAL}_From_{START_DATE}_To_{END_DATE}"
else:
    OUT_NAME = f"compound_backtest_Assets_{A_INITIAL}_From_{START_DATE}_To_{END_DATE}"

OUT_DIR = SCRIPT_DIR / OUT_NAME



# ══════════════════════════════════════════════════════════════════════════════
# IRM helpers (module-level, used in synthetic accrual)
# ══════════════════════════════════════════════════════════════════════════════

def _c_of_U(U, kd=KD, u0=U0):
    """
    AdaptiveCurveIRM curve multiplier at utilisation U with x=0 additional
    supply (i.e. the market's own borrow rate at its current state).

    Piecewise linear curve matching alpha_beta_low / alpha_beta_high:
      U <= U0:  c = (1/kd) + (1 - 1/kd) × U / U0       → c(0)=1/kd, c(U0)=1
      U >  U0:  c = 1 + (kd - 1) × (U - U0) / (1 - U0) → c(U0)=1, c(1)=kd
    """
    if U <= u0:
        return (1.0 / kd) + (1.0 - 1.0 / kd) * U / u0
    else:
        return 1.0 + (kd - 1.0) * (U - u0) / (1.0 - u0)


def _taylor3(x):
    """
    Taylor 3rd-order approximation of exp(x) - 1 = x + x²/2 + x³/6.

    Morpho.sol uses wTaylorCompounded (3 terms) for interest accrual.
    Exact for small r·Δt; at r=10%/yr and Δt=1 hr, x ≈ 1.1e-5 — negligible
    error vs the true exponential.
    """
    return x + x * x / 2.0 + x * x * x / 6.0


def _irm_err(U, u0=U0):
    """Normalised utilisation error (matches AdaptiveCurveIrm.sol)."""
    if U > u0:
        return (U - u0) / (1.0 - u0)
    else:
        return (U - u0) / u0


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_stage1(verbose=True):
    """
    Load and pre-process all raw data needed for the simulation.

    Returns
    -------
    backtest_states : pd.DataFrame
        Wide table of market state (B, Sm, R0) at every ACCRUE event.
        B, Sm are frozen from each market's last event; R0 is IRM-propagated
        to the row's timestamp by build_backtest_states.py.  No further
        forward-fill is applied here — pre-creation rows remain NaN.

    block_arr, time_arr : np.ndarray pair
        Parallel sorted arrays for block → Unix timestamp interpolation.

    market_events : dict  {label -> pd.DataFrame}
        Per-market event table with timestamps, exchange rates, and a
        bad-debt flag.  Sorted by (timestamp, log_index).
    """
    # ── 1A  Load backtest_states ───────────────────────────────────────────
    if verbose:
        print(f"\n[Stage 1A] Loading {STATES_FILE.name} …")
    df_states = pq.read_table(STATES_FILE).to_pandas()
    if verbose:
        print(f"           {len(df_states):,} rows, "
              f"{df_states['timestamp'].min():.0f} → "
              f"{df_states['timestamp'].max():.0f}")

    # NOTE: no forward-fill applied here.
    # build_backtest_states.py already writes a value for every active market
    # at every event row:
    #   B, Sm  — frozen from the market's last event (correct: they only
    #             change at an actual market interaction)
    #   R0     — propagated forward using the IRM formula
    #             R0_new = R0_old × exp(kP/SPY × err(U) × Δt)
    #             NOT a forward-fill — the IRM-adjusted value is pre-computed.
    # Pre-creation rows remain NaN (cur[j] is None in the build script),
    # which is correct: those markets do not yet exist.

    # ── 1B  Block → timestamp interpolation table ──────────────────────────
    if verbose:
        print("[Stage 1B] Building block→timestamp lookup …")
    bt = (df_states[["block_number", "timestamp"]]
          .drop_duplicates("block_number")
          .sort_values("block_number"))
    block_arr = bt["block_number"].to_numpy(dtype=np.float64)
    time_arr  = bt["timestamp"].to_numpy(dtype=np.float64)

    def block_to_ts(blocks):
        return np.interp(np.asarray(blocks, dtype=np.float64),
                         block_arr, time_arr)

    # ── 1C/1D/1E  Load raw events for each market ─────────────────────────
    if verbose:
        print("[Stage 1C–1E] Loading per-market raw events …")
    market_events = {}

    for lbl, full_id in tqdm(zip(LABELS, MARKETS), total=len(LABELS),
                              desc="  Markets", unit="mkt"):
        fpath = MARKET_STATES_DIR / f"{full_id}.parquet"
        if not fpath.exists():
            if verbose:
                tqdm.write(f"    WARNING: {fpath.name} not found — skipping")
            continue

        cols = ["block_number", "log_index", "event_name",
                "totalSupplyAssets", "totalSupplyShares",
                "totalBorrowAssets", "available_liquidity",
                "supply_exchange_rate", "timestamp_unix"]
        df = pq.read_table(fpath, columns=cols).to_pandas()
        df = df.rename(columns={
                'timestamp_unix': 'timestamp'
            })

        # 1D: timestamp via block interpolation
        df = df.sort_values(["timestamp", "log_index"]).reset_index(drop=True)

        # 1E: bad-debt detection — any decrease in supply_exchange_rate
        df["prev_rate"]          = df["supply_exchange_rate"].shift(1)
        df["bad_debt_per_share"] = (
            (df["prev_rate"] - df["supply_exchange_rate"]).clip(lower=0.0)
        )
        df["is_bad_debt"] = df["bad_debt_per_share"] > 0.0

        market_events[lbl] = df

    if verbose:
        print(f"           Loaded {len(market_events)} markets")

    return df_states, block_arr, time_arr, market_events


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Per-market Timeline Index
# ══════════════════════════════════════════════════════════════════════════════

def build_timeline_index(df_states, market_events, verbose=True):
    """
    Pre-compute lookup helpers for fast access to per-market state at any
    given timestamp.

    Returns
    -------
    market_ts_arr : dict {label -> np.ndarray}
        Sorted timestamp array per market (for np.searchsorted lookups).

    market_first_ts : dict {label -> float}
        Unix timestamp of each market's first event.

    market_own_ts_arr : dict {label -> np.ndarray}
        Timestamps of backtest_states rows where event_market == label.
        At these rows, {label}_R0 is the ground-truth on-chain rateAtTarget
        (not propagated) — used for synthetic accrual.

    market_own_R0_arr : dict {label -> np.ndarray}
        Ground-truth R0 values at each market's own ACCRUE rows.
    """
    if verbose:
        print("\n[Stage 2] Building per-market timeline index …")

    market_ts_arr   = {}
    market_first_ts = {}

    for lbl, df in market_events.items():
        ts_arr = df["timestamp"].to_numpy(dtype=np.float64)
        market_ts_arr[lbl]   = ts_arr
        market_first_ts[lbl] = float(ts_arr[0]) if len(ts_arr) > 0 else math.inf

    if verbose:
        for lbl, first_ts in sorted(market_first_ts.items(), key=lambda x: x[1]):
            dt = datetime.fromtimestamp(first_ts, tz=timezone.utc)
            print(f"    {lbl}  first event: {dt:%Y-%m-%d %H:%M UTC}")

    # ── 2C  Per-market ground-truth R0 from own ACCRUE rows ───────────────
    # In backtest_states, rows where event_market == lbl carry the on-chain
    # rateAtTarget for market lbl directly (build_backtest_states.py line 249:
    #   if j == ev_lbl_idx: col_R0[j][row_idx] = st[4]  # ground truth
    # All other markets in that row are IRM-propagated estimates).
    # We use these ground-truth R0 values as the starting point for synthetic
    # accrual: borrow_rate = R0_gt × c(U), from the last own event to t_k.
    if verbose:
        print("    Extracting per-market ground-truth R0 arrays …")

    market_own_ts_arr = {}
    market_own_R0_arr = {}

    for lbl in LABELS:
        mask   = df_states["event_market"] == lbl
        subset = df_states[mask].sort_values("timestamp")
        if len(subset) == 0:
            continue
        market_own_ts_arr[lbl] = subset["timestamp"].to_numpy(dtype=np.float64)
        market_own_R0_arr[lbl] = subset[f"{lbl}_R0"].to_numpy(dtype=np.float64)

    return market_ts_arr, market_first_ts, market_own_ts_arr, market_own_R0_arr


def get_row_at(lbl, timestamp, market_events, market_ts_arr):
    """
    Return the most recent event row for `lbl` with timestamp ≤ `timestamp`.
    Returns None if the market has no data yet at this timestamp.
    """
    ts_arr = market_ts_arr.get(lbl)
    if ts_arr is None:
        return None
    idx = int(np.searchsorted(ts_arr, timestamp, side="right")) - 1
    if idx < 0:
        return None
    return market_events[lbl].iloc[idx]


def _synthetic_exchange_rate(lbl, t_k,
                              market_events, market_ts_arr,
                              market_own_ts_arr, market_own_R0_arr):
    """
    Compute the synthetic supply exchange rate for market `lbl` at time t_k.

    The on-chain exchange rate only updates at AccrueInterest calls.  Between
    calls interest accumulates continuously.  We estimate what the rate would
    be if AccrueInterest were called at t_k:

        borrow_rate  = R0_gt × c(U)           [per second]
        r_dt         = borrow_rate × elapsed   [dimensionless]
        interest     = B × (r_dt + r_dt²/2 + r_dt³/6)   [Taylor 3rd order]
        new_Sm       = Sm + interest × (1 - PHI)
        synth_rate   = new_Sm / totalSupplyShares

    R0_gt is the ground-truth rateAtTarget at the last ACCRUE event for this
    market (from backtest_states rows where event_market == lbl), which is the
    value the on-chain IRM would use as its starting point.

    With PHI = 0 interest accrual adds equally to B and Sm, so free liquidity
    (Sm - B) is unchanged — the last event's available_liquidity is still valid
    for locked-fund calculations.

    Returns
    -------
    synth_rate : float  — synthetic exchange rate at t_k, or last real rate
                          if no R0 is available
    avail_liq  : float  — available liquidity (unchanged by accrual, PHI=0)
    last_row_ts: float  — timestamp of last real event (for optimiser R0)
    """
    row = get_row_at(lbl, t_k, market_events, market_ts_arr)
    if row is None:
        return None, None, None

    t_last        = float(row["timestamp"])
    B             = float(row["totalBorrowAssets"])
    Sm            = float(row["totalSupplyAssets"])
    shares_total  = float(row["totalSupplyShares"])
    real_rate     = float(row["supply_exchange_rate"])
    avail_liq     = float(row["available_liquidity"])

    # If no shares exist, the rate is meaningless; return the stored value.
    if shares_total <= 0 or Sm <= 0:
        return real_rate, avail_liq, t_last

    elapsed = t_k - t_last
    if elapsed <= 0.0:
        return real_rate, avail_liq, t_last

    # ── Locate ground-truth R0 at the last own ACCRUE event ───────────────
    own_ts = market_own_ts_arr.get(lbl)
    own_R0 = market_own_R0_arr.get(lbl)
    if own_ts is None or len(own_ts) == 0:
        return real_rate, avail_liq, t_last

    idx_gt = int(np.searchsorted(own_ts, t_k, side="right")) - 1
    if idx_gt < 0:
        return real_rate, avail_liq, t_last

    R0_gt = float(own_R0[idx_gt])
    if R0_gt <= 0.0 or math.isnan(R0_gt):
        return real_rate, avail_liq, t_last

    # ── Synthetic accrual (Morpho wTaylorCompounded, 3 terms) ─────────────
    # On-chain: borrowRate = _curve(avgRateAtTarget, err)
    # avgRateAtTarget is the trapezoidal average of R0 as it drifts over elapsed:
    #   speed            = KP_SEC * err(U)
    #   linearAdaptation = speed * elapsed
    #   endR0            = R0_gt * exp(linearAdaptation)
    #   midR0            = R0_gt * exp(linearAdaptation / 2)
    #   avgR0            = (R0_gt + endR0 + 2*midR0) / 4      [contract line 124]
    #   borrowRate       = avgR0 * c(U)
    U               = B / Sm
    err             = _irm_err(U)
    lin_adapt       = KP_SEC * err * elapsed
    end_R0          = R0_gt * math.exp(lin_adapt)
    mid_R0          = R0_gt * math.exp(lin_adapt / 2.0)
    avg_R0          = (R0_gt + end_R0 + 2.0 * mid_R0) / 4.0
    borrow_rate     = avg_R0 * _c_of_U(U)     # per-second borrow rate
    r_dt            = borrow_rate * elapsed
    interest        = B * _taylor3(r_dt)

    new_Sm      = Sm + interest * (1.0 - PHI)
    synth_rate  = new_Sm / shares_total

    # Sanity: synthetic rate must be >= real_rate (interest can only add)
    synth_rate = max(synth_rate, real_rate)

    return synth_rate, avail_liq, t_last


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Optimisation Schedule
# ══════════════════════════════════════════════════════════════════════════════

def _parse_date(s):
    """Parse a 'YYYY-MM-DD' string to a UTC Unix timestamp (midnight)."""
    from calendar import timegm
    import time as _time
    return float(timegm(_time.strptime(s, "%Y-%m-%d")))


def build_opt_schedule(df_states, verbose=True):
    """
    Generate the list of Unix timestamps at which the optimizer fires.
    Steps from the global first event by OPT_INTERVAL_SEC until history ends.
    Respects START_DATE / END_DATE if set.
    """
    if verbose:
        print("\n[Stage 3] Building optimisation schedule …")

    global_start = df_states["timestamp"].min()
    global_end   = df_states["timestamp"].max()

    if START_DATE is not None:
        global_start = max(global_start, _parse_date(START_DATE))
    if END_DATE is not None:
        global_end   = min(global_end,   _parse_date(END_DATE))

    t = global_start
    opt_times = []
    while t <= global_end:
        opt_times.append(t)
        t += OPT_INTERVAL_SEC

    if verbose:
        start_dt = datetime.fromtimestamp(global_start, tz=timezone.utc)
        end_dt   = datetime.fromtimestamp(global_end,   tz=timezone.utc)
        print(f"    {len(opt_times):,} optimisations  "
              f"({start_dt:%Y-%m-%d} → {end_dt:%Y-%m-%d}, "
              f"interval = {OPT_INTERVAL_SEC} s)")

    return opt_times


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — Simulation Loop
# ══════════════════════════════════════════════════════════════════════════════

def _get_backtest_state_at(df_states, timestamp):
    """Return the last backtest_states row with timestamp ≤ `timestamp`."""
    arr = df_states["timestamp"].to_numpy()
    idx = int(np.searchsorted(arr, timestamp, side="right")) - 1
    return df_states.iloc[idx] if idx >= 0 else None


def _build_market_states_for_opt(df_states, t_k, active_labels):
    """
    Extract market state dicts for the allocator from the latest backtest row.

    The backtest_states R0 at that snapshot row is already IRM-propagated to
    the row's own timestamp (t_snap).  We further propagate R0 from t_snap to
    t_k using the same IRM formula, so the optimiser sees the rate it would
    encounter if it were called exactly at t_k.

    Parameters
    ----------
    df_states    : backtest_states DataFrame
    t_k          : optimisation timestamp
    active_labels: set of labels that have data at this timestamp

    Returns
    -------
    list of dicts suitable for allocator()
    """
    snap = _get_backtest_state_at(df_states, t_k)
    if snap is None:
        return []

    snap_ts       = float(snap["timestamp"])
    elapsed_extra = t_k - snap_ts        # time from snap row to t_k

    states = []
    for lbl in active_labels:
        full_id = LABEL_TO_ID[lbl]
        B  = snap.get(f"{lbl}_B",  float("nan"))
        Sm = snap.get(f"{lbl}_Sm", float("nan"))
        R0 = snap.get(f"{lbl}_R0", float("nan"))

        if any(math.isnan(v) for v in (B, Sm, R0)):
            continue
        if Sm <= 0 or R0 <= 0:
            continue

        # ── Propagate R0 from snap_ts to t_k ──────────────────────────────
        # The snap's R0 is already correct at t_snap.  Between t_snap and
        # t_k no on-chain event fired for this market, so R0 drifts via
        # the IRM at frozen utilisation U = B / Sm.
        if elapsed_extra > 0.0:
            U   = B / Sm
            err = _irm_err(U)
            R0  = R0 * math.exp(KP_SEC * err * elapsed_extra)

        states.append({
            "Id": full_id, "B": B, "Sm": Sm, "R0": R0,
            "phi": PHI, "T": T, "U0": U0, "kd": KD,
        })
    return states


def run_simulation(df_states, market_events, market_ts_arr,
                   market_first_ts, market_own_ts_arr, market_own_R0_arr,
                   opt_times, verbose=True):
    """
    Core simulation loop.

    For each optimisation timestamp t_k:

      4A  Compute synthetic exchange rates at t_k and update asset values.
          The synthetic rate projects interest accrual from the last real
          AccrueInterest event to t_k using the Taylor 3rd-order formula,
          matching what Morpho's _accrueInterest would compute.

      4B  Detect and log bad-debt events in (t_{k-1}, t_k].

      4C  Compute total assets and x_min per market.

      4D  Skip optimiser if no events occurred in (t_{k-1}, t_k].
          Without events, neither R0 nor B/Sm changed in any market, so the
          optimal allocation is the same as last step.  We record the updated
          asset value but skip the expensive B&B solve.

      4E  Run the faithful allocator (when 4D does not skip).

      4F  Update shares to match new allocation.

      4G  Record results for this step.
    """
    # ── portfolio state ────────────────────────────────────────────────────
    shares      = {lbl: 0.0 for lbl in LABELS}
    idle_assets = A_INITIAL                        # start fully idle
    t_prev      = opt_times[0]

    # x_min per market from the previous step (for locked-funds delta logging)
    prev_x_min  = {lbl: 0.0 for lbl in LABELS}

    # ── result buffers ─────────────────────────────────────────────────────
    opt_rows     = []
    path_rows    = []
    adverse_rows = []

    n_optimised = 0
    n_skipped   = 0

    def ts_utc(ts):
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    for t_k in tqdm(opt_times, desc="Simulating", unit="opt", dynamic_ncols=True):

        # ── 4A  Synthetic exchange rates and asset values ──────────────────
        # For each market, compute what the supply exchange rate would be if
        # _accrueInterest were called at t_k.  This captures interest that
        # has accrued since the last real on-chain ACCRUE event.
        current_assets = {}
        current_rates  = {}    # synthetic rate at t_k
        current_liq    = {}    # available_liquidity (unchanged under accrual)
        last_real_ts   = {}    # timestamp of last real event (for skip logic)

        for lbl in LABELS:
            synth_rate, avail_liq, t_last_real = _synthetic_exchange_rate(
                lbl, t_k,
                market_events, market_ts_arr,
                market_own_ts_arr, market_own_R0_arr,
            )
            if synth_rate is None:
                current_assets[lbl] = 0.0
                current_rates[lbl]  = None
                current_liq[lbl]    = None
                last_real_ts[lbl]   = None
            else:
                current_rates[lbl]  = synth_rate
                current_assets[lbl] = shares[lbl] * synth_rate
                current_liq[lbl]    = avail_liq
                last_real_ts[lbl]   = t_last_real

        # ── 4B  Bad-debt events in (t_prev, t_k] ─────────────────────────
        for lbl in LABELS:
            df_mkt = market_events.get(lbl)
            if df_mkt is None or shares[lbl] == 0.0:
                continue

            ts_arr = market_ts_arr[lbl]
            lo_idx = int(np.searchsorted(ts_arr, t_prev, side="right"))
            hi_idx = int(np.searchsorted(ts_arr, t_k,    side="right"))

            for idx in range(lo_idx, hi_idx):
                row = df_mkt.iloc[idx]
                if not row["is_bad_debt"]:
                    continue

                loss_per_share = float(row["bad_debt_per_share"])
                our_loss       = shares[lbl] * loss_per_share

                adverse_rows.append({
                    "timestamp":       float(row["timestamp"]),
                    "block_number":    int(row["block_number"]),
                    "market_id":       LABEL_TO_ID[lbl],
                    "label":           lbl,
                    "event_type":      "bad_debt",
                    "amount_affected": -our_loss,
                    "x_min_before":    float("nan"),
                    "x_min_after":     float("nan"),
                    "description": (
                        f"Bad debt in market {lbl}: "
                        f"{our_loss:.4f} lost  "
                        f"({loss_per_share:.6e} per share × "
                        f"{shares[lbl]:.6e} shares)"
                    ),
                })
                # Loss already reflected in current_assets via lower synth_rate.

        # ── 4C  Total assets and locked-fund lower bounds ─────────────────
        A_total = sum(current_assets.values()) + idle_assets

        x_min_per_market = {}

        for lbl in LABELS:
            full_id   = LABEL_TO_ID[lbl]
            our_claim = current_assets[lbl]
            liq       = current_liq[lbl]

            if our_claim <= 0 or liq is None:
                continue

            locked  = max(0.0, our_claim - liq)
            x_min_i = locked / A_total if A_total > 0 else 0.0

            if x_min_i > 1e-9:
                x_min_per_market[full_id] = x_min_i

                if x_min_i > prev_x_min[lbl] + 1e-6:
                    adverse_rows.append({
                        "timestamp":       t_k,
                        "block_number":    -1,
                        "market_id":       full_id,
                        "label":           lbl,
                        "event_type":      "locked_funds",
                        "amount_affected": -locked,
                        "x_min_before":    prev_x_min[lbl],
                        "x_min_after":     x_min_i,
                        "description": (
                            f"Locked funds in {lbl}: "
                            f"{locked:.4f} cannot be withdrawn "
                            f"(U ≈ {our_claim / (our_claim + liq):.3f}; "
                            f"x_min = {x_min_i:.4f})"
                        ),
                    })
                prev_x_min[lbl] = x_min_i
            else:
                prev_x_min[lbl] = 0.0

        # ── 4D  Skip optimiser if no market events in (t_prev, t_k] ──────
        # If no AccrueInterest events fired for any market during this hour,
        # the B, Sm, R0, U values are all unchanged — the optimal allocation
        # and locked-fund constraints are identical to the previous step.
        # We record the updated asset value but do not re-solve.
        any_events = False
        for lbl in LABELS:
            ts_arr = market_ts_arr.get(lbl)
            if ts_arr is None:
                continue
            lo = int(np.searchsorted(ts_arr, t_prev, side="right"))
            hi = int(np.searchsorted(ts_arr, t_k,    side="right"))
            if hi > lo:
                any_events = True
                break

        if not any_events:
            n_skipped += 1
            # Still record the updated portfolio value.
            path_rows.append({
                "timestamp":    t_k,
                "event_type":   "skipped",
                "total_assets": A_total,
                "idle_assets":  idle_assets,
                **{f"{lbl}_assets":        current_assets.get(lbl, 0.0) for lbl in LABELS},
                **{f"{lbl}_exchange_rate": (current_rates.get(lbl) or float("nan")) for lbl in LABELS},
            })
            t_prev = t_k
            continue

        # ── 4E  Run the faithful allocator ────────────────────────────────
        active_labels = {
            lbl for lbl in LABELS
            if current_rates.get(lbl) is not None
        }

        mkt_states = _build_market_states_for_opt(df_states, t_k, active_labels)
        n_active   = len(mkt_states)

        t_solver_start = time.perf_counter()
        try:
            best_alloc, obj_val = allocator(
                mkt_states, A_total,
                x_min_per_market=x_min_per_market,
                include_idle=True,
            )
        except Exception as exc:
            tqdm.write(f"  [t={ts_utc(t_k)}] allocator error: {exc}")
            best_alloc, obj_val = None, math.inf
        solver_ms = (time.perf_counter() - t_solver_start) * 1e3

        if best_alloc is None:
            tqdm.write(f"  [t={ts_utc(t_k)}] infeasible — holding positions")
            t_prev = t_k
            continue

        n_optimised += 1

        # ── 4F  Update shares to match new allocation ──────────────────────
        # Treat rebalancing as instantaneous and costless.
        new_idle = 0.0
        for full_id, x_i in best_alloc.items():
            if full_id == IDLE_MARKET_ID:
                new_idle = x_i * A_total
                continue
            lbl  = ID_TO_LABEL.get(full_id)
            rate = current_rates.get(lbl)
            if lbl is None or rate is None or rate <= 0:
                shares[lbl] = 0.0
            else:
                shares[lbl] = (x_i * A_total) / rate

        # Zero out markets not in new allocation
        allocated_ids = set(best_alloc.keys()) - {IDLE_MARKET_ID}
        for lbl in LABELS:
            if LABEL_TO_ID[lbl] not in allocated_ids:
                shares[lbl] = 0.0

        idle_assets = new_idle

        # ── 4G  Record results ─────────────────────────────────────────────
        snap = _get_backtest_state_at(df_states, t_k)

        opt_row = {
            "timestamp":   t_k,
            "A_total":     A_total,
            "idle_assets": idle_assets,
            "n_active":    n_active,
            "obj_value":   obj_val,
            "solver_ms":   solver_ms,
        }
        for lbl in LABELS:
            full_id = LABEL_TO_ID[lbl]
            x_i     = best_alloc.get(full_id, 0.0)
            x_min_i = x_min_per_market.get(full_id, 0.0)
            rate    = current_rates.get(lbl)
            B_i  = float(snap[f"{lbl}_B"])  if snap is not None else float("nan")
            Sm_i = float(snap[f"{lbl}_Sm"]) if snap is not None else float("nan")
            R0_i = float(snap[f"{lbl}_R0"]) if snap is not None else float("nan")
            U_i  = (B_i / Sm_i) if (not math.isnan(B_i) and Sm_i > 0) else float("nan")

            opt_row[f"{lbl}_x"]      = x_i
            opt_row[f"{lbl}_x_min"]  = x_min_i
            opt_row[f"{lbl}_assets"] = current_assets.get(lbl, 0.0)
            opt_row[f"{lbl}_shares"] = shares[lbl]
            opt_row[f"{lbl}_B"]      = B_i
            opt_row[f"{lbl}_Sm"]     = Sm_i
            opt_row[f"{lbl}_R0"]     = R0_i
            opt_row[f"{lbl}_U"]      = U_i

        opt_rows.append(opt_row)

        path_rows.append({
            "timestamp":    t_k,
            "event_type":   "optimization",
            "total_assets": A_total,
            "idle_assets":  idle_assets,
            **{f"{lbl}_assets":        current_assets.get(lbl, 0.0) for lbl in LABELS},
            **{f"{lbl}_exchange_rate": (current_rates.get(lbl) or float("nan")) for lbl in LABELS},
        })

        t_prev = t_k

    if verbose:
        print(f"\n    Optimised: {n_optimised:,}  |  Skipped (no events): {n_skipped:,}")

    return opt_rows, path_rows, adverse_rows


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — Write Output Files
# ══════════════════════════════════════════════════════════════════════════════

def write_outputs(opt_rows, path_rows, adverse_rows, verbose=True):
    """Convert result lists to PyArrow tables and write to Parquet."""
    OUT_DIR.mkdir(exist_ok=True)

    if verbose:
        print(f"\n[Stage 5] Writing output to {OUT_DIR} …")

    if opt_rows:
        opt_file = OUT_DIR / "optimization_log.parquet"
        df_opt   = pd.DataFrame(opt_rows)
        pq.write_table(
            pa.Table.from_pandas(df_opt, preserve_index=False),
            opt_file, compression="snappy"
        )
        if verbose:
            print(f"    optimization_log.parquet  {len(df_opt):,} rows  "
                  f"({opt_file.stat().st_size/1e6:.1f} MB)")

    if path_rows:
        path_file = OUT_DIR / "asset_path.parquet"
        df_path   = pd.DataFrame(path_rows)
        pq.write_table(
            pa.Table.from_pandas(df_path, preserve_index=False),
            path_file, compression="snappy"
        )
        if verbose:
            print(f"    asset_path.parquet         {len(df_path):,} rows  "
                  f"({path_file.stat().st_size/1e6:.1f} MB)")

    adv_file = OUT_DIR / "adverse_events.parquet"
    if adverse_rows:
        df_adv = pd.DataFrame(adverse_rows)
        pq.write_table(
            pa.Table.from_pandas(df_adv, preserve_index=False),
            adv_file, compression="snappy"
        )
        if verbose:
            print(f"    adverse_events.parquet     {len(df_adv):,} events")
            bd  = df_adv[df_adv["event_type"] == "bad_debt"]
            lkd = df_adv[df_adv["event_type"] == "locked_funds"]
            print(f"      bad_debt events:    {len(bd):,}")
            print(f"      locked_funds events:{len(lkd):,}")
            if len(bd):
                print(f"      total bad debt loss: "
                      f"{bd['amount_affected'].sum():.4f} (raw units)")
    else:
        if verbose:
            print("    adverse_events.parquet     0 events (clean run)")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print(f"  Faithful Backtest: {OUT_NAME}")
    print(f"  A_initial = {A_INITIAL:.2e} raw units  |  "
          f"interval = {OPT_INTERVAL_SEC} s  |  "
          f"idle market: enabled")
    print("=" * 70)

    # Stage 1
    df_states, block_arr, time_arr, market_events = load_stage1()

    # Stage 2
    market_ts_arr, market_first_ts, market_own_ts_arr, market_own_R0_arr = \
        build_timeline_index(df_states, market_events)

    # Stage 3
    opt_times = build_opt_schedule(df_states)

    # Stage 4
    print(f"\n[Stage 4] Running simulation ({len(opt_times):,} optimisations) …")
    opt_rows, path_rows, adverse_rows = run_simulation(
        df_states, market_events, market_ts_arr,
        market_first_ts, market_own_ts_arr, market_own_R0_arr,
        opt_times,
    )

    # Stage 5
    write_outputs(opt_rows, path_rows, adverse_rows)

    # ── summary ────────────────────────────────────────────────────────────
    if opt_rows:
        df_opt       = pd.DataFrame(opt_rows)
        final_assets = df_opt["A_total"].iloc[-1]
        init_assets  = A_INITIAL
        total_return = (final_assets - init_assets) / init_assets * 100
        n_days       = (opt_times[-1] - opt_times[0]) / 86400

        print(f"\n{'─'*70}")
        print(f"  Initial capital : {init_assets:.4e}")
        print(f"  Final capital   : {final_assets:.4e}")
        print(f"  Total return    : {total_return:.2f}%  over {n_days:.0f} days")
        print(f"  Annualised      : {total_return / n_days * 365:.2f}%")
        print(f"  Optimisations   : {len(opt_rows):,}")
        print(f"  Adverse events  : {len(adverse_rows):,}")
        print(f"{'─'*70}")


if __name__ == "__main__":
    import os
    os.chdir(SCRIPT_DIR)
    main()