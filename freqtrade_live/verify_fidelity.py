"""Fidelity check: live single-cohort signal vs the research backtest.

Read-only diagnostic. Run from anywhere:

    python freqtrade_live/verify_fidelity.py

Loads the local daily feather data for LTC/XRP, computes the LIVE signal
(``freqtrade_live/user_data/strategies/jansen_signals.compute_state_signal``),
and -- when the research backtest is importable -- compares it against
``jansen_backtest``'s reference position / z-score / hedge-ratio.

Prints diagnostics only; it does not write or modify anything:

- the live signal's shape (warm-up, non-flat days, entry crossings) plus
  regression guards: governed days, ``hr >= 0`` days (entries are vetoed
  there -- expect 0), and unmanaged dead quarters (expect 0);
- the funding-rate drag of the two-perp book (the research backtest does not
  model funding), from the bot's own downloaded funding feathers;
- sign agreement vs the backtest on single-position days, at shift 0 (live
  as-computed) and shift 1 (live moved onto the backtest's fill day -- live
  fills ~close[t], the backtest fills at close[t+1], so the shifted variant
  isolates the one-bar fill-convention offset from real divergence), each
  decomposed into live-only / ref-only / opposite-sign days;
- hr / z divergence on the days both sides publish (expect ~0; residual
  disagreement concentrates on the backtest days holding two overlapping
  cohorts, which the single-cohort live bot cannot represent by design).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))            # .../freqtrade_live
PROJECT = os.path.dirname(HERE)                              # .../Pair_Trading_Jansen
DATA_DIR = os.path.join(PROJECT, "data", "feather")
FUT_DATA_DIR = os.path.join(HERE, "user_data", "data", "binance", "futures")
STRAT_DIR = os.path.join(HERE, "user_data", "strategies")

sys.path.insert(0, STRAT_DIR)
from jansen_signals import _quarter_ends, compute_state_signal  # noqa: E402

KALMAN = {
    "smoother_obs_cov": 1.0,
    "smoother_trans_cov": 0.05,
    "hedge_delta": 0.001,
    "hedge_obs_cov": 2.0,
}
ENTRY_Z = 2.0
LOOKBACK_DAYS = 730

# Funding feathers put the rate in 'open' (h/l/c/volume are zero); keep a few
# fallbacks in case the download format changes.
RATE_COLS = ("open", "fundingRate", "funding_rate", "rate")


def load_close(symbol: str) -> pd.Series:
    path = os.path.join(DATA_DIR, f"{symbol}USDT_1d.feather")
    df = pd.read_feather(path)
    df["open_time_dt"] = pd.to_datetime(df["open_time_dt"])
    df = df.sort_values("open_time_dt").set_index("open_time_dt")
    return df["close"].rename(symbol)


def unmanaged_dead_quarters(
    z: pd.Series, lookback_days: int
) -> list[pd.Timestamp]:
    """Quarters past warm-up whose entire governed range publishes no z.

    Replicates ``compute_state_signal``'s cohort arithmetic (cohort i governs
    ``(q_end, next_q_end]``). The dead-quarter fallback manages a single failed
    cohort under its predecessor's calibration, so an all-NaN governed quarter
    means dates NO calibration covers (e.g. two consecutive failed cohorts) --
    a position held there could only exit via the spread stop.
    """
    idx = z.index
    day = pd.DateOffset(days=1)
    lookback = pd.DateOffset(days=lookback_days)
    q_ends = _quarter_ends(idx)
    dead: list[pd.Timestamp] = []
    for i, q_end in enumerate(q_ends):
        trading_start = q_end + day
        if trading_start - lookback < idx.min():
            continue  # warm-up gate: nothing published here by design
        slice_end = q_ends[i + 1] if i + 1 < len(q_ends) else idx.max()
        gov = z[(idx > q_end) & (idx <= slice_end)]
        if len(gov) and gov.isna().all():
            dead.append(q_end)
    return dead


def _load_funding_daily(symbol: str) -> pd.Series | None:
    path = os.path.join(FUT_DATA_DIR, f"{symbol}_USDT_USDT-8h-funding_rate.feather")
    if not os.path.exists(path):
        return None
    f = pd.read_feather(path)
    if "date" not in f.columns:
        return None
    col = next(
        (c for c in RATE_COLS if c in f.columns and f[c].abs().sum() > 0), None
    )
    if col is None:
        return None
    s = f.assign(date=pd.to_datetime(f["date"])).set_index("date")[col]
    s = s.astype(float).sort_index()
    return s.resample("1D").sum().rename(symbol)  # 3 x 8h settlements per day


def funding_diagnostic() -> None:
    """Funding drag of the two-perp book (not modeled by the backtest).

    Positive rate: longs pay shorts. The book is gross-notional dollar-neutral
    (~equal gross per leg), so the daily drag as a fraction of gross is the
    half-difference of the per-leg daily rates: the long-spread book is long
    LTC (pays LTC funding) and short XRP (receives XRP funding).
    """
    ltc = _load_funding_daily("LTC")
    xrp = _load_funding_daily("XRP")
    if ltc is None or xrp is None:
        print("\n[skip] funding feathers not found under "
              "user_data/data/binance/futures/; diagnostic only.")
        return
    both = pd.concat([ltc, xrp], axis=1).dropna()
    long_drag = (both["LTC"] - both["XRP"]) / 2.0

    print("\n=== funding drag (not modeled in the research backtest) ===")
    print(f"sample              : {both.index.min().date()} .. "
          f"{both.index.max().date()}  ({len(both)} days)")
    print(f"mean daily rate     : LTC {both['LTC'].mean():+.5%}   "
          f"XRP {both['XRP'].mean():+.5%}")
    print(f"long-spread book    : {long_drag.mean():+.5%}/day of gross  "
          f"(~{long_drag.mean() * 365:+.2%}/yr)")
    print(f"short-spread book   : {-long_drag.mean():+.5%}/day of gross  "
          f"(~{-long_drag.mean() * 365:+.2%}/yr)")
    last = both.tail(365)
    drag_1y = ((last["LTC"] - last["XRP"]) / 2.0).mean()
    print(f"last 365d           : long {drag_1y:+.5%}/day "
          f"(~{drag_1y * 365:+.2%}/yr), short {-drag_1y:+.5%}/day")


def main() -> None:
    lead = load_close("LTC")
    lag = load_close("XRP")
    df = pd.concat([lead, lag], axis=1).dropna().sort_index()
    lead_s, lag_s = df["LTC"], df["XRP"]

    sig = compute_state_signal(
        lead_s, lag_s, entry_z=ENTRY_Z, lookback_days=LOOKBACK_DAYS, kalman=KALMAN
    )
    state = sig["state"]
    nonflat = state[state != 0]
    crossings = ((state == 1) & (state.shift() != 1)) | (
        (state == -1) & (state.shift() != -1)
    )

    print("=== LIVE single-cohort signal ===")
    print(f"bars                : {len(state)}  "
          f"({df.index.min().date()} .. {df.index.max().date()})")
    print(f"first non-flat day  : "
          f"{nonflat.index.min().date() if not nonflat.empty else 'never'}")
    print(f"non-flat days       : {int((state != 0).sum())}")
    print(f"entry crossings     : {int(crossings.sum())}")

    # --- Regression guards -------------------------------------------------
    governed_hr = sig["hr"].dropna()
    print(f"governed days (z)   : {int(sig['z'].notna().sum())}")
    print(f"hr >= 0 days        : {int((governed_hr >= 0).sum())}  "
          f"[expect 0 -- the strategy vetoes entries there]")
    dead = unmanaged_dead_quarters(sig["z"], LOOKBACK_DAYS)
    dead_lbl = (
        " ".join(d.date().isoformat() for d in dead) if dead else "[expect 0]"
    )
    print(f"unmanaged dead qtrs : {len(dead)}  {dead_lbl}")

    try:
        funding_diagnostic()
    except Exception as exc:  # noqa: BLE001 -- diagnostic must not block the rest
        print(f"\n[skip] funding diagnostic failed ({exc!r}).")

    # --- Reference comparison (optional: needs the research backtest deps) ---
    sys.path.insert(0, PROJECT)
    try:
        import jansen_backtest as jb
    except Exception as exc:  # noqa: BLE001
        print(f"\n[skip] research backtest not importable ({exc!r}); "
              "live diagnostics above only.")
        return

    config = {
        "data_dir": "data/feather", "feather_dir": "data/feather",
        "output_dir": "output", "interval": "1d", "quote": "USDT",
        "symbol_y": "LTC", "symbol_x": "XRP", "start_equity": 1000.0,
        "entry_z": ENTRY_Z, "risk_limit": -0.2, "fee_rate": 0.001,
        "lookback_days": LOOKBACK_DAYS, "trade_window_months": 6,
        "entry_window_months": 3, "kalman": KALMAN,
    }
    ref = jb.backtest_pair(config, save_output=False).reindex(df.index)

    pos = ref["position"].fillna(0.0)
    npos = ref["n_positions"].fillna(0.0) if "n_positions" in ref else pd.Series(
        0.0, index=ref.index
    )
    ref_sign = np.sign(pos)
    overlap = npos >= 2                             # days live cannot represent

    print("\n=== vs research backtest (LTC/XRP) ===")
    print(f"ref first trade day : "
          f"{pos[pos != 0].index.min().date() if (pos != 0).any() else 'never'}")
    print(f"ref non-flat days   : {int((pos != 0).sum())}")
    print(f"ref max n_positions : {int(npos.max())}")
    print(f"ref overlap (>=2)   : {int(overlap.sum())} days  "
          f"(live cannot represent these by design)")

    # Live fills its signal ~seconds after the daily close (~close[t]); the
    # backtest fills at close[t+1] (z_signal = z.shift(1), backtest L604-605).
    # shift=1 moves the live state onto the backtest's fill day, isolating that
    # one-bar offset: opposite-sign days should drop to ~0 there, leaving only
    # the structural single-vs-overlapping-cohort residual.
    for shift, label in ((0, "as-computed"), (1, "on backtest fill day")):
        live_sign = np.sign(state.shift(shift).fillna(0.0))
        active = (live_sign != 0) | (ref_sign != 0)
        single = active & (npos <= 1)               # days live could match
        n_single = int(single.sum())
        if not n_single:
            continue
        diff = (live_sign != ref_sign) & single
        live_only = int((diff & (ref_sign == 0)).sum())
        ref_only = int((diff & (live_sign == 0)).sum())
        opposite = int((diff & (live_sign != 0) & (ref_sign != 0)).sum())
        n_diff = int(diff.sum())
        print(f"sign disagreement   : {n_diff}/{n_single} single-position days "
              f"({100.0 * n_diff / n_single:.1f}%)  [shift={shift}, {label}]")
        print(f"                      live-only {live_only}, ref-only {ref_only}, "
              f"opposite {opposite}"
              + ("  [expect opposite ~0]" if shift == 1 else ""))

    hr_diff = (sig["hr"] - ref["hedge_ratio"]).abs().dropna()
    z_diff = (sig["z"] - ref["z_score"]).abs().dropna()
    if not hr_diff.empty:
        print(f"hr  |diff| median   : {hr_diff.median():.4f}  "
              f"(max {hr_diff.max():.2f})  [expect ~0]")
    if not z_diff.empty:
        print(f"z   |diff| median   : {z_diff.median():.4f}  "
              f"(max {z_diff.max():.2f})  [expect ~0]")


if __name__ == "__main__":
    main()
