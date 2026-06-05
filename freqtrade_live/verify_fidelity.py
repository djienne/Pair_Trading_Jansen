"""Fidelity check: live single-cohort signal vs the research backtest.

Read-only diagnostic. Run from anywhere:

    python freqtrade_live/verify_fidelity.py

Loads the local daily feather data for LTC/XRP, computes the LIVE signal
(``freqtrade_live/user_data/strategies/jansen_signals.compute_state_signal``),
and -- when the research backtest is importable -- compares it against
``jansen_backtest``'s reference position / z-score / hedge-ratio.

Prints diagnostics only; it does not write or modify anything. The point is to
show the post-fix signal (a) no longer trades for the first ~2 years, (b) agrees
with the backtest sign on the vast majority of single-position days, and (c) has
small hr/z divergence -- with residual disagreement concentrated on the backtest
days that held two overlapping cohorts (which the single-cohort live bot cannot
represent by design).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))            # .../freqtrade_live
PROJECT = os.path.dirname(HERE)                              # .../Pair_Trading_Jansen
DATA_DIR = os.path.join(PROJECT, "data", "feather")
STRAT_DIR = os.path.join(HERE, "user_data", "strategies")

sys.path.insert(0, STRAT_DIR)
from jansen_signals import compute_state_signal  # noqa: E402

KALMAN = {
    "smoother_obs_cov": 1.0,
    "smoother_trans_cov": 0.05,
    "hedge_delta": 0.001,
    "hedge_obs_cov": 2.0,
}
ENTRY_Z = 2.0
LOOKBACK_DAYS = 730


def load_close(symbol: str) -> pd.Series:
    path = os.path.join(DATA_DIR, f"{symbol}USDT_1d.feather")
    df = pd.read_feather(path)
    df["open_time_dt"] = pd.to_datetime(df["open_time_dt"])
    df = df.sort_values("open_time_dt").set_index("open_time_dt")
    return df["close"].rename(symbol)


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
    live_sign = np.sign(state)
    ref_sign = np.sign(pos)

    active = (state != 0) | (pos != 0)              # either side has a position
    single = active & (npos <= 1)                   # days live could match
    overlap = npos >= 2                             # days live cannot represent
    n_single = int(single.sum())
    disagree = int(((live_sign != ref_sign) & single).sum())

    print("\n=== vs research backtest (LTC/XRP) ===")
    print(f"ref first trade day : "
          f"{pos[pos != 0].index.min().date() if (pos != 0).any() else 'never'}")
    print(f"ref non-flat days   : {int((pos != 0).sum())}")
    print(f"ref max n_positions : {int(npos.max())}")
    print(f"ref overlap (>=2)   : {int(overlap.sum())} days  "
          f"(live cannot represent these by design)")
    if n_single:
        print(f"sign DISagreement   : {disagree}/{n_single} single-position days "
              f"({100.0 * disagree / n_single:.1f}%)  "
              f"[reviewer baseline was 501/1061]")

    hr_diff = (sig["hr"] - ref["hedge_ratio"]).abs().dropna()
    z_diff = (sig["z"] - ref["z_score"]).abs().dropna()
    if not hr_diff.empty:
        print(f"hr  |diff| median   : {hr_diff.median():.4f}  "
              f"(max {hr_diff.max():.2f})  [baseline median 4.50]")
    if not z_diff.empty:
        print(f"z   |diff| median   : {z_diff.median():.4f}  "
              f"(max {z_diff.max():.2f})  [baseline median 0.15]")


if __name__ == "__main__":
    main()
