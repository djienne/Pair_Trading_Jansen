"""Jansen pair-trading signal logic, ported from ``jansen_backtest.py``.

This module contains the *pure* math used by the live freqtrade strategy so it
stays a faithful, side-effect-free port of the research backtest:

  * ``kf_smoother``       -- 1-state Kalman price smoother      (backtest L108-120)
  * ``kf_hedge_ratio``    -- 2-state Kalman time-varying hedge  (backtest L123-140)
  * ``estimate_half_life``-- Ornstein-Uhlenbeck half-life       (backtest L143-164)
  * ``compute_state_signal`` -- ties them together into a position-state series

The signal is expressed as a *position state* (+1 long-spread, -1 short-spread,
0 flat) in the spirit of ``INTERMARKET.threshold_revert_signal``: enter when
``|z| > entry_z`` and hold until z crosses zero. This maps cleanly onto
freqtrade's enter/exit columns for each leg.

The Kalman filters, half-life and z-window are recalibrated per calendar quarter
over the trailing ``lookback_days`` (the backtest's per-period cadence), and a
quarter only becomes tradable once a full lookback sits behind it. A quarter
whose spread fails the mean-reversion gate is managed under the *previous*
quarter's calibration (the backtest's months-4-6 coverage). This is a
*single-cohort* projection of the backtest: one continuous position, NOT the
backtest's overlapping period positions. See ``compute_state_signal`` and the
strategy's "Known divergences" note.

Sign conventions (must match the backtest):
    spread = price_y + hr * price_x          # y = lead (LTC), x = lag (XRP)
    hr is the Kalman slope, typically < 0.
    state = -1 (short spread) when z >  entry_z   -> short y / long x
    state = +1 (long  spread) when z < -entry_z   -> long  y / short x
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from pykalman import KalmanFilter
except ImportError as exc:  # pragma: no cover - surfaced at strategy load time
    raise ImportError(
        "pykalman is required. Install with: pip install pykalman"
    ) from exc


# Defaults match jansen_backtest.DEFAULT_KALMAN / config.json "kalman" block.
DEFAULT_KALMAN = {
    "smoother_obs_cov": 1.0,
    "smoother_trans_cov": 0.05,
    "hedge_delta": 0.001,
    "hedge_obs_cov": 2.0,
}


def get_kalman_params(kalman: dict | None) -> dict:
    kalman = kalman or {}
    return {key: kalman.get(key, default) for key, default in DEFAULT_KALMAN.items()}


def kf_smoother(prices: pd.Series, kalman: dict | None = None) -> pd.Series:
    """Kalman-smooth a price series (backtest ``kf_smoother``)."""
    params = get_kalman_params(kalman)
    kf = KalmanFilter(
        transition_matrices=np.eye(1),
        observation_matrices=np.eye(1),
        initial_state_mean=0,
        initial_state_covariance=1,
        observation_covariance=params["smoother_obs_cov"],
        transition_covariance=params["smoother_trans_cov"],
    )
    state_means, _ = kf.filter(prices.values)
    return pd.Series(state_means.flatten(), index=prices.index)


def kf_hedge_ratio(x: pd.Series, y: pd.Series, kalman: dict | None = None) -> np.ndarray:
    """Time-varying hedge ratio via 2-state Kalman (backtest ``kf_hedge_ratio``).

    Returns ``-state_means`` (shape ``(n, 2)``); column 0 is the hedge ratio.
    """
    params = get_kalman_params(kalman)
    delta = params["hedge_delta"]
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)
    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=[0, 0],
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=params["hedge_obs_cov"],
        transition_covariance=trans_cov,
    )
    state_means, _ = kf.filter(y.values)
    return -state_means


def estimate_half_life(spread: pd.Series) -> int | None:
    """OU mean-reversion half-life (backtest ``estimate_half_life``).

    Returns ``None`` when the spread is not mean-reverting (``beta >= 0``) or the
    regression is degenerate, signalling the caller to not trade.
    """
    spread = spread.dropna()
    if len(spread) < 3:
        return None
    X = spread.shift().iloc[1:].to_frame().assign(const=1)
    y = spread.diff().iloc[1:]
    try:
        coeffs, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
    except np.linalg.LinAlgError:
        return None
    beta = coeffs[0]
    if not np.isfinite(beta) or beta >= 0:
        return None
    halflife = int(round(-np.log(2) / beta))
    return max(halflife, 1)


def _state_from_z(z: np.ndarray, entry_z: float) -> np.ndarray:
    """Threshold-revert position state from a z-score array.

    Mirrors the backtest's entry (``|z| > entry_z``) + exit-on-zero-cross logic,
    expressed as a held position so freqtrade can stay in the trade until z
    reverts. NaN warm-up bars carry the previous (flat) state forward.
    """
    state = np.zeros(len(z))
    position = 0
    for i in range(len(z)):
        zi = z[i]
        if np.isnan(zi):
            state[i] = position
            continue
        if zi > entry_z:
            position = -1          # spread too high -> short spread
        if zi < -entry_z:
            position = 1           # spread too low  -> long spread
        if position == -1 and zi <= 0:
            position = 0           # z crossed back through zero -> exit
        if position == 1 and zi >= 0:
            position = 0
        state[i] = position
    return state


def _quarter_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Calendar-quarter boundary dates present in ``index``.

    Matches the research backtest's ``get_quarter_ends`` (backtest L167-170):
    the last available bar in each calendar quarter. These anchor the per-cohort
    recalibration so the live signal recalibrates on exactly the same cadence as
    the backtest, and -- being calendar-anchored, not anchored to the dataframe's
    first row -- they are stable whether computed on a short (live) or long
    (backtest) window.
    """
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.to_datetime(index)
    return index.to_series().resample("QE").last().dropna().index


def compute_state_signal(
    lead_close: pd.Series,
    lag_close: pd.Series,
    *,
    entry_z: float = 2.0,
    lookback_days: int = 730,
    min_history: int = 120,
    kalman: dict | None = None,
) -> pd.DataFrame:
    """Compute the pair-trade state for two aligned daily close series.

    Parameters
    ----------
    lead_close, lag_close
        Aligned (same DatetimeIndex) close series. ``lead`` is y (LTC), ``lag``
        is x (XRP).
    entry_z
        Entry threshold (config ``entry_z``; 2.0 is optimal for LTC/XRP).
    lookback_days
        Trailing window used to estimate the OU half-life (config ``lookback_days``).
    min_history
        Minimum aligned bars required before any signal is produced.

    Returns a DataFrame indexed like the inputs with columns:
        ``hr``     -- time-varying hedge ratio (col 0 of the Kalman state)
        ``spread`` -- price_y + hr * price_x
        ``z``      -- causal rolling z-score of the spread
        ``state``  -- +1 long-spread / -1 short-spread / 0 flat

    **Single-cohort, per-quarter recalibration.** The Kalman filters, half-life
    and z-window are recalibrated at each calendar-quarter boundary over the
    trailing ``lookback_days`` of data -- the same cadence and windowing as the
    research backtest's per-period loop (backtest L563-599) -- rather than a
    single global pass. Each date is governed by exactly one cohort (the most
    recent quarter boundary before it), so unlike the backtest there are NO
    overlapping cohorts: this is one continuous held position, not a sum of
    independent period positions. A quarter only becomes active once a full
    ``lookback_days`` of history sits behind it (the backtest's skip guard,
    L571), so no signal is produced for roughly the first two years of data.

    **Dead-quarter fallback.** A cohort that fails its gates (warm-up,
    ``min_history``, or the half-life mean-reversion test) publishes the
    *previous* passing cohort's hr/spread/z over its quarter, capped at that
    cohort's 6-month ``trading_end`` -- exactly the dates the backtest's
    previous period would still govern (its published months 4-6). When every
    cohort passes (true for the whole LTC/XRP history to date) the fallback
    never runs and the output is bit-identical to the unextended per-quarter
    computation.

    All columns are strictly causal (value at t uses only data <= t): the Kalman
    ``filter`` is a forward pass, the half-life uses only the lookback portion
    (strictly before every governed bar), and the z-window is trailing. No
    explicit ``shift(1)`` is applied -- freqtrade executes a signal on the next
    candle, which already reproduces the backtest's one-bar ``z_signal`` lag
    (backtest L604); shifting here as well would double-lag.
    """
    idx = lead_close.index
    out = pd.DataFrame(
        {"hr": np.nan, "spread": np.nan, "z": np.nan, "state": 0.0}, index=idx
    )
    if len(lead_close) < min_history:
        return out

    data_start = idx.min()
    data_end = idx.max()
    q_ends = _quarter_ends(idx)
    day = pd.DateOffset(days=1)
    lookback = pd.DateOffset(days=lookback_days)

    hr_full = pd.Series(np.nan, index=idx)
    spread_full = pd.Series(np.nan, index=idx)
    z_full = pd.Series(np.nan, index=idx)

    # Most recent PASSING cohort, kept for the dead-quarter fallback below.
    prev: dict | None = None

    for i, q_end in enumerate(q_ends):
        trading_start = q_end + day
        lookback_start = trading_start - lookback

        # Each date belongs to exactly one cohort: cohort i governs
        # (q_end, next_q_end]; the final cohort runs to the end of the data
        # ("now"). No overlap, no gap -> a single continuous position.
        slice_end = q_ends[i + 1] if i + 1 < len(q_ends) else data_end

        # The backtest computes each period over lookback + a 6-MONTH trade
        # window (backtest L567-571), so a period's calibration outlives its own
        # quarter by ~3 months. Mirror that: compute over the extended window so
        # this cohort can also manage the NEXT quarter if that one fails its
        # gates (the backtest's months-4-6 coverage). The Kalman filter and the
        # trailing rolling stats are forward-only recursions, so the appended
        # rows cannot change the values published for this cohort's own quarter.
        trading_end = trading_start + pd.DateOffset(months=6) - day
        ext_end = min(trading_end, data_end)

        # --- Gates, evaluated on the PRIMARY window only (identical pass/fail
        # decisions to the unextended implementation) -------------------------
        # Warm-up gate: the cohort needs a full lookback_days behind it, exactly
        # like the backtest skip guard (backtest L571).
        win_idx = idx[(idx >= lookback_start) & (idx <= slice_end)]
        cohort_ok = lookback_start >= data_start and len(win_idx) >= min_history

        hr_win = spread_win = None
        if cohort_ok:
            ext_idx = idx[(idx >= lookback_start) & (idx <= ext_end)]
            y_win = lead_close.loc[ext_idx]
            x_win = lag_close.loc[ext_idx]

            # Recalibrate the Kalman filters on THIS cohort's window only (the
            # backtest re-initialises them per period, backtest L578-584).
            y_smooth = kf_smoother(y_win, kalman)
            x_smooth = kf_smoother(x_win, kalman)
            hedge_states = kf_hedge_ratio(x_smooth, y_smooth, kalman)
            hr_win = pd.Series(hedge_states[:, 0], index=ext_idx)
            spread_win = y_win + x_win * hr_win

            # Half-life from the lookback portion only [lookback_start : q_end]
            # (backtest L585-589) -- strictly past data relative to every
            # governed bar. Not mean-reverting -> this cohort has no calibration.
            lb_spread = spread_win.loc[spread_win.index <= q_end]
            half_life = estimate_half_life(lb_spread)
            cohort_ok = half_life is not None

        if not cohort_ok:
            # Dead-quarter fallback: the backtest still manages these dates
            # under the PREVIOUS period's calibration (its 6-month window covers
            # them); publish the previous passing cohort's series, capped at
            # that cohort's trading_end. Two consecutive failures publish
            # nothing -- the cap makes the range empty -- exactly like the
            # backtest, where no period covers those dates. A warm-up failure
            # never has a passing predecessor (warm-up is monotone), so this is
            # a structural no-op there. Held positions stay manageable: the
            # fallback z keeps the zero-cross exit alive, on top of the stop.
            if prev is not None:
                fb_idx = idx[
                    (idx >= trading_start)
                    & (idx <= slice_end)
                    & (idx <= prev["trading_end"])
                ]
                if len(fb_idx):
                    logger.warning(
                        "Cohort at %s failed its gates; managing %d bars under "
                        "the previous cohort's calibration (the backtest's "
                        "months-4-6 coverage).", q_end.date(), len(fb_idx),
                    )
                    hr_full.loc[fb_idx] = prev["hr"].loc[fb_idx]
                    spread_full.loc[fb_idx] = prev["spread"].loc[fb_idx]
                    z_full.loc[fb_idx] = prev["z"].loc[fb_idx]
            continue

        window = max(2, int(min(2 * half_life, len(lb_spread))))

        roll = spread_win.rolling(window=window, min_periods=window)
        z_win = (spread_win - roll.mean()) / roll.std()

        # Publish only this cohort's governed dates (trading_start .. slice_end).
        gov_idx = ext_idx[(ext_idx >= trading_start) & (ext_idx <= slice_end)]
        hr_full.loc[gov_idx] = hr_win.loc[gov_idx]
        spread_full.loc[gov_idx] = spread_win.loc[gov_idx]
        z_full.loc[gov_idx] = z_win.loc[gov_idx]

        prev = {
            "hr": hr_win,
            "spread": spread_win,
            "z": z_win,
            "trading_end": trading_end,
        }

    # One continuous held position over the concatenated, per-quarter z. NaN
    # warm-up / skipped-cohort bars carry the previous position forward.
    out["hr"] = hr_full
    out["spread"] = spread_full
    out["z"] = z_full
    out["state"] = _state_from_z(z_full.values, entry_z)
    return out
