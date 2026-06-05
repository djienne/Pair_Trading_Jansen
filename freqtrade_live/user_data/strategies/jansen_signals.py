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

Sign conventions (must match the backtest):
    spread = price_y + hr * price_x          # y = lead (LTC), x = lag (XRP)
    hr is the Kalman slope, typically < 0.
    state = -1 (short spread) when z >  entry_z   -> short y / long x
    state = +1 (long  spread) when z < -entry_z   -> long  y / short x
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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


def _causal_zscore(
    spread: pd.Series, lookback_days: int, min_lookback: int = 90
) -> pd.Series:
    """Rolling z-score with a half-life window recalibrated *causally*.

    The z-score window is re-estimated at each **calendar-quarter boundary**
    from the trailing ``lookback_days`` of spread only (never future data), then
    held constant through the quarter -- exactly the research backtest's
    quarterly recalibration (``get_quarter_ends`` + per-period half-life,
    backtest L563-599). Because every value depends only on past data and the
    recalibration is anchored to the calendar (not to the dataframe's first row),
    z[t] is identical whether computed live (short df) or in backtest (long df),
    so the two stay in lockstep and no lookahead is introduced.
    """
    sp = spread.values
    n = len(sp)
    z = np.full(n, np.nan)
    quarters = spread.index.to_period("Q")

    cur_window: int | None = None
    last_q = None
    for t in range(n):
        q = quarters[t]
        if q != last_q:
            last_q = q
            lb_start = max(0, t - lookback_days + 1)
            if t - lb_start + 1 >= min_lookback:
                hl = estimate_half_life(pd.Series(sp[lb_start : t + 1]))
                if hl is not None:
                    cur_window = max(2, int(min(2 * hl, lookback_days)))
        if cur_window is not None and t + 1 >= cur_window:
            seg = sp[t + 1 - cur_window : t + 1]
            sd = seg.std(ddof=1)
            if sd > 0:
                z[t] = (sp[t] - seg.mean()) / sd
    return pd.Series(z, index=spread.index)


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

    All columns are strictly causal (value at t uses only data <= t): the Kalman
    ``filter`` is a forward pass and the z-window recalibrates only on trailing
    data. This keeps live and freqtrade-backtest results identical per date.
    """
    idx = lead_close.index
    out = pd.DataFrame(
        {"hr": np.nan, "spread": np.nan, "z": np.nan, "state": 0.0}, index=idx
    )
    n = len(lead_close)
    if n < min_history:
        return out

    y_smooth = kf_smoother(lead_close, kalman)
    x_smooth = kf_smoother(lag_close, kalman)

    hedge_states = kf_hedge_ratio(x_smooth, y_smooth, kalman)
    hr = pd.Series(hedge_states[:, 0], index=idx)

    spread = lead_close + lag_close * hr
    z = _causal_zscore(spread, lookback_days)
    state = _state_from_z(z.values, entry_z)

    out["hr"] = hr
    out["spread"] = spread
    out["z"] = z
    out["state"] = state
    return out
