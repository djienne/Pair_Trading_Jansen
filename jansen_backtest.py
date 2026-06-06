import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

pd.set_option('future.no_silent_downcasting', True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils import load_config, resolve_path, get_data_path, get_output_path

try:
    from pykalman import KalmanFilter
except ImportError as exc:
    raise ImportError(
        "pykalman is required. Please install with: pip install pykalman"
    ) from exc


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class Position:
    """Represents an open pair trading position."""
    period: int
    side: float  # +1 for long spread, -1 for short spread
    hr: float  # hedge ratio
    size_y: float
    size_x: float
    entry_value: float  # net position value at entry/rebalance (size_y*py + size_x*px)
    entry_gross: float  # gross notional at entry/rebalance (|size_y|*py + |size_x|*px)

    def compute_value(self, price_y: float, price_x: float) -> float:
        """Compute current position value."""
        return self.size_y * price_y + self.size_x * price_x

    def compute_spread_return(self, price_y: float, price_x: float) -> float:
        """Compute P&L as a fraction of gross notional deployed.

        Using gross notional (always > 0) as the denominator keeps the metric
        stable for dollar-neutral hedged spreads, where the *net* entry value
        can be ~0 and would otherwise make the ratio blow up or disable stops.
        """
        if not np.isfinite(self.entry_gross) or self.entry_gross <= 0:
            return 0.0
        current_value = self.compute_value(price_y, price_x)
        # Positive return = profit, negative return = loss
        return (current_value - self.entry_value) / self.entry_gross


@dataclass
class DayMetrics:
    """Metrics for a single trading day."""
    position: float = 0.0
    n_positions: int = 0
    trade_count: int = 0
    strategy_return: float = 0.0
    equity: float = 0.0
    turnover: float = 0.0
    cash: float = 0.0
    size_y: float = 0.0
    size_x: float = 0.0
    position_value: float = 0.0
    spread_value: float = 0.0
    spread_return: float = 0.0


# ---------------------------------------------------------------------------
# Price Loading
# ---------------------------------------------------------------------------
def load_price_series(symbol: str, config: dict, base_dir: str) -> pd.Series:
    path = get_data_path(base_dir, symbol, config)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing data file: {path}")
    df = pd.read_feather(path)
    df["open_time_dt"] = pd.to_datetime(df["open_time_dt"])
    df = df.sort_values("open_time_dt").set_index("open_time_dt")
    return df["close"].rename(symbol)


# ---------------------------------------------------------------------------
# Kalman Filter Defaults
# ---------------------------------------------------------------------------
DEFAULT_KALMAN = {
    "smoother_obs_cov": 1.0,
    "smoother_trans_cov": 0.05,
    "hedge_delta": 0.001,
    "hedge_obs_cov": 2.0,
}


def get_kalman_params(config: dict) -> dict:
    """Get Kalman filter parameters from config with defaults."""
    kalman_config = config.get("kalman", {})
    return {
        key: kalman_config.get(key, default)
        for key, default in DEFAULT_KALMAN.items()
    }


def kf_smoother(prices: pd.Series, config: dict | None = None) -> pd.Series:
    """Apply Kalman filter smoothing to price series."""
    params = get_kalman_params(config or {})
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


def kf_hedge_ratio(x: pd.Series, y: pd.Series, config: dict | None = None) -> np.ndarray:
    """Estimate time-varying hedge ratio using Kalman filter."""
    params = get_kalman_params(config or {})
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
    """Estimate the mean-reversion half-life of a spread.

    Regresses the spread change on the lagged spread (Ornstein-Uhlenbeck
    discretisation). Returns ``None`` when the spread is not mean-reverting
    (``beta >= 0``) or the regression is degenerate/singular, signalling the
    caller to skip trading that period rather than trade on a garbage window.
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


def get_quarter_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.to_datetime(index)
    return index.to_series().resample("QE").last().dropna().index


def build_period_events(
    z_score: pd.Series,
    entry_z: float,
    entry_window_months: int,
) -> pd.DataFrame:
    empty_events = pd.DataFrame({"side": pd.Series(dtype=float)})
    if z_score.empty:
        return empty_events.copy()

    # Identify Entry Signals
    # Short Spread (-1) when Z > entry_z (Spread is too high, expect reversion down)
    # Long Spread (+1) when Z < -entry_z (Spread is too low, expect reversion up)
    is_above = z_score > entry_z
    is_below = z_score < -entry_z

    # Identify Exit Signals (Crossing Zero).
    # Require both the current and previous sign to be valid so the first bar
    # and any NaN warm-up bars are never flagged as a crossing.
    sign = np.sign(z_score)
    prev_sign = sign.shift(1)
    cross_zero = (sign != prev_sign) & prev_sign.notna() & sign.notna()

    # Create event series
    # We only care about the first instance of crossing the threshold.
    # Use fill_value=False so the shifted boolean masks keep bool dtype; a
    # plain .fillna(False) coerces to object and makes `~` a no-op.
    entry_signals = pd.Series(np.nan, index=z_score.index, dtype=float)
    entry_signals.loc[is_above & ~is_above.shift(1, fill_value=False)] = -1.0
    entry_signals.loc[is_below & ~is_below.shift(1, fill_value=False)] = 1.0

    exit_signals = pd.Series(np.nan, index=z_score.index, dtype=float)
    exit_signals.loc[cross_zero] = 0.0

    # Combine signals
    trades = pd.concat([entry_signals.dropna(), exit_signals.dropna()])
    if trades.empty:
        return empty_events.copy()

    # Sort by date
    trades = trades.sort_index()

    # Remove adjacent duplicates (e.g., 1 -> 1 -> 0 -> 0)
    trades = trades.loc[trades.shift() != trades]

    # Enforce window constraints
    # Entries must happen within the first `entry_window_months`
    first_start = z_score.index.min()
    first_end = (
        first_start
        + pd.DateOffset(months=entry_window_months)
        - pd.DateOffset(days=1)
    )

    # Slice events to the entry window
    # Note: We must allow for an exit event that happens *after* the entry window
    # if we are currently holding a position.
    entry_window_events = trades.loc[first_start:first_end]

    if entry_window_events.empty:
        return empty_events.copy()

    # If the first event in the window is an exit (0), ignore it (we start flat)
    if entry_window_events.iloc[0] == 0.0:
        entry_window_events = entry_window_events.iloc[1:]

    if entry_window_events.empty:
        return empty_events.copy()

    final_events = entry_window_events.copy()

    # If the last event in the entry window leaves us open (non-zero),
    # we look for the next exit event in the full timeline
    if entry_window_events.iloc[-1] != 0.0:
        # Look for events after the entry window
        future_events = trades.loc[first_end:]
        # Find the first '0' (exit)
        future_exits = future_events[future_events == 0.0].head(1)
        if not future_exits.empty:
            final_events = pd.concat([final_events, future_exits]).sort_index()
            # Ensure no duplicates at the boundary
            final_events = final_events.loc[final_events.shift() != final_events]

    # Force-close at the trade-window boundary so a spread that never re-crosses
    # zero cannot leak a position past its own trading window. This is a window
    # boundary, not a signal, so it is intentionally NOT lagged.
    #
    # The simulator applies exits BEFORE entries on a given bar, so the period
    # is "open at the end" iff the last bar carrying events has an entry on it
    # (a same-bar exit only closes a *prior* sub-position). Detect that via the
    # resolved end state rather than the literal last row.
    last_bar = z_score.index.max()

    def _open_at_end(events: pd.Series) -> bool:
        if events.empty:
            return False
        last_ts = events.index.max()
        return bool((events.loc[[last_ts]] != 0.0).any())

    if _open_at_end(final_events) and final_events.index.max() == last_bar:
        # An entry on the very last window bar cannot be held or closed
        # in-window; drop last-bar entries (a same-bar exit is kept, as it may
        # still close a prior sub-position).
        keep = ~((final_events.index == last_bar) & (final_events != 0.0))
        final_events = final_events[keep]

    if _open_at_end(final_events):
        # The forced exit lands on a fresh, strictly-later timestamp than every
        # existing event, so no value-dedup is needed (and must be avoided: an
        # adjacent same-bar exit would otherwise swallow it).
        forced_exit = pd.Series([0.0], index=[last_bar])
        final_events = pd.concat([final_events, forced_exit]).sort_index()

    if final_events.empty:
        return empty_events.copy()

    return final_events.to_frame("side")


# ---------------------------------------------------------------------------
# Position Management Helpers
# ---------------------------------------------------------------------------
def compute_position_sizes(
    side: float, hr: float, target_value: float, price_y: float, price_x: float
) -> tuple[float, float]:
    """Size both legs by the gross notional of one spread unit.

    The spread unit is ``spread = price_y + hr * price_x`` (hr is typically
    negative, so a long spread is long Y / short X). Sizing ``k`` spread units
    so the gross notional ``|size_y|*price_y + |size_x|*price_x == target_value``
    keeps the two legs symmetric between long and short and bounded as
    ``hr -> 0`` or ``hr -> inf`` -- the old ``1/hr`` short-leg sizing exploded
    into unbounded leverage when the hedge ratio was small.
    """
    if not np.isfinite(hr) or abs(hr) <= 1e-12:
        return 0.0, 0.0
    gross_per_unit = price_y + abs(hr) * price_x
    if not np.isfinite(gross_per_unit) or gross_per_unit <= 0:
        return 0.0, 0.0
    k = target_value / gross_per_unit  # spread units; gross notional == target_value
    size_y = side * k
    size_x = side * k * hr
    return size_y, size_x


def close_position(
    pos: Position, price_y: float, price_x: float, fee_rate: float
) -> tuple[float, float]:
    """Close a position and return (cash_delta, turnover)."""
    delta_y = -pos.size_y
    delta_x = -pos.size_x
    trade_notional = abs(delta_y) * price_y + abs(delta_x) * price_x
    cash_delta = -(delta_y * price_y + delta_x * price_x)
    if fee_rate:
        cash_delta -= fee_rate * trade_notional
    return cash_delta, trade_notional


def compute_total_position_value(
    positions: dict[int, Position], price_y: float, price_x: float
) -> float:
    """Compute total value of all positions."""
    return sum(pos.compute_value(price_y, price_x) for pos in positions.values())


def compute_aggregate_metrics(
    positions: dict[int, Position], price_y: float, price_x: float
) -> tuple[float, float, float, float]:
    """Compute aggregate position metrics: (size_y, size_x, position_value, avg_spread_return)."""
    size_y_total = sum(pos.size_y for pos in positions.values())
    size_x_total = sum(pos.size_x for pos in positions.values())
    position_value = compute_total_position_value(positions, price_y, price_x)

    spread_returns = [
        pos.compute_spread_return(price_y, price_x)
        for pos in positions.values()
        if np.isfinite(pos.entry_gross) and pos.entry_gross > 0
    ]
    avg_spread_return = float(np.mean(spread_returns)) if spread_returns else 0.0

    return size_y_total, size_x_total, position_value, avg_spread_return


# ---------------------------------------------------------------------------
# Trade Simulation
# ---------------------------------------------------------------------------
def simulate_pair_trades(
    results: pd.DataFrame,
    trade_events: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Simulate pair trading strategy and return daily metrics."""
    start_equity = float(config.get("start_equity", 1000.0))
    fee_rate = float(config.get("fee_rate", 0.0))
    risk_limit = config.get("risk_limit", -0.2)
    risk_limit = float(risk_limit) if risk_limit is not None else None

    # Prepare events lookup
    if trade_events is None or trade_events.empty:
        trade_events = pd.DataFrame(columns=["period", "side", "hedge_ratio"])
    else:
        trade_events = trade_events.copy()
    trade_events = trade_events.sort_index()
    events_by_date = {ts: df for ts, df in trade_events.groupby(level=0, sort=True)}

    # State
    cash = start_equity
    positions: dict[int, Position] = {}
    trade_count = 0
    prev_value = start_equity
    daily_metrics: list[DayMetrics] = []

    for ts in results.index:
        price_y = results.at[ts, "y"]
        price_x = results.at[ts, "x"]
        portfolio_value_pre = cash + compute_total_position_value(positions, price_y, price_x)
        day_turnover = 0.0

        # Risk limit check - run every day, not just event days
        if risk_limit is not None:
            for period in list(positions.keys()):
                pos = positions[period]
                if pos.compute_spread_return(price_y, price_x) < risk_limit:
                    cash_delta, turnover = close_position(pos, price_y, price_x, fee_rate)
                    cash += cash_delta
                    day_turnover += turnover
                    positions.pop(period)

        events_today = events_by_date.get(ts)
        if events_today is not None and not events_today.empty:
            # Process closes
            close_periods = (
                events_today.loc[events_today["side"] == 0, "period"]
                .dropna().unique()
            )
            for period in close_periods:
                pos = positions.pop(period, None)
                if pos is None:
                    continue
                cash_delta, turnover = close_position(pos, price_y, price_x, fee_rate)
                cash += cash_delta
                day_turnover += turnover

            # Collect new entries
            entry_events = events_today.loc[events_today["side"] != 0]
            entries = [
                (row.period, float(row.side), row.hedge_ratio)
                for row in entry_events.itertuples()
                if row.period not in positions
            ]

            # Rebalance existing + open new positions
            if entries:
                target_value = portfolio_value_pre / (len(positions) + len(entries))

                # Rebalance existing positions
                for pos in positions.values():
                    new_size_y, new_size_x = compute_position_sizes(
                        pos.side, pos.hr, target_value, price_y, price_x
                    )
                    if new_size_y == 0.0 and new_size_x == 0.0:
                        continue
                    delta_y = new_size_y - pos.size_y
                    delta_x = new_size_x - pos.size_x
                    trade_notional = abs(delta_y) * price_y + abs(delta_x) * price_x
                    cash -= delta_y * price_y + delta_x * price_x
                    if fee_rate:
                        cash -= fee_rate * trade_notional
                    day_turnover += trade_notional
                    pos.size_y = new_size_y
                    pos.size_x = new_size_x
                    # Re-baseline risk tracking from the rebalanced book (the
                    # resize cash flow above already realizes the change).
                    pos.entry_value = new_size_y * price_y + new_size_x * price_x
                    pos.entry_gross = (
                        abs(new_size_y) * price_y + abs(new_size_x) * price_x
                    )

                # Open new positions
                for period, side, hr in entries:
                    new_size_y, new_size_x = compute_position_sizes(
                        side, hr, target_value, price_y, price_x
                    )
                    if new_size_y == 0.0 and new_size_x == 0.0:
                        continue
                    trade_notional = abs(new_size_y) * price_y + abs(new_size_x) * price_x
                    cash -= new_size_y * price_y + new_size_x * price_x
                    if fee_rate:
                        cash -= fee_rate * trade_notional
                    day_turnover += trade_notional
                    entry_value = price_y * new_size_y + price_x * new_size_x
                    entry_gross = (
                        abs(new_size_y) * price_y + abs(new_size_x) * price_x
                    )
                    positions[period] = Position(
                        period=period,
                        side=side,
                        hr=hr,
                        size_y=new_size_y,
                        size_x=new_size_x,
                        entry_value=entry_value,
                        entry_gross=entry_gross,
                    )
                    trade_count += 1

        # Compute end-of-day metrics
        size_y, size_x, position_value, avg_spread_return = compute_aggregate_metrics(
            positions, price_y, price_x
        )
        portfolio_value = cash + position_value
        daily_return = 0.0 if prev_value == 0 else portfolio_value / prev_value - 1
        prev_value = portfolio_value

        daily_metrics.append(DayMetrics(
            position=sum(pos.side for pos in positions.values()),
            n_positions=len(positions),
            trade_count=trade_count,
            strategy_return=daily_return,
            equity=portfolio_value,
            turnover=day_turnover,
            cash=cash,
            size_y=size_y,
            size_x=size_x,
            position_value=position_value,
            spread_value=position_value,
            spread_return=avg_spread_return,
        ))

    # Convert to DataFrame
    return pd.DataFrame(
        [
            {
                "position": m.position,
                "n_positions": m.n_positions,
                "trade_count": m.trade_count,
                "strategy_return": m.strategy_return,
                "equity": m.equity,
                "turnover": m.turnover,
                "cash": m.cash,
                "size_y": m.size_y,
                "size_x": m.size_x,
                "position_value": m.position_value,
                "spread_value": m.spread_value,
                "spread_return": m.spread_return,
            }
            for m in daily_metrics
        ],
        index=results.index,
    )


def backtest_pair(
    config: dict,
    save_output: bool = True,
    output_tag: str | None = None,
) -> pd.DataFrame:
    output_dir = resolve_path(BASE_DIR, config.get("output_dir", "output"))
    os.makedirs(output_dir, exist_ok=True)

    y_symbol = config.get("symbol_y", "BNB")
    x_symbol = config.get("symbol_x", "SOL")

    y = load_price_series(y_symbol, config, BASE_DIR)
    x = load_price_series(x_symbol, config, BASE_DIR)

    df = pd.concat([y, x], axis=1).dropna()
    df = df.sort_index()
    if df.empty:
        raise ValueError("No overlapping data between the two symbols.")

    lookback_days = int(config.get("lookback_days", 730))
    trade_window_months = int(config.get("trade_window_months", 6))
    entry_window_months = int(config.get("entry_window_months", 3))
    entry_z = float(config.get("entry_z", 2.0))

    results = pd.DataFrame(
        {
            "y": df[y_symbol],
            "x": df[x_symbol],
            "y_smooth": np.nan,
            "x_smooth": np.nan,
            "hedge_ratio": np.nan,
            "spread": np.nan,
            "z_score": np.nan,
            "z_signal": np.nan,
            "position": 0.0,
        },
        index=df.index,
    )

    trade_events = []
    for period_id, test_end in enumerate(get_quarter_ends(df.index), start=1):
        trading_start = test_end + pd.DateOffset(days=1)
        lookback_start = trading_start - pd.DateOffset(days=lookback_days)
        trading_end = (
            trading_start
            + pd.DateOffset(months=trade_window_months)
            - pd.DateOffset(days=1)
        )
        if lookback_start < df.index.min() or trading_end > df.index.max():
            continue

        period_prices = df.loc[lookback_start:trading_end, [y_symbol, x_symbol]]
        if period_prices.empty:
            continue

        y_smooth = kf_smoother(period_prices[y_symbol], config)
        x_smooth = kf_smoother(period_prices[x_symbol], config)

        hedge_states = kf_hedge_ratio(x_smooth, y_smooth, config)
        hedge_ratio = pd.Series(hedge_states[:, 0], index=period_prices.index)

        spread = period_prices[y_symbol] + period_prices[x_symbol] * hedge_ratio
        lookback_spread = spread.loc[lookback_start:test_end]
        if lookback_spread.empty:
            continue

        half_life = estimate_half_life(lookback_spread)
        if half_life is None:
            # Spread is not mean-reverting in the lookback; skip this period.
            continue
        max_window = len(lookback_spread)
        if max_window < 1:
            continue
        window = min(2 * half_life, max_window)

        rolling = spread.rolling(window=window, min_periods=window)
        z_score = (spread - rolling.mean()) / rolling.std()
        # Lag the signal by one bar to avoid same-bar lookahead: a crossing
        # detected from close[t] is acted on at close[t+1]. Shift the full
        # series BEFORE slicing so the first trade-window bar inherits a valid
        # value from the last lookback bar instead of NaN.
        z_signal = z_score.shift(1)

        trade_slice = slice(trading_start, trading_end)
        trade_index = spread.loc[trade_slice].index
        if trade_index.empty:
            continue
        trade_z = z_score.loc[trade_slice]            # raw z, diagnostics only
        trade_z_signal = z_signal.loc[trade_slice]    # lagged z, drives trades
        events = build_period_events(trade_z_signal, entry_z, entry_window_months)
        if not events.empty:
            events = events.copy()
            events["period"] = period_id
            events["hedge_ratio"] = (
                hedge_ratio.reindex(events.index).ffill().values
            )
            trade_events.append(events)

        period_series = {
            "y_smooth": y_smooth.loc[trade_index],
            "x_smooth": x_smooth.loc[trade_index],
            "hedge_ratio": hedge_ratio.loc[trade_index],
            "spread": spread.loc[trade_index],
            "z_score": trade_z,
            "z_signal": trade_z_signal,
        }
        for column, series in period_series.items():
            results.loc[trade_index, column] = series

    if trade_events:
        trade_events = pd.concat(trade_events).sort_index()
    else:
        trade_events = pd.DataFrame(columns=["period", "side", "hedge_ratio"])
    sim = simulate_pair_trades(results, trade_events, config)
    results["position"] = sim["position"]
    results["n_positions"] = sim["n_positions"]
    results["trade_count"] = sim["trade_count"]
    results["strategy_return"] = sim["strategy_return"]
    results["equity"] = sim["equity"]
    results["turnover"] = sim["turnover"]
    results["cash"] = sim["cash"]
    results["size_y"] = sim["size_y"]
    results["size_x"] = sim["size_x"]
    results["position_value"] = sim["position_value"]
    results["spread_value"] = sim["spread_value"]
    results["spread_return"] = sim["spread_return"]

    if save_output:
        out_path = get_output_path(BASE_DIR, config, y_symbol, x_symbol, output_tag)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        results.reset_index().to_feather(out_path)

    return results


def plot_equity(results: pd.DataFrame, output_dir: str, name: str, show_plot: bool) -> None:
    equity = results["equity"]
    drawdown = equity / equity.cummax() - 1.0

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    axes[0].plot(results.index, equity, label="Equity", linewidth=1.5)
    axes[0].set_title(f"Jansen Method Equity ({name})")
    axes[0].set_ylabel("Equity (USD)")
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(results.index, drawdown, 0.0, color="#e74c3c", alpha=0.35)
    axes[1].plot(results.index, drawdown, color="#c0392b", linewidth=1.0)
    axes[1].set_title("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, f"equity_{name}.png")
    fig.savefig(path, dpi=150)
    if show_plot:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Jansen method pair trading backtest (BNB/SOL)."
    )
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config.json"),
        help="Path to config.json",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = resolve_path(BASE_DIR, config.get("output_dir", "output"))

    results = backtest_pair(config)
    name = f"{config.get('symbol_y', 'BNB')}_{config.get('symbol_x', 'SOL')}"
    show_plot = bool(config.get("show_plot", True))
    plot_equity(results, output_dir, name, show_plot)

    final_equity = results["equity"].iloc[-1]
    print(f"Final equity for {name}: {final_equity:.2f} USD")


if __name__ == "__main__":
    main()
