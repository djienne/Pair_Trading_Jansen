# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
"""Live (dry-run) Jansen pair-trading strategy for LTC/XRP.

A faithful freqtrade port of ``Pair_Trading_Jansen/jansen_backtest.py`` (which
ranked LTC/XRP the #1 pair, Sharpe 0.69, entry_z=2.0). The signal math lives in
``jansen_signals.py``; this class wires it into freqtrade as a *two-leg,
dollar-neutral* spread trade:

  * Whitelist both legs (LTC and XRP perps). ``max_open_trades = 2``.
  * The shared spread state (+1 long-spread / -1 short-spread / 0 flat) is
    computed once per candle from both legs' closes (sibling fetched via the
    DataProvider, the INTERMARKET informative pattern).
  * Each leg trades the opposite direction, sized so the two legs are
    gross-notional dollar-neutral via the time-varying hedge ratio.

           spread state |   LTC leg    |   XRP leg
           -------------+--------------+--------------
            +1 (z<-2)   |  enter_long  |  enter_short
            -1 (z>+2)   |  enter_short |  enter_long
             0 (z->0)   |    exit      |    exit

Exit is z crossing zero (the exit signal) plus a spread stop in ``custom_exit``
(combined P&L of both legs < -20% of gross notional, the backtest's risk_limit).

Switching dry-run -> real money, or Binance -> Hyperliquid, is purely a config
change; this strategy file is unchanged.
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
from pandas import DataFrame

from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade

# Make the sibling jansen_signals.py importable regardless of CWD.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from jansen_signals import compute_state_signal  # noqa: E402

logger = logging.getLogger(__name__)


class PairTradingJansen(IStrategy):

    INTERFACE_VERSION = 3

    # --- Spread definition: y = lead = LTC, x = lag = XRP -------------------
    LEAD_BASE = "LTC"
    LAG_BASE = "XRP"

    # --- Backtest parameters (Pair_Trading_Jansen/config.json) --------------
    entry_z = 2.0
    risk_limit = -0.20          # spread stop: combined PnL / gross notional
    lookback_days = 730         # half-life estimation window
    kalman = {
        "smoother_obs_cov": 1.0,
        "smoother_trans_cov": 0.05,
        "hedge_delta": 0.001,
        "hedge_obs_cov": 2.0,
    }
    target_capital_ratio = 0.95  # fraction of available capital deployed gross

    # --- freqtrade mechanics ------------------------------------------------
    timeframe = "1d"
    can_short = True
    process_only_new_candles = True
    use_exit_signal = True
    use_custom_stoploss = False
    position_adjustment_enable = False

    # ROI / hard stop are disabled; exits are driven by the z-score signal and
    # the spread stop in custom_exit().
    minimal_roi = {"0": 100.0}
    stoploss = -0.99

    trailing_stop = False

    # The Kalman hedge ratio is a forward filter, so the current hr/z need a long
    # warm-up to converge: 730d half-life lookback + Kalman convergence margin.
    # 999 daily candles (~2.7y) fits one Binance futures klines request (limit
    # 1500) and matches the research backtest's ~730d per-period lookback.
    startup_candle_count: int = 999

    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "gtc", "exit": "gtc"}

    # ----------------------------------------------------------------------- #
    # Pair helpers
    # ----------------------------------------------------------------------- #
    def _leg(self, base: str) -> str:
        stake = self.config.get("stake_currency", "USDT")
        if str(self.config.get("trading_mode", "")).lower() == "futures":
            return f"{base}/{stake}:{stake}"
        return f"{base}/{stake}"

    @property
    def lead_pair(self) -> str:
        return self._leg(self.LEAD_BASE)

    @property
    def lag_pair(self) -> str:
        return self._leg(self.LAG_BASE)

    def informative_pairs(self):
        return [(self.lead_pair, self.timeframe), (self.lag_pair, self.timeframe)]

    # ----------------------------------------------------------------------- #
    # Indicators / signal
    # ----------------------------------------------------------------------- #
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["pair_state"] = 0.0
        dataframe["hr"] = np.nan
        dataframe["z"] = np.nan

        lead = self.dp.get_pair_dataframe(self.lead_pair, self.timeframe)
        lag = self.dp.get_pair_dataframe(self.lag_pair, self.timeframe)
        if lead.empty or lag.empty:
            logger.warning(
                "Missing leg data (lead empty=%s, lag empty=%s); no signal for %s.",
                lead.empty, lag.empty, metadata.get("pair"),
            )
            return dataframe

        lead_s = lead.drop_duplicates("date").set_index("date")["close"].sort_index()
        lag_s = lag.drop_duplicates("date").set_index("date")["close"].sort_index()
        common = lead_s.index.intersection(lag_s.index)
        if len(common) < 120:
            return dataframe
        lead_s = lead_s.loc[common]
        lag_s = lag_s.loc[common]

        sig = compute_state_signal(
            lead_s,
            lag_s,
            entry_z=self.entry_z,
            lookback_days=self.lookback_days,
            kalman=self.kalman,
        )

        # Map the date-indexed signal back onto this pair's (date-keyed) rows.
        dataframe["pair_state"] = dataframe["date"].map(sig["state"]).fillna(0.0)
        dataframe["hr"] = dataframe["date"].map(sig["hr"])
        dataframe["z"] = dataframe["date"].map(sig["z"])
        return dataframe

    # ----------------------------------------------------------------------- #
    # Entries / exits (direction depends on which leg this is)
    # ----------------------------------------------------------------------- #
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        state = dataframe["pair_state"]
        if metadata["pair"] == self.lead_pair:           # LTC
            dataframe.loc[state == 1, "enter_long"] = 1
            dataframe.loc[state == -1, "enter_short"] = 1
        else:                                            # XRP (opposite)
            dataframe.loc[state == -1, "enter_long"] = 1
            dataframe.loc[state == 1, "enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        state = dataframe["pair_state"]
        if metadata["pair"] == self.lead_pair:           # LTC
            dataframe.loc[state <= 0, "exit_long"] = 1   # flat or flipped -> close long
            dataframe.loc[state >= 0, "exit_short"] = 1
        else:                                            # XRP (opposite)
            dataframe.loc[state >= 0, "exit_long"] = 1
            dataframe.loc[state <= 0, "exit_short"] = 1
        return dataframe

    # ----------------------------------------------------------------------- #
    # Sizing: gross-notional dollar-neutral across the two legs
    # ----------------------------------------------------------------------- #
    def custom_stake_amount(
        self, pair, current_time, current_rate, proposed_stake, min_stake,
        max_stake, leverage, entry_tag, side, **kwargs,
    ) -> float:
        capital = float(
            self.config.get("available_capital")
            or self.config.get("dry_run_wallet")
            or 1000.0
        )
        target = capital * self.target_capital_ratio

        py = self._last_close(self.lead_pair)   # LTC price
        px = self._last_close(self.lag_pair)    # XRP price
        hr = self._last_value(self.lead_pair, "hr")

        if not (py and px and hr is not None and np.isfinite(hr) and abs(hr) > 1e-12):
            # Fall back to an even split if the hedge ratio is unavailable.
            stake = target / 2.0
        else:
            gross_per_unit = py + abs(hr) * px
            if pair == self.lead_pair:
                stake = target * py / gross_per_unit
            else:
                stake = target * abs(hr) * px / gross_per_unit

        if max_stake:
            stake = min(stake, max_stake)
        if min_stake:
            stake = max(stake, min_stake)
        return float(stake)

    def leverage(
        self, pair, current_time, current_rate, proposed_leverage, max_leverage,
        entry_tag, side, **kwargs,
    ) -> float:
        return 1.0

    # ----------------------------------------------------------------------- #
    # Spread stop: combined PnL of both legs vs gross notional (risk_limit)
    # ----------------------------------------------------------------------- #
    def custom_exit(
        self, pair, trade: Trade, current_time, current_rate, current_profit, **kwargs,
    ):
        legs = [
            t for t in Trade.get_open_trades()
            if t.pair in (self.lead_pair, self.lag_pair)
        ]
        if len(legs) < 2:
            # Hedge incomplete -> fall back to a per-leg stop at the same limit.
            if current_profit < self.risk_limit:
                return "spread_stop_single"
            return None

        total_pnl = 0.0
        total_gross = 0.0
        for t in legs:
            price = current_rate if t.pair == pair else self._last_close(t.pair)
            if not price:
                return None
            direction = -1.0 if t.is_short else 1.0
            total_pnl += direction * (price - t.open_rate) * t.amount
            total_gross += abs(t.open_rate * t.amount)

        if total_gross > 0 and (total_pnl / total_gross) < self.risk_limit:
            return "spread_stop"
        return None

    # ----------------------------------------------------------------------- #
    # Small DataProvider helpers
    # ----------------------------------------------------------------------- #
    def _last_close(self, pair: str):
        return self._last_value(pair, "close")

    def _last_value(self, pair: str, column: str):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or df.empty or column not in df.columns:
            return None
        val = df[column].iloc[-1]
        return None if val is None or (isinstance(val, float) and np.isnan(val)) else float(val)
