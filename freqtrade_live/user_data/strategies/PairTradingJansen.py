# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
"""Live (dry-run) Jansen pair-trading strategy for LTC/XRP.

A *single-cohort* live adaptation of ``Pair_Trading_Jansen/jansen_backtest.py``
(where LTC/XRP now ranks #2 by active-day Sharpe, 1.46, entry_z=2.0). The signal
math lives in ``jansen_signals.py``; this class wires it into freqtrade as a
*two-leg, dollar-neutral* spread trade:

  * Whitelist both legs (LTC and XRP perps). ``max_open_trades = 2``.
  * The shared spread state (+1 long-spread / -1 short-spread / 0 flat) is
    computed once per candle from both legs' closes (sibling fetched via the
    DataProvider, the INTERMARKET informative pattern). The Kalman filters,
    half-life and z-window are recalibrated per calendar quarter over the
    trailing 730d (the backtest's per-period cadence).
  * Entries fire only on the candle where the state *crosses* into +-1 (the
    backtest's first-crossing events); exits fire on z crossing zero, plus the
    spread stop.
  * Each leg trades the opposite direction, sized so the two legs are
    gross-notional dollar-neutral via the time-varying hedge ratio, off the
    *current* wallet equity (compounding).

           spread state |   LTC leg    |   XRP leg
           -------------+--------------+--------------
            +1 (z<-2)   |  enter_long  |  enter_short
            -1 (z>+2)   |  enter_short |  enter_long
             0 (z->0)   |    exit      |    exit

Exits are LEG-COUPLED and centralized in ``custom_exit`` (``populate_exit_trend`` is
a no-op). Both legs decide from ONE shared spread state -- the lead pair's
``pair_state`` -- so they exit on the same iteration and cannot desync into a naked
single leg. ``custom_exit`` applies, in order: (1) an orphan safety-net that
force-closes a surviving leg whose sibling is gone (past a short entry-grace window),
(2) a coupled exit when the shared state reverts to 0 or flips against the entered
direction, and (3) the spread stop (combined P&L of both legs < -20% of gross
notional, the backtest's risk_limit). This replaces the earlier per-leg exit signals,
which could disagree on the boundary candle and orphan a leg.

Known divergences from ``jansen_backtest.py`` (the backtest is inherently
RETROSPECTIVE -- it only activates a quarterly cohort once that cohort's whole
6-month window is historical, jansen_backtest.py L571 -- so a bit-for-bit live
reproduction is impossible). This live strategy is faithful in *methodology*
but, by design:
  * holds ONE spread position; it cannot represent the backtest's overlapping
    cohorts (two simultaneous positions on ~121 of 732 backtest days);
  * has no 3-month entry-window gate and no 6-month forced window close;
  * is anchored to "now", not retrospective.
Use ``jansen_backtest.py`` for performance numbers; this bot is the live trader.

Switching dry-run -> real money, or Binance -> Hyperliquid, is purely a config
change; this strategy file is unchanged.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import timedelta

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
    orphan_grace_seconds = 180   # don't force-close a lone leg during the entry handshake
    # Per-leg liquidation guard (live-only; the backtest assumes cross-netting and
    # has no liquidation concept): with isolated 1x margin a big JOINT rally can
    # liquidate the short leg on-exchange while the combined spread PnL stays ~flat
    # (so the spread stop never fires). Close the whole book once any single leg
    # moves this far against itself. Binance 1x isolated short liquidation sits
    # around +96-99% adverse (maintenance-margin tiers), so 0.60 leaves ample
    # gap-risk margin and should essentially never fire.
    liq_guard_adverse = 0.60

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
        # Dates missing from the two-leg common index (e.g. the sibling lacks a
        # candle) carry the held state forward -- the same NaN-carry semantics as
        # _state_from_z -- instead of snapping to 0, which would fabricate a state
        # transition: a phantom exit plus a fake re-entry "crossing" on resume.
        dataframe["pair_state"] = (
            dataframe["date"].map(sig["state"]).ffill().fillna(0.0)
        )
        dataframe["hr"] = dataframe["date"].map(sig["hr"])
        dataframe["z"] = dataframe["date"].map(sig["z"])
        return dataframe

    # ----------------------------------------------------------------------- #
    # Entries / exits (direction depends on which leg this is)
    # ----------------------------------------------------------------------- #
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        state = dataframe["pair_state"]
        prev = state.shift(1)
        # Threshold-CROSSING entries: fire only on the candle where the spread
        # state transitions INTO +-1 (matching the backtest's first-crossing
        # entry events, jansen_backtest.py L199-201). The held `state` still
        # drives exits, but after a spread stop the state stays +-1 with no fresh
        # transition, so the bot does NOT re-enter without a new zero-cross +
        # threshold crossing.
        entered_long_spread = (state == 1) & (prev != 1)
        entered_short_spread = (state == -1) & (prev != -1)
        if metadata["pair"] == self.lead_pair:           # LTC
            dataframe.loc[entered_long_spread, "enter_long"] = 1
            dataframe.loc[entered_short_spread, "enter_short"] = 1
        else:                                            # XRP (opposite)
            dataframe.loc[entered_short_spread, "enter_long"] = 1
            dataframe.loc[entered_long_spread, "enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exits are CENTRALIZED in custom_exit() so both legs decide from one shared
        # spread state and can never desync into a naked single leg. We deliberately
        # set NO dataframe exit signals here.
        #
        # Why a no-op (not just tidier): freqtrade skips custom_exit() on any candle
        # where an exit signal is already set. The previous per-leg exit_long/exit_short
        # were computed from each leg's *independently recomputed* pair_state, which can
        # disagree on the boundary candle -- that asymmetry (a) orphaned a leg when only
        # one side's signal fired and (b) would suppress the coupled custom_exit. Driving
        # all exits through custom_exit() removes both failure modes.
        return dataframe

    # ----------------------------------------------------------------------- #
    # Sizing: gross-notional dollar-neutral across the two legs
    # ----------------------------------------------------------------------- #
    def custom_stake_amount(
        self, pair, current_time, current_rate, proposed_stake, min_stake,
        max_stake, leverage, entry_tag, side, **kwargs,
    ) -> float:
        # Size from CURRENT equity (compounding), matching the backtest's
        # portfolio_value-based sizing (jansen_backtest.py L425) rather than a
        # static capital number.
        target = self._current_equity() * self.target_capital_ratio

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
        # All exits are decided HERE (populate_exit_trend is a no-op) so the two legs
        # always act on ONE shared spread state and can never orphan.

        # --- 1) Coupled zero-cross exit (the primary exit) ------------------------
        # The literal backtest exit rule, evaluated per entered direction: a held
        # long-spread exits when z >= 0, a held short-spread when z <= 0 (the
        # backtest's cross-zero events / _state_from_z's exit branches). Both legs
        # read the SAME source of truth -- the lead pair's current z -- so they exit
        # on the same iteration (no per-leg recompute divergence). This is checked
        # FIRST so that, when the signal reverts, BOTH legs are tagged
        # "spread_revert" (rather than the second-processed leg being mislabelled an
        # orphan just because its sibling's exit was committed a step earlier).
        #
        # Deliberately NOT the reconstructed `pair_state`: live runs on a rolling
        # ~999-candle window while each cohort needs a 730d lookback, so only the
        # most recent ~270 days carry governed z. A position held longer than that
        # has its entry crossing fall off the window -- the recomputed state
        # collapses to 0 and a state-based exit would fire a phantom "spread_revert"
        # while |z| is still far from zero (same fragility across bot restarts). The
        # direct z test needs no state history; a flip beyond the opposite threshold
        # has necessarily crossed zero, so flips are covered too. NaN/missing z
        # (warm-up, skipped cohort) holds the position, matching the backtest, with
        # the spread stop below still active.
        z = self._last_value(self.lead_pair, "z")
        entered_dir = self._entered_spread_dir(trade)
        if z is not None and (
            (entered_dir == 1 and z >= 0.0) or (entered_dir == -1 and z <= 0.0)
        ):
            return "spread_revert"

        # --- 2) Orphan safety-net (genuine anomaly only) -------------------------
        # We get here only when the shared signal still says HOLD. If this trade's
        # sibling is nonetheless gone (e.g. it was closed by the spread stop, or any
        # desync), a lone leg is a naked directional bet -- force-close it. Guarded so
        # we do NOT fire during the brief two-leg entry handshake: skip while this trade
        # still has a pending order, and require a short grace window since it opened
        # (entries fill within seconds; a real orphan at a candle boundary is hours old).
        legs = [
            t for t in Trade.get_open_trades()
            if t.pair in (self.lead_pair, self.lag_pair)
        ]
        if len(legs) < 2:
            grace = timedelta(seconds=self.orphan_grace_seconds)
            if not trade.has_open_orders and (current_time - trade.open_date_utc) > grace:
                logger.warning(
                    "Orphan leg detected for %s (sibling gone while signal still holds); "
                    "force-closing this leg to restore a flat/neutral book.", pair,
                )
                return "orphan_close"
            return None

        # --- 3) Combined spread stop (both legs present) -------------------------
        # BOTH legs are priced at the last analyzed daily close -- reproducing the
        # backtest's once-per-day close-based risk check (jansen_backtest.py
        # L391-399). Mixing this leg's live tick with the sibling's day-old close
        # would register a joint intraday move (both alts crashing/rallying
        # together) as one-leg PnL and falsely trip the stop on a healthy hedge;
        # `current_rate` is therefore deliberately NOT used here.
        total_pnl = 0.0
        total_gross = 0.0
        max_adverse = 0.0
        for t in legs:
            price = self._last_close(t.pair)
            if not price:
                return None
            direction = -1.0 if t.is_short else 1.0
            total_pnl += direction * (price - t.open_rate) * t.amount
            total_gross += abs(t.open_rate * t.amount)
            leg_ret = price / t.open_rate - 1.0
            max_adverse = max(max_adverse, leg_ret if t.is_short else -leg_ret)

        if total_gross > 0 and (total_pnl / total_gross) < self.risk_limit:
            return "spread_stop"

        # --- 4) Per-leg liquidation guard (see liq_guard_adverse) -----------------
        # Both legs evaluate the same last-close data in the same iteration, so the
        # exit is symmetric (the second-processed leg may get tagged "orphan_close"
        # once its sibling's exit is committed -- same-iteration book closure either
        # way, mirroring the spread-stop note above).
        if max_adverse >= self.liq_guard_adverse:
            logger.warning(
                "Liquidation guard for %s: a leg is %.0f%% adverse vs entry while the "
                "spread PnL has not tripped the stop; closing the book to avoid an "
                "exchange liquidation of one leg.", pair, 100.0 * max_adverse,
            )
            return "leg_liq_guard"
        return None

    def _entered_spread_dir(self, trade: Trade) -> int:
        """Spread direction (+1 long-spread / -1 short-spread) this leg was entered for.

        lead (LTC):  long  -> +1, short -> -1
        lag  (XRP):  short -> +1, long  -> -1   (the opposite leg)
        Returns 0 for an unrecognised pair (defensive; never exits on it).
        """
        if trade.pair == self.lead_pair:
            return -1 if trade.is_short else 1
        if trade.pair == self.lag_pair:
            return 1 if trade.is_short else -1
        return 0

    # ----------------------------------------------------------------------- #
    # Small DataProvider helpers
    # ----------------------------------------------------------------------- #
    def _current_equity(self) -> float:
        """Current total stake (wallet) value for compounding sizing.

        Uses live wallet equity when available (the COPY_HL / DELTA_NEUTRAL
        pattern); falls back to the static config capital during backtest init
        or if the wallet is not yet populated.
        """
        try:
            if getattr(self, "wallets", None) is not None:
                total = float(self.wallets.get_total_stake_amount())
                if total and total > 0:
                    return total
        except Exception:  # pragma: no cover - defensive: wallet not ready
            pass
        return float(
            self.config.get("available_capital")
            or self.config.get("dry_run_wallet")
            or 1000.0
        )

    def _last_close(self, pair: str):
        return self._last_value(pair, "close")

    def _last_value(self, pair: str, column: str):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or df.empty or column not in df.columns:
            return None
        val = df[column].iloc[-1]
        return None if val is None or (isinstance(val, float) and np.isnan(val)) else float(val)
