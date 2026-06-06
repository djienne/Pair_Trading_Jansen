import math
import os
import sys
import unittest

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from jansen_backtest import (
    build_period_events,
    compute_position_sizes,
    simulate_pair_trades,
)
from utils import summarize_results


class BacktestLogicTests(unittest.TestCase):
    def make_results(self, dates, y_prices, x_prices):
        return pd.DataFrame(
            {
                "y": y_prices,
                "x": x_prices,
            },
            index=dates,
        )

    def test_overlapping_period_entries_share_portfolio(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        results = self.make_results(dates, [10, 10, 10], [10, 10, 10])
        trade_events = pd.DataFrame(
            {
                "period": [1, 2, 1, 2],
                "side": [1, 1, 0, 0],
                "hedge_ratio": [1, 1, 1, 1],
            },
            index=[dates[0], dates[0], dates[1], dates[2]],
        )
        config = {
            "start_equity": 1000.0,
            "fee_rate": 0.0,
            "risk_limit": None,
        }
        sim = simulate_pair_trades(results, trade_events, config)
        # Sizing is by gross notional of one spread unit: per position
        # k = target / (price_y + |hr|*price_x). Day 0: target=500/position,
        # k=500/(10+10)=25, so aggregate size_y over 2 positions = 50.
        self.assertEqual(sim.loc[dates[0], "n_positions"], 2)
        self.assertAlmostEqual(sim.loc[dates[0], "size_y"], 50.0)
        # Day 1 only closes period 1 (no new entry -> no rebalance), so the
        # surviving position keeps its day-0 size of 25.
        self.assertEqual(sim.loc[dates[1], "n_positions"], 1)
        self.assertAlmostEqual(sim.loc[dates[1], "size_y"], 25.0)

    def test_risk_limit_closes_losing_positions(self):
        """Risk limit should close positions at loss threshold, not profit."""
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        # Price drops from 100 to 50 - a 50% loss scenario
        results = self.make_results(dates, [100, 50, 50], [100, 50, 50])
        trade_events = pd.DataFrame(
            {
                "period": [1],
                "side": [1],
                "hedge_ratio": [1],
            },
            index=[dates[0]],
        )
        config = {
            "start_equity": 1000.0,
            "fee_rate": 0.0,
            "risk_limit": -0.2,  # Close at 20% loss
        }
        sim = simulate_pair_trades(results, trade_events, config)
        # Day 0: Position opens
        self.assertEqual(sim.loc[dates[0], "n_positions"], 1)
        # Day 1: Price drops 50%, loss > 20%, position should be closed
        self.assertEqual(sim.loc[dates[1], "n_positions"], 0)

    def test_risk_limit_keeps_profitable_positions(self):
        """Risk limit should NOT close profitable positions."""
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        # Price rises from 100 to 200 - a 100% profit scenario
        results = self.make_results(dates, [100, 200, 200], [100, 200, 200])
        trade_events = pd.DataFrame(
            {
                "period": [1],
                "side": [1],
                "hedge_ratio": [1],
            },
            index=[dates[0]],
        )
        config = {
            "start_equity": 1000.0,
            "fee_rate": 0.0,
            "risk_limit": -0.2,
        }
        sim = simulate_pair_trades(results, trade_events, config)
        # Position should remain open because it's profitable
        self.assertEqual(sim.loc[dates[0], "n_positions"], 1)
        self.assertEqual(sim.loc[dates[1], "n_positions"], 1)
        self.assertEqual(sim.loc[dates[2], "n_positions"], 1)

    def test_risk_limit_checked_daily_not_just_event_days(self):
        """Risk limit should be checked every day, not just on event days."""
        dates = pd.date_range("2020-01-01", periods=4, freq="D")
        # Day 0: entry at price 100
        # Day 1: no events, price drops to 70 (30% loss)
        # Day 2: no events, price stays at 70
        # Day 3: new entry event
        results = self.make_results(dates, [100, 70, 70, 70], [100, 70, 70, 70])
        trade_events = pd.DataFrame(
            {
                "period": [1, 2],
                "side": [1, 1],
                "hedge_ratio": [1, 1],
            },
            index=[dates[0], dates[3]],
        )
        config = {
            "start_equity": 1000.0,
            "fee_rate": 0.0,
            "risk_limit": -0.2,  # 20% loss limit
        }
        sim = simulate_pair_trades(results, trade_events, config)
        # Day 0: Position 1 opens
        self.assertEqual(sim.loc[dates[0], "n_positions"], 1)
        # Day 1: Risk check should close position 1 (30% loss > 20% limit)
        # even though there are no events on this day
        self.assertEqual(sim.loc[dates[1], "n_positions"], 0)
        # Day 3: Only position 2 should be open
        self.assertEqual(sim.loc[dates[3], "n_positions"], 1)

    def test_same_day_exit_then_entry(self):
        dates = pd.date_range("2020-01-01", periods=2, freq="D")
        results = self.make_results(dates, [10, 10], [10, 10])
        trade_events = pd.DataFrame(
            {
                "period": [1, 1, 1],
                "side": [1, 0, -1],
                "hedge_ratio": [1, 1, 1],
            },
            index=[dates[0], dates[1], dates[1]],
        )
        config = {
            "start_equity": 1000.0,
            "fee_rate": 0.0,
            "risk_limit": None,
        }
        sim = simulate_pair_trades(results, trade_events, config)
        self.assertEqual(sim.loc[dates[1], "n_positions"], 1)
        self.assertEqual(sim.loc[dates[1], "position"], -1)

    def test_stop_fires_on_neutral_hedged_spread(self):
        """Risk limit must work for a dollar-neutral spread (negative hedge ratio).

        With hedge_ratio=-1 the legs cancel at entry (net value ~0), which the
        old net-value denominator could not handle. The gross-notional
        denominator keeps the stop meaningful.
        """
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        # Long Y / short X. Y falls and X rises -> the spread loses badly.
        results = self.make_results(dates, [100, 70, 70], [100, 120, 120])
        trade_events = pd.DataFrame(
            {"period": [1], "side": [1], "hedge_ratio": [-1]},
            index=[dates[0]],
        )
        config = {"start_equity": 1000.0, "fee_rate": 0.0, "risk_limit": -0.2}
        sim = simulate_pair_trades(results, trade_events, config)
        # Day 0: entry; the position is dollar-neutral (net value ~0).
        self.assertEqual(sim.loc[dates[0], "n_positions"], 1)
        self.assertAlmostEqual(sim.loc[dates[0], "position_value"], 0.0)
        # Day 1: P&L = -25% of gross notional < -20% limit -> closed.
        self.assertEqual(sim.loc[dates[1], "n_positions"], 0)

    def test_rebalance_does_not_corrupt_stop(self):
        """Rebalancing must re-baseline entry_value/entry_gross.

        Prices double before period 2 opens, so period 1 is resized at the new
        prices. If the gross-notional baseline is NOT refreshed on rebalance,
        the stop uses a stale (2x) denominator and fails to fire.
        """
        dates = pd.date_range("2020-01-01", periods=4, freq="D")
        # Day 0 entry @100/100; Day 1-2 prices double (spread flat) -> rebalance
        # at 200/200; Day 3 loss (-25% of the rebalanced gross notional).
        results = self.make_results(
            dates, [100, 200, 200, 140], [100, 200, 200, 240]
        )
        trade_events = pd.DataFrame(
            {"period": [1, 2], "side": [1, 1], "hedge_ratio": [-1, -1]},
            index=[dates[0], dates[2]],
        )
        config = {"start_equity": 1000.0, "fee_rate": 0.0, "risk_limit": -0.2}
        sim = simulate_pair_trades(results, trade_events, config)
        self.assertEqual(sim.loc[dates[2], "n_positions"], 2)
        # Both positions are -25% on the correctly re-baselined denominator.
        # With a stale denominator, period 1 would read -12.5% and survive.
        self.assertEqual(sim.loc[dates[3], "n_positions"], 0)

    def _never_reverting_z(self):
        """Z that crosses +entry_z inside the window and never returns to 0."""
        dates = pd.date_range("2020-01-01", periods=150, freq="D")
        z = pd.Series(0.5, index=dates)
        z.iloc[5:] = 3.0  # crosses above +2 at bar 5, then stays high forever
        return dates, z

    def test_force_close_at_window_end(self):
        """A position that never re-crosses zero is flattened on the last bar."""
        dates, z = self._never_reverting_z()
        events = build_period_events(z, entry_z=2.0, entry_window_months=3)
        # Entry (-1) on the crossing, plus a synthetic exit (0) on the last bar.
        self.assertEqual(len(events), 2)
        self.assertEqual(events["side"].iloc[-1], 0.0)
        self.assertEqual(events.index[-1], dates[-1])

        # End-to-end: the position must be flat on the final bar (no leak).
        results = self.make_results(dates, [100] * len(dates), [100] * len(dates))
        trade_events = events.copy()
        trade_events["period"] = 1
        trade_events["hedge_ratio"] = 1.0
        config = {"start_equity": 1000.0, "fee_rate": 0.0, "risk_limit": None}
        sim = simulate_pair_trades(results, trade_events, config)
        self.assertEqual(sim.loc[dates[10], "n_positions"], 1)
        self.assertEqual(sim.iloc[-1]["n_positions"], 0)

    def test_signal_lag_shifts_events_by_one_bar(self):
        """Lagging z by one bar moves every event timestamp forward one bar."""
        dates = pd.date_range("2020-01-01", periods=60, freq="D")
        z = pd.Series(0.5, index=dates)  # sign +1 baseline (no spurious cross)
        z.iloc[10:20] = 3.0              # above +2 -> entry crossing at bar 10
        z.iloc[20:] = -0.5              # sign flips -> zero-cross exit at bar 20

        raw = build_period_events(z, entry_z=2.0, entry_window_months=3)
        lagged = build_period_events(z.shift(1), entry_z=2.0, entry_window_months=3)

        self.assertEqual(list(raw["side"]), [-1.0, 0.0])
        expected = [ts + pd.Timedelta(days=1) for ts in raw.index]
        self.assertEqual(list(lagged.index), expected)
        self.assertEqual(list(lagged["side"]), list(raw["side"]))

    def test_lag_keeps_forced_close_inside_window(self):
        """The boundary exit stays on the last bar even after the signal lag."""
        dates, z = self._never_reverting_z()
        events = build_period_events(z.shift(1), entry_z=2.0, entry_window_months=3)
        self.assertEqual(events.index.max(), z.index.max())
        self.assertEqual(events["side"].iloc[-1], 0.0)

    def test_position_sizing_symmetric_and_bounded(self):
        """Both legs are sized by gross notional: symmetric long/short, bounded.

        The old short-leg formula used 1/hr, so a small hedge ratio produced
        enormous leverage. The new formula caps gross notional at target_value
        for any hr and makes long/short mirror images.
        """
        target, py, px = 1000.0, 100.0, 50.0
        for hr in (-2.0, -0.5, -0.01, 0.7):
            ly, lx = compute_position_sizes(1.0, hr, target, py, px)
            sy, sx = compute_position_sizes(-1.0, hr, target, py, px)
            gross_long = abs(ly) * py + abs(lx) * px
            gross_short = abs(sy) * py + abs(sx) * px
            self.assertAlmostEqual(gross_long, target)   # bounded == target
            self.assertAlmostEqual(gross_short, target)
            self.assertAlmostEqual(ly, -sy)              # opposite direction,
            self.assertAlmostEqual(lx, -sx)              # same magnitude
        # A tiny hedge ratio must NOT blow up the gross notional (old 1/hr code
        # produced ~2000x leverage here).
        sy, sx = compute_position_sizes(-1.0, -0.001, target, py, px)
        self.assertLessEqual(abs(sy) * py + abs(sx) * px, target * 1.0001)

    def test_force_close_with_same_bar_entry_and_zero_cross(self):
        """Entry coinciding with a zero-cross on one bar must not leak.

        When z jumps from below zero straight past +entry_z, the same bar carries
        both an entry (-1) and a zero-cross exit (0). The simulator applies the
        exit before the entry, so the period is actually OPEN and must be
        force-closed at the window end (a naive "is the last event a 0?" check
        is fooled by the same-bar exit and leaks the position).
        """
        dates = pd.date_range("2020-01-01", periods=150, freq="D")
        z = pd.Series(-0.5, index=dates)  # sign -1
        z.iloc[5:] = 3.0                  # bar 5: crosses zero AND exceeds +2
        events = build_period_events(z, entry_z=2.0, entry_window_months=3)
        # Same bar holds both the entry and a zero-cross exit...
        self.assertIn(-1.0, list(events.loc[[dates[5]], "side"]))
        # ...and the series is force-closed on the final bar.
        self.assertEqual(events["side"].iloc[-1], 0.0)
        self.assertEqual(events.index[-1], dates[-1])

        results = self.make_results(dates, [100] * len(dates), [100] * len(dates))
        trade_events = events.copy()
        trade_events["period"] = 1
        trade_events["hedge_ratio"] = 1.0
        config = {"start_equity": 1000.0, "fee_rate": 0.0, "risk_limit": None}
        sim = simulate_pair_trades(results, trade_events, config)
        self.assertEqual(sim.loc[dates[20], "n_positions"], 1)
        self.assertEqual(sim.iloc[-1]["n_positions"], 0)

    def test_summary_sharpe_uses_active_days_and_365(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        returns = pd.Series([0.0, 0.10, -0.02, 0.03, 0.0], index=dates)
        results = pd.DataFrame(
            {
                "equity": (1.0 + returns).cumprod() * 1000.0,
                "strategy_return": returns,
                "n_positions": [0, 1, 1, 0, 0],
                "trade_count": [0, 1, 1, 1, 1],
            },
            index=dates,
        )

        active_returns = returns.iloc[[1, 2, 3]]
        expected = active_returns.mean() / active_returns.std() * math.sqrt(365)

        summary = summarize_results(results)

        self.assertAlmostEqual(summary["sharpe"], expected)


if __name__ == "__main__":
    unittest.main()
