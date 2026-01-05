import os
import sys
import unittest

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from jansen_backtest import simulate_pair_trades


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
        self.assertEqual(sim.loc[dates[0], "n_positions"], 2)
        self.assertAlmostEqual(sim.loc[dates[0], "size_y"], 100.0)
        self.assertEqual(sim.loc[dates[1], "n_positions"], 1)
        self.assertAlmostEqual(sim.loc[dates[1], "size_y"], 50.0)

    def test_risk_limit_only_on_trade_days(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        results = self.make_results(dates, [100, 200, 200], [100, 200, 200])
        trade_events = pd.DataFrame(
            {
                "period": [1, 2],
                "side": [1, 1],
                "hedge_ratio": [1, 1],
            },
            index=[dates[0], dates[2]],
        )
        config = {
            "start_equity": 1000.0,
            "fee_rate": 0.0,
            "risk_limit": -0.2,
        }
        sim = simulate_pair_trades(results, trade_events, config)
        self.assertEqual(sim.loc[dates[1], "n_positions"], 1)
        self.assertEqual(sim.loc[dates[2], "n_positions"], 1)
        self.assertEqual(sim.loc[dates[2], "position"], 1)

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


if __name__ == "__main__":
    unittest.main()
