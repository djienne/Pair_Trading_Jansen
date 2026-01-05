import argparse
import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from jansen_backtest import backtest_pair, plot_equity
from utils import (
    build_results_table,
    load_config,
    make_run_config,
    print_results_table,
    resolve_path,
    resolve_thresholds,
    ResultTracker,
    summarize_results,
)


# ---------------------------------------------------------------------------
# Sweep Runner
# ---------------------------------------------------------------------------
def run_threshold_sweep(
    config: dict,
    thresholds: list[float],
) -> list[dict]:
    """Run backtest for each threshold and return results."""
    rows = []
    for z in thresholds:
        run_config = make_run_config(
            config,
            config.get("symbol_y", "BNB"),
            config.get("symbol_x", "SOL"),
            z,
        )
        results = backtest_pair(run_config, save_output=False)
        summary = summarize_results(results)
        rows.append({
            "entry_z": z,
            "avg_log_return": summary["avg_log_return"],
            "sharpe": summary["sharpe"],
            "trades": summary["trades"],
        })
    return rows


def find_best_threshold(
    rows: list[dict],
    min_trades: int,
) -> dict | None:
    """Find the best threshold based on Sharpe ratio with minimum trade requirement."""
    tracker = ResultTracker(min_trades=min_trades)
    for row in rows:
        # Create a summary-like dict for the tracker
        summary = {
            "sharpe": row["sharpe"],
            "trades": row["trades"],
            "avg_log_return": row["avg_log_return"],
            "final_equity": 0.0,  # Not tracked per-threshold
        }
        tracker.update(summary, row["entry_z"])
    return tracker.get_best()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep z-score entry thresholds for Jansen backtest."
    )
    parser.add_argument(
        "--config",
        default=os.path.join(BASE_DIR, "config.json"),
        help="Path to config.json",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Comma-separated z-score thresholds to test.",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    thresholds = resolve_thresholds(config, args.thresholds)
    min_trades = int(config.get("min_trades", 20))

    # Run sweep
    rows = run_threshold_sweep(config, thresholds)
    table = build_results_table(rows)
    print_results_table(table)

    # Find best result
    best = find_best_threshold(rows, min_trades)

    if best is None:
        print(f"\nNo result met the minimum trade requirement ({min_trades} trades).")
        return

    best_z = best["best_entry_z"]
    print(
        f"\nBest entry_z by Sharpe ratio (min {min_trades} trades): {best_z} "
        f"(sharpe={best['sharpe']:.4f}, trades={best['trades']})"
    )

    # Re-run best case to generate plot and save output
    symbol_y = config.get("symbol_y", "BNB")
    symbol_x = config.get("symbol_x", "SOL")
    best_config = make_run_config(config, symbol_y, symbol_x, best_z)
    best_results = backtest_pair(best_config, save_output=True)

    output_dir = resolve_path(BASE_DIR, config.get("output_dir", "output"))
    name = f"{symbol_y}_{symbol_x}_z{best_z}"
    show_plot = bool(config.get("show_plot", True))

    plot_equity(best_results, output_dir, name, show_plot)


if __name__ == "__main__":
    main()
