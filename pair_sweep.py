import argparse
import json
import os
import random
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from jansen_backtest import backtest_pair, plot_equity
from utils import (
    compute_signature,
    get_cache_path,
    get_data_path,
    get_output_path,
    list_symbols,
    load_config,
    make_run_config,
    resolve_path,
    resolve_thresholds,
    ResultTracker,
    summarize_results,
)


# ---------------------------------------------------------------------------
# Cache Management
# ---------------------------------------------------------------------------
def load_cached_result(cache_path: str, expected_signature: str) -> dict | None:
    """Load cached result if signature matches, otherwise return None."""
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, "r", encoding="utf-8") as f:
        cached = json.load(f)
    if cached.get("signature") == expected_signature:
        return cached.get("summary")
    return None


def save_cached_result(
    cache_path: str,
    signature: str,
    summary: dict,
    output_path: str,
) -> None:
    """Save result to cache."""
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "signature": signature,
                "summary": summary,
                "output_path": output_path,
            },
            f,
            indent=2,
        )


# ---------------------------------------------------------------------------
# Pair Backtest Runner
# ---------------------------------------------------------------------------
def run_pair_backtest(
    y_symbol: str,
    x_symbol: str,
    thresholds: list[float],
    config: dict,
    cache_dir: str,
    code_paths: list[str],
    min_trades: int,
) -> dict | None:
    """
    Run backtest for a single pair across all thresholds.
    Returns the best result dict or None if no valid result found.
    """
    data_paths = [
        get_data_path(BASE_DIR, y_symbol, config),
        get_data_path(BASE_DIR, x_symbol, config),
    ]

    pair_tracker = ResultTracker(min_trades=min_trades)

    for z in thresholds:
        run_config = make_run_config(
            config, y_symbol, x_symbol, z, output_dir=cache_dir
        )

        signature = compute_signature(run_config, data_paths + code_paths)
        cache_path = get_cache_path(cache_dir, y_symbol, x_symbol, float(z))

        # Try to load from cache
        summary = load_cached_result(cache_path, signature)

        if summary is None:
            # Run backtest
            results = backtest_pair(run_config, save_output=True, output_tag=f"z{z}")
            summary = summarize_results(results)

            # Cache the result
            output_path = get_output_path(BASE_DIR, run_config, y_symbol, x_symbol, f"z{z}")
            save_cached_result(cache_path, signature, summary, output_path)

        pair_tracker.update(summary, float(z))

    return pair_tracker.get_best()


def load_best_results(
    y_symbol: str,
    x_symbol: str,
    best_z: float,
    config: dict,
    cache_dir: str,
) -> pd.DataFrame:
    """Load the best results from cache or re-run if necessary."""
    cached_file_path = os.path.join(
        cache_dir, f"jansen_backtest_{y_symbol}_{x_symbol}_z{best_z}.feather"
    )

    if os.path.exists(cached_file_path):
        df = pd.read_feather(cached_file_path)
        df["open_time_dt"] = pd.to_datetime(df["open_time_dt"])
        return df.sort_values("open_time_dt").set_index("open_time_dt")

    # Re-run if cache file missing
    best_config = make_run_config(
        config, y_symbol, x_symbol, best_z, output_dir=cache_dir
    )
    return backtest_pair(best_config, save_output=False)


# ---------------------------------------------------------------------------
# Results Reporting
# ---------------------------------------------------------------------------
def build_results_table(rows: list[dict]) -> pd.DataFrame:
    """Build and sort results table by Sharpe ratio."""
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


def print_results_table(table: pd.DataFrame) -> None:
    """Print formatted results table."""
    print(table.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank available symbol pairs by best Sharpe ratio."
    )
    parser.add_argument(
        "--config",
        default=os.path.join(BASE_DIR, "config.json"),
        help="Path to config.json",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Comma-separated z-score thresholds to test per pair.",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    interval = config.get("interval", "1d")
    quote = config.get("quote", "USDT")
    data_dir = resolve_path(BASE_DIR, config.get("data_dir", "../data/feather"))
    min_history_days = int(config.get("min_history_days", 1000))
    min_trades = int(config.get("min_trades", 20))

    # Find symbols and build pairs
    symbols = list_symbols(data_dir, interval, quote, min_history_days)
    if len(symbols) < 2:
        raise ValueError("Not enough symbols found to build pairs.")

    pairs = [
        (y_symbol, x_symbol)
        for i, y_symbol in enumerate(symbols)
        for x_symbol in symbols[i + 1:]
    ]
    random.shuffle(pairs)

    # Setup
    thresholds = resolve_thresholds(config, args.thresholds)
    cache_dir = os.path.join(BASE_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    code_paths = [
        os.path.join(BASE_DIR, "jansen_backtest.py"),
        os.path.join(BASE_DIR, "pair_sweep.py"),
        os.path.join(BASE_DIR, "utils.py"),
    ]

    print(f"Testing {len(pairs)} pair combinations (min_trades={min_trades})...")

    # Process all pairs
    rows = []
    global_tracker = ResultTracker(min_trades=min_trades)
    best_pair_info = None

    for idx, (y_symbol, x_symbol) in enumerate(pairs, 1):
        print(f"Starting pair {y_symbol}/{x_symbol}...")

        best_of_pair = run_pair_backtest(
            y_symbol, x_symbol, thresholds, config, cache_dir, code_paths, min_trades
        )

        if best_of_pair:
            if global_tracker.update(best_of_pair, best_of_pair["best_entry_z"]):
                best_pair_info = (y_symbol, x_symbol, best_of_pair["best_entry_z"])
            rows.append({"symbol_y": y_symbol, "symbol_x": x_symbol, **best_of_pair})
        else:
            rows.append({
                "symbol_y": y_symbol,
                "symbol_x": x_symbol,
                "best_entry_z": None,
                "sharpe": None,
                "avg_log_return": None,
                "final_equity": None,
                "trades": None,
            })

        # Progress report
        if idx % 5 == 0 or idx == len(pairs):
            current_best = global_tracker.best_sharpe or 0.0
            pair_sharpe = best_of_pair["sharpe"] if best_of_pair else 0.0
            print(
                f"Processed {idx}/{len(pairs)} pairs. "
                f"Latest: {y_symbol}/{x_symbol} sharpe={pair_sharpe:.4f} "
                f"| Global Best Sharpe: {current_best:.4f}"
            )

    # Print final results
    table = build_results_table(rows)
    print_results_table(table)

    # Generate plot for best pair
    if best_pair_info:
        best_y, best_x, best_z = best_pair_info
        output_dir = resolve_path(BASE_DIR, config.get("output_dir", "output"))
        name = f"{best_y}_{best_x}_z{best_z}"
        show_plot = bool(config.get("show_plot", True))

        best_results = load_best_results(best_y, best_x, best_z, config, cache_dir)
        plot_equity(best_results, output_dir, name, show_plot)


if __name__ == "__main__":
    main()
