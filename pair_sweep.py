import argparse
import json
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
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
def _sanitize_for_json(obj: dict) -> dict:
    """Replace NaN/Inf values with None for JSON serialization."""
    result = {}
    for key, value in obj.items():
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            result[key] = None
        elif isinstance(value, dict):
            result[key] = _sanitize_for_json(value)
        else:
            result[key] = value
    return result


def load_cached_result(cache_path: str, expected_signature: str) -> dict | None:
    """Load cached result if signature matches, otherwise return None."""
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("signature") == expected_signature:
            return cached.get("summary")
    except (json.JSONDecodeError, KeyError, TypeError):
        # Corrupted cache file - will be regenerated
        pass
    return None


def save_cached_result(
    cache_path: str,
    signature: str,
    summary: dict,
    output_path: str,
) -> None:
    """Save result to cache."""
    # Sanitize summary to handle NaN values
    clean_summary = _sanitize_for_json(summary)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "signature": signature,
                "summary": clean_summary,
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


def save_results_csv(table: pd.DataFrame, output_dir: str) -> str:
    """Save ranked results to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "pair_rankings.csv")
    table.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\nResults saved to: {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Summary Visualizations
# ---------------------------------------------------------------------------
def plot_summary(table: pd.DataFrame, output_dir: str, top_n: int = 15) -> None:
    """Generate summary visualization plots for pair sweep results."""
    os.makedirs(output_dir, exist_ok=True)

    # Filter valid results (non-null Sharpe)
    valid = table.dropna(subset=["sharpe"]).copy()
    if valid.empty:
        print("No valid results to plot.")
        return

    # Create pair labels
    valid["pair"] = valid["symbol_y"] + "/" + valid["symbol_x"]

    # Setup figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Pair Sweep Summary", fontsize=14, fontweight="bold")

    # 1. Top N pairs by Sharpe ratio (horizontal bar chart)
    ax1 = axes[0, 0]
    top_pairs = valid.nlargest(top_n, "sharpe")
    colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in top_pairs["sharpe"]]
    bars = ax1.barh(top_pairs["pair"], top_pairs["sharpe"], color=colors)
    ax1.set_xlabel("Sharpe Ratio")
    ax1.set_title(f"Top {len(top_pairs)} Pairs by Sharpe Ratio")
    ax1.axvline(x=0, color="black", linewidth=0.5)
    ax1.invert_yaxis()
    # Add value labels
    for bar, val in zip(bars, top_pairs["sharpe"]):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                 f"{val:.2f}", va="center", fontsize=8)

    # 2. Sharpe vs Number of Trades (scatter)
    ax2 = axes[0, 1]
    scatter = ax2.scatter(
        valid["trades"],
        valid["sharpe"],
        c=valid["sharpe"],
        cmap="RdYlGn",
        alpha=0.7,
        edgecolors="white",
        linewidth=0.5
    )
    ax2.set_xlabel("Number of Trades")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_title("Sharpe Ratio vs Trade Count")
    ax2.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    plt.colorbar(scatter, ax=ax2, label="Sharpe")

    # 3. Distribution of Sharpe ratios (histogram)
    ax3 = axes[1, 0]
    n_bins = min(30, len(valid) // 2 + 1)
    ax3.hist(valid["sharpe"], bins=n_bins, color="#3498db", edgecolor="white", alpha=0.8)
    ax3.axvline(x=0, color="red", linewidth=1.5, linestyle="--", label="Zero")
    ax3.axvline(x=valid["sharpe"].median(), color="orange", linewidth=1.5,
                linestyle="-", label=f"Median: {valid['sharpe'].median():.2f}")
    ax3.set_xlabel("Sharpe Ratio")
    ax3.set_ylabel("Count")
    ax3.set_title("Distribution of Sharpe Ratios")
    ax3.legend()

    # 4. Summary statistics text box
    ax4 = axes[1, 1]
    ax4.axis("off")

    stats_text = f"""
    PAIR SWEEP SUMMARY
    {"="*40}

    Total pairs tested:     {len(table):>10}
    Valid results:          {len(valid):>10}
    Pairs with Sharpe > 0:  {(valid['sharpe'] > 0).sum():>10}
    Pairs with Sharpe > 1:  {(valid['sharpe'] > 1).sum():>10}

    SHARPE RATIO STATISTICS
    {"="*40}
    Mean:                   {valid['sharpe'].mean():>10.3f}
    Median:                 {valid['sharpe'].median():>10.3f}
    Std Dev:                {valid['sharpe'].std():>10.3f}
    Min:                    {valid['sharpe'].min():>10.3f}
    Max:                    {valid['sharpe'].max():>10.3f}

    BEST PAIR
    {"="*40}
    Pair:                   {valid.iloc[0]['pair']:>10}
    Sharpe:                 {valid.iloc[0]['sharpe']:>10.3f}
    Trades:                 {int(valid.iloc[0]['trades']):>10}
    Best Z-Score:           {valid.iloc[0]['best_entry_z']:>10.1f}
    """

    ax4.text(0.1, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()

    # Save figure
    plot_path = os.path.join(output_dir, "pair_sweep_summary.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Summary plot saved to: {plot_path}")

    plt.close(fig)


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

    # Save results to CSV and generate summary plots
    output_dir = resolve_path(BASE_DIR, config.get("output_dir", "output"))
    save_results_csv(table, output_dir)
    plot_summary(table, output_dir)

    # Generate equity plot for best pair
    if best_pair_info:
        best_y, best_x, best_z = best_pair_info
        name = f"{best_y}_{best_x}_z{best_z}"
        show_plot = bool(config.get("show_plot", True))

        best_results = load_best_results(best_y, best_x, best_z, config, cache_dir)
        plot_equity(best_results, output_dir, name, show_plot)


if __name__ == "__main__":
    main()
