import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from jansen_backtest import backtest_pair, plot_equity
from utils import (
    build_results_table,
    compute_signature,
    get_cache_path,
    get_data_path,
    get_output_path,
    list_symbols,
    load_config,
    make_run_config,
    print_results_table,
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
def _pair_label(pair: tuple[str, str] | None) -> str | None:
    """Return the display label for a selected pair."""
    return f"{pair[0]}/{pair[1]}" if pair else None


def _plot_top_pairs_chart(
    ax,
    valid: pd.DataFrame,
    top_n: int,
    selected_pair: tuple[str, str] | None = None,
) -> None:
    """Plot horizontal bar chart of top pairs by active-day Sharpe ratio."""
    selected_label = _pair_label(selected_pair)
    top_pairs = valid.nlargest(top_n, "sharpe")
    if selected_label and selected_label not in set(top_pairs["pair"]):
        selected = valid.loc[valid["pair"] == selected_label]
        if not selected.empty:
            top_pairs = pd.concat([top_pairs, selected.iloc[:1]], ignore_index=True)

    colors = [
        "#f1c40f" if pair == selected_label else "#2ecc71" if sharpe > 0 else "#e74c3c"
        for pair, sharpe in zip(top_pairs["pair"], top_pairs["sharpe"])
    ]
    bars = ax.barh(top_pairs["pair"], top_pairs["sharpe"], color=colors)
    ax.set_xlabel("Active-Day Sharpe Ratio (365d)")
    ax.set_title(f"Top {len(top_pairs)} Pairs by Active-Day Sharpe")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.invert_yaxis()
    for bar, pair, val in zip(bars, top_pairs["pair"], top_pairs["sharpe"]):
        if pair == selected_label:
            bar.set_edgecolor("black")
            bar.set_linewidth(2.0)
            bar.set_hatch("//")
        label = f"{val:.2f}" + ("  selected" if pair == selected_label else "")
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=8, fontweight="bold" if pair == selected_label else "normal")


def _plot_sharpe_vs_trades(
    ax,
    valid: pd.DataFrame,
    selected_pair: tuple[str, str] | None = None,
) -> None:
    """Plot scatter of active-day Sharpe ratio vs trade count."""
    selected_label = _pair_label(selected_pair)
    colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in valid["sharpe"]]
    ax.scatter(
        valid["trades"], valid["sharpe"],
        c=colors, alpha=0.7, edgecolors="white", linewidth=0.5
    )
    if selected_label:
        selected = valid.loc[valid["pair"] == selected_label]
        if not selected.empty:
            row = selected.iloc[0]
            ax.scatter(
                row["trades"], row["sharpe"],
                marker="*", s=260, c="#f1c40f", edgecolors="black",
                linewidth=1.2, zorder=5, label=f"{selected_label} selected"
            )
            ax.legend(loc="best")
    ax.set_xlabel("Number of Trades")
    ax.set_ylabel("Active-Day Sharpe Ratio (365d)")
    ax.set_title("Active-Day Sharpe vs Trade Count")
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")


def _plot_sharpe_distribution(
    ax,
    valid: pd.DataFrame,
    selected_pair: tuple[str, str] | None = None,
) -> None:
    """Plot histogram of active-day Sharpe ratio distribution."""
    selected_label = _pair_label(selected_pair)
    n_bins = min(30, len(valid) // 2 + 1)
    ax.hist(valid["sharpe"], bins=n_bins, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(x=0, color="red", linewidth=1.5, linestyle="--", label="Zero")
    ax.axvline(x=valid["sharpe"].median(), color="orange", linewidth=1.5,
               linestyle="-", label=f"Median: {valid['sharpe'].median():.2f}")
    if selected_label:
        selected = valid.loc[valid["pair"] == selected_label]
        if not selected.empty:
            ax.axvline(
                x=selected.iloc[0]["sharpe"], color="#f1c40f", linewidth=2.5,
                linestyle="-", label=f"{selected_label}: {selected.iloc[0]['sharpe']:.2f}"
            )
    ax.set_xlabel("Active-Day Sharpe Ratio (365d)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Active-Day Sharpe Ratios")
    ax.legend()


def _plot_summary_stats(
    ax,
    table: pd.DataFrame,
    valid: pd.DataFrame,
    selected_pair: tuple[str, str] | None = None,
) -> None:
    """Plot summary statistics text box."""
    ax.axis("off")
    selected_label = _pair_label(selected_pair)
    selected = valid.loc[valid["pair"] == selected_label] if selected_label else pd.DataFrame()
    selected_text = ""
    if not selected.empty:
        selected_row = selected.iloc[0]
        rank = int(valid["sharpe"].rank(method="min", ascending=False).loc[selected.index[0]])
        selected_text = f"""
    SELECTED PAIR
    {"="*40}
    Pair:                   {selected_label:>10}
    Rank:                   {rank:>10}
    Active-Day Sharpe:      {selected_row['sharpe']:>10.3f}
    Trades:                 {int(selected_row['trades']):>10}
    Best Z-Score:           {selected_row['best_entry_z']:>10.1f}
"""
    stats_text = f"""
    PAIR SWEEP SUMMARY
    {"="*40}

    Total pairs tested:     {len(table):>10}
    Valid results:          {len(valid):>10}
    Pairs with AD Sharpe > 0:{(valid['sharpe'] > 0).sum():>10}
    Pairs with AD Sharpe > 1:{(valid['sharpe'] > 1).sum():>10}

    ACTIVE-DAY SHARPE STATISTICS
    {"="*40}
    Mean:                   {valid['sharpe'].mean():>10.3f}
    Median:                 {valid['sharpe'].median():>10.3f}
    Std Dev:                {valid['sharpe'].std():>10.3f}
    Min:                    {valid['sharpe'].min():>10.3f}
    Max:                    {valid['sharpe'].max():>10.3f}

    BEST PAIR
    {"="*40}
    Pair:                   {valid.iloc[0]['pair']:>10}
    Active-Day Sharpe:      {valid.iloc[0]['sharpe']:>10.3f}
    Trades:                 {int(valid.iloc[0]['trades']):>10}
    Best Z-Score:           {valid.iloc[0]['best_entry_z']:>10.1f}
{selected_text}    """
    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))


def plot_summary(
    table: pd.DataFrame,
    output_dir: str,
    top_n: int = 15,
    selected_pair: tuple[str, str] | None = None,
) -> None:
    """Generate summary visualization plots for pair sweep results."""
    os.makedirs(output_dir, exist_ok=True)

    valid = table.dropna(subset=["sharpe"]).copy()
    if valid.empty:
        print("No valid results to plot.")
        return

    valid["pair"] = valid["symbol_y"] + "/" + valid["symbol_x"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Pair Sweep Summary", fontsize=14, fontweight="bold")

    _plot_top_pairs_chart(axes[0, 0], valid, top_n, selected_pair)
    _plot_sharpe_vs_trades(axes[0, 1], valid, selected_pair)
    _plot_sharpe_distribution(axes[1, 0], valid, selected_pair)
    _plot_summary_stats(axes[1, 1], table, valid, selected_pair)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "pair_sweep_summary.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Summary plot saved to: {plot_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Parallel execution helpers
# ---------------------------------------------------------------------------
def _run_pair_worker(task: tuple):
    """Module-level worker (picklable for the spawn start method on Windows).

    One pathological pair must not abort the whole pool, so failures degrade to
    a None result, which is reported as an empty row.
    """
    y_symbol, x_symbol, thresholds, config, cache_dir, code_paths, min_trades = task
    try:
        best = run_pair_backtest(
            y_symbol, x_symbol, thresholds, config, cache_dir, code_paths, min_trades
        )
    except Exception as exc:  # noqa: BLE001 - keep the sweep going
        print(f"  pair {y_symbol}/{x_symbol} failed: {exc}")
        best = None
    return y_symbol, x_symbol, best


def _row_from_best(y_symbol: str, x_symbol: str, best: dict | None) -> dict:
    """Build a results-table row from a pair's best result (or a blank row)."""
    if best:
        return {"symbol_y": y_symbol, "symbol_x": x_symbol, **best}
    return {
        "symbol_y": y_symbol,
        "symbol_x": x_symbol,
        "best_entry_z": None,
        "sharpe": None,
        "avg_log_return": None,
        "final_equity": None,
        "trades": None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank available symbol pairs by best active-day Sharpe ratio."
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
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker processes (default: config 'workers' or 1).",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    interval = config.get("interval", "1d")
    quote = config.get("quote", "USDT")
    data_dir = resolve_path(BASE_DIR, config.get("data_dir", "../data/feather"))
    min_history_days = int(config.get("min_history_days", 1000))
    min_trades = int(config.get("min_trades", 20))
    test_both_directions = bool(config.get("test_both_directions", True))
    workers = args.workers if args.workers is not None else int(config.get("workers", 1))
    workers = max(1, workers)

    # Find symbols and build pairs
    symbols = list_symbols(data_dir, interval, quote, min_history_days)
    if len(symbols) < 2:
        raise ValueError("Not enough symbols found to build pairs.")

    # Pair order matters: symbol_y is the regression-dependent series in the
    # hedge-ratio estimate, so (A, B) != (B, A). Test both directions by default.
    if test_both_directions:
        pairs = [
            (y_symbol, x_symbol)
            for y_symbol in symbols
            for x_symbol in symbols
            if y_symbol != x_symbol
        ]
    else:
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

    print(
        f"Testing {len(pairs)} pair combinations "
        f"(min_trades={min_trades}, workers={workers})..."
    )

    # Process all pairs (serially, or fanned out across worker processes).
    tasks = [
        (y, x, thresholds, config, cache_dir, code_paths, min_trades)
        for (y, x) in pairs
    ]
    rows = []
    total = len(tasks)
    if workers == 1:
        for idx, task in enumerate(tasks, 1):
            y_symbol, x_symbol, best = _run_pair_worker(task)
            rows.append(_row_from_best(y_symbol, x_symbol, best))
            if idx % 5 == 0 or idx == total:
                print(f"Processed {idx}/{total} pairs. Latest: {y_symbol}/{x_symbol}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_pair_worker, task) for task in tasks]
            for idx, future in enumerate(as_completed(futures), 1):
                y_symbol, x_symbol, best = future.result()
                rows.append(_row_from_best(y_symbol, x_symbol, best))
                if idx % 5 == 0 or idx == total:
                    print(f"Processed {idx}/{total} pairs. Latest: {y_symbol}/{x_symbol}")

    # Print final results (sorted by active-day Sharpe desc)
    table = build_results_table(rows)
    print_results_table(table)

    # Save results to CSV and generate summary plots
    output_dir = resolve_path(BASE_DIR, config.get("output_dir", "output"))
    save_results_csv(table, output_dir)
    selected_pair = (
        config.get("symbol_y", "LTC"),
        config.get("symbol_x", "XRP"),
    )
    plot_summary(table, output_dir, selected_pair=selected_pair)

    # Best pair = top valid row of the Sharpe-sorted table.
    valid = table.dropna(subset=["sharpe"])
    if not valid.empty:
        top = valid.iloc[0]
        best_y, best_x, best_z = top["symbol_y"], top["symbol_x"], top["best_entry_z"]
        name = f"{best_y}_{best_x}_z{best_z}"
        show_plot = bool(config.get("show_plot", True))
        print(
            f"Best pair: {best_y}/{best_x} z={best_z} "
            f"active_day_sharpe={top['sharpe']:.4f}"
        )

        best_results = load_best_results(best_y, best_x, best_z, config, cache_dir)
        plot_equity(best_results, output_dir, name, show_plot)


if __name__ == "__main__":
    main()
