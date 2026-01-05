import json
import os
import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252
MIN_RETURN_CLIP = -0.999  # Clip returns to avoid log(0) issues


# ---------------------------------------------------------------------------
# Configuration Helpers
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(base_dir: str, path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(base_dir, path_value))


def get_data_path(base_dir: str, symbol: str, config: dict) -> str:
    interval = config.get("interval", "1d")
    quote = config.get("quote", "USDT")
    data_dir = resolve_path(base_dir, config.get("data_dir", "../data/feather"))
    filename = f"{symbol}{quote}_{interval}.feather"
    return os.path.join(data_dir, filename)


def get_output_path(base_dir: str, config: dict, symbol_y: str, symbol_x: str, tag: str | None = None) -> str:
    output_dir = resolve_path(base_dir, config.get("output_dir", "output"))
    suffix = f"_{tag}" if tag else ""
    filename = f"jansen_backtest_{symbol_y}_{symbol_x}{suffix}.feather"
    return os.path.join(output_dir, filename)


def get_cache_path(cache_dir: str, symbol_y: str, symbol_x: str, z: float) -> str:
    cache_key = f"{symbol_y}_{symbol_x}_z{z}".replace(".", "p")
    return os.path.join(cache_dir, f"{cache_key}.json")


def parse_thresholds(raw: str) -> list[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [float(p) for p in parts]


def resolve_thresholds(config: dict, raw_thresholds: str | None) -> list[float]:
    if raw_thresholds:
        return parse_thresholds(raw_thresholds)
    grid = config.get("threshold_grid", [1.0, 1.5, 2.0, 2.5, 3.0])
    return [float(value) for value in grid]


def make_run_config(
    base_config: dict,
    symbol_y: str,
    symbol_x: str,
    entry_z: float,
    **overrides: Any,
) -> dict:
    """Create a run configuration from base config with symbol pair and z-score threshold."""
    cfg = dict(base_config)
    cfg["symbol_y"] = symbol_y
    cfg["symbol_x"] = symbol_x
    cfg["entry_z"] = float(entry_z)
    cfg.update(overrides)
    return cfg


def list_symbols(
    data_dir: str,
    interval: str,
    quote: str,
    min_history_days: int = 0,
) -> list[str]:
    symbols = []
    suffix = f"_{interval}.feather"
    for name in os.listdir(data_dir):
        if not name.endswith(suffix):
            continue
        base = name[: -len(suffix)]
        if not base.endswith(quote):
            continue
        symbol = base[: -len(quote)]
        if min_history_days:
            path = os.path.join(data_dir, name)
            df = pd.read_feather(path, columns=["open_time_dt"])
            if df["open_time_dt"].nunique() < min_history_days:
                continue
        symbols.append(symbol)
    return sorted(set(symbols))


def summarize_results(results: pd.DataFrame) -> dict:
    """Compute summary statistics from backtest results."""
    returns = results["strategy_return"].fillna(0.0)
    log_returns = np.log1p(np.clip(returns, MIN_RETURN_CLIP, None))
    total_return = results["equity"].iloc[-1] / results["equity"].iloc[0] - 1
    avg_log_return = log_returns.mean()
    std = returns.std()

    sharpe = np.nan
    if std and np.isfinite(std):
        sharpe = (returns.mean() / std) * np.sqrt(TRADING_DAYS_PER_YEAR)

    if "trade_count" in results.columns:
        trades = results["trade_count"].iloc[-1]
    else:
        trades = results["position"].diff().abs().fillna(0.0).sum() / 2

    return {
        "final_equity": float(results["equity"].iloc[-1]),
        "total_return": float(total_return),
        "avg_log_return": float(avg_log_return),
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "trades": int(trades) if "trade_count" in results.columns else float(trades),
    }


@dataclass
class BestResult:
    """Stores the best result found during optimization."""
    sharpe: float
    best_entry_z: float
    avg_log_return: float
    final_equity: float
    trades: int | float

    def to_dict(self) -> dict:
        return {
            "sharpe": self.sharpe,
            "best_entry_z": self.best_entry_z,
            "avg_log_return": self.avg_log_return,
            "final_equity": self.final_equity,
            "trades": self.trades,
        }


class ResultTracker:
    """Tracks the best backtest result based on Sharpe ratio."""

    def __init__(self, min_trades: int = 20):
        self.min_trades = min_trades
        self.best_sharpe: float | None = None
        self.best_result: BestResult | None = None

    def update(self, summary: dict, z: float) -> bool:
        """Update with new result. Returns True if this is the new best."""
        sharpe = summary.get("sharpe", np.nan)
        trades = summary.get("trades", 0)

        # Skip if insufficient trades or invalid sharpe
        if trades < self.min_trades or not np.isfinite(sharpe):
            return False

        if self.best_sharpe is None or sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            self.best_result = BestResult(
                sharpe=sharpe,
                best_entry_z=z,
                avg_log_return=summary.get("avg_log_return", 0.0),
                final_equity=summary.get("final_equity", 0.0),
                trades=trades,
            )
            return True
        return False

    def get_best(self) -> dict | None:
        """Return the best result as a dictionary, or None if no valid result."""
        return self.best_result.to_dict() if self.best_result else None


def compute_signature(config: dict, paths: list[str]) -> str:
    payload = {
        "config": config,
        "files": [],
    }
    for path in paths:
        stat = os.stat(path)
        payload["files"].append(
            {
                "path": os.path.normpath(path),
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
            }
        )
    serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
