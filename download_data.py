import argparse
import json
import os
import time
from typing import Dict, Optional

import pandas as pd
import requests

from utils import load_config

COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "num_trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "taker_buy_base",
    "taker_buy_quote",
]

INT_COLUMNS = ["open_time", "close_time", "num_trades"]


def interval_to_millis(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == "m":
        return value * 60 * 1000
    if unit == "h":
        return value * 60 * 60 * 1000
    if unit == "d":
        return value * 24 * 60 * 60 * 1000
    if unit == "w":
        return value * 7 * 24 * 60 * 60 * 1000
    raise ValueError(f"Unsupported interval: {interval}")


def fetch_klines(
    base_url: str,
    symbol: str,
    interval: str,
    start_time: Optional[int],
    end_time: Optional[int],
    limit: int,
) -> list:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_time is not None:
        params["startTime"] = int(start_time)
    if end_time is not None:
        params["endTime"] = int(end_time)

    response = requests.get(
        f"{base_url}/fapi/v1/klines", params=params, timeout=30
    )
    response.raise_for_status()
    return response.json()


def klines_to_frame(klines: list) -> pd.DataFrame:
    df = pd.DataFrame(klines, columns=COLUMNS)
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in INT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce", downcast="integer")

    df["open_time_dt"] = (
        pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
    )
    df.drop(columns=["ignore"], inplace=True)
    return df


def update_symbol_data(
    symbol: str,
    interval: str,
    out_path: str,
    base_url: str,
    limit: int,
) -> pd.DataFrame:
    interval_ms = interval_to_millis(interval)
    end_time = int(time.time() * 1000)

    existing: Optional[pd.DataFrame] = None
    if os.path.exists(out_path):
        existing = pd.read_feather(out_path)

    if existing is None or existing.empty:
        start_time = 0
    else:
        last_open = int(existing["open_time"].max())
        start_time = last_open + interval_ms

    if start_time >= end_time:
        print(f"{symbol} {interval}: already up to date.")
        return existing

    all_rows = []
    current_start = start_time
    while True:
        data = fetch_klines(
            base_url=base_url,
            symbol=symbol,
            interval=interval,
            start_time=current_start,
            end_time=end_time,
            limit=limit,
        )
        if not data:
            break

        all_rows.extend(data)
        last_open = data[-1][0]
        next_start = last_open + interval_ms
        if next_start >= end_time or len(data) < limit:
            break

        current_start = next_start
        time.sleep(0.2)

    if not all_rows:
        print(f"{symbol} {interval}: no new data.")
        return existing

    new_df = klines_to_frame(all_rows)
    if existing is not None and not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["open_time"]).sort_values(
            "open_time"
        )
    else:
        combined = new_df.sort_values("open_time")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    combined.to_feather(out_path)
    print(f"{symbol} {interval}: saved {len(combined)} rows to {out_path}")
    return combined


def main(config_path: str = "config.json", symbols_override: Optional[list] = None) -> None:
    config = load_config(config_path)
    interval = config.get("interval", "1d")
    quote = config.get("quote", "USDT")
    base_url = config.get("binance_base_url", "https://fapi.binance.com")
    limit = int(config.get("max_klines_per_request", 1500))

    feather_dir = config.get("feather_dir") or config.get("data_dir", "data/feather")
    os.makedirs(feather_dir, exist_ok=True)

    if symbols_override:
        # CLI override: assume full symbol names (e.g., BTCUSDT)
        symbols = set(symbols_override)
    else:
        # Config: simple list like ["BTC", "ETH"] -> append quote
        symbols = set(f"{s}{quote}" for s in config.get("symbols", []))

    if not symbols:
        raise ValueError("No symbols provided. Add 'symbols' list to config or use --symbols flag.")

    for symbol in sorted(symbols):
        out_path = os.path.join(feather_dir, f"{symbol}_{interval}.feather")
        update_symbol_data(symbol, interval, out_path, base_url, limit)

    print("Download complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and cache Binance futures klines."
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Optional list of symbols to download (e.g., BTCUSDT ETHUSDT).",
    )
    args = parser.parse_args()

    main(args.config, args.symbols)