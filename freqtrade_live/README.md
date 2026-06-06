# freqtrade_live — Jansen LTC/XRP pair trade (dry-run)

A live (dry-run) freqtrade deployment of the Kalman-filter pairs-trading strategy
from `../jansen_backtest.py`, on the **LTC/XRP** pair -- now the sweep's #2 pair
under active-day Sharpe (1.46, 365-day annualized, `entry_z = 2.0`). Runs in
Docker like the other freqtrade
projects in this tree (`FLAGS_100pairs_HYPE_REAL`, `ADVANCED_MM_HL`, …).

## What it does

It trades the spread `spread = LTC + hr·XRP` as **two coordinated perpetual
legs** that are gross-notional dollar-neutral:

| spread state            | LTC leg      | XRP leg      |
|-------------------------|--------------|--------------|
| `+1` long-spread (z<−2) | long         | short        |
| `−1` short-spread (z>+2)| short        | long         |
| `0`  (z crosses 0)      | exit         | exit         |

- **Hedge ratio** `hr` and **price smoothing**: 2-state / 1-state Kalman filters
  (`pykalman`), parameters from the research `config.json`. **Recalibrated per
  calendar quarter** over the trailing `lookback_days = 730` — the backtest's
  per-period cadence — not one global pass.
- **Z-score window**: `min(2·half_life, lookback)` where the half-life is an
  Ornstein–Uhlenbeck estimate over the trailing 730d, re-estimated each quarter.
  A quarter only becomes tradable once a full 730d sits behind it, so **the bot
  emits no signal for roughly the first two years of data** (≈ mid-2022 for this
  dataset), matching the reference.
- **Entry**: `|z| > 2.0`, fired only on the candle where the state **crosses**
  into ±1 (the backtest's first-crossing events — it will not re-enter after a
  stop without a fresh crossing). **Exit**: z crosses zero, or the **spread
  stop** — combined unrealized PnL of both legs `< −20%` of gross notional
  (`custom_exit`, the backtest's `risk_limit`).
- **Sizing**: `custom_stake_amount` splits the deployed capital between the legs
  by the hedge ratio so `|notional_LTC| + |notional_XRP|` ≈ deployed capital,
  sized off the **current wallet equity (compounding)**, like the backtest.

The strategy evaluates once per **daily** candle, so trades are infrequent
(~tens of trades over years) but high-quality — this is expected, not a bug.

## Known divergences from `jansen_backtest.py`

The research backtest is **inherently retrospective**: it only activates a
quarterly cohort once that cohort's *entire* 6-month trading window is historical
(`jansen_backtest.py:571`). At "now" the current cohort is always skipped, so a
bit-for-bit live reproduction is impossible. This bot is faithful in
*methodology* (per-quarter Kalman/half-life/z recalibration, 730d warm-up,
crossing entries, compounding sizing) but **by design**:

- holds **one** spread position — it cannot represent the backtest's
  **overlapping cohorts** (two simultaneous positions on ~121 of 732 backtest
  days);
- has **no 3-month entry-window gate** and **no 6-month forced window close**;
- is anchored to "now", not retrospective.

So the live equity path will not match the backtest one-for-one (most of the
difference concentrates on the overlap days). **Use `jansen_backtest.py` for
performance numbers** — this folder is the live trader. `verify_fidelity.py`
quantifies the remaining signal-level divergence on the local data.

## Layout

```
freqtrade_live/
├── docker-compose.yml           # freqtradeorg/freqtrade:2025.9, API on 127.0.0.1:3012
│                                 # (compose project name: pair_ltc_xrp_jansen)
├── Dockerfile.technical         # adds pykalman
├── commands.txt                 # copy-paste cheatsheet
└── user_data/
    ├── config.json              # Binance USDT-M futures, dry_run, both legs
    ├── config_hyperliquid.json  # Hyperliquid USDC variant (fill wallet creds)
    └── strategies/
        ├── jansen_signals.py        # ported Kalman / half-life / z-score math
        └── PairTradingJansen.py     # the IStrategy
```

## Run it

```bash
docker compose build                      # installs pykalman (once)

# Download warm-up history for both legs
docker compose run --rm freqtrade download-data \
  --exchange binance --trading-mode futures \
  --pairs LTC/USDT:USDT XRP/USDT:USDT --timeframe 1d --timerange 20200101-

docker compose up -d                      # start the dry-run bot
docker compose logs -f                    # watch it compute z and open legs
```

API/UI: `http://127.0.0.1:3012` (user `david`, password `lolalola` — change in
`config.json`). See `commands.txt` for backtest + management commands.

## Why Binance futures (not Hyperliquid) for now

The strategy is daily and needs a ~2-year warm-up; the backtest itself was built
on Binance USDⓈ-M futures daily data, so that history is guaranteed to exist.
Hyperliquid has little daily history for these coins. Because this is freqtrade,
switching is config-only:

1. Fill `walletAddress` / `privateKey` in `config_hyperliquid.json`.
2. Download HL data (`--exchange hyperliquid`, pairs `LTC/USDC:USDC`,
   `XRP/USDC:USDC`).
3. Point the compose `command: --config` at `config_hyperliquid.json`.

## Going live (real money)

Set `"dry_run": false` in the active config. **The strategy code does not
change.** Start small; the spread stop is the only risk control beyond the
z-score exit.
