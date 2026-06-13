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
- is anchored to "now", not retrospective;
- **fills one bar earlier**: both decide from the same z[t], but live fills at
  ~open[t+1] (seconds after the daily close, ≈ close[t]) while the backtest's
  `z_signal = z.shift(1)` fills at close[t+1]. `verify_fidelity.py` quantifies
  it: shifting the live state one bar drops the sign disagreement from 23.2%
  to 15.9% and the opposite-sign days from 4 to 0;
- **re-enters after a stop** only on z reverting through zero plus a fresh
  ±entry_z crossing; the backtest can re-enter on a mere threshold re-cross
  inside its 3-month entry window;
- deploys **95% of equity** (`target_capital_ratio`, head-room for fees and
  rounding) vs the backtest's 100%;
- pays **perp funding** on both legs — not modeled by the backtest. The
  gross-neutral book pays the half-difference of the per-leg rates (~0.1%/yr
  full-sample, ~1%/yr over the last year; `verify_fidelity.py` prints the
  current numbers);
- runs **isolated 1× margin**, so a large *joint* rally could liquidate the
  short leg on-exchange while spread PnL is ~flat — liquidation doesn't exist
  in the backtest. Mitigated by the `leg_liq_guard` exit (any leg 60% adverse
  from entry, well inside Binance's ~96-99% liquidation distance);
- manages **dead quarters** (a cohort failing the half-life gate) under the
  *previous* cohort's calibration — the backtest's months-4-6 coverage. Never
  triggered in LTC/XRP history to date.

So the live equity path will not match the backtest one-for-one (most of the
difference concentrates on the overlap days). **Use `jansen_backtest.py` for
performance numbers** — this folder is the live trader; the docker backtest in
`commands.txt` is a wiring check only. `verify_fidelity.py` quantifies the
remaining signal-level divergence on the local data and prints regression
guards (hr-sign, dead quarters, funding drag).

## Layout

```
freqtrade_live/
├── docker-compose.yml           # freqtradeorg/freqtrade:2025.9, API on 127.0.0.1:3012
│                                 # (compose project name: pair_ltc_xrp_jansen)
├── Dockerfile.technical         # adds pykalman
├── commands.txt                 # copy-paste cheatsheet
├── show_PnL.py                  # host-side PnL/performance table (reads docker ps + the API)
├── verify_fidelity.py           # signal-level live-vs-backtest fidelity + regression guards
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

### Check PnL / performance

From this folder on the **host** (needs `pip install requests`):

```bash
python show_PnL.py
```

prints a color-coded table for the `PAIR_LTC_XRP` container — trades, PnL
(all/closed), win rate, profit factor, Sharpe, max drawdown, days live and CAGR
— reading the API credentials straight from `config.json`. With no open trades
it shows `0` trades / `0.00%`: the strategy only enters on a fresh ±2 z-crossing,
so a bot that booted mid-episode (state already ±1) correctly sits flat until the
next crossing.

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
