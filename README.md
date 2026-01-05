# Jansen Method Pair Trading

This project implements a pairs trading strategy inspired by the "Pairs trading in practice" section of `pair_trading_in_practice.pdf` and the ML4Trading book (chapters 06-07). It utilizes Kalman filtering to smooth price series and estimate a time-varying hedge ratio. A z-score is built from the resulting spread, and trades are executed when the z-score crosses defined thresholds.

## Features

- **Kalman Filter Smoothing**: Reduces noise in individual price series.
- **Dynamic Hedge Ratio**: Uses a Kalman Filter to estimate the cointegration relationship between two assets in real-time.
- **Mean Reversion Strategy**: Trades the z-score of the spread, entering at `entry_z` and exiting when the z-score changes sign (mean reversion).
- **Rolling Windows**: Recomputes hedge ratio/half-life/z-score quarterly, trades a 6-month window, and only opens trades in the first 3 months.
- **Risk Management**: Configurable stop-loss via `risk_limit` (default -20% spread return).
- **Optimization Tools**: Scripts to sweep z-score thresholds and find the best-performing pairs.
- **Caching**: `pair_sweep.py` uses a signature-based caching mechanism to speed up repeated runs.
- **Unit Tests**: Test suite for backtest logic validation.

## Example Output

<img src="output/equity_BTC_XRP_z1.0.png" width="700" alt="Equity curve for BTC/XRP (z=1.0)" />
## Scripts

### 1. `jansen_backtest.py`
The core backtest engine for a single pair.
```powershell
python jansen_backtest.py
```
It reads `symbol_x` and `symbol_y` from `config.json`, runs the backtest, and saves the results/plots to the `output/` directory.
The backtest recalibrates quarterly using a rolling lookback window.

### 2. `zscore_sweep.py`
Sweeps a grid of z-score thresholds for the pair defined in `config.json`.
```powershell
python zscore_sweep.py --thresholds "1.0,1.5,2.0,2.5,3.0"
```
Useful for finding the optimal entry threshold for a specific pair. It filters results based on `min_trades` and ranks them by `sharpe` ratio.

### 3. `pair_sweep.py`
Ranks all possible pairs from the available data based on their performance.
```powershell
python pair_sweep.py
```
- It filters symbols based on `min_history_days`.
- It tests each pair against the `threshold_grid` defined in `config.json`.
- Results are cached in the `cache/` folder using a SHA-256 signature of the data and code.

**Outputs:**
- `output/pair_rankings.csv` - Ranked results table with Sharpe, trades, best z-score
- `output/pair_sweep_summary.png` - Summary visualization with:
  - Top pairs bar chart (by Sharpe ratio)
  - Sharpe vs trade count scatter plot
  - Sharpe ratio distribution histogram
  - Summary statistics panel
- `output/equity_{Y}_{X}_z{Z}.png` - Equity curve for the best-performing pair

## Configuration (`config.json`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `data_dir` | Path to directory containing `.feather` price files | `data/feather` |
| `output_dir` | Path where backtest results and plots are saved | `output` |
| `interval` | Candle interval (e.g., `1d`) | `1d` |
| `quote` | Quote currency (e.g., `USDT`) | `USDT` |
| `symbol_y` | Primary symbol (Lead) | - |
| `symbol_x` | Secondary symbol (Lag/Hedge) | - |
| `start_equity` | Initial capital | `1000.0` |
| `entry_z` | Z-score threshold for entering a trade | `2.0` |
| `risk_limit` | Stop-loss trigger as spread return (e.g., `-0.2` = -20%) | `-0.2` |
| `threshold_grid` | List of z-score thresholds to test in sweep scripts | `[1.0, 1.5, 2.0, 2.5, 3.0]` |
| `min_history_days` | Minimum data points required for a symbol in `pair_sweep.py` | `1000` |
| `min_trades` | Minimum trades required for a valid backtest result | `20` |
| `fee_rate` | Transaction fee rate (e.g., `0.001` for 0.1%) | `0.001` |
| `lookback_days` | Rolling lookback (in days) before each quarterly test window | `730` |
| `trade_window_months` | Months traded after each quarterly test end | `6` |
| `entry_window_months` | Months within trading window where entries are allowed | `3` |
| `show_plot` | Toggle display of equity curve window | `true` |

## Installation

Ensure you have the following dependencies installed:

```powershell
pip install pandas numpy pykalman matplotlib pyarrow
```

The data files should be in Feather format with a `close` price column and an `open_time_dt` datetime column.
Files should be named following the pattern `{symbol}{quote}_{interval}.feather` (e.g., `BNBUSDT_1d.feather`).

## Testing

Run the unit tests with:

```powershell
python -m unittest tests.test_backtest_logic -v
```

## Project Structure

```
Jansen_method/
├── jansen_backtest.py      # Core backtest engine
├── utils.py                # Shared utilities and helpers
├── pair_sweep.py           # Brute-force pair ranking
├── zscore_sweep.py         # Z-score threshold optimization
├── config.json             # Configuration file
├── tests/
│   └── test_backtest_logic.py
├── data/feather/           # Price data (not tracked in git)
├── cache/                  # Cached results (not tracked in git)
├── output/                 # Backtest results and plots (not tracked in git)
└── 06_*.ipynb, 07_*.ipynb  # Reference notebooks from ML4Trading
```

## References

- `pair_trading_in_practice.pdf` - Original methodology paper
- `06_statistical_arbitrage_with_cointegrated_pairs.ipynb` - Cointegration testing reference
- `07_pairs_trading_backtest.ipynb` - Backtest implementation reference (uses Backtrader)
