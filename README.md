# Moving Average Crossover vs Buy & Hold (10-Year Backtest)

A robust, event-driven backtesting engine in Python that compares a Moving Average Crossover strategy against a standard Buy & Hold strategy on the S&P 500 (SPY).

## Features
- **Real Data:** Pulls 10 years of daily historical data directly from Yahoo Finance using `yfinance`.
- **Transaction Costs:** Accounts for realistic trading friction (defaults to 5 basis points per trade).
- **Institutional Metrics:** Calculates CAGR, Annualized Volatility, Sharpe Ratio, Sortino Ratio, Calmar Ratio, Max Drawdown, Hit Rate, Turnover, Beta, and Alpha.
- **Parameter Sweep:** Automatically runs a grid search across different Fast/Slow MA combinations and plots a Sharpe Ratio heatmap.
- **Cost Sensitivity:** Tests how the strategy's edge decays as transaction costs increase.
- **Regime Analysis:** Breaks down performance by market regime (Bull/Bear) and volatility (Low/Mid/High) to see exactly when the strategy outperforms.

## What it does
1. Downloads SPY data from 2015 to 2024.
2. Runs a baseline test (50/200 MA vs Buy & Hold) and prints a comparison table.
3. Runs a sweep of 36 different moving average combinations.
4. Generates a heatmap (`sharpe_heatmap_10y.png`) showing the best parameters.
5. Outputs an equity curve and drawdown chart (`equity_drawdown_10y.png`).

## How to Run

1. Install the required dependencies:
```bash
pip install pandas numpy yfinance matplotlib