"""
live_ma_backtest_10y.py - MA crossover vs buy-and-hold, 10 years of real SPY.
Run: python live_ma_backtest_10y.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

TICKER   = "SPY"
START    = "2015-01-01"
END      = "2024-12-31"
COST_BPS = 5.0
FAST     = 50
SLOW     = 200
LONG_ONLY = True

# ============================================================ 
print("Downloading %s %s -> %s ..." % (TICKER, START, END))
prices = yf.download(TICKER, start=START, end=END,
                     auto_adjust=True, progress=False)

if isinstance(prices.columns, pd.MultiIndex):
    prices.columns = prices.columns.get_level_values(0)

prices = prices[["Open", "High", "Low", "Close", "Volume"]].dropna()

assert len(prices) > 2000, "Only %d bars - download failed?" % len(prices)#checks if total non 0 values more than 2000
print("Got %d bars: %s -> %s"
      % (len(prices), prices.index[0].date(), prices.index[-1].date()))
print("First close = %.2f" % prices["Close"].iloc[0])

# ============================================================ 
def backtest(prices, signal_fn, cost_bps=5.0):
    ann = 252#considering ideal case(constant in all considered years)

    raw = signal_fn(prices).reindex(prices.index).fillna(0.0)
    pos = raw.shift(1).fillna(0.0)                      

    ret   = prices["Close"].pct_change().fillna(0.0)
    gross = pos * ret
    cost  = pos.diff().abs().fillna(pos.abs()) * (cost_bps / 1e4)
    net   = gross - cost

    eq   = (1 + net).cumprod()
    yrs  = len(net) / ann
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol  = net.std() * np.sqrt(ann)
    sharpe  = net.mean() * ann / vol if vol > 0 else np.nan
    dn = net[net < 0].std() * np.sqrt(ann)
    sortino = net.mean() * ann / dn if dn > 0 else np.nan
    dd = eq / eq.cummax() - 1
    maxdd = dd.min()
    calmar = cagr / abs(maxdd) if maxdd < 0 else np.nan

    b = ret.reindex(net.index).fillna(0.0)
    beta  = np.cov(net, b)[0, 1] / np.var(b, ddof=0) if np.var(b) > 0 else np.nan
    alpha = net.mean() * ann - beta * b.mean() * ann

    metrics = {
        "CAGR": cagr, "AnnVol": vol, "Sharpe": sharpe, "Sortino": sortino,
        "Calmar": calmar, "MaxDD": maxdd,
        "HitRate": (net[pos > 0] > 0).mean() if (pos > 0).any() else np.nan,
        "Turnover": pos.diff().abs().sum() / yrs if yrs > 0 else np.nan,
        "Exposure": (pos.abs() > 0).mean(),
        "Beta": beta, "Alpha": alpha,
        "TotalCost": float(cost.sum()),
        "Years": yrs,
    }
    return {"returns": net, "positions": pos, "metrics": metrics,
            "equity": eq, "drawdown": dd}


# ============================================================ 

def ma_signal(fast=FAST, slow=SLOW, long_only=None):
    if long_only is None:
        long_only = LONG_ONLY
    short_val = 0.0 if long_only else -1.0
    def _sig(df):
        f = df["Close"].rolling(fast, min_periods=fast).mean()
        s = df["Close"].rolling(slow, min_periods=slow).mean()
        sig = pd.Series(0.0, index=df.index)
        sig[f > s] = 1.0
        sig[f < s] = short_val
        return sig
    return _sig


def bh_signal(df):
    return pd.Series(1.0, index=df.index)


# ============================================================ 
ma = backtest(prices, ma_signal(FAST, SLOW), COST_BPS)
bh = backtest(prices, bh_signal,             COST_BPS)

cols = ["CAGR", "AnnVol", "Sharpe", "Sortino", "Calmar", "MaxDD",
        "HitRate", "Exposure", "Turnover", "Beta", "Alpha", "TotalCost", "Years"]
table = pd.DataFrame({
    "MA(%d,%d)" % (FAST, SLOW): ma["metrics"],
    "Buy&Hold": bh["metrics"],
}).T
print("\n=== Baseline: %s, %s-%s (%.1f yrs), cost=%.0fbps ==="
      % (TICKER, START[:4], END[:4], ma["metrics"]["Years"], COST_BPS))
print(table[cols].round(3).to_string())

# ============================================================ 
print("\n=== Parameter sweep ===")
rows = []
for f in [5, 10, 20, 30, 50, 100]:
    for s in [20, 50, 100, 150, 200, 250]:
        if f >= s:
            continue
        m = backtest(prices, ma_signal(f, s), COST_BPS)["metrics"]
        rows.append({"fast": f, "slow": s, **m})
sweep = pd.DataFrame(rows)

print("\nTop-10 by Sharpe:")
print(sweep.nlargest(10, "Sharpe")[["fast", "slow", "CAGR", "Sharpe", "MaxDD", "Turnover"]]
      .round(3).to_string(index=False))
print("\nDistribution:")
print(sweep[["CAGR", "Sharpe", "MaxDD"]].describe().round(3).to_string())

pivot = sweep.pivot(index="fast", columns="slow", values="Sharpe")
fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", origin="lower")
ax.set_xticks(range(len(pivot.columns)), pivot.columns)
ax.set_yticks(range(len(pivot.index)),  pivot.index)
ax.set_xlabel("slow MA"); ax.set_ylabel("fast MA")
ax.set_title("%s Sharpe heatmap (%s-%s)" % (TICKER, START[:4], END[:4]))
fig.colorbar(im, ax=ax); fig.tight_layout()
fig.savefig("sharpe_heatmap_10y.png", dpi=120)

# ============================================================ 
print("\n=== Cost sensitivity ===")
cost_rows = []
for c in [0, 2, 5, 10, 20, 50]:
    m = backtest(prices, ma_signal(FAST, SLOW), c)["metrics"]
    b = backtest(prices, bh_signal,              c)["metrics"]
    cost_rows.append({"cost_bps": c,
                      "MA_Sharpe": m["Sharpe"], "MA_CAGR": m["CAGR"],
                      "BH_Sharpe": b["Sharpe"], "BH_CAGR": b["CAGR"],
                      "MA_minus_BH_Sharpe": m["Sharpe"] - b["Sharpe"]})
print(pd.DataFrame(cost_rows).round(3).to_string(index=False))

# ============================================================ 
print("\n=== Regime analysis (bull/bear x low/mid/high vol) ===")
ret60 = prices["Close"].pct_change(60)
vol60 = prices["Close"].pct_change().rolling(60).std()
regime = pd.Series(np.where(ret60 > 0, "bull", "bear"), index=prices.index)
vol_b = pd.qcut(vol60, 3, labels=["low", "mid", "high"])

def bucket(ret_series, label):
    df = pd.concat([ret_series.rename("r"),
                    regime.rename("regime"),
                    vol_b.rename("vol")], axis=1).dropna()
    out = df.groupby(["regime", "vol"], observed=True)["r"].agg(["mean", "std", "count"])
    out["Sharpe"] = out["mean"] / out["std"] * np.sqrt(252)
    out["strategy"] = label
    return out.reset_index()

print(pd.concat([bucket(ma["returns"], "MA"),
                 bucket(bh["returns"], "BH")]).round(3).to_string(index=False))

# ============================================================ 
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

axes[0].plot(ma["equity"], label="MA(%d,%d)" % (FAST, SLOW), lw=1.4)
axes[0].plot(bh["equity"], label="Buy&Hold", lw=1.4, alpha=0.85)
axes[0].set_ylabel("Growth of $1")
axes[0].set_title("%s: net-of-cost equity (%s-%s, %.0fbps)"
                  % (TICKER, START[:4], END[:4], COST_BPS))
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].fill_between(ma["drawdown"].index, ma["drawdown"], 0, alpha=0.6,
                     color="tab:blue", label="MA(%d,%d) DD" % (FAST, SLOW))
axes[1].fill_between(bh["drawdown"].index, bh["drawdown"], 0, alpha=0.3,
                     color="tab:orange", label="Buy&Hold DD")
axes[1].set_ylabel("Drawdown"); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout(); plt.savefig("equity_drawdown_10y.png", dpi=120)
plt.show()
# ============================================================ 9. CONCLUSION
ma_m, bh_m = ma["metrics"], bh["metrics"]
print("\n=== CONCLUSION ===")
print("%s, %s-%s (%.1f yrs), real Yahoo data, %.0fbps."
      % (TICKER, START, END, ma_m["Years"], COST_BPS))
print("")
print("  Return    : MA CAGR %.1f%%  vs  BH CAGR %.1f%%"
      % (ma_m["CAGR"] * 100, bh_m["CAGR"] * 100))
print("  Risk-adj. : MA Sharpe %.2f  vs  BH Sharpe %.2f"
      % (ma_m["Sharpe"], bh_m["Sharpe"]))
print("  Downside  : MA MaxDD %.1f%%  vs  BH MaxDD %.1f%%"
      % (ma_m["MaxDD"] * 100, bh_m["MaxDD"] * 100))
print("  Costs     : MA total %.3f%%  vs  BH total %.3f%%"
      % (ma_m["TotalCost"] * 100, bh_m["TotalCost"] * 100))
print("  Robustness: Sharpe median %.2f, max %.2f"
      % (sweep["Sharpe"].median(), sweep["Sharpe"].max()))