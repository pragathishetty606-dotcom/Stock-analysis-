"""
Comparative analysis of five Indian listed companies:
Paytm, Zydus Lifesciences, LG Electronics India, Siemens Energy India (ENRIN), Lodha.

What it does
  1. Downloads 2 years of daily prices from Yahoo Finance (yfinance)
  2. Computes returns, volatility, max drawdown, Sharpe ratio, beta vs Nifty 50
  3. Pulls key fundamentals (market cap, P/E, P/B, margins, ROE, debt/equity)
  4. Saves charts (PNG) and an Excel workbook with all results

Setup:   pip install yfinance pandas numpy matplotlib openpyxl
Run:     python company_analysis.py
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Configuration  (edit tickers here if Yahoo uses a different symbol)
# --------------------------------------------------------------------------
COMPANIES = {
    "Paytm": "PAYTM.NS",
    "Zydus Life": "ZYDUSLIFE.NS",
    "LG India": "LGEINDIA.NS",
    "ENRIN": "ENRIN.NS",
    "Lodha": "LODHA.NS",
}
BENCHMARK = "^NSEI"          # Nifty 50
PERIOD = "2y"
RISK_FREE = 0.07             # annual, approx. Indian 10-yr G-sec yield
TRADING_DAYS = 252
OUT_DIR = "output"

os.makedirs(OUT_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# 1. Price data
# --------------------------------------------------------------------------
def load_prices() -> pd.DataFrame:
    tickers = list(COMPANIES.values()) + [BENCHMARK]
    raw = yf.download(tickers, period=PERIOD, auto_adjust=True, progress=False)
    prices = raw["Close"].copy()
    inverse = {v: k for k, v in COMPANIES.items()}
    inverse[BENCHMARK] = "Nifty 50"
    prices = prices.rename(columns=inverse)
    prices = prices.dropna(how="all")
    missing = [n for n in list(COMPANIES) + ["Nifty 50"] if n not in prices.columns]
    if missing:
        print(f"Warning: no price data for {missing} (check ticker symbols)")
    return prices


# --------------------------------------------------------------------------
# 2. Risk / return metrics
# --------------------------------------------------------------------------
def max_drawdown(series: pd.Series) -> float:
    s = series.dropna()
    return float((s / s.cummax() - 1).min())


def performance_table(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().dropna(how="all")
    rows = {}
    for name in prices.columns:
        s = prices[name].dropna()
        if len(s) < 30:
            continue
        r = returns[name].dropna()
        years = (s.index[-1] - s.index[0]).days / 365.25
        total_ret = s.iloc[-1] / s.iloc[0] - 1
        cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else np.nan
        vol = r.std() * np.sqrt(TRADING_DAYS)
        sharpe = (cagr - RISK_FREE) / vol if vol else np.nan

        beta = np.nan
        if name != "Nifty 50" and "Nifty 50" in returns:
            joined = pd.concat([r, returns["Nifty 50"]], axis=1, join="inner").dropna()
            if len(joined) > 30:
                cov = np.cov(joined.iloc[:, 0], joined.iloc[:, 1])
                beta = cov[0, 1] / cov[1, 1]

        rows[name] = {
            "Last Price (INR)": s.iloc[-1],
            "1M Return %": (s.iloc[-1] / s.iloc[max(-22, -len(s))] - 1) * 100,
            "6M Return %": (s.iloc[-1] / s.iloc[max(-126, -len(s))] - 1) * 100,
            "Total Return %": total_ret * 100,
            "CAGR %": cagr * 100,
            "Volatility %": vol * 100,
            "Max Drawdown %": max_drawdown(s) * 100,
            "Sharpe": sharpe,
            "Beta vs Nifty": beta,
            "52W High": s.tail(TRADING_DAYS).max(),
            "52W Low": s.tail(TRADING_DAYS).min(),
        }
    return pd.DataFrame(rows).T.round(2)


# --------------------------------------------------------------------------
# 3. Fundamentals
# --------------------------------------------------------------------------
FUNDAMENTAL_FIELDS = {
    "marketCap": "Market Cap (INR Cr)",
    "trailingPE": "P/E (TTM)",
    "forwardPE": "Forward P/E",
    "priceToBook": "P/B",
    "enterpriseToEbitda": "EV/EBITDA",
    "profitMargins": "Net Margin %",
    "operatingMargins": "Operating Margin %",
    "returnOnEquity": "ROE %",
    "returnOnAssets": "ROA %",
    "revenueGrowth": "Revenue Growth %",
    "earningsGrowth": "Earnings Growth %",
    "debtToEquity": "Debt/Equity",
    "dividendYield": "Dividend Yield %",
    "sector": "Sector",
}
PERCENT_FIELDS = {
    "Net Margin %", "Operating Margin %", "ROE %", "ROA %",
    "Revenue Growth %", "Earnings Growth %",
}


def fundamentals_table() -> pd.DataFrame:
    rows = {}
    for name, ticker in COMPANIES.items():
        try:
            info = yf.Ticker(ticker).info
        except Exception as exc:  # network / symbol problems
            print(f"Could not fetch fundamentals for {name}: {exc}")
            info = {}
        row = {}
        for key, label in FUNDAMENTAL_FIELDS.items():
            val = info.get(key)
            if isinstance(val, (int, float)):
                if key == "marketCap":
                    val = val / 1e7                      # INR -> crore
                elif label in PERCENT_FIELDS:
                    val = val * 100
                elif key == "dividendYield":
                    # yfinance versions differ: fraction vs. already a percent
                    val = val * 100 if val < 1 else val
                val = round(val, 2)
            row[label] = val
        rows[name] = row
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------
# 4. Charts
# --------------------------------------------------------------------------
def save_charts(prices: pd.DataFrame, perf: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    # a) Rebased price (100 = start)
    rebased = prices.apply(lambda s: s / s.dropna().iloc[0] * 100)
    ax = rebased.plot(figsize=(12, 6), linewidth=1.6)
    ax.set_title("Price performance (rebased to 100)")
    ax.set_ylabel("Index")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/rebased_prices.png", dpi=150)
    plt.close()

    # b) Risk vs return
    comp = perf.drop(index="Nifty 50", errors="ignore")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(comp["Volatility %"], comp["CAGR %"], s=120)
    for name, row in comp.iterrows():
        ax.annotate(name, (row["Volatility %"], row["CAGR %"]),
                    xytext=(6, 6), textcoords="offset points")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Annualised volatility %")
    ax.set_ylabel("CAGR %")
    ax.set_title("Risk vs return")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/risk_return.png", dpi=150)
    plt.close()

    # c) Drawdowns
    dd = prices.apply(lambda s: s / s.cummax() - 1) * 100
    ax = dd.plot(figsize=(12, 5), linewidth=1.2)
    ax.set_title("Drawdown from peak (%)")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/drawdowns.png", dpi=150)
    plt.close()

    # d) Correlation heatmap of daily returns
    corr = prices.pct_change().corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdYlBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im)
    ax.set_title("Daily return correlation")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/correlation.png", dpi=150)
    plt.close()


# --------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------
def main() -> None:
    print("Downloading prices ...")
    prices = load_prices()
    if prices.empty:
        raise SystemExit("No price data returned. Check your internet connection.")

    print("Computing performance metrics ...")
    perf = performance_table(prices)

    print("Fetching fundamentals ...")
    fund = fundamentals_table()

    print("Drawing charts ...")
    save_charts(prices, perf)

    path = f"{OUT_DIR}/company_analysis.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        perf.to_excel(xw, sheet_name="Performance")
        fund.to_excel(xw, sheet_name="Fundamentals")
        prices.round(2).to_excel(xw, sheet_name="Daily Prices")
        prices.pct_change().corr().round(3).to_excel(xw, sheet_name="Correlation")

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n=== PERFORMANCE ===")
    print(perf)
    print("\n=== FUNDAMENTALS ===")
    print(fund)
    print(f"\nSaved Excel and charts to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
