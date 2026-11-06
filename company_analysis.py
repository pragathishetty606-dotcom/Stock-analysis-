import warnings
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Company Stock Analysis",
    page_icon="📈",
    layout="wide",
)

# -------------------------------------------------------------------
# Configuration: the same five companies from your original file
# -------------------------------------------------------------------
COMPANIES = {
    "Paytm": "PAYTM.NS",
    "Zydus Life": "ZYDUSLIFE.NS",
    "LG India": "LGEINDIA.NS",
    "ENRIN": "ENRIN.NS",
    "Lodha": "LODHA.NS",
}

BENCHMARK = "^NSEI"  # Nifty 50
PERIOD = "2y"
RISK_FREE = 0.07
TRADING_DAYS = 252

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
    "Net Margin %",
    "Operating Margin %",
    "ROE %",
    "ROA %",
    "Revenue Growth %",
    "Earnings Growth %",
}


# -------------------------------------------------------------------
# 1. Download share prices
# -------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_prices():
    tickers = list(COMPANIES.values()) + [BENCHMARK]

    raw = yf.download(
        tickers,
        period=PERIOD,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        return pd.DataFrame()

    # yfinance usually returns MultiIndex columns when downloading
    # multiple tickers. This handles either common column arrangement.
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        level1 = raw.columns.get_level_values(1)

        if "Close" in level0:
            prices = raw.xs("Close", axis=1, level=0)
        elif "Close" in level1:
            prices = raw.xs("Close", axis=1, level=1)
        else:
            raise ValueError("Yahoo Finance did not return closing prices.")
    else:
        if "Close" not in raw.columns:
            raise ValueError("Yahoo Finance did not return closing prices.")
        prices = raw[["Close"]].copy()
        prices.columns = [tickers[0]]

    inverse = {ticker: name for name, ticker in COMPANIES.items()}
    inverse[BENCHMARK] = "Nifty 50"
    prices = prices.rename(columns=inverse)
    prices = prices.dropna(how="all")
    return prices


# -------------------------------------------------------------------
# 2. Calculate risk and return measures
# -------------------------------------------------------------------
def max_drawdown(series):
    series = series.dropna()
    if series.empty:
        return np.nan
    return float((series / series.cummax() - 1).min())


def performance_table(prices):
    returns = prices.pct_change(fill_method=None)
    rows = {}

    for name in prices.columns:
        series = prices[name].dropna()

        if len(series) < 30:
            continue

        daily_returns = returns[name].dropna()
        years = (series.index[-1] - series.index[0]).days / 365.25

        total_return = series.iloc[-1] / series.iloc[0] - 1
        cagr = (
            (1 + total_return) ** (1 / years) - 1
            if years > 0 and total_return > -1
            else np.nan
        )

        volatility = daily_returns.std() * np.sqrt(TRADING_DAYS)
        sharpe = (
            (cagr - RISK_FREE) / volatility
            if pd.notna(volatility) and volatility != 0
            else np.nan
        )

        beta = np.nan
        if name != "Nifty 50" and "Nifty 50" in returns.columns:
            joined = pd.concat(
                [daily_returns, returns["Nifty 50"]],
                axis=1,
                join="inner",
            ).dropna()

            if len(joined) > 30:
                market_variance = np.var(joined.iloc[:, 1], ddof=1)
                if market_variance != 0:
                    beta = np.cov(
                        joined.iloc[:, 0], joined.iloc[:, 1], ddof=1
                    )[0, 1] / market_variance

        one_month_start = series.iloc[max(0, len(series) - 22)]
        six_month_start = series.iloc[max(0, len(series) - 126)]

        rows[name] = {
            "Last Price (INR)": series.iloc[-1],
            "1M Return %": (series.iloc[-1] / one_month_start - 1) * 100,
            "6M Return %": (series.iloc[-1] / six_month_start - 1) * 100,
            "Total Return %": total_return * 100,
            "CAGR %": cagr * 100,
            "Volatility %": volatility * 100,
            "Max Drawdown %": max_drawdown(series) * 100,
            "Sharpe": sharpe,
            "Beta vs Nifty": beta,
            "52W High": series.tail(TRADING_DAYS).max(),
            "52W Low": series.tail(TRADING_DAYS).min(),
        }

    return pd.DataFrame(rows).T.round(2)


# -------------------------------------------------------------------
# 3. Download company fundamentals
# -------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fundamentals_table():
    rows = {}

    for name, ticker in COMPANIES.items():
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            info = {}

        row = {}
        for key, label in FUNDAMENTAL_FIELDS.items():
            value = info.get(key)

            if isinstance(value, (int, float, np.integer, np.floating)):
                if key == "marketCap":
                    value = value / 1e7  # convert rupees to crore
                elif label in PERCENT_FIELDS:
                    value = value * 100
                elif key == "dividendYield":
                    value = value * 100 if value < 1 else value

                value = round(value, 2)

            row[label] = value

        rows[name] = row

    return pd.DataFrame(rows).T


# -------------------------------------------------------------------
# 4. Create charts
# -------------------------------------------------------------------
def make_charts(prices, performance):
    figures = {}

    # A. Price performance, rebased to 100
    valid_prices = prices.dropna(how="all")
    rebased = valid_prices.apply(
        lambda column: column / column.dropna().iloc[0] * 100
        if column.notna().any()
        else column
    )

    fig1, ax1 = plt.subplots(figsize=(11, 5))
    rebased.plot(ax=ax1, linewidth=1.5)
    ax1.set_title("Price Performance (Rebased to 100)")
    ax1.set_ylabel("Index")
    ax1.set_xlabel("Date")
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    figures["Price Performance"] = fig1

    # B. Risk versus return
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    comparison = performance.drop(index="Nifty 50", errors="ignore")

    if not comparison.empty:
        ax2.scatter(
            comparison["Volatility %"],
            comparison["CAGR %"],
            s=90,
        )
        for company, row in comparison.iterrows():
            if pd.notna(row["Volatility %"]) and pd.notna(row["CAGR %"]):
                ax2.annotate(
                    company,
                    (row["Volatility %"], row["CAGR %"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                )

    ax2.axhline(0, color="grey", linewidth=0.8)
    ax2.set_title("Risk vs Return")
    ax2.set_xlabel("Annualised Volatility %")
    ax2.set_ylabel("CAGR %")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    figures["Risk vs Return"] = fig2

    # C. Drawdowns from previous peaks
    drawdowns = prices.apply(
        lambda column: (column / column.cummax() - 1) * 100
    )
    fig3, ax3 = plt.subplots(figsize=(11, 5))
    drawdowns.plot(ax=ax3, linewidth=1.2)
    ax3.set_title("Drawdown from Peak")
    ax3.set_ylabel("Drawdown %")
    ax3.set_xlabel("Date")
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    figures["Drawdowns"] = fig3

    # D. Correlation heatmap
    correlation = prices.pct_change(fill_method=None).corr()
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    image = ax4.imshow(correlation, vmin=-1, vmax=1, cmap="RdYlBu_r")
    ax4.set_xticks(range(len(correlation.columns)))
    ax4.set_yticks(range(len(correlation.index)))
    ax4.set_xticklabels(correlation.columns, rotation=45, ha="right")
    ax4.set_yticklabels(correlation.index)

    for i in range(len(correlation.index)):
        for j in range(len(correlation.columns)):
            value = correlation.iloc[i, j]
            if pd.notna(value):
                ax4.text(j, i, f"{value:.2f}", ha="center", va="center")

    fig4.colorbar(image, ax=ax4)
    ax4.set_title("Daily Return Correlation")
    fig4.tight_layout()
    figures["Correlation"] = fig4

    return figures


# -------------------------------------------------------------------
# 5. Build downloadable Excel workbook
# -------------------------------------------------------------------
def create_excel(performance, fundamentals, prices):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        performance.to_excel(writer, sheet_name="Performance")
        fundamentals.to_excel(writer, sheet_name="Fundamentals")
        prices.round(2).to_excel(writer, sheet_name="Daily Prices")
        prices.pct_change(fill_method=None).corr().round(3).to_excel(
            writer, sheet_name="Correlation"
        )

    output.seek(0)
    return output.getvalue()


# -------------------------------------------------------------------
# 6. Streamlit webpage
# -------------------------------------------------------------------
st.title("📈 Comparative Stock Analysis")
st.write(
    "Compare the price performance, risk, return and key financial "
    "fundamentals of five Indian listed companies against the Nifty 50."
)

st.caption(
    "Data source: Yahoo Finance | Period: Last 2 years | "
    "Risk-free rate assumption: 7% annually"
)

with st.sidebar:
    st.header("Analysis Details")
    st.write("**Companies included:**")
    for company in COMPANIES:
        st.write(f"• {company}")
    st.write("**Benchmark:** Nifty 50")
    if st.button("Refresh data"):
        load_prices.clear()
        fundamentals_table.clear()
        st.rerun()

try:
    with st.spinner("Downloading share-price data from Yahoo Finance..."):
        prices = load_prices()

    if prices.empty:
        st.error(
            "No price data was returned. Please check the internet connection "
            "or Yahoo Finance ticker symbols."
        )
        st.stop()

    available_companies = [
        name for name in COMPANIES if name in prices.columns
    ]
    missing_companies = [
        name for name in COMPANIES if name not in prices.columns
    ]

    if missing_companies:
        st.warning(
            "Yahoo Finance did not return price data for: "
            + ", ".join(missing_companies)
        )

    if not available_companies:
        st.error("No company price data is available to display.")
        st.stop()

    prices = prices[available_companies + (
        ["Nifty 50"] if "Nifty 50" in prices.columns else []
    )]

    with st.spinner("Calculating performance and risk measures..."):
        performance = performance_table(prices)

    st.subheader("1. Company Performance Summary")
    company_performance = performance.drop(
        index="Nifty 50", errors="ignore"
    )
    st.dataframe(company_performance, use_container_width=True)

    # Key metric cards
    st.subheader("2. Latest Share Prices")
    metric_columns = st.columns(min(5, len(available_companies)))

    for index, company in enumerate(available_companies):
        latest_price = prices[company].dropna()
        if not latest_price.empty:
            with metric_columns[index % len(metric_columns)]:
                st.metric(
                    label=company,
                    value=f"₹{latest_price.iloc[-1]:,.2f}",
                )

    st.subheader("3. Price and Risk Charts")
    try:
        charts = make_charts(prices, performance)

        for chart_title, figure in charts.items():
            st.markdown(f"**{chart_title}**")
            st.pyplot(figure, clear_figure=True)
            plt.close(figure)

    except Exception as chart_error:
        st.warning(f"Some charts could not be created: {chart_error}")

    st.subheader("4. Company Fundamentals")
    with st.spinner("Fetching company fundamentals..."):
        fundamentals = fundamentals_table()

    fundamentals = fundamentals.loc[
        [name for name in available_companies if name in fundamentals.index]
    ]
    st.dataframe(fundamentals, use_container_width=True)

    st.subheader("5. Nifty 50 Comparison")
    if "Nifty 50" in performance.index:
        st.dataframe(
            performance.loc[["Nifty 50"]],
            use_container_width=True,
        )
    else:
        st.info("Nifty 50 benchmark data was not available.")

    st.subheader("6. Download Analysis")
    excel_file = create_excel(performance, fundamentals, prices)

    st.download_button(
        label="📥 Download Complete Analysis (Excel)",
        data=excel_file,
        file_name="company_analysis.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
    )

    st.success("Stock analysis dashboard loaded.")

except Exception as error:
    st.error("The app could not complete the stock analysis.")
    st.exception(error)
    st.info(
        "Please take a screenshot of this error and send it here "
        "so we can fix the exact issue."
    )
