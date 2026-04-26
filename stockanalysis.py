import yfinance as yf
import pandas as pd
import streamlit as st
import mplfinance as mpf
from openai import OpenAI
from streamlit_autorefresh import st_autorefresh
import os


# ✅ MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Forex & Stock Technical Analysis App",
    layout="wide"
)
st_autorefresh(interval=300000, key="market_refresh")

st.title("Forex & Stock Technical Analysis App")

# Inputs
ticker = st.text_input("Enter ticker", "EURUSD=X")
use_ai = st.checkbox("Use ChatGPT AI Summary", value=False)
scan_market = st.button("Scan Major Forex Pairs", key="scan_btn")
analyze_single = st.button("Analyze Single Ticker", key="analyze_btn")


pairs = [
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "AUDUSD=X",
    "USDCAD=X",
    "USDCHF=X",
    "NZDUSD=X",
]


save_path = st.text_input(
    "Save location",
    r"C:\Users\lewis\Documents\trade_history.csv"
)


# ---------------- DATA ----------------
def get_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(inplace=True)
    return df


# ---------------- INDICATORS ----------------
def add_indicators(df):
    df["EMA_20"] = df["Close"].ewm(span=20).mean()
    df["EMA_50"] = df["Close"].ewm(span=50).mean()
    df["EMA_200"] = df["Close"].ewm(span=200).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    high_low = df["High"] - df["Low"]
    high_close = abs(df["High"] - df["Close"].shift())
    low_close = abs(df["Low"] - df["Close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(14).mean()

    df["EMA_12"] = df["Close"].ewm(span=12).mean()
    df["EMA_26"] = df["Close"].ewm(span=26).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()

    df.dropna(inplace=True)
    return df


# ---------------- LOGIC ----------------
def trend_direction(df):
    latest = df.iloc[-1]

    if latest["EMA_20"] > latest["EMA_50"] > latest["EMA_200"]:
        return "Bullish"
    elif latest["EMA_20"] < latest["EMA_50"] < latest["EMA_200"]:
        return "Bearish"
    return "Mixed / Sideways"


def support_resistance(df, lookback=50):
    recent = df.tail(lookback)
    return recent["Low"].min(), recent["High"].max()


def trendline_detection(df, lookback=30):
    recent = df.tail(lookback)

    if recent["Close"].iloc[-1] > recent["Close"].iloc[0]:
        return "Uptrend"
    elif recent["Close"].iloc[-1] < recent["Close"].iloc[0]:
        return "Downtrend"
    return "Sideways"


def generate_trade_logic(daily_trend, four_hour_trend, df):
    latest = df.iloc[-1]

    price = latest["Close"]
    atr = latest["ATR"]
    support, resistance = support_resistance(df)

    if daily_trend == "Bullish":
        bias = "Buy Only"
        stop_loss = price - atr * 1.5
        take_profit = price + atr * 3
    elif daily_trend == "Bearish":
        bias = "Sell Only"
        stop_loss = price + atr * 1.5
        take_profit = price - atr * 3
    else:
        bias = "No Trade"
        stop_loss = None
        take_profit = None

    return {
        "price": price,
        "rsi": latest["RSI"],
        "atr": atr,
        "support": support,
        "resistance": resistance,
        "bias": bias,
        "stop_loss": stop_loss,
        "take_profit": take_profit
    }

def score_trade(daily_trend, four_hour_trend, one_hour_trend, rsi):
    score = 0

    if daily_trend in ["Bullish", "Bearish"]:
        score += 30

    if four_hour_trend == daily_trend:
        score += 30

    if one_hour_trend == daily_trend:
        score += 25

    if 40 <= rsi <= 65:
        score += 15

    return score


def scan_pairs(pairs):
    results = []

    for pair in pairs:
        df_1h = get_data(pair, "2mo", "1h")
        df_4h = get_data(pair, "6mo", "4h")
        df_1d = get_data(pair, "2y", "1d")

        if df_1h is None or df_4h is None or df_1d is None:
            continue

        df_1h = add_indicators(df_1h)
        df_4h = add_indicators(df_4h)
        df_1d = add_indicators(df_1d)

        daily_trend = trend_direction(df_1d)
        four_hour_trend = trend_direction(df_4h)
        one_hour_trend = trend_direction(df_1h)

        trade = generate_trade_logic(daily_trend, four_hour_trend, df_1h)

        score = score_trade(
            daily_trend,
            four_hour_trend,
            one_hour_trend,
            trade["rsi"]
        )

        results.append({
            "Pair": pair,
            "Score": score,
            "1D Trend": daily_trend,
            "4H Trend": four_hour_trend,
            "1H Trend": one_hour_trend,
            "Bias": trade["bias"],
            "Price": round(trade["price"], 5),
            "RSI": round(trade["rsi"], 2),
            "ATR": round(trade["atr"], 5),
            "Support": round(trade["support"], 5),
            "Resistance": round(trade["resistance"], 5),
            "Stop Loss": None if trade["stop_loss"] is None else round(trade["stop_loss"], 5),
            "Take Profit": None if trade["take_profit"] is None else round(trade["take_profit"], 5),
        })

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df = results_df.sort_values(by="Score", ascending=False)

    return results_df

# ---------------- CHART ----------------
def create_chart(df, ticker, timeframe, trade=None):
    chart_df = df.tail(100)

    add_plots = [
        mpf.make_addplot(chart_df["EMA_20"]),
        mpf.make_addplot(chart_df["EMA_50"]),
        mpf.make_addplot(chart_df["EMA_200"]),
    ]

    hlines = []
    if trade:
        hlines += [trade["support"], trade["resistance"]]

        if trade["stop_loss"]:
            hlines.append(trade["stop_loss"])
        if trade["take_profit"]:
            hlines.append(trade["take_profit"])

    fig, _ = mpf.plot(
        chart_df,
        type="candle",
        style="yahoo",
        addplot=add_plots,
        hlines=dict(hlines=hlines),
        returnfig=True,
        title=f"{ticker} {timeframe}",
        figsize=(12,6)
    )

    return fig

if scan_market:
    st.subheader("Major Forex Pair Scanner")

    results_df = scan_pairs(pairs)

    if results_df.empty:
        st.error("No scan results found.")
    else:
        st.dataframe(results_df)

        # ✅ PUT IT RIGHT HERE
        top_3 = results_df.head(3)

        st.subheader("Top 3 Trade Setups")
        st.table(top_3)

        results_df.to_csv("trade_history.csv", mode="a", header=False, index=False)
        st.caption("Scan saved to trade_history.csv")

        top_score = results_df["Score"].max()

        if top_score >= 80:
            st.success("High-quality setup found.")
        elif top_score >= 60:
            st.warning("Moderate setup found.")
        else:
            st.info("No strong setups.")


# ---------------- MAIN ----------------
prompt = None

if st.button("Analyze"):

    df_1h = get_data(ticker, "2mo", "1h")
    df_4h = get_data(ticker, "6mo", "4h")
    df_1d = get_data(ticker, "2y", "1d")

    if df_1h is None or df_4h is None or df_1d is None:
        st.error("Invalid ticker")
        st.stop()

    df_1h = add_indicators(df_1h)
    df_4h = add_indicators(df_4h)
    df_1d = add_indicators(df_1d)

    daily_trend = trend_direction(df_1d)
    four_hour_trend = trend_direction(df_4h)
    one_hour_trend = trend_direction(df_1h)
    trendline = trendline_detection(df_1h)

    trade = generate_trade_logic(daily_trend, four_hour_trend, df_1h)

    st.subheader("Multi-Timeframe Analysis")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("1D Trend", daily_trend)
    col2.metric("4H Trend", four_hour_trend)
    col3.metric("1H Trend", one_hour_trend)
    col4.metric("Bias", trade["bias"])

    st.subheader("Charts")

    tab1, tab2, tab3 = st.tabs(["1H", "4H", "1D"])

    with tab1:
        st.pyplot(create_chart(df_1h, ticker, "1H", trade))

    with tab2:
        st.pyplot(create_chart(df_4h, ticker, "4H"))

    with tab3:
        st.pyplot(create_chart(df_1d, ticker, "1D"))

    prompt = f"""
Analyze {ticker}

1D Trend: {daily_trend}
4H Trend: {four_hour_trend}
1H Trend: {one_hour_trend}
Trendline: {trendline}

Price: {trade['price']}
RSI: {trade['rsi']}
ATR: {trade['atr']}
Support: {trade['support']}
Resistance: {trade['resistance']}
Bias: {trade['bias']}
Stop Loss: {trade['stop_loss']}
Take Profit: {trade['take_profit']}

Give a clean trading summary.
"""

   
    # -------- AI --------
if use_ai and prompt:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.warning("Set OPENAI_API_KEY")
    else:
        client = OpenAI(api_key="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")

        try:
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=str(prompt)
            )

            st.subheader("AI Analysis")
            st.write(response.output_text)

        except Exception as e:
            st.error(f"AI failed: {e}")