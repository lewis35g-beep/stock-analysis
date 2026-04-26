import os
import yfinance as yf
import pandas as pd
import streamlit as st
import mplfinance as mpf
from openai import OpenAI

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Forex & Stock Technical Analysis App",
    layout="wide"
)

st.title("Forex & Stock Technical Analysis App")

ticker = st.text_input("Enter ticker", "EURUSD=X")
use_ai = st.checkbox("Use AI Summary")

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

    df.dropna(inplace=True)
    return df

# ---------------- LOGIC ----------------
def trend_direction(df):
    latest = df.iloc[-1]

    if latest["EMA_20"] > latest["EMA_50"] > latest["EMA_200"]:
        return "Bullish"
    elif latest["EMA_20"] < latest["EMA_50"] < latest["EMA_200"]:
        return "Bearish"
    return "Sideways"

def support_resistance(df):
    return df["Low"].tail(50).min(), df["High"].tail(50).max()

def generate_trade_logic(daily_trend, four_hour_trend, df):
    latest = df.iloc[-1]

    price = latest["Close"]
    atr = latest["ATR"]
    rsi = latest["RSI"]

    support, resistance = support_resistance(df)

    if daily_trend == "Bullish":
        bias = "Buy"
        sl = price - atr * 1.5
        tp = price + atr * 3
    elif daily_trend == "Bearish":
        bias = "Sell"
        sl = price + atr * 1.5
        tp = price - atr * 3
    else:
        bias = "No Trade"
        sl, tp = None, None

    return {
        "price": price,
        "rsi": rsi,
        "atr": atr,
        "support": support,
        "resistance": resistance,
        "bias": bias,
        "stop_loss": sl,
        "take_profit": tp
    }

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
        returnfig=True
    )

    return fig

# ---------------- MAIN ----------------
if st.button("Analyze"):

    df_1h = get_data(ticker, "2mo", "1h")
    df_4h = get_data(ticker, "6mo", "4h")
    df_1d = get_data(ticker, "2y", "1d")

    if df_1h is None:
        st.error("Invalid ticker")
        st.stop()

    df_1h = add_indicators(df_1h)
    df_4h = add_indicators(df_4h)
    df_1d = add_indicators(df_1d)

    daily_trend = trend_direction(df_1d)
    four_hour_trend = trend_direction(df_4h)
    one_hour_trend = trend_direction(df_1h)

    trade = generate_trade_logic(daily_trend, four_hour_trend, df_1h)

    st.subheader("Trends")
    st.write({
        "1D": daily_trend,
        "4H": four_hour_trend,
        "1H": one_hour_trend,
        "Bias": trade["bias"]
    })

    st.subheader("Charts")
    st.pyplot(create_chart(df_1h, ticker, "1H", trade))

    # ---------------- PROMPT ----------------
    prompt = f"""
Analyze {ticker}

1D Trend: {daily_trend}
4H Trend: {four_hour_trend}
1H Trend: {one_hour_trend}

Price: {trade['price']}
RSI: {trade['rsi']}
ATR: {trade['atr']}
Support: {trade['support']}
Resistance: {trade['resistance']}
Bias: {trade['bias']}

Give a short trading summary.
"""

    # ---------------- AI ----------------
    if use_ai:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            st.warning("OPENAI_API_KEY not set")
        else:
            client = OpenAI(api_key=api_key)

            try:
                response = client.responses.create(
                    model="gpt-4.1-mini",
                    input=prompt
                )

                st.subheader("AI Analysis")
                st.write(response.output_text)

            except Exception as e:
                st.error(f"AI failed: {e}")