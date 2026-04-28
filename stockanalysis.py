import os
import yfinance as yf
import pandas as pd
import streamlit as st
import mplfinance as mpf
from openai import OpenAI
import feedparser
from urllib.parse import quote_plus

st.set_page_config(
    page_title="Forex & Stock Technical Analysis App",
    layout="wide"
)

st.title("Forex & Stock Technical Analysis App")

ticker_input = st.text_input("Enter stock, crypto, or forex pair", "EURUSD")
use_ai = st.checkbox("Use AI Summary")


def normalize_ticker(ticker):
    ticker = ticker.strip().upper()

    forex_pairs = [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
        "USDCAD", "USDCHF", "NZDUSD",
        "EURJPY", "GBPJPY", "EURGBP", "EURAUD",
        "AUDJPY", "CADJPY", "CHFJPY"
    ]

    if ticker in forex_pairs:
        return ticker + "=X"

    return ticker


def get_data(ticker, period, interval):
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(inplace=True)
    return df


def resample_to_4h(df):
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df_4h = df.resample("4h").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    })

    df_4h.dropna(inplace=True)
    return df_4h


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
    true_range = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    df["ATR"] = true_range.rolling(14).mean()

    df["EMA_12"] = df["Close"].ewm(span=12).mean()
    df["EMA_26"] = df["Close"].ewm(span=26).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()

    df.dropna(inplace=True)
    return df


def trend_direction(df):
    latest = df.iloc[-1]

    if latest["EMA_20"] > latest["EMA_50"] > latest["EMA_200"]:
        return "Bullish"
    elif latest["EMA_20"] < latest["EMA_50"] < latest["EMA_200"]:
        return "Bearish"
    else:
        return "Mixed / Sideways"


def support_resistance(df, lookback=50):
    recent = df.tail(lookback)
    support = recent["Low"].min()
    resistance = recent["High"].max()
    return support, resistance


def trendline_detection(df, lookback=30):
    recent = df.tail(lookback)

    first_close = recent["Close"].iloc[0]
    last_close = recent["Close"].iloc[-1]

    if last_close > first_close:
        return "Uptrend line rising"
    elif last_close < first_close:
        return "Downtrend line falling"
    else:
        return "Flat trendline / consolidation"


def generate_trade_logic(daily_trend, four_hour_trend, one_hour_df):
    latest = one_hour_df.iloc[-1]

    price = latest["Close"]
    atr = latest["ATR"]
    rsi = latest["RSI"]

    support, resistance = support_resistance(one_hour_df)

    if daily_trend == "Bullish" and four_hour_trend in ["Bullish", "Mixed / Sideways"]:
        bias = "Buy Only"
        stop_loss = price - atr * 1.5
        take_profit = price + atr * 3
    elif daily_trend == "Bearish" and four_hour_trend in ["Bearish", "Mixed / Sideways"]:
        bias = "Sell Only"
        stop_loss = price + atr * 1.5
        take_profit = price - atr * 3
    else:
        bias = "No Trade / Wait"
        stop_loss = None
        take_profit = None

    return {
        "price": price,
        "rsi": rsi,
        "atr": atr,
        "support": support,
        "resistance": resistance,
        "bias": bias,
        "stop_loss": stop_loss,
        "take_profit": take_profit
    }


def create_chart(df, ticker, timeframe, trade=None):
    chart_df = df.tail(100).copy()

    add_plots = [
        mpf.make_addplot(chart_df["EMA_20"]),
        mpf.make_addplot(chart_df["EMA_50"]),
        mpf.make_addplot(chart_df["EMA_200"]),
    ]

    hlines = []

    if trade:
        hlines.append(trade["support"])
        hlines.append(trade["resistance"])

        if trade["stop_loss"] is not None:
            hlines.append(trade["stop_loss"])

        if trade["take_profit"] is not None:
            hlines.append(trade["take_profit"])

    fig, _ = mpf.plot(
        chart_df,
        type="candle",
        style="yahoo",
        addplot=add_plots,
        hlines=dict(hlines=hlines),
        returnfig=True,
        figsize=(12, 6),
        title=f"{ticker} {timeframe}"
    )

    return fig

def get_news(ticker, max_articles=10):
    query = quote_plus(ticker)
    url = f"https://news.google.com/rss/search?q={query}+stock+forex+market&hl=en-US&gl=US&ceid=US:en"

    feed = feedparser.parse(url)

    articles = []

    for entry in feed.entries[:max_articles]:
        articles.append({
            "title": entry.title,
            "link": entry.link,
            "published": entry.get("published", "N/A"),
            "summary": entry.get("summary", "")
        })

    return articles

def analyze_news_with_ai(ticker, articles):
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return "OPENAI_API_KEY not set."

    client = OpenAI(api_key=api_key)

    news_text = ""

    for i, article in enumerate(articles, 1):
        news_text += f"""
Article {i}
Title: {article['title']}
Published: {article['published']}
Summary: {article['summary']}
Link: {article['link']}
"""

    prompt = f"""
You are a market news analyst.

Analyze the following public news headlines for {ticker}.

Determine whether the news is:
- Bullish for price
- Bearish for price
- Neutral / unclear

For each article, explain briefly why.

Then give an overall news sentiment:
Bullish, Bearish, or Neutral.

Also explain whether the news supports or conflicts with the current technical trade setup.

News:
{news_text}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text

if st.button("Analyze"):
    ticker = normalize_ticker(ticker_input)

    df_1h = get_data(ticker, "60d", "1h")
    df_1d = get_data(ticker, "2y", "1d")

    if df_1h is None or df_1d is None:
        st.error("Invalid ticker or no data found.")
        st.stop()

    df_4h = resample_to_4h(df_1h)

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
    col1.metric("1D Confirmation", daily_trend)
    col2.metric("4H Trend", four_hour_trend)
    col3.metric("1H Entry Trend", one_hour_trend)
    col4.metric("Bias", trade["bias"])

    st.subheader("Trade Setup")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"{trade['price']:.5f}")
    c2.metric("RSI", f"{trade['rsi']:.2f}")
    c3.metric("ATR", f"{trade['atr']:.5f}")
    c4.metric("Trendline", trendline)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Support", f"{trade['support']:.5f}")
    c6.metric("Resistance", f"{trade['resistance']:.5f}")
    c7.metric(
        "Stop Loss",
        "N/A" if trade["stop_loss"] is None else f"{trade['stop_loss']:.5f}"
    )
    c8.metric(
        "Take Profit",
        "N/A" if trade["take_profit"] is None else f"{trade['take_profit']:.5f}"
    )

    st.subheader("Charts")

    tab1, tab2, tab3 = st.tabs(["1H Entry", "4H Trend", "1D Confirmation"])

    with tab1:
        st.pyplot(create_chart(df_1h, ticker, "1H Entry", trade))

    with tab2:
        st.pyplot(create_chart(df_4h, ticker, "4H Trend"))

    with tab3:
        st.pyplot(create_chart(df_1d, ticker, "1D Confirmation"))

    prompt = f"""
You are a professional forex and stock technical analyst.

Analyze {ticker} using this multi-timeframe setup.

Rules:
- 1D = confirmation
- 4H = trend direction
- 1H = entry
- Only take buys if Daily trend is bullish
- Only take sells if Daily trend is bearish
- Use ATR for stop loss planning
- Use support and resistance for targets

Market Summary:
1D Trend: {daily_trend}
4H Trend: {four_hour_trend}
1H Trend: {one_hour_trend}
Trendline: {trendline}

Trade Data:
Current Price: {trade['price']}
RSI: {trade['rsi']}
ATR: {trade['atr']}
Support: {trade['support']}
Resistance: {trade['resistance']}
Bias: {trade['bias']}
Stop Loss: {trade['stop_loss']}
Take Profit: {trade['take_profit']}

Give a clean trading summary with:
1. Market bias
2. Buy, sell, or no-trade setup
3. Entry idea
4. Stop loss logic
5. Take profit logic
6. What invalidates the setup
7. Risk warning
"""

    if use_ai:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            st.warning("OPENAI_API_KEY is not set.")
        else:
            client = OpenAI(api_key=api_key)

            try:
                response = client.responses.create(
                    model="gpt-4.1-mini",
                    input=str(prompt)
                )

                st.subheader("AI Analysis")
                st.write(response.output_text)

            except Exception as e:
                st.error(f"AI failed: {e}")

    st.subheader("News Feed Sentiment")

    articles = get_news(ticker, max_articles=10)

    if not articles:
        st.warning("No news found.")
    else:
        for article in articles:
            st.markdown(f"**{article['title']}**")
            st.caption(article["published"])
            st.markdown(f"[Read article]({article['link']})")
            st.divider()

        if use_ai:
            try:
                news_analysis = analyze_news_with_ai(ticker, articles)

                st.subheader("AI News Impact Analysis")
                st.write(news_analysis)

            except Exception as e:
                st.error(f"News AI analysis failed: {e}")