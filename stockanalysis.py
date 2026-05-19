import os
from pathlib import Path
from urllib.parse import quote_plus

import yfinance as yf
import pandas as pd
import streamlit as st
import mplfinance as mpf
import feedparser
from openai import OpenAI


st.set_page_config(
    page_title="Forex & Stock Technical Analysis App",
    layout="wide"
)

st.title("Forex & Stock Technical Analysis App")


ticker_input = st.text_input("Enter stock, crypto, or forex pair", "EURUSD")
use_ai = st.checkbox("Use AI Summary")

trading_style = st.selectbox(
    "Select Trading Style",
    ["Scalping", "Day Trading", "Swing Trading"]
)

major_forex_pairs = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
    "USDCAD", "USDCHF", "NZDUSD",
    "EURJPY", "GBPJPY", "EURGBP"
]


@st.cache_data
def load_book_rules():
    rules_file = Path("book_rules.txt")

    if not rules_file.exists():
        return ""

    return rules_file.read_text(encoding="utf-8")


book_rules = load_book_rules()

if book_rules:
    st.sidebar.success("Book trading rules loaded.")
else:
    st.sidebar.warning("No book_rules.txt found.")


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    return OpenAI(api_key=api_key)


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


def clean_news_ticker(ticker):
    ticker = ticker.replace("=X", "").upper()

    forex_news_names = {
        "EURUSD": "EUR USD forex euro dollar",
        "GBPUSD": "GBP USD forex pound dollar",
        "USDJPY": "USD JPY forex dollar yen",
        "AUDUSD": "AUD USD forex australian dollar",
        "USDCAD": "USD CAD forex dollar canadian",
        "USDCHF": "USD CHF forex dollar swiss franc",
        "NZDUSD": "NZD USD forex new zealand dollar",
        "EURJPY": "EUR JPY forex euro yen",
        "GBPJPY": "GBP JPY forex pound yen",
        "EURGBP": "EUR GBP forex euro pound",
        "EURAUD": "EUR AUD forex euro australian dollar",
        "AUDJPY": "AUD JPY forex australian dollar yen",
        "CADJPY": "CAD JPY forex canadian dollar yen",
        "CHFJPY": "CHF JPY forex swiss franc yen"
    }

    return forex_news_names.get(ticker, ticker)


def get_timeframe_settings(style):
    if style == "Scalping":
        return {
            "entry_interval": "5m",
            "entry_period": "5d",
            "trend_interval": "15m",
            "trend_period": "30d",
            "confirm_interval": "1h",
            "confirm_period": "60d",
            "entry_label": "5M Entry",
            "trend_label": "15M Trend",
            "confirm_label": "1H Confirmation",
            "atr_multiplier_sl": 1.0,
            "atr_multiplier_tp": 1.5,
            "lookahead": 6
        }

    if style == "Day Trading":
        return {
            "entry_interval": "15m",
            "entry_period": "30d",
            "trend_interval": "1h",
            "trend_period": "60d",
            "confirm_interval": "1d",
            "confirm_period": "1y",
            "entry_label": "15M Entry",
            "trend_label": "1H Trend",
            "confirm_label": "1D Confirmation",
            "atr_multiplier_sl": 1.25,
            "atr_multiplier_tp": 2.0,
            "lookahead": 8
        }

    return {
        "entry_interval": "1h",
        "entry_period": "60d",
        "trend_interval": "1h",
        "trend_period": "60d",
        "confirm_interval": "1d",
        "confirm_period": "2y",
        "entry_label": "1H Entry",
        "trend_label": "4H Trend",
        "confirm_label": "1D Confirmation",
        "atr_multiplier_sl": 1.5,
        "atr_multiplier_tp": 3.0,
        "lookahead": 10
    }


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


def generate_trade_logic(confirm_trend, trend_trend, entry_df, sl_mult=1.5, tp_mult=3.0):
    latest = entry_df.iloc[-1]

    price = latest["Close"]
    atr = latest["ATR"]
    rsi = latest["RSI"]

    support, resistance = support_resistance(entry_df)

    if confirm_trend == "Bullish" and trend_trend in ["Bullish", "Mixed / Sideways"]:
        bias = "Buy Only"
        stop_loss = price - atr * sl_mult
        take_profit = price + atr * tp_mult
    elif confirm_trend == "Bearish" and trend_trend in ["Bearish", "Mixed / Sideways"]:
        bias = "Sell Only"
        stop_loss = price + atr * sl_mult
        take_profit = price - atr * tp_mult
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


def historical_pattern_analysis(df, lookahead=10):
    data = df.copy()

    data["future_return"] = (
        data["Close"].shift(-lookahead) / data["Close"] - 1
    )

    current = data.iloc[-1]
    current_rsi = current["RSI"]

    similar = data[
        (data["RSI"].between(current_rsi - 5, current_rsi + 5)) &
        (data["EMA_20"] > data["EMA_50"])
    ]

    if len(similar) < 5:
        return {
            "samples": len(similar),
            "avg_forward_return": None,
            "win_rate": None,
            "message": "Not enough similar historical setups found."
        }

    avg_return = similar["future_return"].mean()
    win_rate = (similar["future_return"] > 0).mean()

    return {
        "samples": len(similar),
        "avg_forward_return": avg_return,
        "win_rate": win_rate,
        "message": "Historical pattern analysis completed."
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
    url = (
        f"https://news.google.com/rss/search?"
        f"q={query}+stock+forex+market&hl=en-US&gl=US&ceid=US:en"
    )

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
    client = get_openai_client()

    if client is None:
        return "OPENAI_API_KEY is not set."

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

News:
{news_text}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text


def quick_pair_scan(pair, style):
    settings = get_timeframe_settings(style)
    ticker = normalize_ticker(pair)

    df_entry = get_data(ticker, settings["entry_period"], settings["entry_interval"])
    df_confirm = get_data(ticker, settings["confirm_period"], settings["confirm_interval"])

    if df_entry is None or df_confirm is None:
        return None

    if style == "Swing Trading":
        df_trend = resample_to_4h(df_entry)
    else:
        df_trend = get_data(ticker, settings["trend_period"], settings["trend_interval"])

    if df_trend is None:
        return None

    df_entry = add_indicators(df_entry)
    df_trend = add_indicators(df_trend)
    df_confirm = add_indicators(df_confirm)

    confirm_trend = trend_direction(df_confirm)
    trend_trend = trend_direction(df_trend)
    entry_trend = trend_direction(df_entry)

    trade = generate_trade_logic(
        confirm_trend,
        trend_trend,
        df_entry,
        settings["atr_multiplier_sl"],
        settings["atr_multiplier_tp"]
    )

    score = 0

    if confirm_trend in ["Bullish", "Bearish"]:
        score += 30

    if trend_trend == confirm_trend:
        score += 30

    if entry_trend == confirm_trend:
        score += 20

    if 40 <= trade["rsi"] <= 65:
        score += 10

    if trade["bias"] != "No Trade / Wait":
        score += 10

    return {
        "Pair": pair,
        "Style": style,
        "Yahoo Ticker": ticker,
        "Score": score,
        "Confirmation Trend": confirm_trend,
        "Trend Timeframe": trend_trend,
        "Entry Trend": entry_trend,
        "Bias": trade["bias"],
        "Price": round(trade["price"], 5),
        "RSI": round(trade["rsi"], 2),
        "ATR": round(trade["atr"], 5),
        "Support": round(trade["support"], 5),
        "Resistance": round(trade["resistance"], 5),
        "Stop Loss": None if trade["stop_loss"] is None else round(trade["stop_loss"], 5),
        "Take Profit": None if trade["take_profit"] is None else round(trade["take_profit"], 5),
    }


def scan_forex_watchlist(style):
    results = []

    for pair in major_forex_pairs:
        result = quick_pair_scan(pair, style)

        if result:
            results.append(result)

    df = pd.DataFrame(results)

    if not df.empty:
        df = df.sort_values(by="Score", ascending=False)

    return df


def ask_openai_assistant(question, context=""):
    client = get_openai_client()

    if client is None:
        return "OPENAI_API_KEY is not set."

    prompt = f"""
You are an AI forex trading assistant.

Use this backend trading rulebook:
{book_rules}

Context:
{context}

User question:
{question}

Answer clearly. Focus on:
- risk management
- trade quality
- technical logic
- rulebook alignment
- scalping vs day trading vs swing trading
- what to watch next
- what would invalidate the setup
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text


def analyze_trade_journal_with_ai(trade_data_text):
    client = get_openai_client()

    if client is None:
        return "OPENAI_API_KEY is not set."

    prompt = f"""
You are a professional trading coach.

Use this backend trading rulebook:
{book_rules}

Analyze the following trade journal or trade data.

Find:
1. What went right
2. What went wrong
3. Whether the trade followed the rulebook
4. Entry quality
5. Stop loss quality
6. Take profit quality
7. Risk/reward quality
8. Psychological mistakes
9. Repeated patterns
10. Concrete improvement recommendations

Trade Data:
{trade_data_text}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text


st.divider()
st.header("AI Forex Watchlist")

if st.button("Scan Forex Watchlist"):
    with st.spinner("Scanning major forex pairs..."):
        watchlist_df = scan_forex_watchlist(trading_style)

    if watchlist_df.empty:
        st.warning("No forex setups found.")
    else:
        st.dataframe(watchlist_df)

        top_pairs = watchlist_df.head(3)

        st.subheader("Top Forex Pairs to Watch")
        st.table(top_pairs)

        if use_ai:
            watchlist_context = watchlist_df.to_string(index=False)

            try:
                watchlist_ai = ask_openai_assistant(
                    f"Based on this {trading_style} forex watchlist, which pairs should I watch first and why?",
                    context=watchlist_context
                )

                st.subheader("AI Watchlist Recommendation")
                st.write(watchlist_ai)

            except Exception as e:
                st.error(f"AI watchlist failed: {e}")


st.divider()
st.header("Ask AI Trading Assistant")

assistant_question = st.text_area(
    "Ask a market, strategy, risk, or trade-management question",
    height=120
)

if st.button("Ask Assistant"):
    if not assistant_question.strip():
        st.warning("Enter a question first.")
    else:
        try:
            answer = ask_openai_assistant(assistant_question)
            st.subheader("Assistant Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Assistant failed: {e}")


st.divider()
st.header("Trade Journal Analyzer")

journal_upload = st.file_uploader(
    "Upload trade journal CSV",
    type=["csv"]
)

manual_trade = st.text_area(
    "Or enter a trade manually",
    placeholder=(
        "Example:\n"
        "Pair: EURUSD\n"
        "Style: Scalping / Day Trading / Swing Trading\n"
        "Direction: Buy\n"
        "Entry: 1.0850\n"
        "Stop Loss: 1.0810\n"
        "Take Profit: 1.0930\n"
        "Reason: Daily bullish, 4H pullback, RSI 52\n"
        "Outcome: Win/Loss\n"
        "Notes: ..."
    ),
    height=180
)

if st.button("Analyze Trade Journal"):
    trade_text = ""

    if journal_upload is not None:
        journal_df = pd.read_csv(journal_upload)
        st.dataframe(journal_df)
        trade_text += journal_df.to_string(index=False)

    if manual_trade.strip():
        trade_text += "\n\nManual Trade:\n" + manual_trade

    if not trade_text.strip():
        st.warning("Upload a CSV or enter a trade first.")
    else:
        try:
            journal_ai = analyze_trade_journal_with_ai(trade_text)

            st.subheader("AI Trade Review")
            st.write(journal_ai)

        except Exception as e:
            st.error(f"Trade journal AI failed: {e}")


st.divider()
st.header("Single Ticker / Forex Pair Analysis")

if st.button("Analyze"):
    settings = get_timeframe_settings(trading_style)
    ticker = normalize_ticker(ticker_input)

    df_entry = get_data(
        ticker,
        settings["entry_period"],
        settings["entry_interval"]
    )

    df_confirm = get_data(
        ticker,
        settings["confirm_period"],
        settings["confirm_interval"]
    )

    if df_entry is None or df_confirm is None:
        st.error("Invalid ticker or no data found.")
        st.stop()

    if trading_style == "Swing Trading":
        df_trend = resample_to_4h(df_entry)
    else:
        df_trend = get_data(
            ticker,
            settings["trend_period"],
            settings["trend_interval"]
        )

    if df_trend is None:
        st.error("Could not load trend timeframe data.")
        st.stop()

    df_entry = add_indicators(df_entry)
    df_trend = add_indicators(df_trend)
    df_confirm = add_indicators(df_confirm)

    confirm_trend = trend_direction(df_confirm)
    trend_trend = trend_direction(df_trend)
    entry_trend = trend_direction(df_entry)
    trendline = trendline_detection(df_entry)

    trade = generate_trade_logic(
        confirm_trend,
        trend_trend,
        df_entry,
        settings["atr_multiplier_sl"],
        settings["atr_multiplier_tp"]
    )

    history_stats = historical_pattern_analysis(
        df_entry,
        lookahead=settings["lookahead"]
    )

    st.subheader(f"{trading_style} Multi-Timeframe Analysis")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(settings["confirm_label"], confirm_trend)
    col2.metric(settings["trend_label"], trend_trend)
    col3.metric(settings["entry_label"], entry_trend)
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

    st.subheader("Historical Pattern Analysis")

    st.write(f"Similar Setups Found: {history_stats['samples']}")

    if history_stats["avg_forward_return"] is not None:
        st.write(
            f"Average Forward Return: "
            f"{history_stats['avg_forward_return']:.4%}"
        )
        st.write(
            f"Win Rate: "
            f"{history_stats['win_rate']:.2%}"
        )
    else:
        st.info(history_stats["message"])

    st.subheader("Charts")

    tab1, tab2, tab3 = st.tabs([
        settings["entry_label"],
        settings["trend_label"],
        settings["confirm_label"]
    ])

    with tab1:
        st.pyplot(create_chart(df_entry, ticker, settings["entry_label"], trade))

    with tab2:
        st.pyplot(create_chart(df_trend, ticker, settings["trend_label"]))

    with tab3:
        st.pyplot(create_chart(df_confirm, ticker, settings["confirm_label"]))

    prompt = f"""
You are a professional forex and stock technical analyst.

Trading Style:
{trading_style}

You must follow this backend trading rulebook when analyzing trades.

BACKEND TRADING RULEBOOK:
{book_rules}

Use the rulebook to judge:
- Whether the setup is valid for {trading_style}
- Whether the entry is high quality
- Whether risk/reward is acceptable
- Whether the setup matches proven technical patterns
- Whether the trade should be buy, sell, or no trade

Do not merely summarize the rulebook. Apply it directly to this market setup.

Analyze {ticker} using this multi-timeframe setup.

Technical Framework:
- Confirmation timeframe: {settings["confirm_label"]}
- Trend timeframe: {settings["trend_label"]}
- Entry timeframe: {settings["entry_label"]}

Market Summary:
Confirmation Trend: {confirm_trend}
Trend Timeframe Direction: {trend_trend}
Entry Timeframe Direction: {entry_trend}
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

Historical Pattern Analysis:
Similar Setups: {history_stats['samples']}
Average Forward Return: {history_stats['avg_forward_return']}
Win Rate: {history_stats['win_rate']}

Give a clean trading summary with:
1. Market bias
2. Rulebook validation
3. Buy, sell, or no-trade decision
4. Entry idea
5. Stop loss logic
6. Take profit logic
7. What invalidates the setup
8. Confidence score from 1 to 10
9. Risk warning
"""

    if use_ai:
        client = get_openai_client()

        if client is None:
            st.warning("OPENAI_API_KEY is not set.")
        else:
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

    news_query = clean_news_ticker(ticker)
    articles = get_news(news_query, max_articles=10)

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
                news_analysis = analyze_news_with_ai(news_query, articles)

                st.subheader("AI News Impact Analysis")
                st.write(news_analysis)

            except Exception as e:
                st.error(f"News AI analysis failed: {e}")