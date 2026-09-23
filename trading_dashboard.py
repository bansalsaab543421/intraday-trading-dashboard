import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Intraday Trading Dashboard",
    page_icon="📈",
    layout="wide"
)


# =========================================================
# DARK UI
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #0b0f14;
    color: #ffffff;
}

.block-container {
    padding-top: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

h1, h2, h3 {
    color: white;
}

.metric-card {
    background: #111820;
    border: 1px solid #26313d;
    border-radius: 12px;
    padding: 15px;
    text-align: center;
}

.metric-title {
    color: #8d9aaa;
    font-size: 12px;
}

.metric-value {
    color: white;
    font-size: 22px;
    font-weight: bold;
}

.buy {
    color: #00e676;
    font-weight: bold;
}

.sell {
    color: #ff5252;
    font-weight: bold;
}

.hold {
    color: #ffc107;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.title("📈 AI Intraday Technical Dashboard")

st.caption(
    "Groww / TradingView-style technical analysis dashboard "
    "using Yahoo Finance data"
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Dashboard Settings")

symbol = st.sidebar.text_input(
    "Stock Symbol",
    "RELIANCE.NS"
).upper().strip()

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["5m", "15m", "30m", "1h"]
)

period = st.sidebar.selectbox(
    "Data Period",
    ["5d", "1mo", "3mo", "6mo"]
)

capital = st.sidebar.number_input(
    "Trading Capital (₹)",
    min_value=1000,
    value=100000,
    step=5000
)

risk_percent = st.sidebar.number_input(
    "Risk per Trade (%)",
    min_value=0.1,
    max_value=10.0,
    value=1.0,
    step=0.1
)

analyze = st.sidebar.button(
    "🔍 Analyze Stock",
    use_container_width=True
)


# =========================================================
# FUNCTIONS
# =========================================================

@st.cache_data(ttl=30)
def get_data(symbol, period, interval):

    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False
    )

    if data.empty:
        return data

    # Handle yfinance MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.copy()

    required = ["Open", "High", "Low", "Close", "Volume"]

    for col in required:
        if col not in data.columns:
            return pd.DataFrame()

    data = data[required]

    for col in required:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data.dropna(inplace=True)

    return data


def calculate_indicators(data):

    data = data.copy()

    # EMA
    data["EMA_9"] = data["Close"].ewm(
        span=9,
        adjust=False
    ).mean()

    data["EMA_21"] = data["Close"].ewm(
        span=21,
        adjust=False
    ).mean()

    # =====================================================
    # RSI
    # =====================================================

    delta = data["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    data["RSI"] = 100 - (
        100 / (1 + rs)
    )

    # =====================================================
    # MACD
    # =====================================================

    ema12 = data["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = data["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    data["MACD"] = ema12 - ema26

    data["MACD_SIGNAL"] = data["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    data["MACD_HIST"] = (
        data["MACD"] -
        data["MACD_SIGNAL"]
    )

    # =====================================================
    # ATR
    # =====================================================

    high_low = (
        data["High"] -
        data["Low"]
    )

    high_close = (
        data["High"] -
        data["Close"].shift()
    ).abs()

    low_close = (
        data["Low"] -
        data["Close"].shift()
    ).abs()

    tr = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    data["ATR"] = tr.rolling(14).mean()

    # =====================================================
    # VWAP
    # =====================================================

    typical_price = (
        data["High"] +
        data["Low"] +
        data["Close"]
    ) / 3

    cumulative_volume = data["Volume"].cumsum()

    data["VWAP"] = (
        typical_price *
        data["Volume"]
    ).cumsum() / cumulative_volume

    # =====================================================
    # Volume Average
    # =====================================================

    data["VOL_AVG"] = data["Volume"].rolling(20).mean()

    return data


def generate_signal(data):

    latest = data.iloc[-1]

    score = 0

    reasons = []

    # EMA
    if latest["EMA_9"] > latest["EMA_21"]:
        score += 2
        reasons.append("EMA 9 is above EMA 21")
    else:
        score -= 2
        reasons.append("EMA 9 is below EMA 21")

    # VWAP
    if latest["Close"] > latest["VWAP"]:
        score += 1
        reasons.append("Price is above VWAP")
    else:
        score -= 1
        reasons.append("Price is below VWAP")

    # RSI
    if 50 < latest["RSI"] < 70:
        score += 1
        reasons.append("RSI supports bullish momentum")

    elif 30 < latest["RSI"] < 50:
        score -= 1
        reasons.append("RSI shows weak momentum")

    # MACD
    if latest["MACD_HIST"] > 0:
        score += 1
        reasons.append("MACD histogram is positive")
    else:
        score -= 1
        reasons.append("MACD histogram is negative")

    # Volume
    if latest["Volume"] > latest["VOL_AVG"]:
        if score > 0:
            score += 1
            reasons.append("Volume is above average")
        else:
            score -= 1
            reasons.append("High volume with bearish momentum")

    # Final signal
    if score >= 3:
        signal = "BUY"
    elif score <= -3:
        signal = "SELL"
    else:
        signal = "HOLD"

    return signal, score, reasons


def trade_levels(data, signal):

    latest = data.iloc[-1]

    price = float(latest["Close"])
    atr = float(latest["ATR"])

    if not np.isfinite(atr) or atr <= 0:
        atr = price * 0.005

    if signal == "BUY":

        entry = price

        stop_loss = entry - (
            atr * 1.2
        )

        target1 = entry + (
            atr * 1.5
        )

        target2 = entry + (
            atr * 2.5
        )

        exit_price = stop_loss

    elif signal == "SELL":

        entry = price

        stop_loss = entry + (
            atr * 1.2
        )

        target1 = entry - (
            atr * 1.5
        )

        target2 = entry - (
            atr * 2.5
        )

        exit_price = stop_loss

    else:

        entry = None
        stop_loss = None
        target1 = None
        target2 = None
        exit_price = None

    return (
        entry,
        stop_loss,
        target1,
        target2,
        exit_price
    )


# =========================================================
# MAIN
# =========================================================

if analyze:

    with st.spinner("📊 Fetching market data..."):

        try:

            data = get_data(
                symbol,
                period,
                timeframe
            )

            if data.empty:

                st.error(
                    "❌ Data nahi mila. "
                    "Stock symbol check karein."
                )

                st.stop()

            data = calculate_indicators(data)

            signal, score, reasons = generate_signal(data)

            (
                entry,
                stop_loss,
                target1,
                target2,
                exit_price
            ) = trade_levels(
                data,
                signal
            )

            latest = data.iloc[-1]

            current_price = float(
                latest["Close"]
            )

            rsi = float(
                latest["RSI"]
            )

            macd_hist = float(
                latest["MACD_HIST"]
            )

            ema9 = float(
                latest["EMA_9"]
            )

            ema21 = float(
                latest["EMA_21"]
            )

            vwap = float(
                latest["VWAP"]
            )

            atr = float(
                latest["ATR"]
            )

            # =================================================
            # TOP STATUS
            # =================================================

            if ema9 > ema21:
                trend = "UP"
            else:
                trend = "DOWN"

            if signal == "BUY":
                signal_class = "buy"
                signal_icon = "🟢"
            elif signal == "SELL":
                signal_class = "sell"
                signal_icon = "🔴"
            else:
                signal_class = "hold"
                signal_icon = "🟡"

            st.markdown(
                f"""
                ### {symbol}

                **Trend:** `{trend}` &nbsp;&nbsp;
                **Signal:** <span class="{signal_class}">
                {signal_icon} {signal}
                </span>
                """,
                unsafe_allow_html=True
            )

            # =================================================
            # METRICS
            # =================================================

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "LAST PRICE",
                f"₹{current_price:,.2f}"
            )

            c2.metric(
                "TREND",
                trend
            )

            c3.metric(
                "SIGNAL",
                signal
            )

            c4.metric(
                "RSI (14)",
                f"{rsi:.2f}"
            )

            # =================================================
            # TRADE SETUP
            # =================================================

            st.markdown("## 🎯 Trade Setup")

            t1, t2, t3, t4, t5 = st.columns(5)

            if entry is not None:

                t1.metric(
                    "ENTRY",
                    f"₹{entry:,.2f}"
                )

                t2.metric(
                    "STOP-LOSS",
                    f"₹{stop_loss:,.2f}"
                )

                t3.metric(
                    "TARGET 1",
                    f"₹{target1:,.2f}"
                )

                t4.metric(
                    "TARGET 2",
                    f"₹{target2:,.2f}"
                )

                t5.metric(
                    "EXIT",
                    f"₹{exit_price:,.2f}"
                )

            else:

                t1.metric("ENTRY", "--")
                t2.metric("STOP-LOSS", "--")
                t3.metric("TARGET 1", "--")
                t4.metric("TARGET 2", "--")
                t5.metric("EXIT", "--")

                st.info(
                    "No entry condition met. "
                    "Market ko monitor karein."
                )

            # =================================================
            # POSITION SIZE
            # =================================================

            if entry is not None:

                risk_amount = (
                    capital *
                    risk_percent /
                    100
                )

                risk_per_share = abs(
                    entry - stop_loss
                )

                if risk_per_share > 0:

                    quantity = int(
                        risk_amount /
                        risk_per_share
                    )

                    position_value = (
                        quantity *
                        entry
                    )

                else:

                    quantity = 0
                    position_value = 0

                p1, p2, p3 = st.columns(3)

                p1.metric(
                    "Risk Amount",
                    f"₹{risk_amount:,.0f}"
                )

                p2.metric(
                    "Suggested Qty",
                    f"{quantity:,}"
                )

                p3.metric(
                    "Position Value",
                    f"₹{position_value:,.0f}"
                )

            # =================================================
            # CHART
            # =================================================

            st.markdown("## 📊 Price Chart")

            chart_data = data.tail(200).copy()

            fig = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[
                    0.60,
                    0.20,
                    0.20
                ],
                subplot_titles=[
                    "Price",
                    "RSI",
                    "MACD"
                ]
            )

            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=chart_data.index,
                    open=chart_data["Open"],
                    high=chart_data["High"],
                    low=chart_data["Low"],
                    close=chart_data["Close"],
                    name="Price"
                ),
                row=1,
                col=1
            )

            # EMA 9
            fig.add_trace(
                go.Scatter(
                    x=chart_data.index,
                    y=chart_data["EMA_9"],
                    mode="lines",
                    name="EMA 9",
                    line=dict(width=1.5)
                ),
                row=1,
                col=1
            )

            # EMA 21
            fig.add_trace(
                go.Scatter(
                    x=chart_data.index,
                    y=chart_data["EMA_21"],
                    mode="lines",
                    name="EMA 21",
                    line=dict(width=1.5)
                ),
                row=1,
                col=1
            )

            # VWAP
            fig.add_trace(
                go.Scatter(
                    x=chart_data.index,
                    y=chart_data["VWAP"],
                    mode="lines",
                    name="VWAP",
                    line=dict(width=1.5)
                ),
                row=1,
                col=1
            )

            # =================================================
            # BUY / SELL MARKERS
            # =================================================

            buy_condition = (
                (chart_data["EMA_9"] >
                 chart_data["EMA_21"]) &
                (chart_data["EMA_9"].shift(1) <=
                 chart_data["EMA_21"].shift(1))
            )

            sell_condition = (
                (chart_data["EMA_9"] <
                 chart_data["EMA_21"]) &
                (chart_data["EMA_9"].shift(1) >=
                 chart_data["EMA_21"].shift(1))
            )

            buy_points = chart_data[
                buy_condition
            ]

            sell_points = chart_data[
                sell_condition
            ]

            fig.add_trace(
                go.Scatter(
                    x=buy_points.index,
                    y=buy_points["Low"] * 0.995,
                    mode="markers",
                    name="BUY",
                    marker=dict(
                        symbol="triangle-up",
                        size=12
                    )
                ),
                row=1,
                col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=sell_points.index,
                    y=sell_points["High"] * 1.005,
                    mode="markers",
                    name="SELL",
                    marker=dict(
                        symbol="triangle-down",
                        size=12
                    )
                ),
                row=1,
                col=1
            )

            # =================================================
            # TRADE LEVEL LINES
            # =================================================

            if entry is not None:

                fig.add_hline(
                    y=entry,
                    line_dash="dash",
                    annotation_text="ENTRY",
                    row=1,
                    col=1
                )

                fig.add_hline(
                    y=stop_loss,
                    line_dash="dash",
                    annotation_text="STOP LOSS",
                    row=1,
                    col=1
                )

                fig.add_hline(
                    y=target1,
                    line_dash="dash",
                    annotation_text="TARGET 1",
                    row=1,
                    col=1
                )

                fig.add_hline(
                    y=target2,
                    line_dash="dash",
                    annotation_text="TARGET 2",
                    row=1,
                    col=1
                )

            # =================================================
            # RSI
            # =================================================

            fig.add_trace(
                go.Scatter(
                    x=chart_data.index,
                    y=chart_data["RSI"],
                    mode="lines",
                    name="RSI"
                ),
                row=2,
                col=1
            )

            fig.add_hline(
                y=70,
                line_dash="dot",
                row=2,
                col=1
            )

            fig.add_hline(
                y=30,
                line_dash="dot",
                row=2,
                col=1
            )

            # =================================================
            # MACD
            # =================================================

            fig.add_trace(
                go.Scatter(
                    x=chart_data.index,
                    y=chart_data["MACD"],
                    mode="lines",
                    name="MACD"
                ),
                row=3,
                col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=chart_data.index,
                    y=chart_data["MACD_SIGNAL"],
                    mode="lines",
                    name="Signal"
                ),
                row=3,
                col=1
            )

            # =================================================
            # CHART DESIGN
            # =================================================

            fig.update_layout(
                height=850,
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                margin=dict(
                    l=20,
                    r=20,
                    t=40,
                    b=20
                ),
                legend=dict(
                    orientation="h",
                    y=1.02
                )
            )

            fig.update_yaxes(
                showgrid=True,
                gridcolor="#202832"
            )

            fig.update_xaxes(
                showgrid=True,
                gridcolor="#202832"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # =================================================
            # INDICATORS
            # =================================================

            st.markdown("## 📌 Technical Indicators")

            i1, i2, i3, i4 = st.columns(4)

            i1.metric(
                "EMA 9",
                f"₹{ema9:,.2f}"
            )

            i2.metric(
                "EMA 21",
                f"₹{ema21:,.2f}"
            )

            i3.metric(
                "VWAP",
                f"₹{vwap:,.2f}"
            )

            i4.metric(
                "MACD HIST",
                f"{macd_hist:.4f}"
            )

            # =================================================
            # WHY SIGNAL
            # =================================================

            st.markdown("## 🔎 Why This Signal?")

            for reason in reasons:
                st.write("• " + reason)

            # =================================================
            # WARNING
            # =================================================

            st.warning(
                "⚠️ This dashboard is for technical analysis "
                "and educational purposes only. "
                "Signals are not guaranteed trading advice."
            )

        except Exception as e:

            st.error(
                "❌ Error while analyzing the stock"
            )

            st.code(
                str(e)
            )

else:

    st.info(
        "👈 Sidebar se stock symbol select karke "
        "**Analyze Stock** button press karein."
    )

    st.markdown("""
    ### Example

    **RELIANCE.NS**

    **TCS.NS**

    **INFY.NS**

    **HDFCBANK.NS**

    **SBIN.NS**

    **ICICIBANK.NS**
    """)
