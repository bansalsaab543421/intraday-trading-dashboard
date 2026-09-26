import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Indian Stock Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */
    .stApp {
        background-color: #ffffff;
        color: #222222;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1600px;
    }

    /* Remove excessive Streamlit spacing */
    div[data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }

    /* Header */
    .dashboard-title {
        font-size: 26px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 4px;
    }

    .dashboard-subtitle {
        font-size: 13px;
        color: #6b7280;
        margin-bottom: 15px;
    }

    /* Timeframe selector */
    div[data-testid="stRadio"] > div {
        gap: 5px;
    }

    div[data-testid="stRadio"] label {
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        padding: 6px 14px;
        cursor: pointer;
        transition: all 0.15s ease;
    }

    div[data-testid="stRadio"] label:hover {
        background-color: #f3f4f6;
        border-color: #9ca3af;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px;
    }

    /* Signal cards */
    .signal-card {
        border-radius: 8px;
        padding: 14px 18px;
        margin: 5px 0 12px 0;
        font-weight: 600;
    }

    .signal-hold {
        background-color: #fff7ed;
        border: 1px solid #fed7aa;
        color: #9a3412;
    }

    .signal-buy {
        background-color: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #047857;
    }

    .signal-sell {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        color: #dc2626;
    }

    /* Price header */
    .price-header {
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
        margin-bottom: 8px;
    }

    .stock-name {
        font-size: 20px;
        font-weight: 700;
        color: #111827;
    }

    .timeframe-label {
        font-size: 14px;
        color: #6b7280;
        background: #f3f4f6;
        padding: 4px 8px;
        border-radius: 5px;
    }

    .price-positive {
        color: #059669;
        font-weight: 600;
    }

    .price-negative {
        color: #dc2626;
        font-weight: 600;
    }

    /* Buy sell buttons */
    .buy-box {
        background-color: #10b981;
        color: white;
        border-radius: 7px;
        padding: 9px 16px;
        text-align: center;
        font-weight: 700;
    }

    .sell-box {
        background-color: #ef4444;
        color: white;
        border-radius: 7px;
        padding: 9px 16px;
        text-align: center;
        font-weight: 700;
    }

    /* Section title */
    .section-title {
        font-size: 17px;
        font-weight: 700;
        color: #111827;
        margin-top: 15px;
        margin-bottom: 5px;
    }

    /* Footer */
    .footer-note {
        color: #6b7280;
        font-size: 12px;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="dashboard-title">🇮🇳 Pro Indian Stock Trading Dashboard</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="dashboard-subtitle">'
    "TradingView-style candlestick chart • Technical indicators • Signal analysis"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def get_period_and_interval(timeframe):
    """
    Convert the selected UI timeframe into Yahoo Finance
    period + interval.

    Important:
    Yahoo Finance places restrictions on historical intraday
    data. Intraday intervals are therefore limited to a practical
    recent period.
    """

    if timeframe == "15m":
        return "5d", "15m"

    if timeframe == "30m":
        return "30d", "30m"

    if timeframe == "60m":
        return "60d", "60m"

    if timeframe == "1D":
        return "1mo", "1d"

    if timeframe == "1W":
        return "7d", "1d"

    if timeframe == "1Y":
        return "1y", "1d"

    if timeframe == "5Y":
        return "5y", "1d"

    if timeframe == "MAX":
        return "max", "1d"

    return "1y", "1d"


@st.cache_data(ttl=60, show_spinner=False)
def download_stock_data(symbol, period, interval):
    """
    Download stock data from Yahoo Finance.
    Cached for 60 seconds to reduce unnecessary requests.
    """

    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    return df


def clean_dataframe(df):
    """
    Handle Yahoo Finance MultiIndex columns and clean data.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Yahoo Finance sometimes returns MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            df.columns = df.columns.droplevel(-1)

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for column in required_columns:
        if column not in df.columns:
            return pd.DataFrame()

    df = df[required_columns].copy()

    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ],
        inplace=True,
    )

    return df


# ============================================================
# TECHNICAL INDICATORS
# ============================================================


def compute_indicators(df):
    """
    Calculate EMA, RSI, MACD and ATR.
    """

    df = df.copy()

    # ---------------- EMA ----------------

    df["EMA_20"] = (
        df["Close"]
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )

    df["EMA_50"] = (
        df["Close"]
        .ewm(
            span=50,
            adjust=False,
        )
        .mean()
    )

    # ---------------- RSI ----------------

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = (
        gain.ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False,
        )
        .mean()
    )

    avg_loss = (
        loss.ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False,
        )
        .mean()
    )

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan,
    )

    df["RSI"] = (
        100
        - (
            100
            / (
                1
                + rs
            )
        )
    )

    df["RSI"] = df["RSI"].fillna(50)

    # ---------------- MACD ----------------

    ema12 = (
        df["Close"]
        .ewm(
            span=12,
            adjust=False,
        )
        .mean()
    )

    ema26 = (
        df["Close"]
        .ewm(
            span=26,
            adjust=False,
        )
        .mean()
    )

    df["MACD"] = ema12 - ema26

    df["MACD_signal"] = (
        df["MACD"]
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    df["MACD_hist"] = (
        df["MACD"]
        - df["MACD_signal"]
    )

    # ---------------- ATR ----------------

    previous_close = df["Close"].shift(1)

    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (
                df["High"]
                - previous_close
            ).abs(),
            (
                df["Low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ATR"] = (
        true_range
        .ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False,
        )
        .mean()
    )

    return df


# ============================================================
# SIGNAL GENERATION
# ============================================================


def generate_signal(
    df,
    rsi_overbought=70,
    rsi_oversold=30,
    atr_stop_mult=1.5,
    atr_target_mult=2.5,
):
    """
    Generate BUY / SELL / HOLD based on:

    BUY:
        EMA20 > EMA50
        +
        fresh bullish MACD crossover
        +
        RSI below overbought

    SELL:
        EMA20 < EMA50
        +
        fresh bearish MACD crossover
        +
        RSI above oversold

    Otherwise HOLD.
    """

    if len(df) < 2:
        return {
            "trend": "UNKNOWN",
            "action": "HOLD",
            "reason": "Not enough data",
            "ltp": 0,
            "entry": None,
            "sl": None,
            "target": None,
            "rsi": 50,
            "macd_hist": 0,
        }

    last = df.iloc[-1]
    previous = df.iloc[-2]

    # ---------------- Trend ----------------

    if (
        pd.notna(last["EMA_20"])
        and pd.notna(last["EMA_50"])
    ):
        trend = (
            "UP"
            if last["EMA_20"]
            > last["EMA_50"]
            else "DOWN"
        )
    else:
        trend = "UNKNOWN"

    action = "HOLD"
    reason = "Waiting for stronger confirmation"

    # ---------------- MACD crossover ----------------

    bullish_cross = False
    bearish_cross = False

    if (
        pd.notna(previous["MACD"])
        and pd.notna(previous["MACD_signal"])
        and pd.notna(last["MACD"])
        and pd.notna(last["MACD_signal"])
    ):

        bullish_cross = (
            previous["MACD"]
            <= previous["MACD_signal"]
            and
            last["MACD"]
            > last["MACD_signal"]
        )

        bearish_cross = (
            previous["MACD"]
            >= previous["MACD_signal"]
            and
            last["MACD"]
            < last["MACD_signal"]
        )

    # ---------------- BUY ----------------

    if (
        trend == "UP"
        and bullish_cross
        and last["RSI"] < rsi_overbought
    ):

        action = "BUY"

        reason = (
            "Uptrend + bullish MACD crossover + "
            "RSI confirmation"
        )

    # ---------------- SELL ----------------

    elif (
        trend == "DOWN"
        and bearish_cross
        and last["RSI"] > rsi_oversold
    ):

        action = "SELL"

        reason = (
            "Downtrend + bearish MACD crossover + "
            "RSI confirmation"
        )

    # ---------------- Levels ----------------

    ltp = float(last["Close"])

    atr = (
        float(last["ATR"])
        if pd.notna(last["ATR"])
        else 0
    )

    entry = None
    stop_loss = None
    target = None

    if action == "BUY":

        entry = round(
            ltp,
            2,
        )

        stop_loss = round(
            ltp
            - (
                atr
                * atr_stop_mult
            ),
            2,
        )

        target = round(
            ltp
            + (
                atr
                * atr_target_mult
            ),
            2,
        )

    elif action == "SELL":

        entry = round(
            ltp,
            2,
        )

        stop_loss = round(
            ltp
            + (
                atr
                * atr_stop_mult
            ),
            2,
        )

        target = round(
            ltp
            - (
                atr
                * atr_target_mult
            ),
            2,
        )

    return {
        "trend": trend,
        "action": action,
        "reason": reason,
        "ltp": round(
            ltp,
            2,
        ),
        "entry": entry,
        "sl": stop_loss,
        "target": target,
        "rsi": round(
            float(last["RSI"]),
            2,
        ),
        "macd_hist": round(
            float(last["MACD_hist"]),
            4,
        ),
    }


# ============================================================
# TOP CONTROLS
# ============================================================


control_col1, control_col2 = st.columns(
    [5, 1]
)

with control_col1:

    symbol = st.text_input(
        "Stock Symbol",
        value="RELIANCE.NS",
        placeholder="Example: RELIANCE.NS",
        label_visibility="collapsed",
    ).strip().upper()


with control_col2:

    if st.button(
        "🔄 Refresh",
        use_container_width=True,
    ):

        st.cache_data.clear()
        st.rerun()


# ============================================================
# TIMEFRAME
# ============================================================


st.markdown(
    '<div class="section-title">Chart Timeframe</div>',
    unsafe_allow_html=True,
)

timeframe = st.radio(
    "Select timeframe",
    [
        "15m",
        "30m",
        "60m",
        "1D",
        "1W",
        "1Y",
        "5Y",
        "MAX",
    ],
    index=3,
    horizontal=True,
    label_visibility="collapsed",
)


# ============================================================
# INDICATOR CONTROLS
# ============================================================


indicator_col1, indicator_col2, indicator_col3 = st.columns(
    [1, 1, 4]
)

with indicator_col1:

    show_ema = st.checkbox(
        "EMA",
        value=True,
    )

with indicator_col2:

    show_volume = st.checkbox(
        "Volume",
        value=True,
    )


# ============================================================
# DOWNLOAD DATA
# ============================================================


period, interval = get_period_and_interval(
    timeframe
)


try:

    with st.spinner(
        f"Loading {symbol}..."
    ):

        raw_df = download_stock_data(
            symbol,
            period,
            interval,
        )

    df = clean_dataframe(
        raw_df
    )

    if df.empty:

        st.error(
            f"No data found for `{symbol}`."
        )

        st.info(
            "Use an NSE symbol such as "
            "`RELIANCE.NS`, `TCS.NS`, "
            "`INFY.NS`, `SBIN.NS`."
        )

        st.stop()

    # Calculate indicators
    df = compute_indicators(
        df
    )

    # Need at least 2 candles
    if len(df) < 2:

        st.warning(
            "Not enough candles available "
            "for this timeframe."
        )

        st.stop()


    # ========================================================
    # CURRENT MARKET DATA
    # ========================================================

    last = df.iloc[-1]

    previous = df.iloc[-2]

    current_price = float(
        last["Close"]
    )

    open_price = float(
        last["Open"]
    )

    high_price = float(
        last["High"]
    )

    low_price = float(
        last["Low"]
    )

    close_price = float(
        last["Close"]
    )

    volume = float(
        last["Volume"]
    )

    previous_close = float(
        previous["Close"]
    )

    price_change = (
        current_price
        - previous_close
    )

    price_change_pct = (
        price_change
        / previous_close
        * 100
        if previous_close
        else 0
    )


    # ========================================================
    # SIGNAL
    # ========================================================

    sig = generate_signal(
        df
    )


    # ========================================================
    # STOCK HEADER
    # ========================================================

    change_class = (
        "price-positive"
        if price_change >= 0
        else "price-negative"
    )

    change_symbol = (
        "+"
        if price_change >= 0
        else ""
    )

    st.markdown(
        f"""
        <div class="price-header">

            <span class="stock-name">
                {symbol}
            </span>

            <span class="timeframe-label">
                {timeframe}
            </span>

            <span>
                O {open_price:,.2f}
            </span>

            <span>
                H {high_price:,.2f}
            </span>

            <span>
                L {low_price:,.2f}
            </span>

            <span>
                C {close_price:,.2f}
            </span>

            <span class="{change_class}">
                {change_symbol}{price_change:,.2f}
                ({change_symbol}{price_change_pct:.2f}%)
            </span>

        </div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # BUY / SELL QUOTES
    # ========================================================

    quote_col1, quote_col2, quote_col3 = st.columns(
        [1, 1, 6]
    )

    with quote_col1:

        st.markdown(
            f"""
            <div class="sell-box">
                <div style="font-size:20px;">
                    ₹{current_price:,.2f}
                </div>
                <div style="font-size:12px;">
                    SELL
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with quote_col2:

        st.markdown(
            f"""
            <div class="buy-box">
                <div style="font-size:20px;">
                    ₹{current_price:,.2f}
                </div>
                <div style="font-size:12px;">
                    BUY
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


    # ========================================================
    # VOLUME HEADER
    # ========================================================

    if volume >= 1_000_000:

        volume_text = (
            f"{volume / 1_000_000:.2f} M"
        )

    elif volume >= 1_000:

        volume_text = (
            f"{volume / 1_000:.2f} K"
        )

    else:

        volume_text = (
            f"{volume:,.0f}"
        )


    st.caption(
        f"Volume  {volume_text}"
    )


    # ========================================================
    # SIGNAL BANNER
    # ========================================================

    if sig["action"] == "BUY":

        signal_class = "signal-buy"

        signal_text = (
            f"🟢 BUY SIGNAL — "
            f"Entry ₹{sig['entry']:,.2f} | "
            f"Target ₹{sig['target']:,.2f} | "
            f"Stop ₹{sig['sl']:,.2f}"
        )

    elif sig["action"] == "SELL":

        signal_class = "signal-sell"

        signal_text = (
            f"🔴 SELL SIGNAL — "
            f"Entry ₹{sig['entry']:,.2f} | "
            f"Target ₹{sig['target']:,.2f} | "
            f"Stop ₹{sig['sl']:,.2f}"
        )

    else:

        signal_class = "signal-hold"

        signal_text = (
            f"🟠 HOLD — "
            f"{sig['reason']}"
        )

    st.markdown(
        f"""
        <div class="signal-card {signal_class}">
            {signal_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # MAIN TRADINGVIEW STYLE CHART
    # ========================================================

    chart_rows = 2 if show_volume else 1

    if show_volume:

        row_heights = [
            0.80,
            0.20,
        ]

    else:

        row_heights = [
            1.0
        ]


    fig = make_subplots(
        rows=chart_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.015,
        row_heights=row_heights,
        specs=[
            [{"secondary_y": False}]
            for _ in range(chart_rows)
        ],
    )


    # ========================================================
    # CANDLESTICKS
    # ========================================================

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],

            increasing_line_color="#00b386",
            increasing_fillcolor="#00b386",

            decreasing_line_color="#ff5b45",
            decreasing_fillcolor="#ff5b45",

            name="Candles",

            showlegend=False,
        ),
        row=1,
        col=1,
    )


    # ========================================================
    # EMA 20 / 50
    # ========================================================

    if show_ema:

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EMA_20"],
                mode="lines",
                line=dict(
                    color="#f59e0b",
                    width=1.4,
                ),
                name="EMA 20",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EMA_50"],
                mode="lines",
                line=dict(
                    color="#8b5cf6",
                    width=1.4,
                ),
                name="EMA 50",
            ),
            row=1,
            col=1,
        )


    # ========================================================
    # VOLUME
    # ========================================================

    if show_volume:

        volume_colors = np.where(
            df["Close"]
            >= df["Open"],
            "#00b894",
            "#ff7b6b",
        )

        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                marker_color=volume_colors,
                opacity=0.70,
                name="Volume",
                showlegend=False,
            ),
            row=2,
            col=1,
        )


    # ========================================================
    # CURRENT PRICE LINE
    # ========================================================

    fig.add_hline(
        y=current_price,
        line_dash="dot",
        line_color="#ff5b45",
        line_width=1,
        row=1,
        col=1,
    )


    # ========================================================
    # ENTRY / STOP / TARGET
    # ========================================================

    if sig["action"] in (
        "BUY",
        "SELL",
    ):

        # Entry
        fig.add_hline(
            y=sig["entry"],
            line_dash="dash",
            line_color="#3b82f6",
            annotation_text="Entry",
            annotation_position="right",
            row=1,
            col=1,
        )

        # Stop
        fig.add_hline(
            y=sig["sl"],
            line_dash="dash",
            line_color="#ef4444",
            annotation_text="SL",
            annotation_position="right",
            row=1,
            col=1,
        )

        # Target
        fig.add_hline(
            y=sig["target"],
            line_dash="dash",
            line_color="#10b981",
            annotation_text="Target",
            annotation_position="right",
            row=1,
            col=1,
        )


        marker_color = (
            "#10b981"
            if sig["action"]
            == "BUY"
            else "#ef4444"
        )

        marker_symbol = (
            "triangle-up"
            if sig["action"]
            == "BUY"
            else "triangle-down"
        )


        fig.add_trace(
            go.Scatter(
                x=[df.index[-1]],
                y=[current_price],
                mode="markers",
                marker=dict(
                    symbol=marker_symbol,
                    size=15,
                    color=marker_color,
                    line=dict(
                        color="#ffffff",
                        width=1,
                    ),
                ),
                name=sig["action"],
            ),
            row=1,
            col=1,
        )


    # ========================================================
    # CHART LAYOUT
    # ========================================================

    fig.update_layout(

        height=650,

        paper_bgcolor="#ffffff",

        plot_bgcolor="#ffffff",

        margin=dict(
            l=10,
            r=60,
            t=10,
            b=10,
        ),

        hovermode="x unified",

        xaxis_rangeslider_visible=False,

        showlegend=True,

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0)",
        ),

        font=dict(
            color="#374151",
            size=12,
        ),
    )


    # ========================================================
    # X AXIS
    # ========================================================

    fig.update_xaxes(

        showgrid=True,

        gridcolor="#eeeeee",

        gridwidth=1,

        zeroline=False,

        showline=False,

        rangeslider_visible=False,

        fixedrange=False,
    )


    # ========================================================
    # PRICE Y AXIS
    # ========================================================

    fig.update_yaxes(

        showgrid=True,

        gridcolor="#eeeeee",

        gridwidth=1,

        zeroline=False,

        showline=False,

        fixedrange=False,

        side="right",

        row=1,

        col=1,
    )


    # ========================================================
    # VOLUME AXIS
    # ========================================================

    if show_volume:

        fig.update_yaxes(

            showgrid=False,

            showticklabels=False,

            zeroline=False,

            showline=False,

            row=2,

            col=1,
        )


    # ========================================================
    # RENDER CHART
    # ========================================================

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "scrollZoom": True,
            "doubleClick": "reset",
            "responsive": True,
        },
    )


    # ========================================================
    # RSI + MACD
    # ========================================================

    indicator_col1, indicator_col2 = st.columns(
        2
    )


    # ========================================================
    # RSI
    # ========================================================

    with indicator_col1:

        st.markdown(
            "### RSI (14)"
        )

        fig_rsi = go.Figure()

        fig_rsi.add_trace(
            go.Scatter(
                x=df.index,
                y=df["RSI"],
                mode="lines",
                line=dict(
                    color="#2563eb",
                    width=1.5,
                ),
                name="RSI",
            )
        )

        fig_rsi.add_hline(
            y=70,
            line_dash="dot",
            line_color="#ef4444",
        )

        fig_rsi.add_hline(
            y=30,
            line_dash="dot",
            line_color="#10b981",
        )

        fig_rsi.update_layout(

            height=220,

            margin=dict(
                l=5,
                r=5,
                t=5,
                b=5,
            ),

            paper_bgcolor="#ffffff",

            plot_bgcolor="#ffffff",

            showlegend=False,

            yaxis=dict(
                range=[0, 100],
                showgrid=True,
                gridcolor="#eeeeee",
            ),

            xaxis=dict(
                showgrid=False,
            ),
        )

        st.plotly_chart(
            fig_rsi,
            use_container_width=True,
        )


    # ========================================================
    # MACD
    # ========================================================

    with indicator_col2:

        st.markdown(
            "### MACD"
        )

        fig_macd = go.Figure()

        macd_colors = np.where(
            df["MACD_hist"] >= 0,
            "#10b981",
            "#ef4444",
        )

        fig_macd.add_trace(
            go.Bar(
                x=df.index,
                y=df["MACD_hist"],
                marker_color=macd_colors,
                opacity=0.65,
                name="Histogram",
            )
        )

        fig_macd.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD"],
                line=dict(
                    color="#2563eb",
                    width=1.4,
                ),
                name="MACD",
            )
        )

        fig_macd.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD_signal"],
                line=dict(
                    color="#f59e0b",
                    width=1.4,
                ),
                name="Signal",
            )
        )

        fig_macd.update_layout(

            height=220,

            margin=dict(
                l=5,
                r=5,
                t=5,
                b=5,
            ),

            paper_bgcolor="#ffffff",

            plot_bgcolor="#ffffff",

            showlegend=True,

            legend=dict(
                orientation="h",
                y=1.1,
            ),

            yaxis=dict(
                showgrid=True,
                gridcolor="#eeeeee",
            ),

            xaxis=dict(
                showgrid=False,
            ),
        )

        st.plotly_chart(
            fig_macd,
            use_container_width=True,
        )


    # ========================================================
    # MARKET OVERVIEW
    # ========================================================

    st.markdown(
        "### 📊 Market Overview"
    )

    metric1, metric2, metric3, metric4, metric5 = st.columns(
        5
    )

    metric1.metric(
        "LAST PRICE",
        f"₹{current_price:,.2f}",
        f"{price_change:+.2f} ({price_change_pct:+.2f}%)",
    )

    metric2.metric(
        "RSI (14)",
        f"{sig['rsi']:.1f}",
    )

    metric3.metric(
        "TREND",
        sig["trend"],
    )

    metric4.metric(
        "MACD",
        f"{sig['macd_hist']:.3f}",
    )

    metric5.metric(
        "SIGNAL",
        sig["action"],
    )


    # ========================================================
    # TRADE SETUP
    # ========================================================

    st.markdown(
        "### 🎯 Trade Setup"
    )

    setup1, setup2, setup3, setup4 = st.columns(
        4
    )

    setup1.metric(
        "ENTRY",
        (
            f"₹{sig['entry']:,.2f}"
            if sig["entry"]
            else "--"
        ),
    )

    setup2.metric(
        "TARGET",
        (
            f"₹{sig['target']:,.2f}"
            if sig["target"]
            else "--"
        ),
    )

    setup3.metric(
        "STOP LOSS",
        (
            f"₹{sig['sl']:,.2f}"
            if sig["sl"]
            else "--"
        ),
    )

    if (
        sig["entry"]
        and sig["target"]
        and sig["sl"]
    ):

        if sig["action"] == "BUY":

            risk = (
                sig["entry"]
                - sig["sl"]
            )

            reward = (
                sig["target"]
                - sig["entry"]
            )

        else:

            risk = (
                sig["sl"]
                - sig["entry"]
            )

            reward = (
                sig["entry"]
                - sig["target"]
            )

        rr = (
            reward / risk
            if risk > 0
            else 0
        )

        setup4.metric(
            "RISK : REWARD",
            f"1 : {rr:.2f}",
        )

    else:

        setup4.metric(
            "RISK : REWARD",
            "--",
        )


    # ========================================================
    # SIGNAL REASON
    # ========================================================

    st.caption(
        f"Signal reasoning: {sig['reason']}"
    )


    # ========================================================
    # DATA INFORMATION
    # ========================================================

    if timeframe in (
        "15m",
        "30m",
        "60m",
    ):

        st.info(
            "ℹ️ Intraday historical data from "
            "Yahoo Finance is limited. The "
            "15m/30m/60m charts therefore use "
            "recent available intraday candles."
        )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.markdown(
        """
        <div class="footer-note">

        ⚠️ <b>Data & risk notice:</b>
        Yahoo Finance is a free market-data source and may be
        delayed, incomplete, rate-limited, or temporarily unavailable.
        It should not be treated as a broker's live execution feed.

        This dashboard is for educational and analytical purposes.
        Technical signals are algorithmic indicators, not guarantees
        of future market movements. Always verify prices and orders
        through your broker before taking action.

        </div>
        """,
        unsafe_allow_html=True,
    )


except Exception as e:

    st.error(
        "Unable to load the stock data."
    )

    st.info(
        "Please check the symbol. "
        "For NSE stocks use `.NS`, for example "
        "`RELIANCE.NS`, `TCS.NS`, `INFY.NS`, "
        "`HDFCBANK.NS`."
    )

    with st.expander(
        "Technical details"
    ):

        st.exception(e)
