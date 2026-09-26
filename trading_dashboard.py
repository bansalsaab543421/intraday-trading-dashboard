import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Upstox Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CONSTANTS
# ============================================================

UPSTOX_BASE = "https://api.upstox.com"

HISTORICAL_V3 = (
    "https://api.upstox.com/v3/historical-candle"
)

INSTRUMENT_SEARCH = (
    "https://api.upstox.com/v2/instruments/search"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #ffffff;
        color: #111827;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }

    /* Remove excessive spacing */
    div[data-testid="stVerticalBlock"] {
        gap: 0.35rem;
    }

    /* Top header */
    .top-title {
        font-size: 22px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 2px;
    }

    .top-subtitle {
        color: #6b7280;
        font-size: 12px;
        margin-bottom: 12px;
    }

    /* Chart header */
    .chart-header {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
        padding: 5px 0;
    }

    .chart-symbol {
        font-size: 18px;
        font-weight: 700;
        color: #111827;
    }

    .chart-timeframe {
        color: #6b7280;
        font-size: 13px;
    }

    .ohlc {
        font-size: 13px;
        color: #374151;
    }

    .positive {
        color: #00a878;
        font-weight: 600;
    }

    .negative {
        color: #ef4444;
        font-weight: 600;
    }

    /* TradingView / Upstox style selector */
    div[data-testid="stRadio"] > div {
        gap: 2px;
    }

    div[data-testid="stRadio"] label {
        border: 1px solid #e5e7eb;
        background: white;
        border-radius: 5px;
        padding: 4px 10px;
        font-size: 13px;
        cursor: pointer;
    }

    div[data-testid="stRadio"] label:hover {
        background: #f3f4f6;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        border: 1px solid #e5e7eb;
        border-radius: 7px;
        padding: 10px;
        background: #ffffff;
    }

    /* Signal */
    .signal {
        border-radius: 7px;
        padding: 10px 14px;
        margin: 5px 0;
        font-size: 13px;
        font-weight: 600;
    }

    .buy {
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
    }

    .sell {
        background: #fef2f2;
        color: #dc2626;
        border: 1px solid #fecaca;
    }

    .hold {
        background: #fff7ed;
        color: #c2410c;
        border: 1px solid #fed7aa;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "access_token" not in st.session_state:
    st.session_state.access_token = None

if "instrument_key" not in st.session_state:
    st.session_state.instrument_key = None

if "instrument_name" not in st.session_state:
    st.session_state.instrument_name = None


# ============================================================
# GET ACCESS TOKEN
# ============================================================

def get_access_token():

    # First preference: Streamlit secrets
    try:

        token = st.secrets.get(
            "UPSTOX_ACCESS_TOKEN",
            None
        )

        if token:
            return token

    except Exception:
        pass

    # Second preference: session
    return st.session_state.access_token


# ============================================================
# API HEADERS
# ============================================================

def api_headers(access_token):

    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }


# ============================================================
# SEARCH INSTRUMENT
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def find_nse_equity(
    symbol,
    access_token,
):

    symbol = (
        symbol.upper()
        .replace(".NS", "")
        .strip()
    )

    headers = api_headers(
        access_token
    )

    params = {
        "query": symbol,
        "exchanges": "NSE",
        "segments": "EQ",
        "page_number": 1,
        "records": 30,
    }

    response = requests.get(
        INSTRUMENT_SEARCH,
        headers=headers,
        params=params,
        timeout=15,
    )

    if response.status_code != 200:

        raise RuntimeError(
            f"Instrument search failed: "
            f"{response.status_code} "
            f"{response.text}"
        )

    payload = response.json()

    instruments = payload.get(
        "data",
        []
    )

    if not instruments:

        raise ValueError(
            f"No NSE equity found for "
            f"{symbol}"
        )

    # Prefer exact trading symbol
    exact = [
        x
        for x in instruments
        if str(
            x.get(
                "trading_symbol",
                ""
            )
        ).upper()
        == symbol
    ]

    if exact:
        return exact[0]

    # Otherwise first NSE EQ result
    for item in instruments:

        if item.get(
            "segment"
        ) == "NSE_EQ":

            return item

    return instruments[0]


# ============================================================
# HISTORICAL DATA
# ============================================================

def fetch_historical_candles(
    instrument_key,
    unit,
    interval,
    from_date,
    to_date,
    access_token,
):

    url = (
        f"{HISTORICAL_V3}/"
        f"{instrument_key}/"
        f"{unit}/"
        f"{interval}/"
        f"{to_date}/"
        f"{from_date}"
    )

    response = requests.get(
        url,
        headers=api_headers(
            access_token
        ),
        timeout=30,
    )

    if response.status_code != 200:

        raise RuntimeError(
            f"Historical API failed: "
            f"{response.status_code}\n"
            f"{response.text}"
        )

    payload = response.json()

    candles = (
        payload
        .get("data", {})
        .get("candles", [])
    )

    if not candles:

        return pd.DataFrame(
            columns=[
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "OI",
            ]
        )

    rows = []

    for candle in candles:

        if len(candle) < 6:
            continue

        rows.append(
            {
                "Timestamp": candle[0],
                "Open": candle[1],
                "High": candle[2],
                "Low": candle[3],
                "Close": candle[4],
                "Volume": candle[5],
                "OI": (
                    candle[6]
                    if len(candle) > 6
                    else 0
                ),
            }
        )

    df = pd.DataFrame(
        rows
    )

    if df.empty:
        return df

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["Timestamp"]
    )

    # Upstox timestamps include IST offset.
    # Convert to Asia/Kolkata for display.
    if df["Timestamp"].dt.tz is not None:

        df["Timestamp"] = (
            df["Timestamp"]
            .dt.tz_convert(
                "Asia/Kolkata"
            )
            .dt.tz_localize(None)
        )

    df.set_index(
        "Timestamp",
        inplace=True
    )

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "OI",
    ]

    for column in numeric_columns:

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

    df.sort_index(
        inplace=True
    )

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    return df


# ============================================================
# DATE RANGE
# ============================================================

def get_dates(
    candle_interval,
    chart_range,
):

    today = date.today()

    # -----------------------------
    # Daily candles
    # -----------------------------

    if candle_interval == "1D":

        if chart_range == "1W":
            start = today - timedelta(
                days=14
            )

        elif chart_range == "1M":
            start = today - timedelta(
                days=45
            )

        elif chart_range == "6M":
            start = today - timedelta(
                days=210
            )

        elif chart_range == "1Y":
            start = today - timedelta(
                days=400
            )

        elif chart_range == "5Y":
            start = today - timedelta(
                days=1900
            )

        elif chart_range == "MAX":
            start = date(
                2000,
                1,
                1
            )

        else:
            start = today - timedelta(
                days=400
            )

        return start, today


    # -----------------------------
    # 60 minute
    # -----------------------------

    if candle_interval == "60m":

        if chart_range == "1D":
            days = 2

        elif chart_range == "1W":
            days = 14

        elif chart_range == "1M":
            days = 45

        elif chart_range == "6M":
            days = 190

        elif chart_range == "1Y":
            days = 365

        elif chart_range == "5Y":
            days = 365 * 5

        else:
            days = 365

        return (
            today - timedelta(days=days),
            today
        )


    # -----------------------------
    # 30 minute
    # -----------------------------

    if candle_interval == "30m":

        if chart_range == "1D":
            days = 2

        elif chart_range == "1W":
            days = 14

        elif chart_range == "1M":
            days = 45

        elif chart_range == "6M":
            days = 190

        elif chart_range == "1Y":
            days = 365

        elif chart_range == "5Y":
            days = 365 * 5

        else:
            days = 365

        return (
            today - timedelta(days=days),
            today
        )


    # -----------------------------
    # 15 minute
    # -----------------------------

    if candle_interval == "15m":

        if chart_range == "1D":
            days = 2

        elif chart_range == "1W":
            days = 14

        elif chart_range == "1M":
            days = 31

        elif chart_range == "6M":
            days = 180

        elif chart_range == "1Y":
            days = 365

        elif chart_range == "5Y":
            days = 365 * 5

        else:
            days = 365

        return (
            today - timedelta(days=days),
            today
        )

    return (
        today - timedelta(days=400),
        today
    )


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    # EMA
    df["EMA20"] = (
        df["Close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    df["EMA50"] = (
        df["Close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    # RSI
    delta = df["Close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain
        .ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False
        )
        .mean()
    )

    avg_loss = (
        loss
        .ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False
        )
        .mean()
    )

    rs = (
        avg_gain
        / avg_loss.replace(
            0,
            np.nan
        )
    )

    df["RSI"] = (
        100
        - (
            100
            / (
                1 + rs
            )
        )
    )

    df["RSI"] = (
        df["RSI"]
        .fillna(50)
    )

    # MACD
    ema12 = (
        df["Close"]
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        df["Close"]
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    df["MACD"] = (
        ema12 - ema26
    )

    df["MACDSignal"] = (
        df["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    df["MACDHist"] = (
        df["MACD"]
        - df["MACDSignal"]
    )

    # ATR
    previous_close = (
        df["Close"]
        .shift(1)
    )

    tr = pd.concat(
        [
            df["High"]
            - df["Low"],

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
        tr
        .ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False
        )
        .mean()
    )

    return df


# ============================================================
# SIGNAL
# ============================================================

def generate_signal(df):

    if len(df) < 2:

        return {
            "action": "HOLD",
            "trend": "UNKNOWN",
            "reason": "Not enough candles",
            "entry": None,
            "target": None,
            "sl": None,
        }

    last = df.iloc[-1]
    prev = df.iloc[-2]

    trend = (
        "UP"
        if last["EMA20"]
        > last["EMA50"]
        else "DOWN"
    )

    bullish_cross = (
        prev["MACD"]
        <= prev["MACDSignal"]
        and
        last["MACD"]
        > last["MACDSignal"]
    )

    bearish_cross = (
        prev["MACD"]
        >= prev["MACDSignal"]
        and
        last["MACD"]
        < last["MACDSignal"]
    )

    action = "HOLD"

    reason = (
        "Waiting for stronger confirmation"
    )

    if (
        trend == "UP"
        and bullish_cross
        and last["RSI"] < 70
    ):

        action = "BUY"

        reason = (
            "Uptrend + bullish MACD crossover "
            "+ RSI confirmation"
        )

    elif (
        trend == "DOWN"
        and bearish_cross
        and last["RSI"] > 30
    ):

        action = "SELL"

        reason = (
            "Downtrend + bearish MACD crossover "
            "+ RSI confirmation"
        )

    entry = None
    target = None
    sl = None

    atr = (
        float(last["ATR"])
        if pd.notna(last["ATR"])
        else 0
    )

    price = float(
        last["Close"]
    )

    if action == "BUY":

        entry = round(
            price,
            2
        )

        sl = round(
            price - 1.5 * atr,
            2
        )

        target = round(
            price + 2.5 * atr,
            2
        )

    elif action == "SELL":

        entry = round(
            price,
            2
        )

        sl = round(
            price + 1.5 * atr,
            2
        )

        target = round(
            price - 2.5 * atr,
            2
        )

    return {
        "action": action,
        "trend": trend,
        "reason": reason,
        "entry": entry,
        "target": target,
        "sl": sl,
    }


# ============================================================
# CHART
# ============================================================

def create_chart(
    df,
    candle_interval,
    show_volume,
    show_ema,
):

    rows = (
        2
        if show_volume
        else 1
    )

    heights = (
        [0.78, 0.22]
        if show_volume
        else [1]
    )

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.015,
        row_heights=heights,
    )

    # Candles
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],

            increasing_line_color="#00b386",
            increasing_fillcolor="#00b386",

            decreasing_line_color="#ff563f",
            decreasing_fillcolor="#ff563f",

            name="Price",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # EMA
    if show_ema:

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EMA20"],
                mode="lines",
                line=dict(
                    color="#ff9800",
                    width=1.2,
                ),
                name="EMA 20",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EMA50"],
                mode="lines",
                line=dict(
                    color="#7047eb",
                    width=1.2,
                ),
                name="EMA 50",
            ),
            row=1,
            col=1,
        )

    # Volume
    if show_volume:

        colors = np.where(
            df["Close"]
            >= df["Open"],
            "#7edac5",
            "#ff9b9b",
        )

        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                marker_color=colors,
                opacity=0.75,
                name="Volume",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Layout
    fig.update_layout(

        height=650,

        paper_bgcolor="#ffffff",

        plot_bgcolor="#ffffff",

        margin=dict(
            l=5,
            r=55,
            t=5,
            b=5,
        ),

        hovermode="x unified",

        xaxis_rangeslider_visible=False,

        dragmode="pan",

        showlegend=True,

        legend=dict(
            orientation="h",
            y=1.01,
            x=0,
        ),

        font=dict(
            color="#374151",
            size=11,
        ),
    )

    # Price axis
    fig.update_yaxes(
        side="right",
        showgrid=True,
        gridcolor="#eeeeee",
        zeroline=False,
        row=1,
        col=1,
    )

    # Volume
    if show_volume:

        fig.update_yaxes(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            row=2,
            col=1,
        )

    # X axis
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#eeeeee",
        zeroline=False,
        rangeslider_visible=False,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        showline=False,
    )

    return fig


# ============================================================
# AUTHENTICATION UI
# ============================================================

st.markdown(
    """
    <div class="top-title">
        📈 Upstox Market Dashboard
    </div>

    <div class="top-subtitle">
        Upstox V3 market data • Candlestick chart • Technical analysis
    </div>
    """,
    unsafe_allow_html=True,
)


access_token = get_access_token()


if not access_token:

    st.warning(
        "Upstox access token is not configured."
    )

    st.markdown(
        """
        Add your Upstox access token to Streamlit secrets.

        Create:

        `.streamlit/secrets.toml`

        and add:

        `UPSTOX_ACCESS_TOKEN = "YOUR_ACCESS_TOKEN"`

        Do **not** put the token directly inside `app.py`
        or commit it to GitHub.
        """
    )

    st.stop()


# ============================================================
# TOP CONTROLS
# ============================================================

col_symbol, col_refresh = st.columns(
    [6, 1]
)

with col_symbol:

    symbol = st.text_input(
        "NSE Symbol",
        value="RELIANCE",
        placeholder="RELIANCE",
        label_visibility="collapsed",
    ).upper().strip()


with col_refresh:

    if st.button(
        "↻ Refresh",
        use_container_width=True,
    ):

        st.cache_data.clear()
        st.rerun()


# ============================================================
# FIND INSTRUMENT
# ============================================================

try:

    instrument = find_nse_equity(
        symbol,
        access_token,
    )

except Exception as e:

    st.error(
        f"Unable to find `{symbol}` in Upstox."
    )

    st.exception(e)

    st.stop()


instrument_key = instrument[
    "instrument_key"
]

display_symbol = instrument.get(
    "trading_symbol",
    symbol,
)

instrument_name = instrument.get(
    "short_name",
    display_symbol,
)


# ============================================================
# TIMEFRAME CONTROLS
# ============================================================

st.markdown(
    "**Candle interval**"
)

candle_interval = st.radio(
    "Candle interval",
    [
        "15m",
        "30m",
        "60m",
        "1D",
    ],
    horizontal=True,
    index=0,
    label_visibility="collapsed",
)


st.markdown(
    "**Chart range**"
)

chart_range = st.radio(
    "Chart range",
    [
        "1D",
        "1W",
        "1M",
        "6M",
        "1Y",
        "5Y",
        "MAX",
    ],
    horizontal=True,
    index=1,
    label_visibility="collapsed",
)


# ============================================================
# CHART OPTIONS
# ============================================================

option1, option2 = st.columns(
    [1, 1]
)

with option1:

    show_volume = st.checkbox(
        "Volume",
        value=True,
    )

with option2:

    show_ema = st.checkbox(
        "EMA 20 / EMA 50",
        value=True,
    )


# ============================================================
# FETCH DATA
# ============================================================

try:

    from_date, to_date = get_dates(
        candle_interval,
        chart_range,
    )

    # -----------------------------
    # Upstox V3 interval
    # -----------------------------

    if candle_interval == "15m":

        unit = "minutes"
        interval = "15"

    elif candle_interval == "30m":

        unit = "minutes"
        interval = "30"

    elif candle_interval == "60m":

        unit = "hours"
        interval = "1"

    else:

        unit = "days"
        interval = "1"


    with st.spinner(
        f"Loading {display_symbol}..."
    ):

        df = fetch_historical_candles(
            instrument_key,
            unit,
            interval,
            from_date.isoformat(),
            to_date.isoformat(),
            access_token,
        )


except Exception as e:

    st.error(
        "Unable to load Upstox historical data."
    )

    st.exception(e)

    st.stop()


if df.empty:

    st.warning(
        "No candles were returned for this "
        "timeframe/range."
    )

    st.stop()


# ============================================================
# ADD INDICATORS
# ============================================================

df = add_indicators(
    df
)


# ============================================================
# CURRENT DATA
# ============================================================

last = df.iloc[-1]

previous = (
    df.iloc[-2]
    if len(df) >= 2
    else df.iloc[-1]
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

change = (
    close_price
    - previous_close
)

change_pct = (
    change
    / previous_close
    * 100
    if previous_close
    else 0
)


# ============================================================
# SIGNAL
# ============================================================

signal = generate_signal(
    df
)


# ============================================================
# CHART HEADER
# ============================================================

change_class = (
    "positive"
    if change >= 0
    else "negative"
)

sign = (
    "+"
    if change >= 0
    else ""
)

st.markdown(
    f"""
    <div class="chart-header">

        <span class="chart-symbol">
            {display_symbol}
        </span>

        <span class="chart-timeframe">
            · {candle_interval}
        </span>

        <span class="ohlc">
            O {open_price:,.2f}
        </span>

        <span class="ohlc">
            H {high_price:,.2f}
        </span>

        <span class="ohlc">
            L {low_price:,.2f}
        </span>

        <span class="ohlc">
            C {close_price:,.2f}
        </span>

        <span class="{change_class}">
            {sign}{change:,.2f}
            ({sign}{change_pct:.2f}%)
        </span>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# VOLUME TEXT
# ============================================================

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


# ============================================================
# MAIN CHART
# ============================================================

fig = create_chart(
    df,
    candle_interval,
    show_volume,
    show_ema,
)


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


# ============================================================
# RSI / MACD
# ============================================================

rsi_col, macd_col = st.columns(
    2
)


# ---------------- RSI ----------------

with rsi_col:

    st.markdown(
        "### RSI (14)"
    )

    rsi_fig = go.Figure()

    rsi_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["RSI"],
            line=dict(
                color="#2563eb",
                width=1.4,
            ),
            name="RSI",
        )
    )

    rsi_fig.add_hline(
        y=70,
        line_dash="dot",
        line_color="#ef4444",
    )

    rsi_fig.add_hline(
        y=30,
        line_dash="dot",
        line_color="#00a878",
    )

    rsi_fig.update_layout(
        height=210,
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
        rsi_fig,
        use_container_width=True,
    )


# ---------------- MACD ----------------

with macd_col:

    st.markdown(
        "### MACD"
    )

    macd_fig = go.Figure()

    hist_colors = np.where(
        df["MACDHist"] >= 0,
        "#00a878",
        "#ef6b5b",
    )

    macd_fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["MACDHist"],
            marker_color=hist_colors,
            opacity=0.65,
            name="Histogram",
        )
    )

    macd_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD"],
            line=dict(
                color="#2563eb",
                width=1.3,
            ),
            name="MACD",
        )
    )

    macd_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACDSignal"],
            line=dict(
                color="#f59e0b",
                width=1.3,
            ),
            name="Signal",
        )
    )

    macd_fig.update_layout(
        height=210,
        margin=dict(
            l=5,
            r=5,
            t=5,
            b=5,
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        yaxis=dict(
            showgrid=True,
            gridcolor="#eeeeee",
        ),
        xaxis=dict(
            showgrid=False,
        ),
        legend=dict(
            orientation="h",
            y=1.08,
        ),
    )

    st.plotly_chart(
        macd_fig,
        use_container_width=True,
    )


# ============================================================
# MARKET OVERVIEW
# ============================================================

st.markdown(
    "### 📊 Market Overview"
)

m1, m2, m3, m4, m5 = st.columns(
    5
)

m1.metric(
    "LTP",
    f"₹{close_price:,.2f}",
    f"{change:+.2f} ({change_pct:+.2f}%)",
)

m2.metric(
    "RSI",
    f"{last['RSI']:.1f}",
)

m3.metric(
    "TREND",
    signal["trend"],
)

m4.metric(
    "MACD",
    f"{last['MACDHist']:.2f}",
)

m5.metric(
    "SIGNAL",
    signal["action"],
)


# ============================================================
# TRADE SETUP
# ============================================================

st.markdown(
    "### 🎯 Trade Setup"
)

t1, t2, t3, t4 = st.columns(
    4
)

t1.metric(
    "ENTRY",
    (
        f"₹{signal['entry']:,.2f}"
        if signal["entry"]
        else "--"
    ),
)

t2.metric(
    "TARGET",
    (
        f"₹{signal['target']:,.2f}"
        if signal["target"]
        else "--"
    ),
)

t3.metric(
    "STOP LOSS",
    (
        f"₹{signal['sl']:,.2f}"
        if signal["sl"]
        else "--"
    ),
)

if (
    signal["entry"]
    and signal["target"]
    and signal["sl"]
):

    if signal["action"] == "BUY":

        risk = (
            signal["entry"]
            - signal["sl"]
        )

        reward = (
            signal["target"]
            - signal["entry"]
        )

    else:

        risk = (
            signal["sl"]
            - signal["entry"]
        )

        reward = (
            signal["entry"]
            - signal["target"]
        )

    rr = (
        reward / risk
        if risk > 0
        else 0
    )

    t4.metric(
        "RISK : REWARD",
        f"1 : {rr:.2f}",
    )

else:

    t4.metric(
        "RISK : REWARD",
        "--",
    )


# ============================================================
# SIGNAL MESSAGE
# ============================================================

if signal["action"] == "BUY":

    st.markdown(
        f"""
        <div class="signal buy">
            🟢 BUY — {signal['reason']}
        </div>
        """,
        unsafe_allow_html=True,
    )

elif signal["action"] == "SELL":

    st.markdown(
        f"""
        <div class="signal sell">
            🔴 SELL — {signal['reason']}
        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        f"""
        <div class="signal hold">
            🟠 HOLD — {signal['reason']}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DATA INFORMATION
# ============================================================

st.caption(
    f"Upstox instrument: {instrument_key} • "
    f"Data range: {from_date} → {to_date} • "
    f"Candle: {candle_interval}"
)

st.caption(
    "Market data and technical indicators are for "
    "analysis/education. Verify live prices and "
    "orders through your broker before execution."
)
