import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
from io import StringIO
import base64
import requests

# =========================================================
# AI QUOTEX PRO V2.4 | Theme switch and developer credit
# =========================================================

st.set_page_config(
    page_title="AI QUOTEX PRO",
    page_icon="📈",
    layout="wide"
)

# Appearance control: user can switch themes without editing config files.
if "app_theme" not in st.session_state:
    st.session_state["app_theme"] = "Dark Mode"

with st.sidebar:
    st.markdown("## ⚙️ App Settings")
    selected_theme = st.radio(
        "Appearance",
        ["Dark Mode", "White Mode"],
        index=0 if st.session_state["app_theme"] == "Dark Mode" else 1,
        key="app_theme_picker",
        horizontal=True
    )
    st.session_state["app_theme"] = selected_theme
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;padding:12px 4px;color:#8b98aa;"
        "font-size:13px;font-weight:600;letter-spacing:.3px;'>"
        "Developed by <span style='color:#4f9cf9;'>Dip</span></div>",
        unsafe_allow_html=True
    )

if st.session_state["app_theme"] == "Dark Mode":
    page_bg, text_color, panel_bg, border_color, muted = (
        "#0b1220", "#f4f7fb", "#172338", "#263750", "#9fb0c7"
    )
    buy_bg, sell_bg, wait_bg = "#123b2b", "#451e2a", "#3b3218"
else:
    page_bg, text_color, panel_bg, border_color, muted = (
        "#f5f7fb", "#182230", "#ffffff", "#dce3ec", "#64748b"
    )
    buy_bg, sell_bg, wait_bg = "#e1f5e9", "#fde8eb", "#fff4d6"

st.markdown(f"""
<style>
.stApp {{background:{page_bg};color:{text_color}}}
[data-testid="stHeader"] {{background:transparent}}
[data-testid="stSidebar"] {{background:{panel_bg};}}
[data-testid="stMetric"] {{
    background:{panel_bg};
    padding:14px;
    border-radius:12px;
    border:1px solid {border_color};
}}
div[data-testid="stDataFrame"] {{border:1px solid {border_color}}}
.signal-buy {{background:{buy_bg};padding:14px;border-radius:12px}}
.signal-sell {{background:{sell_bg};padding:14px;border-radius:12px}}
.signal-wait {{background:{wait_bg};padding:14px;border-radius:12px}}
.small-note {{color:{muted};font-size:0.85rem}}
</style>
""", unsafe_allow_html=True)

st.title("📈 AI QUOTEX PRO")
st.caption("Forex • OTC Screenshot Review • Binance Spot & Futures | Version 2.4")

st.warning(
    "Research prototype only. Signals are not guaranteed, scores are not "
    "win probabilities, and no automatic trading is performed."
)

# -------------------------
# BINANCE PUBLIC MARKET ANALYZER
# -------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def binance_symbol_catalog(market):
    """Fetch every currently tradable USDT-quoted symbol from Binance public exchange info."""
    if market == "Futures":
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    else:
        url = "https://api.binance.com/api/v3/exchangeInfo"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    payload = response.json()
    symbols = []
    for item in payload.get("symbols", []):
        if item.get("status") != "TRADING":
            continue
        if item.get("quoteAsset") != "USDT":
            continue
        if market == "Futures" and item.get("contractType") != "PERPETUAL":
            continue
        pair = item.get("symbol", "")
        base = item.get("baseAsset", "")
        if pair and base:
            symbols.append({"pair": pair, "base": base})
    return sorted(symbols, key=lambda x: x["pair"])


@st.cache_data(ttl=30, show_spinner=False)
def binance_klines(symbol, market, interval, limit=300):
    base = "https://fapi.binance.com" if market == "Futures" else "https://api.binance.com"
    path = "/fapi/v1/klines" if market == "Futures" else "/api/v3/klines"
    r = requests.get(base + path, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=15)
    r.raise_for_status()
    raw = r.json()
    if not isinstance(raw, list) or len(raw) < 80:
        raise ValueError("Binance returned insufficient candle history.")
    cols = ["open_time","Open","High","Low","Close","Volume","close_time","quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("Time")[["Open","High","Low","Close","Volume"]].dropna()
    # Exclude the still-forming candle.
    return df.iloc[:-1]


def binance_indicators(df):
    d = df.copy()
    for n in [9, 21, 50]:
        d[f"EMA{n}"] = d.Close.ewm(span=n, adjust=False).mean()
    delta = d.Close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - 100/(1+rs)
    d["RSI"] = d.RSI.mask((loss == 0) & (gain > 0), 100).mask((gain == 0) & (loss > 0), 0)
    e12 = d.Close.ewm(span=12, adjust=False).mean()
    e26 = d.Close.ewm(span=26, adjust=False).mean()
    d["MACD"] = e12-e26
    d["MACD_SIGNAL"] = d.MACD.ewm(span=9, adjust=False).mean()
    d["BB_MID"] = d.Close.rolling(20).mean()
    d["BB_STD"] = d.Close.rolling(20).std()
    d["BB_UPPER"] = d.BB_MID + 2*d.BB_STD
    d["BB_LOWER"] = d.BB_MID - 2*d.BB_STD
    prev = d.Close.shift(1)
    tr = pd.concat([d.High-d.Low, (d.High-prev).abs(), (d.Low-prev).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    d["VOL_AVG"] = d.Volume.rolling(20).mean()
    return d


def binance_trend(row):
    if row.EMA21 > row.EMA50 and row.Close > row.EMA21:
        return "BULLISH"
    if row.EMA21 < row.EMA50 and row.Close < row.EMA21:
        return "BEARISH"
    return "SIDEWAYS"


def binance_signal(frames, market):
    # Confirmation from 15m, 1h and 4h; selected chart can be any timeframe.
    rows = [frames[x].iloc[-1] for x in ["15m", "1h", "4h"]]
    score = 0
    notes = []
    for tf, row in zip(["15m", "1h", "4h"], rows):
        if row.EMA21 > row.EMA50:
            score += 2; notes.append(f"{tf}: EMA trend bullish")
        elif row.EMA21 < row.EMA50:
            score -= 2; notes.append(f"{tf}: EMA trend bearish")
        else:
            notes.append(f"{tf}: sideways EMA structure")
    last = frames["selected"].iloc[-1]
    if last.MACD > last.MACD_SIGNAL:
        score += 1; notes.append("Selected timeframe MACD bullish")
    else:
        score -= 1; notes.append("Selected timeframe MACD bearish")
    if 50 < last.RSI < 68:
        score += 1; notes.append("RSI supports bullish momentum")
    elif 32 < last.RSI < 50:
        score -= 1; notes.append("RSI supports bearish momentum")
    elif last.RSI >= 72 or last.RSI <= 28:
        notes.append("RSI extended; avoid chasing")
        return "WAIT", score, notes
    if last.Volume > last.VOL_AVG:
        notes.append("Volume above 20-candle average")
    else:
        notes.append("Volume not above average")
    if score >= 5:
        return ("BUY" if market == "Spot" else "LONG"), score, notes
    if score <= -5:
        return ("SELL" if market == "Spot" else "SHORT"), score, notes
    return "WAIT", score, notes


def render_binance_page():
    st.header("₿ Binance AI Analyzer")
    st.caption("Public Binance market data • No API key • No order execution")
    st.info("Signals are rule-based research, not guaranteed predictions. Spot SELL is not a short position. Futures can lose money rapidly; no leverage or auto-trading is configured.")
    c1,c2 = st.columns(2)
    with c1:
        market = st.selectbox("Market", ["Spot", "Futures"], key="bn_market")
    with c2:
        tf = st.selectbox("Chart timeframe", ["5m","15m","1h","4h","1d"], index=1, key="bn_tf")

    try:
        catalog = binance_symbol_catalog(market)
        st.caption(f"Binance {market}: {len(catalog):,} currently tradable USDT pairs loaded. List refreshes automatically about once per hour.")
        search_text = st.text_input(
            "🔎 Search coin or pair (e.g. BTC, ETH, SOLUSDT)",
            key="bn_symbol_search",
            placeholder="Type coin name / ticker / pair..."
        ).strip().upper()
        matches = [
            item for item in catalog
            if not search_text
            or search_text in item["base"].upper()
            or search_text in item["pair"].upper()
        ]
        if not matches:
            st.warning("No matching active USDT pair found. Try another ticker, or check the spelling.")
            return
        options = [item["pair"] for item in matches]
        preferred = st.session_state.get("bn_symbol", "BTCUSDT")
        default_index = options.index(preferred) if preferred in options else 0
        symbol = st.selectbox(
            f"Select trading pair ({len(matches):,} matches)",
            options,
            index=default_index,
            key="bn_symbol"
        )
    except Exception as e:
        st.error(f"Could not load Binance's live symbol list: {e}. Check internet access and press Rerun.")
        return

    if st.button("🔄 ANALYZE BINANCE", type="primary", use_container_width=True):
        try:
            with st.spinner("Fetching Binance public candles and checking multiple timeframes..."):
                frames = {x: binance_indicators(binance_klines(symbol, market, x)) for x in ["15m","1h","4h"]}
                if tf in frames:
                    frames["selected"] = frames[tf]
                else:
                    frames["selected"] = binance_indicators(binance_klines(symbol, market, tf))
                sig, score, notes = binance_signal(frames, market)
                st.session_state["bn_result"] = {"frames":frames,"signal":sig,"score":score,"notes":notes,"symbol":symbol,"market":market,"tf":tf}
                hist = st.session_state.get("bn_history", [])
                hist.append({"Time UTC":datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),"Market":market,"Pair":symbol,"Timeframe":tf,"Signal":sig,"Confluence score":score})
                st.session_state["bn_history"] = hist[-300:]
        except Exception as e:
            st.error(f"Binance data could not be loaded: {e}. Check internet access and try again.")
    result = st.session_state.get("bn_result")
    if not result:
        st.warning("Choose Spot/Futures, pair and timeframe, then press ANALYZE BINANCE.")
        return
    if result["symbol"] != symbol or result["market"] != market or result["tf"] != tf:
        st.info("Settings changed. Press ANALYZE BINANCE to update the result.")
        return
    d = result["frames"]["selected"]
    last = d.iloc[-1]
    st.subheader("Signal overview")
    a,b,c,dcol = st.columns(4)
    a.metric("Signal", result["signal"])
    b.metric("Rule confluence", f"{result['score']:+d} / 9")
    c.metric("Last price", f"{last.Close:,.6g}")
    dcol.metric("RSI (14)", f"{last.RSI:.1f}")
    st.caption("Confluence score is indicator alignment, NOT a probability of winning.")
    st.write("**Why this signal:** " + " • ".join(result["notes"]))
    trend_cols = st.columns(3)
    for col, period in zip(trend_cols,["15m","1h","4h"]):
        col.metric(f"{period} trend", binance_trend(result["frames"][period].iloc[-1]))
    entry = float(last.Close)
    atr = float(last.ATR) if pd.notna(last.ATR) else 0
    direction = 1 if result["signal"] in ["BUY","LONG"] else -1 if result["signal"] in ["SELL","SHORT"] else 0
    if direction and atr > 0:
        sl = entry - direction*1.5*atr
        tp1 = entry + direction*1.5*atr
        tp2 = entry + direction*2.5*atr
        x,y,z = st.columns(3)
        x.metric("Reference entry", f"{entry:,.6g}")
        y.metric("ATR stop reference", f"{sl:,.6g}")
        z.metric("TP2 reference (2.5 ATR)", f"{tp2:,.6g}")
        st.caption(f"TP1 reference (1.5 ATR): {tp1:,.6g}. Illustrative levels only; fees, slippage and volatility can change outcomes.")
    else:
        st.info("No actionable setup. WAIT is a valid signal; do not force a trade.")
    if market == "Futures":
        try:
            fr = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",params={"symbol":symbol},timeout=12)
            fr.raise_for_status(); premium = fr.json()
            oi = requests.get("https://fapi.binance.com/fapi/v1/openInterest",params={"symbol":symbol},timeout=12)
            oi.raise_for_status(); oi_data = oi.json()
            f1,f2 = st.columns(2)
            f1.metric("Funding rate (latest)", f"{float(premium.get('lastFundingRate',0))*100:.4f}%")
            f2.metric("Open interest (contracts)", f"{float(oi_data.get('openInterest',0)):,.2f}")
        except Exception as e:
            st.caption(f"Funding/Open Interest unavailable right now: {e}")
        st.warning("Futures risk: use isolated margin if trading manually, keep leverage conservative, and define risk before entry. This app does not set margin mode or place orders.")
    chart = d.tail(160)
    fig = go.Figure(data=[go.Candlestick(x=chart.index,open=chart.Open,high=chart.High,low=chart.Low,close=chart.Close,name=symbol)])
    fig.add_trace(go.Scatter(x=chart.index,y=chart.EMA21,name="EMA 21"))
    fig.add_trace(go.Scatter(x=chart.index,y=chart.EMA50,name="EMA 50"))
    fig.update_layout(template="plotly_dark",height=500,xaxis_rangeslider_visible=False)
    st.plotly_chart(fig,use_container_width=True)
    st.subheader("Signal history")
    history = st.session_state.get("bn_history", [])
    if history:
        hist_df = pd.DataFrame(history)
        st.dataframe(hist_df.tail(30),use_container_width=True,hide_index=True)
        st.download_button("Download Binance signal history CSV",hist_df.to_csv(index=False).encode("utf-8"),"binance_signal_history.csv","text/csv")

# Main navigation is placed in the page body so it is easy to find on desktop and mobile.
st.markdown("### Select Analyzer")
page = st.radio(
    "Choose a section",
    ["Forex / OTC", "Binance Spot & Futures"],
    horizontal=True,
    key="main_dashboard_navigation",
    label_visibility="collapsed"
)
if page == "Binance Spot & Futures":
    render_binance_page()
    st.stop()

PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X",
    "USD/CHF": "CHF=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X"
}

# -------------------------
# DATA
# -------------------------

@st.cache_data(ttl=60, show_spinner=False)
def get_market_data(symbol):
    df = yf.Ticker(symbol).history(
        period="5d",
        interval="1m",
        auto_adjust=False
    )
    if df.empty:
        raise ValueError("No market data received from Yahoo Finance.")

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing market columns: " + ", ".join(missing))

    df = df[required].copy().dropna(subset=["Open", "High", "Low", "Close"])
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Ignore the newest minute because it may still be forming.
    if len(df) > 2:
        df = df.iloc[:-1]

    if len(df) < 100:
        raise ValueError("Not enough 1-minute candle history.")

    return df


def resample_data(df, minutes):
    rule = f"{minutes}min"
    out = df.resample(rule, label="right", closed="right").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna(subset=["Open", "High", "Low", "Close"])

    # Keep only fully completed higher-timeframe candles.
    now = pd.Timestamp.now(tz=df.index.tz) if df.index.tz is not None else pd.Timestamp.now()
    cutoff = now.floor(rule)
    out = out.loc[out.index <= cutoff]
    return out


# -------------------------
# INDICATORS
# -------------------------

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
    return rsi


def calculate_indicators(df):
    df = df.copy()

    for span in [9, 20, 50]:
        df[f"EMA{span}"] = df["Close"].ewm(span=span, adjust=False).mean()

    df["RSI"] = calculate_rsi(df["Close"], 14)

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    # Bollinger Bands
    df["BB_MID"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_UPPER"] = df["BB_MID"] + 2 * bb_std
    df["BB_LOWER"] = df["BB_MID"] - 2 * bb_std

    # Stochastic RSI
    rsi_min = df["RSI"].rolling(14).min()
    rsi_max = df["RSI"].rolling(14).max()
    denom = (rsi_max - rsi_min).replace(0, np.nan)
    df["STOCH_RSI_K"] = 100 * (df["RSI"] - rsi_min) / denom
    df["STOCH_RSI_D"] = df["STOCH_RSI_K"].rolling(3).mean()

    # ATR
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    # Supertrend
    hl2 = (df["High"] + df["Low"]) / 2
    basic_upper = hl2 + 3 * df["ATR"]
    basic_lower = hl2 - 3 * df["ATR"]
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    st_line = pd.Series(index=df.index, dtype=float)
    st_dir = pd.Series(index=df.index, dtype=float)

    for i in range(len(df)):
        if pd.isna(df["ATR"].iloc[i]):
            continue
        if i == 0 or pd.isna(st_dir.iloc[i - 1]):
            st_dir.iloc[i] = 1
            st_line.iloc[i] = basic_lower.iloc[i]
            continue

        prev_fu = final_upper.iloc[i - 1]
        prev_fl = final_lower.iloc[i - 1]
        prev_c = df["Close"].iloc[i - 1]

        if basic_upper.iloc[i] < prev_fu or prev_c > prev_fu:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = prev_fu

        if basic_lower.iloc[i] > prev_fl or prev_c < prev_fl:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = prev_fl

        if st_dir.iloc[i - 1] == -1 and df["Close"].iloc[i] > final_upper.iloc[i]:
            st_dir.iloc[i] = 1
        elif st_dir.iloc[i - 1] == 1 and df["Close"].iloc[i] < final_lower.iloc[i]:
            st_dir.iloc[i] = -1
        else:
            st_dir.iloc[i] = st_dir.iloc[i - 1]

        st_line.iloc[i] = final_lower.iloc[i] if st_dir.iloc[i] == 1 else final_upper.iloc[i]

    df["SUPERTREND"] = st_line
    df["SUPERTREND_DIR"] = st_dir
    return df


# -------------------------
# MARKET STRUCTURE / PATTERNS
# -------------------------

def get_trend(df):
    if len(df) < 60:
        return "INSUFFICIENT DATA"
    last = df.iloc[-1]
    if last["EMA20"] > last["EMA50"] and last["Close"] > last["EMA20"]:
        return "BULLISH"
    if last["EMA20"] < last["EMA50"] and last["Close"] < last["EMA20"]:
        return "BEARISH"
    return "SIDEWAYS"


def detect_pattern(df):
    if len(df) < 3:
        return "Not enough candles"

    prev = df.iloc[-2]
    cur = df.iloc[-1]
    body = abs(cur["Close"] - cur["Open"])
    candle_range = max(cur["High"] - cur["Low"], 1e-12)
    upper_wick = cur["High"] - max(cur["Open"], cur["Close"])
    lower_wick = min(cur["Open"], cur["Close"]) - cur["Low"]

    if body / candle_range < 0.1:
        return "Doji"
    if (prev["Close"] < prev["Open"] and cur["Close"] > cur["Open"]
            and cur["Open"] <= prev["Close"] and cur["Close"] >= prev["Open"]):
        return "Bullish Engulfing"
    if (prev["Close"] > prev["Open"] and cur["Close"] < cur["Open"]
            and cur["Open"] >= prev["Close"] and cur["Close"] <= prev["Open"]):
        return "Bearish Engulfing"
    if lower_wick > body * 2 and upper_wick < body:
        return "Hammer-like candle"
    if upper_wick > body * 2 and lower_wick < body:
        return "Shooting-star-like candle"
    return "No clear pattern"


def support_resistance(df, window=100):
    sample = df.tail(window)
    return float(sample["Low"].min()), float(sample["High"].max())


# -------------------------
# SIGNAL ENGINE
# -------------------------

def generate_signal(df15, df5, df1):
    if min(len(df15), len(df5), len(df1)) < 60:
        return "WAIT", 0, ["Insufficient completed candle history"]

    a, b, c = df15.iloc[-1], df5.iloc[-1], df1.iloc[-1]
    score = 0
    reasons = []

    if a["EMA20"] > a["EMA50"]:
        score += 2
        reasons.append("15m trend bullish")
    elif a["EMA20"] < a["EMA50"]:
        score -= 2
        reasons.append("15m trend bearish")

    if b["EMA20"] > b["EMA50"]:
        score += 2
        reasons.append("5m trend bullish")
    elif b["EMA20"] < b["EMA50"]:
        score -= 2
        reasons.append("5m trend bearish")

    if b["MACD"] > b["MACD_SIGNAL"]:
        score += 1
        reasons.append("5m MACD positive")
    elif b["MACD"] < b["MACD_SIGNAL"]:
        score -= 1
        reasons.append("5m MACD negative")

    if c["EMA9"] > c["EMA20"]:
        score += 1
        reasons.append("1m EMA momentum bullish")
    elif c["EMA9"] < c["EMA20"]:
        score -= 1
        reasons.append("1m EMA momentum bearish")

    if pd.notna(c["RSI"]):
        if 50 < c["RSI"] < 68:
            score += 1
            reasons.append("1m RSI supports bullish momentum")
        elif 32 < c["RSI"] < 50:
            score -= 1
            reasons.append("1m RSI supports bearish momentum")
        elif c["RSI"] >= 75 or c["RSI"] <= 25:
            return "WAIT", score, ["RSI is strongly extended; avoid chasing"]

    if pd.notna(c["SUPERTREND_DIR"]):
        if c["SUPERTREND_DIR"] == 1:
            score += 1
            reasons.append("1m Supertrend bullish")
        else:
            score -= 1
            reasons.append("1m Supertrend bearish")

    if score >= 5:
        signal = "BUY"
    elif score <= -5:
        signal = "SELL"
    else:
        signal = "WAIT"

    return signal, score, reasons or ["No clear directional confirmation"]


def analyze_pair(symbol):
    raw = get_market_data(symbol)
    df1 = calculate_indicators(raw)
    df5 = calculate_indicators(resample_data(raw, 5))
    df15 = calculate_indicators(resample_data(raw, 15))
    signal, score, reasons = generate_signal(df15, df5, df1)
    return {
        "signal": signal, "score": score, "reasons": reasons,
        "df1": df1, "df5": df5, "df15": df15
    }


# -------------------------
# SIDEBAR
# -------------------------

st.sidebar.header("⚙️ Market Settings")
market_type = st.sidebar.selectbox("Market Type", ["Forex", "OTC"])
pair = st.sidebar.selectbox("Select Pair", list(PAIRS.keys()))
st.sidebar.caption("Data source: Yahoo Finance (Forex reference feed)")
st.sidebar.info(
    "OTC analysis is disabled until a verified, authorized OTC candle "
    "data provider is connected. Forex quotes are not OTC quotes."
)

st.sidebar.divider()
st.sidebar.subheader("Analysis Options")
auto_refresh = st.sidebar.checkbox("Enable auto refresh", value=False)
refresh_seconds = st.sidebar.select_slider("Refresh interval (seconds)", options=[15, 30, 60, 120], value=60)
if auto_refresh:
    st_autorefresh(interval=refresh_seconds * 1000, key="market_auto_refresh")
show_bollinger = st.sidebar.checkbox("Show Bollinger Bands", value=True)
show_ema = st.sidebar.checkbox("Show EMA 20", value=True)
show_supertrend = st.sidebar.checkbox("Show Supertrend", value=True)
chart_candles = st.sidebar.slider("Chart candles", 50, 250, 100, 10)

# -------------------------
# MAIN ANALYSIS
# -------------------------

st.subheader(f"Market: {pair}")

left, right = st.columns([1, 3])
with left:
    analyze_clicked = st.button("🔄 ANALYZE MARKET", type="primary", use_container_width=True)
with right:
    st.caption("Press Analyze to fetch the latest available completed candles.")


if market_type == "OTC":
    st.subheader("🖼️ Quotex OTC Screenshot Analysis")
    st.warning(
        "Upload a clear screenshot of the Quotex OTC candlestick chart. "
        "This reviews only the visible image; it does not connect to Quotex "
        "or receive live OTC prices."
    )
    otc_image = st.file_uploader(
        "Upload Quotex OTC chart screenshot",
        type=["png", "jpg", "jpeg"],
        key="otc_chart_upload"
    )
    st.caption(
        "For best results, show the asset name, timeframe, and at least 30–50 "
        "visible candles. Hide account balances and personal information."
    )
    if otc_image is not None:
        st.image(otc_image, caption="Uploaded OTC chart", use_container_width=True)

    api_key = st.text_input(
        "OpenAI API key (required for AI image analysis)",
        type="password",
        help="The key is used for this analysis and is not saved by this app."
    )
    if st.button("🔍 ANALYZE OTC SCREENSHOT", type="primary", use_container_width=True):
        if otc_image is None:
            st.error("Please upload a chart screenshot first.")
        elif not api_key.strip():
            st.error("Please enter your OpenAI API key to enable image analysis.")
        else:
            try:
                from openai import OpenAI
                encoded = base64.b64encode(otc_image.getvalue()).decode("utf-8")
                mime = otc_image.type or "image/png"
                client = OpenAI(api_key=api_key.strip())
                prompt = (
                    "Carefully review this Quotex OTC chart screenshot as a cautious "
                    "technical research assistant. Do not claim access to live data or "
                    "invent unreadable prices/indicators. Identify asset/timeframe only "
                    "if legible. Assess visible candle structure and momentum. Return "
                    "Signal: BUY, SELL, or WAIT; qualitative confidence LOW/MEDIUM/HIGH "
                    "(not a win probability); visible evidence; uncertainty; and what "
                    "would invalidate the view. If unclear or insufficient candles, "
                    "choose WAIT. Do not promise profit or prescribe a guaranteed entry "
                    "or expiry. This is not financial advice."
                )
                with st.spinner("AI is reviewing the visible chart..."):
                    response = client.responses.create(
                        model="gpt-4.1-mini",
                        input=[{"role": "user", "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image",
                             "image_url": f"data:{mime};base64,{encoded}"}
                        ]}]
                    )
                st.session_state["otc_image_analysis"] = response.output_text
                st.success("Screenshot review completed.")
            except ImportError:
                st.error("OpenAI package is missing. Run: python -m pip install openai")
            except Exception as exc:
                st.error(f"OTC image analysis failed: {exc}")

    if st.session_state.get("otc_image_analysis"):
        st.subheader("🎯 OTC Screenshot Signal Review")
        st.markdown(st.session_state["otc_image_analysis"])
        st.caption(
            "Visual screenshot review only, not a live Quotex feed. OTC prices can "
            "change quickly; results may be wrong or stale. Verify independently."
        )
    st.stop()

if analyze_clicked or auto_refresh:
    if analyze_clicked:
        st.cache_data.clear()
    try:
        with st.spinner("Fetching candles and calculating indicators..."):
            result = analyze_pair(PAIRS[pair])
            previous = st.session_state.get("last_signal_by_pair", {}).get(pair)
            st.session_state["result"] = result
            st.session_state["pair"] = pair
            st.session_state["analyzed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if previous is not None and previous != result["signal"]:
                st.toast(f"{pair}: signal changed from {previous} to {result['signal']}", icon="🔔")
            last_signals = st.session_state.get("last_signal_by_pair", {})
            last_signals[pair] = result["signal"]
            st.session_state["last_signal_by_pair"] = last_signals

            history = st.session_state.get("signal_history", [])
            latest_time = result["df1"].index[-1].strftime("%Y-%m-%d %H:%M:%S")
            if not history or not (history[-1]["Pair"] == pair and history[-1]["Candle Time"] == latest_time):
                history.append({
                    "Analyzed At": st.session_state["analyzed_at"],
                    "Pair": pair,
                    "Candle Time": latest_time,
                    "Signal": result["signal"],
                    "Score": result["score"],
                    "Pattern": detect_pattern(result["df1"])
                })
            st.session_state["signal_history"] = history[-500:]
    except Exception as exc:
        st.error(f"Market data error: {exc}")

if "result" not in st.session_state:
    st.info("Select a Forex pair and press ANALYZE MARKET to begin.")
    st.stop()

if st.session_state.get("pair") != pair:
    st.info("You selected a different pair. Press ANALYZE MARKET to refresh its analysis.")
    st.stop()

result = st.session_state["result"]
signal = result["signal"]
df1, df5, df15 = result["df1"], result["df5"], result["df15"]
latest = df1.iloc[-1]
support, resistance = support_resistance(df1)

# -------------------------
# SIGNAL SUMMARY
# -------------------------

st.divider()
st.subheader("🎯 Trading Signal")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Signal", signal)
m2.metric("Analysis Score", f"{result['score']:+d} / 8")
m3.metric("Latest Close", f"{latest['Close']:.5f}")
m4.metric("Latest Candle", df1.index[-1].strftime("%H:%M"))

if signal == "BUY":
    st.markdown('<div class="signal-buy"><b>BUY BIAS</b> — research signal only</div>', unsafe_allow_html=True)
elif signal == "SELL":
    st.markdown('<div class="signal-sell"><b>SELL BIAS</b> — research signal only</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="signal-wait"><b>WAIT</b> — no sufficiently strong setup</div>', unsafe_allow_html=True)

st.write("**Signal explanation:** " + " | ".join(result["reasons"]))
st.caption("The score measures rule alignment only. It is not a probability or expected win rate.")

# -------------------------
# MULTI-TIMEFRAME
# -------------------------

st.subheader("🧭 Multi-Timeframe Trends")
t1, t2, t3 = st.columns(3)
t1.metric("15 Minute", get_trend(df15))
t2.metric("5 Minute", get_trend(df5))
t3.metric("1 Minute", get_trend(df1))

# -------------------------
# INDICATORS
# -------------------------

st.subheader("📊 Technical Indicators")
i1, i2, i3, i4, i5 = st.columns(5)
i1.metric("RSI (14)", f"{latest['RSI']:.2f}" if pd.notna(latest["RSI"]) else "N/A")
i2.metric("EMA 9", f"{latest['EMA9']:.5f}")
i3.metric("EMA 20", f"{latest['EMA20']:.5f}")
i4.metric("MACD", f"{latest['MACD']:.6f}")
i5.metric("Stoch RSI K", f"{latest['STOCH_RSI_K']:.1f}" if pd.notna(latest["STOCH_RSI_K"]) else "N/A")

j1, j2, j3 = st.columns(3)
j1.metric("Supertrend", "Bullish" if latest["SUPERTREND_DIR"] == 1 else "Bearish" if latest["SUPERTREND_DIR"] == -1 else "N/A")
j2.metric("Support (100 candles)", f"{support:.5f}")
j3.metric("Resistance (100 candles)", f"{resistance:.5f}")

# -------------------------
# CHART
# -------------------------

st.subheader("🕯️ 1 Minute Candlestick Chart")
chart_data = df1.tail(chart_candles)
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=chart_data.index,
    open=chart_data["Open"], high=chart_data["High"],
    low=chart_data["Low"], close=chart_data["Close"],
    name=pair
))

if show_ema:
    fig.add_trace(go.Scatter(
        x=chart_data.index, y=chart_data["EMA20"],
        name="EMA 20", line=dict(color="orange", width=1.5)
    ))

if show_bollinger:
    fig.add_trace(go.Scatter(
        x=chart_data.index, y=chart_data["BB_UPPER"],
        name="BB Upper", line=dict(color="rgba(100,160,255,0.6)", width=1)
    ))
    fig.add_trace(go.Scatter(
        x=chart_data.index, y=chart_data["BB_LOWER"],
        name="BB Lower", line=dict(color="rgba(100,160,255,0.6)", width=1),
        fill="tonexty", fillcolor="rgba(100,160,255,0.06)"
    ))

if show_supertrend:
    fig.add_trace(go.Scatter(
        x=chart_data.index, y=chart_data["SUPERTREND"],
        name="Supertrend", line=dict(color="lime", width=1.4)
    ))

fig.update_layout(
    template="plotly_dark", height=560,
    xaxis_rangeslider_visible=False,
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", y=1.02, x=0)
)
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# PATTERN + MARKET LEVELS
# -------------------------

st.subheader("🔍 Price Action & Market Structure")
p1, p2, p3 = st.columns(3)
p1.metric("Latest Candle Pattern", detect_pattern(df1))
p2.metric("Support", f"{support:.5f}")
p3.metric("Resistance", f"{resistance:.5f}")
st.caption("Support/resistance are simple recent-window extremes, not guaranteed reversal levels.")

# -------------------------
# PAIR SCANNER
# -------------------------

st.divider()
st.subheader("🌐 Multi-Pair Scanner")
st.write("Scan the supported Forex pairs using the same rule-based signal engine.")
if st.button("🔎 SCAN ALL PAIRS", use_container_width=True):
    scan_rows = []
    progress = st.progress(0)
    status = st.empty()
    for idx, (name, symbol) in enumerate(PAIRS.items()):
        status.write(f"Scanning {name}...")
        try:
            item = analyze_pair(symbol)
            scan_rows.append({
                "Pair": name,
                "Signal": item["signal"],
                "Score": item["score"],
                "15m Trend": get_trend(item["df15"]),
                "5m Trend": get_trend(item["df5"]),
                "1m Trend": get_trend(item["df1"])
            })
        except Exception as exc:
            scan_rows.append({
                "Pair": name, "Signal": "ERROR", "Score": "—",
                "15m Trend": str(exc)[:45], "5m Trend": "—", "1m Trend": "—"
            })
        progress.progress((idx + 1) / len(PAIRS))
    st.session_state["scan_results"] = pd.DataFrame(scan_rows)
    status.success("Pair scan completed.")

if "scan_results" in st.session_state:
    st.dataframe(st.session_state["scan_results"], use_container_width=True, hide_index=True)

# -------------------------
# SIGNAL HISTORY + EXPORT
# -------------------------

st.divider()
st.subheader("🗂️ Signal History")
history = st.session_state.get("signal_history", [])
if history:
    history_df = pd.DataFrame(history).iloc[::-1]
    st.dataframe(history_df, use_container_width=True, hide_index=True)
    csv_data = pd.DataFrame(history).to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ DOWNLOAD SIGNAL HISTORY (CSV)",
        data=csv_data,
        file_name="ai_quotex_pro_signal_history.csv",
        mime="text/csv"
    )
    if st.button("🧹 CLEAR SESSION HISTORY"):
        st.session_state["signal_history"] = []
        st.rerun()
else:
    st.info("No signal history yet. Analyze a market to record the first signal.")

# -------------------------
# ALERT STATUS / DISCLAIMERS
# -------------------------

st.divider()
st.subheader("🔔 Alert & Refresh")
st.info(
    "This version records signal changes when you run an analysis. "
    "Auto-refresh and on-screen signal-change notifications are available. "
    "Sound alerts are not included because browser autoplay permissions vary by device."
)
if st.button("♻️ REFRESH DATA NOW"):
    st.cache_data.clear()
    st.rerun()

st.subheader("⚠️ Research Disclaimer")
st.info(
    "AI QUOTEX PRO V2.1 is a rule-based market research prototype. "
    "It does not predict guaranteed outcomes, does not calculate a verified "
    "win rate, and does not place trades. Yahoo Finance data can be delayed, "
    "incomplete, unavailable, or different from a broker's OTC/live feed. "
    "Test any strategy on historical and demo data before considering risk."
)

st.caption(
    "AI QUOTEX PRO V2.1 | Manual analysis only | "
    + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
)


st.markdown(
    "<div style='text-align:center;padding:22px 8px 10px 8px;"
    "margin-top:28px;border-top:1px solid rgba(128,128,128,.25);"
    "font-size:13px;opacity:.8;'>Developed by <b>Dip</b> · AI QUOTEX PRO</div>",
    unsafe_allow_html=True
)
