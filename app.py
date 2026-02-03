import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
from streamlit_lightweight_charts import renderLightweightCharts

# ---------------------------------------------------------
# 1. 頁面設定：寬版模式，模擬看盤軟體
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Futu Style Analyzer")

st.subheader("台美股專業看盤 (仿富途牛牛 - 亮色版)")

# ---------------------------------------------------------
# 2. 側邊欄設定 (整合市場選擇功能)
# ---------------------------------------------------------
st.sidebar.header("股票設定")

# 市場選擇 (自動補齊代碼後綴)
market_mode = st.sidebar.radio(
    "選擇市場",
    options=["台股 (上市)", "台股 (上櫃)", "美股/其他"],
    index=0
)

# 輸入股票代碼 (使用者只需輸入數字或代碼)
raw_symbol = st.sidebar.text_input("輸入代碼", value="2481")

# 週期選擇
period = st.sidebar.selectbox("K棒週期", options=["1y", "2y", "5y"], index=0)

# 自動組裝代碼邏輯
if market_mode == "台股 (上市)":
    # 如果使用者沒打 .TW，自動加上
    ticker = f"{raw_symbol}.TW" if not raw_symbol.upper().endswith(".TW") else raw_symbol
elif market_mode == "台股 (上櫃)":
    ticker = f"{raw_symbol}.TWO" if not raw_symbol.upper().endswith(".TWO") else raw_symbol
else:
    ticker = raw_symbol.upper() # 美股通常是大寫

st.sidebar.caption(f"目前查詢代碼: {ticker}")

# ---------------------------------------------------------
# 3. 數據抓取與計算 (使用 pandas_ta)
# ---------------------------------------------------------
@st.cache_data(ttl=60)
def get_data(ticker, period):
    # 下載數據
    try:
        data = yf.download(ticker, period=period)
    except Exception:
        return None

    # 處理 MultiIndex columns (yfinance 新版修正)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    if data.empty:
        return None

    # --- 核心指標計算 ---
    # 1. 均線 (MA)
    data['MA5'] = ta.sma(data['Close'], length=5)
    data['MA10'] = ta.sma(data['Close'], length=10)
    data['MA20'] = ta.sma(data['Close'], length=20)
    
    # 2. 布林通道 (BBands)
    # pandas_ta 的 bbands 會回傳三欄：BBL (下), BBM (中), BBU (上)
    bbands = ta.bbands(data['Close'], length=20, std=2)
    if bbands is not None:
        data = pd.concat([data, bbands], axis=1)
    
    # 3. MACD
    macd = ta.macd(data['Close'])
    if macd is not None:
        data = pd.concat([data, macd], axis=1)
    
    # 4. RSI
    data['RSI'] = ta.rsi(data['Close'], length=14)
    
    # 5. KD (Stochastic)
    stoch = ta.stoch(data['High'], data['Low'], data['Close'])
    if stoch is not None:
        data = pd.concat([data, stoch], axis=1)
    
    # 6. OBV (On Balance Volume)
    data['OBV'] = ta.obv(data['Close'], data['Volume'])
    
    # 7. 乖離率 (Bias) - 以 20日均線為基準
    # 公式: (收盤價 - 均線) / 均線 * 100
    data['BIAS'] = (data['Close'] - data['MA20']) / data['MA20'] * 100
    
    # 格式整理給圖表庫使用 (轉小寫方便操作)
    data = data.reset_index()
    data.columns = [col.lower() for col in data.columns]
    return data

try:
    df = get_data(ticker, period)
    if df is None:
        st.error(f"找不到代碼 {ticker} 的資料，請確認輸入是否正確。")
        st.stop()
except Exception as e:
    st.error(f"發生錯誤: {e}")
    st.stop()

# ---------------------------------------------------------
# 4. 圖表配置 (Futu 亮色風格核心)
# ---------------------------------------------------------
# 定義顏色 (台股習慣：紅漲綠跌)
COLOR_UP = '#FF5252'    # 鮮紅 (漲)
COLOR_DOWN = '#00B746'  # 鮮綠 (跌)
COLOR_MA5 = '#FFA500'   # 橘色
COLOR_MA10 = '#40E0D0'  # 藍綠色
COLOR_MA20 = '#9370DB'  # 紫色

# 準備數據格式
chart_data = []
for index, row in df.iterrows():
    # K線圖基本數據
    candle = {
        'time': row['date'].strftime('%Y-%m-%d'),
        'open': row['open'],
        'high': row['high'],
        'low': row['low'],
        'close': row['close']
    }
    # 整合所有指標數據 (使用 get 避免有些指標計算初期為 NaN 導致錯誤)
    candle.update({
        'volume': row['volume'] if not pd.isna(row['volume']) else 0,
        'ma5': row.get('ma5'), 'ma10': row.get('ma10'), 'ma20': row.get('ma20'),
        'bbu': row.get('bbu_20_2.0'), 'bbl': row.get('bbl_20_2.0'),
        'macd': row.get('macd_12_26_9'), 'signal': row.get('macds_12_26_9'), 'hist': row.get('macdh_12_26_9'),
        'rsi': row.get('rsi'),
        'k': row.get('stochk_14_3_3'), 'd': row.get('stochd_14_3_3'),
        'obv': row.get('obv'),
        'bias': row.get('bias')
    })
    chart_data.append(candle)

# --- 設定圖表外觀 (亮色模式關鍵) ---
chartOptions = {
    "layout": {
        "backgroundColor": "#FFFFFF",  # 【亮色模式】背景白
        "textColor": "#333333"         # 文字深灰
    },
    "grid": {
        "vertLines": {"color": "#F0F0F0"}, # 極淡網格
        "horzLines": {"color": "#F0F0F0"}
    },
    "crosshair": {
        "mode": 1  # 磁吸游標
    },
    "rightPriceScale": {
        "borderColor": "#E0E0E0"
    },
    "timeScale": {
        "borderColor": "#E0E0E0"
    }
}

# ---------------------------------------------------------
# 5. 定義圖表系列 (Series) - 層層堆疊
# ---------------------------------------------------------
series = [
    # --- Pane 0: 主圖 (K線 + MA + 布林) ---
    {
        "type": "Candlestick",
        "data": chart_data,
        "options": {
            "upColor": COLOR_UP,
            "downColor": COLOR_DOWN,
            "borderUpColor": COLOR_UP,
            "borderDownColor": COLOR_DOWN,
            "wickUpColor": COLOR_UP,
            "wickDownColor": COLOR_DOWN
        }
    },
    {"type": "Line", "data": chart_data, "options": {"color": COLOR_MA5, "lineWidth": 1, "title": "MA5"}, "valueField": "ma5"},
    {"type": "Line", "data": chart_data, "options": {"color": COLOR_MA10, "lineWidth": 1, "title": "MA10"}, "valueField": "ma10"},
    {"type": "Line", "data": chart_data, "options": {"color": COLOR_MA20, "lineWidth": 2, "title": "MA20"}, "valueField": "ma20"},
    # 布林通道 (用半透明線)
    {"type": "Line", "data": chart_data, "options": {"color": "rgba(0, 0, 255, 0.3)", "lineWidth": 1, "lineStyle": 2, "title": "BBU"}, "valueField": "bbu"},
    {"type": "Line", "data": chart_data, "options": {"color": "rgba(0, 0, 255, 0.3)", "lineWidth": 1, "lineStyle": 2, "title": "BBL"}, "valueField": "bbl"},

    # --- Pane 1: 成交量 ---
    {
        "type": "Histogram",
        "data": chart_data,
        "options": {
            "color": "#26a69a",
            "priceFormat": {"type": "volume"},
            "priceScaleId": "volume" # 獨立座標軸 ID
        },
        "valueField": "volume"
    },
    
    # --- Pane 2: MACD ---
    { # DIF (快線)
        "type": "Line", "data": chart_data, "options": {"color": "#2962FF", "lineWidth": 1, "priceScaleId": "macd"}, "valueField": "macd"
    },
    { # DEM (慢線)
        "type": "Line", "data": chart_data, "options": {"color": "#FF6D00", "lineWidth": 1, "priceScaleId": "macd"}, "valueField": "signal"
    },
    { # 柱狀圖
        "type": "Histogram", "data": chart_data, "options": {"color": "#26a69a", "priceScaleId": "macd"}, "valueField": "hist"
    },

    # --- Pane 3: KDJ (KD) ---
    {"type": "Line", "data": chart_data, "options": {"color": "#E91E63", "title": "K", "priceScaleId": "kdj"}, "valueField": "k"},
    {"type": "Line", "data": chart_data, "options": {"color": "#2196F3", "title": "D", "priceScaleId": "kdj"}, "valueField": "d"},

    # --- Pane 4: RSI ---
    {"type": "Line", "data": chart_data, "options": {"color": "#9C27B0", "title": "RSI", "priceScaleId": "rsi"}, "valueField": "rsi"},
    
    # --- Pane 5: OBV ---
    {"type": "Line", "data": chart_data, "options": {"color": "#FF9800", "title": "OBV", "priceScaleId": "obv"}, "valueField": "obv"},
    
    # --- Pane 6: 乖離率 (Bias) ---
    {"type": "Line", "data": chart_data, "options": {"color": "#607D8B", "title": "Bias(20)", "priceScaleId": "bias"}, "valueField": "bias"},
]

# ---------------------------------------------------------
# 6. 渲染圖表 (設定各個面板的高度比例)
# ---------------------------------------------------------
st.markdown("### 技術分析圖表")

renderLightweightCharts(
    [
        {"chart": chartOptions, "series": series[:6], "height": 400}, # 主圖
        {"chart": chartOptions, "series": [series[6]], "height": 100}, # 成交量
        {"chart": chartOptions, "series": series[7:10], "height": 150}, # MACD
        {"chart": chartOptions, "series": series[10:12], "height": 100}, # KD
        {"chart": chartOptions, "series": [series[12]], "height": 100}, # RSI
        {"chart": chartOptions, "series": [series[13]], "height": 100}, # OBV
        {"chart": chartOptions, "series": [series[14]], "height": 100}, # Bias
    ], 
    key="multi_pane_chart"
)
