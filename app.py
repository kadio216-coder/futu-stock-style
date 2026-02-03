import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np
from streamlit_lightweight_charts import renderLightweightCharts

# ---------------------------------------------------------
# 1. 頁面設定
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Futu Style Analyzer")
st.subheader("台美股專業看盤 (仿富途牛牛 - 亮色版)")

# ---------------------------------------------------------
# 2. 側邊欄設定
# ---------------------------------------------------------
st.sidebar.header("股票設定")

# 市場與代碼
market_mode = st.sidebar.radio("選擇市場", options=["台股 (上市)", "台股 (上櫃)", "美股/其他"], index=2) # 預設改為美股方便測試
raw_symbol = st.sidebar.text_input("輸入代碼", value="MU")

# K 棒週期對照表
interval_map = {
    "日 K": "1d",
    "週 K": "1wk",
    "月 K": "1mo",
    "季 K": "3mo",
    "年 K": "1y" 
}
selected_interval_label = st.sidebar.selectbox("K 棒週期", options=list(interval_map.keys()), index=0)
interval = interval_map[selected_interval_label]

# 自動組裝代碼
if market_mode == "台股 (上市)":
    ticker = f"{raw_symbol}.TW" if not raw_symbol.upper().endswith(".TW") else raw_symbol
elif market_mode == "台股 (上櫃)":
    ticker = f"{raw_symbol}.TWO" if not raw_symbol.upper().endswith(".TWO") else raw_symbol
else:
    ticker = raw_symbol.upper()

st.sidebar.caption(f"查詢代碼: {ticker}")

# ---------------------------------------------------------
# 3. 數據抓取與計算 (強力修復版)
# ---------------------------------------------------------
@st.cache_data(ttl=60)
def get_data(ticker, interval_label, interval_code):
    try:
        # 根據週期自動決定要抓多久的資料
        if interval_label == "日 K":
            period = "2y"
        elif interval_label == "週 K":
            period = "5y"
        else:
            period = "max"

        # 下載資料
        download_interval = "1mo" if interval_label == "年 K" else interval_code
        data = yf.download(ticker, period=period, interval=download_interval, progress=False)
        
        if data.empty:
            return None

        # --- 【修復 1】處理 MultiIndex 與時區 ---
        # 如果欄位是多層索引 (Price, Ticker)，只取第一層 (Price)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # 移除時區 (最關鍵的一步，解決空白圖表的元兇)
        data.index = data.index.tz_localize(None)

        # --- 【修復 2】年 K 線重算 ---
        if interval_label == "年 K":
            ohlc_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            data = data.resample('YE').agg(ohlc_dict).dropna()

        # --- 核心指標計算 ---
        data['MA5'] = ta.sma(data['Close'], length=5)
        data['MA10'] = ta.sma(data['Close'], length=10)
        data['MA20'] = ta.sma(data['Close'], length=20)
        
        bbands = ta.bbands(data['Close'], length=20, std=2)
        if bbands is not None:
            data = pd.concat([data, bbands], axis=1)
        
        macd = ta.macd(data['Close'])
        if macd is not None:
            data = pd.concat([data, macd], axis=1)
        
        data['RSI'] = ta.rsi(data['Close'], length=14)
        
        stoch = ta.stoch(data['High'], data['Low'], data['Close'])
        if stoch is not None:
            data = pd.concat([data, stoch], axis=1)
            
        data['OBV'] = ta.obv(data['Close'], data['Volume'])
        data['BIAS'] = (data['Close'] - data['MA20']) / data['MA20'] * 100
        
        # 格式整理
        data = data.reset_index()
        # 統一欄位名稱為小寫 (open, high, low...)
        data.columns = [col.lower() for col in data.columns]
        
        # 確保有 date_str 欄位
        if 'date' in data.columns:
            data['date_str'] = data['date'].dt.strftime('%Y-%m-%d')
        elif 'index' in data.columns:
            data['date_str'] = data['index'].dt.strftime('%Y-%m-%d')
            
        return data
    except Exception as e:
        st.error(f"數據處理錯誤: {e}")
        return None

# 執行抓取
df = get_data(ticker, selected_interval_label, interval)

# --- Debug 區塊：如果圖表還是空的，打開這個看看有沒有資料 ---
with st.expander("🛠️ 開發者診斷工具 (點擊展開查看原始數據)"):
    if df is not None:
        st.write("前 5 筆數據預覽：", df.head())
    else:
        st.write("沒有抓到數據 (DataFrame is None)")

if df is None or df.empty:
    st.error(f"無法取得代碼 {ticker} 的資料。")
    st.stop()

# ---------------------------------------------------------
# 4. 圖表配置 (強力轉型版)
# ---------------------------------------------------------
COLOR_UP = '#FF5252'
COLOR_DOWN = '#00B746'

# 輔助函式：安全轉型 (處理 None 和 NaN)
def safe_float(val):
    if val is None or pd.isna(val):
        return None
    return float(val)

chart_data = []
for index, row in df.iterrows():
    # --- 【修復 3】強制將 numpy float 轉為 python float ---
    # JSON 序列化非常挑剔，這裡我們手動一個一個轉，確保萬無一失
    candle = {
        'time': row['date_str'], 
        'open': safe_float(row['open']), 
        'high': safe_float(row['high']), 
        'low': safe_float(row['low']), 
        'close': safe_float(row['close']),
        'volume': int(row['volume']) if not pd.isna(row['volume']) else 0,
        
        'ma5': safe_float(row.get('ma5')), 
        'ma10': safe_float(row.get('ma10')), 
        'ma20': safe_float(row.get('ma20')),
        
        'bbu': safe_float(row.get('bbu_20_2.0')), 
        'bbl': safe_float(row.get('bbl_20_2.0')),
        
        'macd': safe_float(row.get('macd_12_26_9')), 
        'signal': safe_float(row.get('macds_12_26_9')), 
        'hist': safe_float(row.get('macdh_12_26_9')),
        
        'rsi': safe_float(row.get('rsi')),
        'k': safe_float(row.get('stochk_14_3_3')), 
        'd': safe_float(row.get('stochd_14_3_3')),
        'obv': safe_float(row.get('obv')),
        'bias': safe_float(row.get('bias'))
    }
    chart_data.append(candle)

# 圖表外觀設定
chartOptions = {
    "layout": {
        "backgroundColor": "#FFFFFF", "textColor": "#333333"
    },
    "grid": {
        "vertLines": {"color": "#F0F0F0"}, "horzLines": {"color": "#F0F0F0"}
    },
    "crosshair": {"mode": 1},
    "rightPriceScale": {"borderColor": "#E0E0E0"},
    "timeScale": {"borderColor": "#E0E0E0"}
}

series = [
    {
        "type": "Candlestick",
        "data": chart_data,
        "options": {
            "upColor": COLOR_UP, "downColor": COLOR_DOWN,
            "borderUpColor": COLOR_UP, "borderDownColor": COLOR_DOWN,
            "wickUpColor": COLOR_UP, "wickDownColor": COLOR_DOWN
        }
    },
    {"type": "Line", "data": chart_data, "options": {"color": '#FFA500', "lineWidth": 1, "title": "MA5"}, "valueField": "ma5"},
    {"type": "Line", "data": chart_data, "options": {"color": '#40E0D0', "lineWidth": 1, "title": "MA10"}, "valueField": "ma10"},
    {"type": "Line", "data": chart_data, "options": {"color": '#9370DB', "lineWidth": 2, "title": "MA20"}, "valueField": "ma20"},
    {"type": "Line", "data": chart_data, "options": {"color": "rgba(0, 0, 255, 0.3)", "lineWidth": 1, "lineStyle": 2}, "valueField": "bbu"},
    {"type": "Line", "data": chart_data, "options": {"color": "rgba(0, 0, 255, 0.3)", "lineWidth": 1, "lineStyle": 2}, "valueField": "bbl"},
    
    # 副圖
    {"type": "Histogram", "data": chart_data, "options": {"color": "#26a69a", "priceFormat": {"type": "volume"}, "priceScaleId": "volume"}, "valueField": "volume"},
    {"type": "Line", "data": chart_data, "options": {"color": "#2962FF", "lineWidth": 1, "priceScaleId": "macd"}, "valueField": "macd"},
    {"type": "Line", "data": chart_data, "options": {"color": "#FF6D00", "lineWidth": 1, "priceScaleId": "macd"}, "valueField": "signal"},
    {"type": "Histogram", "data": chart_data, "options": {"color": "#26a69a", "priceScaleId": "macd"}, "valueField": "hist"},
    {"type": "Line", "data": chart_data, "options": {"color": "#E91E63", "priceScaleId": "kdj"}, "valueField": "k"},
    {"type": "Line", "data": chart_data, "options": {"color": "#2196F3", "priceScaleId": "kdj"}, "valueField": "d"},
    {"type": "Line", "data": chart_data, "options": {"color": "#9C27B0", "priceScaleId": "rsi"}, "valueField": "rsi"},
    {"type": "Line", "data": chart_data, "options": {"color": "#FF9800", "priceScaleId": "obv"}, "valueField": "obv"},
    {"type": "Line", "data": chart_data, "options": {"color": "#607D8B", "priceScaleId": "bias"}, "valueField": "bias"},
]

renderLightweightCharts([
    {"chart": chartOptions, "series": series[:6], "height": 400},
    {"chart": chartOptions, "series": [series[6]], "height": 100},
    {"chart": chartOptions, "series": series[7:10], "height": 150},
    {"chart": chartOptions, "series": series[10:12], "height": 100},
    {"chart": chartOptions, "series": [series[12]], "height": 100},
    {"chart": chartOptions, "series": [series[13]], "height": 100},
    {"chart": chartOptions, "series": [series[14]], "height": 100},
], key="multi_pane_chart")
