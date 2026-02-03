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
market_mode = st.sidebar.radio("選擇市場", options=["台股 (上市)", "台股 (上櫃)", "美股/其他"], index=0)
raw_symbol = st.sidebar.text_input("輸入代碼", value="2481")

# --- 關鍵修改：真正的 K 棒週期選擇 ---
interval_map = {
    "日 K": "1d",
    "週 K": "1wk",
    "月 K": "1mo",
    "季 K": "3mo",
    "年 K": "1y"  # yfinance 原生不支援年K，稍後用 pandas 重算 (Resample)
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
# 3. 數據抓取與計算
# ---------------------------------------------------------
@st.cache_data(ttl=60)
def get_data(ticker, interval_label, interval_code):
    try:
        # 根據週期自動決定要抓多久的資料 (Period)，避免資料太少或太多
        if interval_label == "日 K":
            period = "2y"
        elif interval_label == "週 K":
            period = "5y"
        else:
            period = "max" # 月、季、年抓全部歷史

        # 下載資料
        # 注意: yfinance 不支援直接下載 "1y" (年K)，所以年K我們先下載 "1mo" 再合成
        download_interval = "1mo" if interval_label == "年 K" else interval_code
        
        data = yf.download(ticker, period=period, interval=download_interval)
        
        # 處理資料結構 (MultiIndex 修復)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if data.empty:
            return None

        # --- 特殊處理：年 K 線 (Resample) ---
        if interval_label == "年 K":
            # 將月線資料重採樣為年線
            ohlc_dict = {
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }
            data = data.resample('YE').agg(ohlc_dict).dropna()

        # 確保索引是 Datetime
        data.index = pd.to_datetime(data.index)

        # --- 核心指標計算 ---
        # 1. 均線 (MA)
        data['MA5'] = ta.sma(data['Close'], length=5)
        data['MA10'] = ta.sma(data['Close'], length=10)
        data['MA20'] = ta.sma(data['Close'], length=20)
        
        # 2. 布林通道
        bbands = ta.bbands(data['Close'], length=20, std=2)
        if bbands is not None:
            data = pd.concat([data, bbands], axis=1)
        
        # 3. MACD
        macd = ta.macd(data['Close'])
        if macd is not None:
            data = pd.concat([data, macd], axis=1)
        
        # 4. RSI & KD & OBV
        data['RSI'] = ta.rsi(data['Close'], length=14)
        stoch = ta.stoch(data['High'], data['Low'], data['Close'])
        if stoch is not None:
            data = pd.concat([data, stoch], axis=1)
        data['OBV'] = ta.obv(data['Close'], data['Volume'])
        data['BIAS'] = (data['Close'] - data['MA20']) / data['MA20'] * 100
        
        # 格式整理：轉小寫、重設索引
        data = data.reset_index()
        data.columns = [col.lower() for col in data.columns]
        
        # 【關鍵修復】確保日期是字串格式 YYYY-MM-DD
        # yfinance 的 date 有時候會帶時區 (timezone)，導致圖表掛掉，這裡強制轉
        if 'date' in data.columns:
            data['date_str'] = data['date'].dt.strftime('%Y-%m-%d')
        else:
            # 如果 reset_index 後 index 變成了 'index' 欄位
            data['date_str'] = data['index'].dt.strftime('%Y-%m-%d')

        # 替換 NaN 為 None
        data = data.replace({np.nan: None})
        
        return data
    except Exception as e:
        print(f"Error: {e}") # Debug 用
        return None

# 執行抓取
df = get_data(ticker, selected_interval_label, interval)

if df is None or df.empty:
    st.error(f"無法取得代碼 {ticker} 的資料 (或是該週期無資料)。")
    st.stop()

# ---------------------------------------------------------
# 4. 圖表配置 (資料映射)
# ---------------------------------------------------------
COLOR_UP = '#FF5252'
COLOR_DOWN = '#00B746'
COLOR_MA5 = '#FFA500'
COLOR_MA10 = '#40E0D0'
COLOR_MA20 = '#9370DB'

chart_data = []
for index, row in df.iterrows():
    # 這裡使用剛剛強制轉好的 date_str，解決空白月曆問題
    candle = {
        'time': row['date_str'], 
        'open': row['open'], 'high': row['high'], 'low': row['low'], 'close': row['close'],
        'volume': row['volume'] if row['volume'] is not None else 0,
        'ma5': row.get('ma5'), 'ma10': row.get('ma10'), 'ma20': row.get('ma20'),
        'bbu': row.get('bbu_20_2.0'), 'bbl': row.get('bbl_20_2.0'),
        'macd': row.get('macd_12_26_9'), 'signal': row.get('macds_12_26_9'), 'hist': row.get('macdh_12_26_9'),
        'rsi': row.get('rsi'),
        'k': row.get('stochk_14_3_3'), 'd': row.get('stochd_14_3_3'),
        'obv': row.get('obv'),
        'bias': row.get('bias')
    }
    chart_data.append(candle)

# 圖表設定
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
    {"type": "Line", "data": chart_data, "options": {"color": COLOR_MA5, "lineWidth": 1, "title": "MA5"}, "valueField": "ma5"},
    {"type": "Line", "data": chart_data, "options": {"color": COLOR_MA10, "lineWidth": 1, "title": "MA10"}, "valueField": "ma10"},
    {"type": "Line", "data": chart_data, "options": {"color": COLOR_MA20, "lineWidth": 2, "title": "MA20"}, "valueField": "ma20"},
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
