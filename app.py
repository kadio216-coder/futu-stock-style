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
st.subheader("台美股專業看盤 (仿富途牛牛 - 終極修復版)")

# ---------------------------------------------------------
# 2. 側邊欄設定
# ---------------------------------------------------------
st.sidebar.header("股票設定")

# 強制清除快取按鈕 (如果圖表怪怪的，按這個)
if st.sidebar.button("🔄 強制重整數據"):
    st.cache_data.clear()

market_mode = st.sidebar.radio("選擇市場", options=["台股 (上市)", "台股 (上櫃)", "美股/其他"], index=2)
raw_symbol = st.sidebar.text_input("輸入代碼", value="MU")

interval_map = {
    "日 K": "1d",
    "週 K": "1wk",
    "月 K": "1mo",
    "季 K": "3mo",
    "年 K": "1y" 
}
selected_interval_label = st.sidebar.selectbox("K 棒週期", options=list(interval_map.keys()), index=0)
interval = interval_map[selected_interval_label]

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
@st.cache_data(ttl=10) # 縮短快取時間方便測試
def get_data(ticker, interval_label, interval_code):
    try:
        if interval_label == "日 K":
            period = "2y"
        elif interval_label == "週 K":
            period = "5y"
        else:
            period = "max"

        download_interval = "1mo" if interval_label == "年 K" else interval_code
        data = yf.download(ticker, period=period, interval=download_interval, progress=False)
        
        if data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # 移除時區
        data.index = data.index.tz_localize(None)

        if interval_label == "年 K":
            ohlc_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            data = data.resample('YE').agg(ohlc_dict).dropna()

        # 計算指標
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
        
        data = data.reset_index()
        data.columns = [col.lower() for col in data.columns]
        
        # 統一日期格式
        if 'date' in data.columns:
            data['date_str'] = data['date'].dt.strftime('%Y-%m-%d')
        elif 'index' in data.columns:
            data['date_str'] = data['index'].dt.strftime('%Y-%m-%d')
            
        return data
    except Exception as e:
        st.error(f"數據錯誤: {e}")
        return None

df = get_data(ticker, selected_interval_label, interval)

if df is None or df.empty:
    st.error("無數據")
    st.stop()

# ---------------------------------------------------------
# 4. 圖表配置 (核彈級拆解法)
# ---------------------------------------------------------
# 我們不把所有資料塞在同一個字典，而是拆成獨立的 List
# 這樣可以避開所有 "Key Error" 或 "None" 的問題

candlestick_data = []
volume_data = []
ma5_data = []
ma10_data = []
ma20_data = []
bbu_data = []
bbl_data = []
macd_dif_data = []
macd_dea_data = []
macd_hist_data = []
k_data = []
d_data = []
rsi_data = []
obv_data = []
bias_data = []

# 安全轉型函數
def is_valid(val):
    return val is not None and not pd.isna(val)

for index, row in df.iterrows():
    time_str = row['date_str']
    
    # 1. 基礎 K 線 (如果沒有價格，這一行就跳過)
    if is_valid(row['open']) and is_valid(row['close']):
        candlestick_data.append({
            'time': time_str,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
        })
        
    # 2. 成交量
    if is_valid(row['volume']):
        volume_data.append({'time': time_str, 'value': float(row['volume'])})
        
    # 3. 均線 (如果不夠天數算出來是 NaN，我們就不要加入 List)
    if is_valid(row.get('ma5')): ma5_data.append({'time': time_str, 'value': float(row['ma5'])})
    if is_valid(row.get('ma10')): ma10_data.append({'time': time_str, 'value': float(row['ma10'])})
    if is_valid(row.get('ma20')): ma20_data.append({'time': time_str, 'value': float(row['ma20'])})
    
    # 4. 布林
    if is_valid(row.get('bbu_20_2.0')): bbu_data.append({'time': time_str, 'value': float(row['bbu_20_2.0'])})
    if is_valid(row.get('bbl_20_2.0')): bbl_data.append({'time': time_str, 'value': float(row['bbl_20_2.0'])})

    # 5. MACD
    if is_valid(row.get('macd_12_26_9')): macd_dif_data.append({'time': time_str, 'value': float(row['macd_12_26_9'])})
    if is_valid(row.get('macds_12_26_9')): macd_dea_data.append({'time': time_str, 'value': float(row['macds_12_26_9'])})
    if is_valid(row.get('macdh_12_26_9')): macd_hist_data.append({'time': time_str, 'value': float(row['macdh_12_26_9'])})

    # 6. KD, RSI, OBV, BIAS
    if is_valid(row.get('stochk_14_3_3')): k_data.append({'time': time_str, 'value': float(row['stochk_14_3_3'])})
    if is_valid(row.get('stochd_14_3_3')): d_data.append({'time': time_str, 'value': float(row['stochd_14_3_3'])})
    if is_valid(row.get('rsi')): rsi_data.append({'time': time_str, 'value': float(row['rsi'])})
    if is_valid(row.get('obv')): obv_data.append({'time': time_str, 'value': float(row['obv'])})
    if is_valid(row.get('bias')): bias_data.append({'time': time_str, 'value': float(row['bias'])})


# 圖表設定
chartOptions = {
    "layout": { "backgroundColor": "#FFFFFF", "textColor": "#333333" },
    "grid": { "vertLines": {"color": "#F0F0F0"}, "horzLines": {"color": "#F0F0F0"} },
    "crosshair": { "mode": 1 },
    "rightPriceScale": { "borderColor": "#E0E0E0" },
    "timeScale": { "borderColor": "#E0E0E0" }
}

# 顏色定義
COLOR_UP = '#FF5252'
COLOR_DOWN = '#00B746'

series = [
    # 主圖
    {
        "type": "Candlestick",
        "data": candlestick_data, # 直接餵入獨立的 List
        "options": {
            "upColor": COLOR_UP, "downColor": COLOR_DOWN,
            "borderUpColor": COLOR_UP, "borderDownColor": COLOR_DOWN,
            "wickUpColor": COLOR_UP, "wickDownColor": COLOR_DOWN
        }
    },
    {"type": "Line", "data": ma5_data, "options": {"color": '#FFA500', "lineWidth": 1, "title": "MA5"}},
    {"type": "Line", "data": ma10_data, "options": {"color": '#40E0D0', "lineWidth": 1, "title": "MA10"}},
    {"type": "Line", "data": ma20_data, "options": {"color": '#9370DB', "lineWidth": 2, "title": "MA20"}},
    {"type": "Line", "data": bbu_data, "options": {"color": "rgba(0, 0, 255, 0.3)", "lineWidth": 1, "lineStyle": 2}},
    {"type": "Line", "data": bbl_data, "options": {"color": "rgba(0, 0, 255, 0.3)", "lineWidth": 1, "lineStyle": 2}},
    
    # 成交量
    {"type": "Histogram", "data": volume_data, "options": {"color": "#26a69a", "priceFormat": {"type": "volume"}, "priceScaleId": "volume"}},
    
    # MACD
    {"type": "Line", "data": macd_dif_data, "options": {"color": "#2962FF", "lineWidth": 1, "priceScaleId": "macd"}},
    {"type": "Line", "data": macd_dea_data, "options": {"color": "#FF6D00", "lineWidth": 1, "priceScaleId": "macd"}},
    {"type": "Histogram", "data": macd_hist_data, "options": {"color": "#26a69a", "priceScaleId": "macd"}},
    
    # KD
    {"type": "Line", "data": k_data, "options": {"color": "#E91E63", "priceScaleId": "kdj"}},
    {"type": "Line", "data": d_data, "options": {"color": "#2196F3", "priceScaleId": "kdj"}},
    
    # RSI
    {"type": "Line", "data": rsi_data, "options": {"color": "#9C27B0", "priceScaleId": "rsi"}},
    
    # OBV
    {"type": "Line", "data": obv_data, "options": {"color": "#FF9800", "priceScaleId": "obv"}},
    
    # Bias
    {"type": "Line", "data": bias_data, "options": {"color": "#607D8B", "priceScaleId": "bias"}},
]

st.markdown("### 技術分析圖表")
renderLightweightCharts([
    {"chart": chartOptions, "series": series[:6], "height": 400},
    {"chart": chartOptions, "series": [series[6]], "height": 100},
    {"chart": chartOptions, "series": series[7:10], "height": 150},
    {"chart": chartOptions, "series": series[10:12], "height": 100},
    {"chart": chartOptions, "series": [series[12]], "height": 100},
    {"chart": chartOptions, "series": [series[13]], "height": 100},
    {"chart": chartOptions, "series": [series[14]], "height": 100},
], key="multi_pane_chart_v2") # 改 key 強制刷新
