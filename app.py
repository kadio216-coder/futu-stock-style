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
st.subheader("台美股K線圖") # <--- 看到這個標題才算更新成功

# ---------------------------------------------------------
# 2. 側邊欄設定 (已移除重整按鈕)
# ---------------------------------------------------------
st.sidebar.header("股票設定")

market_mode = st.sidebar.radio("選擇市場", options=["台股 (上市)", "台股 (上櫃)", "美股/其他"], index=2)
raw_symbol = st.sidebar.text_input("輸入代碼", value="MU")

interval_map = {"日 K": "1d", "週 K": "1wk", "月 K": "1mo", "季 K": "3mo", "年 K": "1y"}
selected_interval_label = st.sidebar.selectbox("K 棒週期", options=list(interval_map.keys()), index=0)

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
def get_clean_data(ticker, interval_label):
    try:
        # 下載資料
        interval = interval_map[interval_label]
        period = "2y" if interval_label == "日 K" else "max"
        download_interval = "1mo" if interval_label == "年 K" else interval
        
        data = yf.download(ticker, period=period, interval=download_interval, progress=False)
        
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        
        data.index = data.index.tz_localize(None)
        
        if interval_label == "年 K":
            data = data.resample('YE').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()

        data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        # 計算指標
        data['MA5'] = ta.sma(data['Close'], length=5)
        data['MA10'] = ta.sma(data['Close'], length=10)
        data['MA20'] = ta.sma(data['Close'], length=20)
        
        bbands = ta.bbands(data['Close'], length=20, std=2)
        if bbands is not None: data = pd.concat([data, bbands], axis=1)
        
        macd = ta.macd(data['Close'])
        if macd is not None: data = pd.concat([data, macd], axis=1)
        
        data['RSI'] = ta.rsi(data['Close'], length=14)
        
        stoch = ta.stoch(data['High'], data['Low'], data['Close'])
        if stoch is not None: data = pd.concat([data, stoch], axis=1)
        
        data['OBV'] = ta.obv(data['Close'], data['Volume'])
        data['BIAS'] = (data['Close'] - data['MA20']) / data['MA20'] * 100
        
        data = data.reset_index()
        data.columns = [col.lower() for col in data.columns]
        
        if 'date' in data.columns:
            data['time'] = data['date'].astype('int64') // 10**9
        elif 'index' in data.columns:
            data['time'] = data['index'].astype('int64') // 10**9
            
        return data
    except Exception as e:
        st.error(f"Error: {e}")
        return None

df = get_clean_data(ticker, selected_interval_label)

if df is None or df.empty:
    st.error("無資料")
    st.stop()

# ---------------------------------------------------------
# 4. 數據打包 (顏色與格式設定)
# ---------------------------------------------------------
COLOR_UP = '#FF5252'    # 漲 (紅)
COLOR_DOWN = '#00B746'  # 跌 (綠)

def is_safe(val):
    if val is None or pd.isna(val) or np.isinf(val): return False
    return True

candles = []
vols = []
ma5, ma10, ma20 = [], [], []
bbu, bbl = [], []
macd_dif, macd_dea, macd_hist = [], [], []
k_line, d_line, rsi_line, obv_line, bias_line = [], [], [], [], []

for _, row in df.iterrows():
    t = int(row['time'])
    
    # K線
    if is_safe(row['open']) and is_safe(row['close']):
        candles.append({
            'time': t, 
            'open': float(row['open']), 'high': float(row['high']), 
            'low': float(row['low']), 'close': float(row['close'])
        })
        
        # 成交量 (漲紅跌綠)
        if is_safe(row['volume']):
            # 如果收盤 >= 開盤，用紅色，否則綠色
            bar_color = COLOR_UP if row['close'] >= row['open'] else COLOR_DOWN
            vols.append({'time': t, 'value': float(row['volume']), 'color': bar_color})
            
    # 指標
    if is_safe(row.get('ma5')): ma5.append({'time': t, 'value': float(row['ma5'])})
    if is_safe(row.get('ma10')): ma10.append({'time': t, 'value': float(row['ma10'])})
    if is_safe(row.get('ma20')): ma20.append({'time': t, 'value': float(row['ma20'])})
    
    if is_safe(row.get('bbu_20_2.0')): bbu.append({'time': t, 'value': float(row['bbu_20_2.0'])})
    if is_safe(row.get('bbl_20_2.0')): bbl.append({'time': t, 'value': float(row['bbl_20_2.0'])})
    
    if is_safe(row.get('macd_12_26_9')): macd_dif.append({'time': t, 'value': float(row['macd_12_26_9'])})
    if is_safe(row.get('macds_12_26_9')): macd_dea.append({'time': t, 'value': float(row['macds_12_26_9'])})
    if is_safe(row.get('macdh_12_26_9')): 
        hist_val = float(row['macdh_12_26_9'])
        macd_hist.append({'time': t, 'value': hist_val, 'color': COLOR_UP if hist_val > 0 else COLOR_DOWN})
    
    if is_safe(row.get('stochk_14_3_3')): k_line.append({'time': t, 'value': float(row['stochk_14_3_3'])})
    if is_safe(row.get('stochd_14_3_3')): d_line.append({'time': t, 'value': float(row['stochd_14_3_3'])})
    if is_safe(row.get('rsi')): rsi_line.append({'time': t, 'value': float(row['rsi'])})
    if is_safe(row.get('obv')): obv_line.append({'time': t, 'value': float(row['obv'])})
    if is_safe(row.get('bias')): bias_line.append({'time': t, 'value': float(row['bias'])})

# ---------------------------------------------------------
# 5. 渲染圖表 (隱藏標籤、顯示標題)
# ---------------------------------------------------------
chartOptions = {
    "layout": { "backgroundColor": "#FFFFFF", "textColor": "#333333" },
    "grid": { "vertLines": {"color": "#F0F0F0"}, "horzLines": {"color": "#F0F0F0"} },
    "rightPriceScale": { "borderColor": "#E0E0E0", "scaleMargins": {"top": 0.1, "bottom": 0.1} },
    "timeScale": { "borderColor": "#E0E0E0", "timeVisible": True }
}

series_config = [
    {
        "type": "Candlestick",
        "data": candles,
        "options": {
            "upColor": COLOR_UP, "downColor": COLOR_DOWN,
            "borderUpColor": COLOR_UP, "borderDownColor": COLOR_DOWN,
            "wickUpColor": COLOR_UP, "wickDownColor": COLOR_DOWN
        }
    }
]

# 均線設定：lastValueVisible: False (不顯示Y軸標籤，避免擋住K線)
if ma5: series_config.append({"type": "Line", "data": ma5, "options": {"color": '#FFA500', "lineWidth": 1, "title": "MA5", "lastValueVisible": False, "priceLineVisible": False}})
if ma10: series_config.append({"type": "Line", "data": ma10, "options": {"color": '#40E0D0', "lineWidth": 1, "title": "MA10", "lastValueVisible": False, "priceLineVisible": False}})
if ma20: series_config.append({"type": "Line", "data": ma20, "options": {"color": '#9370DB', "lineWidth": 2, "title": "MA20", "lastValueVisible": False, "priceLineVisible": False}})

# 布林通道
if bbu: series_config.append({"type": "Line", "data": bbu, "options": {"color": "rgba(0, 0, 255, 0.3)", "lineWidth": 1, "lineStyle": 2, "lastValueVisible": False, "priceLineVisible": False}})
if bbl: series_config.append({"type": "Line", "data": bbl, "options": {"color": "rgba(0, 0, 255, 0.3)", "lineWidth": 1, "lineStyle": 2, "lastValueVisible": False, "priceLineVisible": False}})

# 面板組合
panes = [{"chart": chartOptions, "series": series_config, "height": 400}]

# 成交量 (加入 title)
if vols: panes.append({"chart": chartOptions, "series": [{"type": "Histogram", "data": vols, "options": {"priceFormat": {"type": "volume"}, "title": "成交量 (Vol)"}}], "height": 100})

# MACD
macd_series = []
if macd_dif: macd_series.append({"type": "Line", "data": macd_dif, "options": {"color": "#2962FF", "lineWidth": 1, "title": "DIF"}})
if macd_dea: macd_series.append({"type": "Line", "data": macd_dea, "options": {"color": "#FF6D00", "lineWidth": 1, "title": "DEA"}})
if macd_hist: macd_series.append({"type": "Histogram", "data": macd_hist, "options": {"title": "MACD"}})
if macd_series: panes.append({"chart": chartOptions, "series": macd_series, "height": 150})

# KDJ
kdj_series = []
if k_line: kdj_series.append({"type": "Line", "data": k_line, "options": {"color": "#E91E63", "title": "K"}})
if d_line: kdj_series.append({"type": "Line", "data": d_line, "options": {"color": "#2196F3", "title": "D"}})
if kdj_series: panes.append({"chart": chartOptions, "series": kdj_series, "height": 100})

# RSI
if rsi_line: panes.append({"chart": chartOptions, "series": [{"type": "Line", "data": rsi_line, "options": {"color": "#9C27B0", "title": "RSI(14)"}}], "height": 100})

# OBV
if obv_line: panes.append({"chart": chartOptions, "series": [{"type": "Line", "data": obv_line, "options": {"color": "#FF9800", "title": "OBV"}}], "height": 100})

# Bias
if bias_line: panes.append({"chart": chartOptions, "series": [{"type": "Line", "data": bias_line, "options": {"color": "#607D8B", "title": "乖離率"}}], "height": 100})

st.markdown("### 📊 技術分析圖表")
if len(candles) > 0:
    renderLightweightCharts(panes, key="final_v3") # 改 Key 強制刷新
else:
    st.error("錯誤：無數據")
