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
st.subheader("台美股K線")

# ---------------------------------------------------------
# 2. 側邊欄設定 (移除重整按鈕)
# ---------------------------------------------------------
st.sidebar.header("股票設定")

market_mode = st.sidebar.radio("選擇市場", options=["台股 (上市)", "台股 (上櫃)", "美股/其他"], index=2)
raw_symbol = st.sidebar.text_input("輸入代碼", value="MU")

interval_map = {"日 K": "1d", "週 K": "1wk", "月 K": "1mo", "季 K": "3mo", "年 K": "1y"}
selected_interval_label = st.sidebar.selectbox("K 棒週期", options=list(interval_map.keys()), index=0)

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
def get_clean_data(ticker, interval_label):
    try:
        # 1. 下載資料
        interval = interval_map[interval_label]
        period = "2y" if interval_label == "日 K" else "max"
        download_interval = "1mo" if interval_label == "年 K" else interval
        
        data = yf.download(ticker, period=period, interval=download_interval, progress=False)
        
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        
        # 2. 移除時區
        data.index = data.index.tz_localize(None)
        
        # 3. 年K處理
        if interval_label == "年 K":
            data = data.resample('YE').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()

        # 4. 嚴格清洗
        data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        # 5. 計算指標
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
        
        # 6. 轉換為 Unix Timestamp
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
# 4. 數據打包 (顏色邏輯與標籤優化)
# ---------------------------------------------------------
# 定義顏色
COLOR_UP = '#FF5252'    # 漲 (紅)
COLOR_DOWN = '#00B746'  # 跌 (綠)

def is_safe(val):
    if val is None: return False
    if pd.isna(val): return False
    if np.isinf(val): return False
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
        
        # 【牛牛風格成交量】: 漲紅跌綠
        if is_safe(row['volume']):
            vol_color = COLOR_UP if row['close'] >= row['open'] else COLOR_DOWN
            vols.append({'time': t, 'value': float(row['volume']), 'color': vol_color})
            
    # 指標
    if is_safe(row.get('ma5')): ma5.append({'time': t, 'value': float(row['ma5'])})
    if is_safe(row.get('ma10')): ma10.append({'time': t, 'value': float(row['ma10'])})
    if is_safe(row.get('ma20')): ma20.append({'time': t, 'value': float(row['ma20'])})
    
    if is_safe(row.get('bbu_20_2.0')): bbu.append({'time': t, 'value': float(row['bbu_20_2.0'])})
    if is_safe(row.get('bbl_20_2.0')): bbl.append({'time': t, 'value': float(row['bbl_20_2.0'])})
    
    if is_safe(row.get('macd_12_26_9')): macd_dif.append({'time': t, 'value': float(row['macd_12_26_9'])})
    if is_safe(row.get('macds_12_26_9')): macd_dea.append({'time': t, 'value': float(row['macds_12_26_9'])})
    if is_safe(row.get('macdh_12_26_9')): macd_hist.append({'time': t, 'value': float(row['macdh_12_26_9']), 'color': COLOR_UP if row.get('macdh_12_26_9') > 0 else COLOR_DOWN})
    
    if is_safe(row.get('stochk_14_3_3')): k_line.append({'time': t, 'value': float(row['stochk_14_3_3'])})
    if is_safe(row.get('stochd_14_3_3')): d_line.append({'time': t, 'value': float(row['stochd_14_3_3'])})
    if is_safe(row.get('rsi')): rsi_line.append({'time': t, 'value': float(row['rsi'])})
    if is_safe(row.get('obv')): obv_line.append({'time': t, 'value': float(row['obv'])})
    if is_safe(row.get('bias')): bias_line.append({'time': t, 'value': float(row['bias'])})

# ---------------------------------------------------------
# 5. 渲染圖表配置 (標籤與標題設定)
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

# 加入均線 (關鍵修改：lastValueVisible=False 隱藏Y軸標籤，避免擋住K棒)
if ma5: series_config.append({"type": "Line", "data": ma5, "options": {"color": '#FFA500', "lineWidth": 1, "title": "MA5", "lastValueVisible": False, "priceLineVisible": False}})
if ma10: series_config.append({"type": "Line", "data": ma10, "options": {"color": '#40E0D0', "lineWidth": 1, "title": "MA10", "lastValueVisible": False, "priceLineVisible": False}})
if ma20: series_config.append({"type": "Line", "data": ma20, "options": {"color": '#9370DB', "lineWidth": 2, "title": "MA20", "lastValueVisible": False, "priceLineVisible": False}})

if bbu: series_config.append({"type": "Line", "data": bbu, "options": {"color": "rgba(0, 0, 255, 0.3)", "lineWidth": 1, "lineStyle": 2, "lastValueVisible": False, "priceLineVisible": False}})
if bbl: series_config.append({"type": "Line", "data": bbl, "options": {"color": "rgba(0, 0, 255, 0.3)", "lineWidth": 1, "lineStyle": 2, "lastValueVisible": False, "priceLineVisible": False}})

# 副圖配置 (確保 title 都有設定)
vol_series = [{"type": "Histogram", "data": vols, "options": {"priceFormat": {"type": "volume"}, "title": "成交量 (Vol)"}}] if vols else []

macd_series = []
if macd_dif: macd_series.append({"type": "Line", "data": macd_dif, "options": {"color": "#2962FF", "lineWidth": 1, "title": "DIF"}})
if macd_dea: macd_series.append({"type": "Line", "data": macd_dea, "options": {"color": "#FF6D00", "lineWidth": 1, "title": "DEA"}})
if macd_hist: macd_series.append({"type": "Histogram", "data": macd_hist, "options": {"title": "MACD"}})

kdj_series = []
if k_line: kdj_series.append({"type": "Line", "data": k_line, "options": {"color": "#E91E63", "title": "K"}})
if d_line: kdj_series.append({"type": "Line", "data": d_line, "options": {"color": "#2196F3", "title": "D"}})

rsi_series = [{"type": "Line", "data": rsi_line, "options": {"color": "#9C27B0", "title": "RSI (14)"}}] if rsi_line else []
obv_series = [{"type": "Line", "data": obv_line, "options": {"color": "#FF9800", "title": "OBV"}}] if obv_line else []
bias_series = [{"type": "Line", "data": bias_line, "options": {"color": "#607D8B", "title": "乖離率 (Bias)"}}] if bias_line else []

# 組合面板
panes = [
    {"chart": chartOptions, "series": series_config, "height": 400},
]
if vol_series: panes.append({"chart": chartOptions, "series": vol_series, "height": 100})
if macd_series: panes.append({"chart": chartOptions, "series": macd_series, "height": 150})
if kdj_series: panes.append({"chart": chartOptions, "series": kdj_series, "height": 100})
if rsi_series: panes.append({"chart": chartOptions, "series": rsi_series, "height": 100})
if obv_series: panes.append({"chart": chartOptions, "series": obv_series, "height": 100})
if bias_series: panes.append({"chart": chartOptions, "series": bias_series, "height": 100})

st.markdown("### 📊 技術分析圖表")
if len(candles) > 0:
    renderLightweightCharts(panes, key="final_chart_v3")
else:
    st.error("嚴重錯誤：沒有可用的 K 線數據。請檢查股票代碼是否正確。")
