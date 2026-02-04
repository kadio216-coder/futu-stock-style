import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from streamlit_lightweight_charts import renderLightweightCharts

# ---------------------------------------------------------
# 1. 頁面設定
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Futu Style Analyzer")
st.subheader("台美股專業看盤 (仿富途牛牛 - V5.0 旗艦混合版)")

# ---------------------------------------------------------
# 2. 側邊欄設定
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
        interval = interval_map[interval_label]
        # 抓取長一點的資料，讓滑桿有空間發揮
        period = "5y" if interval_label in ["日 K", "週 K"] else "max"
        download_interval = "1mo" if interval_label == "年 K" else interval
        
        data = yf.download(ticker, period=period, interval=download_interval, progress=False)
        
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        
        data.index = data.index.tz_localize(None)
        
        if interval_label == "年 K":
            data = data.resample('YE').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()

        # 保留所有行以便後續指標計算，最後再清洗
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
            data['date_obj'] = data['date'] 
            data['time'] = data['date'].astype('int64') // 10**9
        elif 'index' in data.columns:
            data['date_obj'] = data['index']
            data['time'] = data['index'].astype('int64') // 10**9
            
        return data
    except Exception as e:
        st.error(f"Error: {e}")
        return None

full_df = get_clean_data(ticker, selected_interval_label)

if full_df is None or full_df.empty:
    st.error("無資料")
    st.stop()

# --- 滑桿與篩選邏輯 ---
st.sidebar.divider()
st.sidebar.write("### 📅 資料區間篩選")
min_date = full_df['date_obj'].min().to_pydatetime()
max_date = full_df['date_obj'].max().to_pydatetime()

# 預設選取最近 1 年 (比較適中的範圍)
default_start = max_date - timedelta(days=365)
if default_start < min_date: default_start = min_date

start_date, end_date = st.sidebar.slider(
    "載入資料範圍 (滑桿控制總量，圖表內可縮放)",
    min_value=min_date,
    max_value=max_date,
    value=(default_start, max_date),
    format="YYYY-MM-DD"
)

df = full_df[(full_df['date_obj'] >= start_date) & (full_df['date_obj'] <= end_date)]

if df.empty:
    st.warning("選取區間無數據")
    st.stop()

# ---------------------------------------------------------
# 4. 數據打包 (包含空白填充)
# ---------------------------------------------------------
COLOR_UP = '#FF5252'
COLOR_DOWN = '#00B746'

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
    
    if is_safe(row['open']) and is_safe(row['close']):
        candles.append({
            'time': t, 
            'open': float(row['open']), 'high': float(row['high']), 
            'low': float(row['low']), 'close': float(row['close'])
        })
    else:
        continue 

    if is_safe(row['volume']):
        bar_color = COLOR_UP if row['close'] >= row['open'] else COLOR_DOWN
        vols.append({'time': t, 'value': float(row['volume']), 'color': bar_color})
    else:
        vols.append({'time': t, 'value': 0, 'color': 'rgba(0,0,0,0)'})

    ma5.append({'time': t, 'value': float(row['ma5'])} if is_safe(row.get('ma5')) else {'time': t})
    ma10.append({'time': t, 'value': float(row['ma10'])} if is_safe(row.get('ma10')) else {'time': t})
    ma20.append({'time': t, 'value': float(row['ma20'])} if is_safe(row.get('ma20')) else {'time': t})
    
    bbu.append({'time': t, 'value': float(row['bbu_20_2.0'])} if is_safe(row.get('bbu_20_2.0')) else {'time': t})
    bbl.append({'time': t, 'value': float(row['bbl_20_2.0'])} if is_safe(row.get('bbl_20_2.0')) else {'time': t})
    
    macd_dif.append({'time': t, 'value': float(row['macd_12_26_9'])} if is_safe(row.get('macd_12_26_9')) else {'time': t})
    macd_dea.append({'time': t, 'value': float(row['macds_12_26_9'])} if is_safe(row.get('macds_12_26_9')) else {'time': t})
    
    if is_safe(row.get('macdh_12_26_9')):
        hist_val = float(row['macdh_12_26_9'])
        macd_hist.append({'time': t, 'value': hist_val, 'color': COLOR_UP if hist_val > 0 else COLOR_DOWN})
    else:
        macd_hist.append({'time': t})
        
    k_line.append({'time': t, 'value': float(row['stochk_14_3_3'])} if is_safe(row.get('stochk_14_3_3')) else {'time': t})
    d_line.append({'time': t, 'value': float(row['stochd_14_3_3'])} if is_safe(row.get('stochd_14_3_3')) else {'time': t})
    
    rsi_line.append({'time': t, 'value': float(row['rsi'])} if is_safe(row.get('rsi')) else {'time': t})
    obv_line.append({'time': t, 'value': float(row['obv'])} if is_safe(row.get('obv')) else {'time': t})
    bias_line.append({'time': t, 'value': float(row['bias'])} if is_safe(row.get('bias')) else {'time': t})


# ---------------------------------------------------------
# 5. 渲染圖表 (解鎖互動 + 維持對齊)
# ---------------------------------------------------------
common_chart_options = {
    "layout": { "backgroundColor": "#FFFFFF", "textColor": "#333333" },
    "grid": { "vertLines": {"color": "#F0F0F0"}, "horzLines": {"color": "#F0F0F0"} },
    # 右側寬度鎖定
    "rightPriceScale": { 
        "borderColor": "#E0E0E0", 
        "scaleMargins": {"top": 0.1, "bottom": 0.1},
        "minimumWidth": 120, 
        "visible": True,
    },
    "leftPriceScale": { "visible": False },
    "timeScale": { "borderColor": "#E0E0E0", "timeVisible": True, "rightOffset": 12 },
    
    # 【關鍵修改】重新開放互動功能，但保留手機防誤觸
    "handleScroll": { 
        "mouseWheel": True,      # 允許滑鼠滾輪縮放
        "pressedMouseMove": True,# 允許按住拖曳
        "horzTouchDrag": True,   # 允許手機左右滑動 K 線
        "vertTouchDrag": False   # 禁止手機上下滑動 K 線 (避免網頁卡住)
    },
    "handleScale": {
        "axisPressedMouseMove": True, 
        "mouseWheel": True, 
        "pinch": True # 允許手機雙指縮放
    },
}

format_2f = {"type": "price", "precision": 2, "minMove": 0.01}
format_volume = {"type": "volume"} 

panes = []

# Pane 0
series_main = [
    {"type": "Candlestick", "data": candles, "options": {"upColor": COLOR_UP, "downColor": COLOR_DOWN, "borderUpColor": COLOR_UP, "borderDownColor": COLOR_DOWN, "wickUpColor": COLOR_UP, "wickDownColor": COLOR_DOWN}}
]
if ma5: series_main.append({"type": "Line", "data": ma5, "options": {"color": '#FFA500', "lineWidth": 1, "title": "MA5", "lastValueVisible": False, "priceLineVisible": False}})
if ma10: series_main.append({"type": "Line", "data": ma10, "options": {"color": '#40E0D0', "lineWidth": 1, "title": "MA10", "lastValueVisible": False, "priceLineVisible": False}})
if ma20: series_main.append({"type": "Line", "data": ma20, "options": {"color": '#9370DB', "lineWidth": 2, "title": "MA20", "lastValueVisible": False, "priceLineVisible": False}})
if bbu: series_main.append({"type": "Line", "data": bbu, "options": {"color": "rgba(0, 0, 255, 0.3)", "lineWidth": 1, "lineStyle": 2, "lastValueVisible": False, "priceLineVisible": False}})
if bbl: series_main.append({"type": "Line", "data": bbl, "options": {"color": "rgba(0, 0, 255, 0.3)", "lineWidth": 1, "lineStyle": 2, "lastValueVisible": False, "priceLineVisible": False}})
panes.append({"chart": common_chart_options, "series": series_main, "height": 400})

# Pane 1-N
if vols: panes.append({"chart": common_chart_options, "series": [{"type": "Histogram", "data": vols, "options": {"priceFormat": {"type": "volume"}, "title": "成交量 (Vol)"}}], "height": 100})

macd_series = []
if macd_dif: macd_series.append({"type": "Line", "data": macd_dif, "options": {"color": "#2962FF", "lineWidth": 1, "title": "DIF", "priceFormat": format_2f}})
if macd_dea: macd_series.append({"type": "Line", "data": macd_dea, "options": {"color": "#FF6D00", "lineWidth": 1, "title": "DEA", "priceFormat": format_2f}})
if macd_hist: macd_series.append({"type": "Histogram", "data": macd_hist, "options": {"title": "MACD", "priceFormat": format_2f}})
if macd_series: panes.append({"chart": common_chart_options, "series": macd_series, "height": 150})

kdj_series = []
if k_line: kdj_series.append({"type": "Line", "data": k_line, "options": {"color": "#E91E63", "title": "K", "priceFormat": format_2f}})
if d_line: kdj_series.append({"type": "Line", "data": d_line, "options": {"color": "#2196F3", "title": "D", "priceFormat": format_2f}})
if kdj_series: panes.append({"chart": common_chart_options, "series": kdj_series, "height": 100})

if rsi_line: panes.append({"chart": common_chart_options, "series": [{"type": "Line", "data": rsi_line, "options": {"color": "#9C27B0", "title": "RSI(14)", "priceFormat": format_2f}}], "height": 100})
if obv_line: panes.append({"chart": common_chart_options, "series": [{"type": "Line", "data": obv_line, "options": {"color": "#FF9800", "title": "OBV", "priceFormat": format_volume}}], "height": 100})
if bias_line: panes.append({"chart": common_chart_options, "series": [{"type": "Line", "data": bias_line, "options": {"color": "#607D8B", "title": "乖離率", "priceFormat": format_2f}}], "height": 100})

st.markdown("### 📊 技術分析圖表 (混合互動模式)")
if len(candles) > 0:
    renderLightweightCharts(panes, key=f"final_v5_hybrid_{start_date}_{end_date}")
else:
    st.error("錯誤：無數據")
