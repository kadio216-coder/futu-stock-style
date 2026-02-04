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
st.set_page_config(layout="wide", page_title="Futu Desktop Replica")

st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 1rem; padding-left: 1rem; padding-right: 1rem;}
    h3 {margin-bottom: 0px;}
    .stRadio > div {flex-direction: row;} 
    div[data-testid="column"] {background-color: #FAFAFA; padding: 10px; border-radius: 5px;}
    div.stCheckbox {margin-bottom: -10px;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. 資料層
# ---------------------------------------------------------
@st.cache_data(ttl=60)
def get_data(ticker, period="2y", interval="1d"):
    try:
        dl_interval = "1mo" if interval == "1y" else interval
        data = yf.download(ticker, period=period, interval=dl_interval, progress=False)
        
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data.index = data.index.tz_localize(None)
        
        if interval == "1y":
            data = data.resample('YE').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()

        data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
        data.columns = [str(col).lower() for col in data.columns]
        
        close_col = 'close' if 'close' in data.columns else 'adj close'
        if close_col not in data.columns: return None

        # --- 指標計算 ---
        # 1. 均線 (EMA)
        data['MA5'] = ta.ema(data[close_col], length=5)
        data['MA10'] = ta.ema(data[close_col], length=10)
        data['MA20'] = ta.ema(data[close_col], length=20)
        data['MA60'] = ta.ema(data[close_col], length=60)
        
        # 2. 布林 (典型價格)
        data['tp'] = (data['high'] + data['low'] + data[close_col]) / 3
        data['boll_mid'] = data['tp'].rolling(window=20).mean()
        data['boll_std'] = data['tp'].rolling(window=20).std()
        data['boll_upper'] = data['boll_mid'] + (2 * data['boll_std'])
        data['boll_lower'] = data['boll_mid'] - (2 * data['boll_std'])
        
        # 3. 其他
        macd = ta.macd(data[close_col])
        if macd is not None: data = pd.concat([data, macd], axis=1)
        
        data['RSI'] = ta.rsi(data[close_col], length=14)
        
        stoch = ta.stoch(data['high'], data['low'], data[close_col])
        if stoch is not None: data = pd.concat([data, stoch], axis=1)
        
        data['OBV'] = ta.obv(data[close_col], data['volume'])
        data['BIAS'] = (data[close_col] - data['MA20']) / data['MA20'] * 100
        
        # 日期處理
        data = data.reset_index()
        data.columns = [str(col).lower() for col in data.columns]
        
        date_col = None
        for name in ['date', 'datetime', 'timestamp', 'index']:
            if name in data.columns:
                date_col = name
                break
        if date_col is None:
            for col in data.columns:
                if pd.api.types.is_datetime64_any_dtype(data[col]):
                    date_col = col
                    break
        if date_col is None: return None
            
        data['date_obj'] = pd.to_datetime(data[date_col])
        data['time'] = data['date_obj'].astype('int64') // 10**9
            
        return data
    except Exception as e:
        print(f"Error: {e}")
        return None

# ---------------------------------------------------------
# 3. 佈局架構
# ---------------------------------------------------------
with st.sidebar:
    st.header("🔍 股票搜尋")
    market_mode = st.radio("市場", ["台股(市)", "台股(櫃)", "美股"], index=2, horizontal=True)
    raw_symbol = st.text_input("代碼", value="MU")
    
    if market_mode == "台股(市)": ticker = f"{raw_symbol}.TW" if not raw_symbol.upper().endswith(".TW") else raw_symbol
    elif market_mode == "台股(櫃)": ticker = f"{raw_symbol}.TWO" if not raw_symbol.upper().endswith(".TWO") else raw_symbol
    else: ticker = raw_symbol.upper()

col_main, col_tools = st.columns([0.82, 0.18])

# ---------------------------------------------------------
# 4. 右側指標面板
# ---------------------------------------------------------
with col_tools:
    st.markdown("#### ⚙️ 指標")
    st.caption("主圖 (無損切換)")
    show_ma = st.checkbox("MA 均線", value=True)
    show_boll = st.checkbox("BOLL 布林", value=True)
    
    st.divider()
    st.caption("副圖 (切換會重排)")
    show_vol = st.checkbox("VOL 成交量", value=True)
    show_macd = st.checkbox("MACD", value=True)
    show_kdj = st.checkbox("KDJ", value=True)
    show_rsi = st.checkbox("RSI", value=True)
    show_obv = st.checkbox("OBV", value=False)
    show_bias = st.checkbox("BIAS", value=False)

# ---------------------------------------------------------
# 5. 主圖表區
# ---------------------------------------------------------
with col_main:
    c_top1, c_top2 = st.columns([0.6, 0.4])
    with c_top1:
        st.subheader(f"{ticker} 走勢圖")
    with c_top2:
        interval_label = st.radio("週期", ["日K", "週K", "月K", "年K"], index=0, horizontal=True, label_visibility="collapsed")
    
    interval_map = {"日K": "1d", "週K": "1wk", "月K": "1mo", "年K": "1y"}
    full_df = get_data(ticker, period="max", interval=interval_map[interval_label])
    
    if full_df is None:
        st.error(f"無法取得 {ticker} 資料")
        st.stop()
        
    min_d, max_d = full_df['date_obj'].min().to_pydatetime(), full_df['date_obj'].max().to_pydatetime()
    default_start = max_d - timedelta(days=365)
    if default_start < min_d: default_start = min_d
    
    start_date, end_date = st.slider("", min_d, max_d, (default_start, max_d), format="YYYY-MM-DD", label_visibility="collapsed")
    
    df = full_df[(full_df['date_obj'] >= start_date) & (full_df['date_obj'] <= end_date)]
    if df.empty: st.stop()

    # --- 數據打包 ---
    COLOR_UP = '#FF5252'
    COLOR_DOWN = '#00B746'
    
    def is_valid(val): return val is not None and not pd.isna(val) and not np.isinf(val)

    candles, vols = [], []
    ma5, ma10, ma20, ma60 = [], [], [], []
    bbu, bbm, bbl = [], [], [] 
    macd_dif, macd_dea, macd_hist = [], [], []
    k_line, d_line, rsi_line, obv_line, bias_line = [], [], [], [], []

    for _, row in df.iterrows():
        t = int(row['time'])
        if is_valid(row['open']):
            candles.append({'time': t, 'open': row['open'], 'high': row['high'], 'low': row['low'], 'close': row['close']})
        else: continue

        # [策略調整] 主圖指標：永遠計算並加入列表 (只是後面決定顯不顯示)
        # 這樣資料結構不變，Key 不變，縮放就不會重置
        if is_valid(row.get('ma5')): ma5.append({'time': t, 'value': row['ma5']})
        if is_valid(row.get('ma10')): ma10.append({'time': t, 'value': row['ma10']})
        if is_valid(row.get('ma20')): ma20.append({'time': t, 'value': row['ma20']})
        if is_valid(row.get('ma60')): ma60.append({'time': t, 'value': row['ma60']})
        
        if is_valid(row.get('boll_upper')): bbu.append({'time': t, 'value': row['boll_upper']})
        if is_valid(row.get('boll_mid')):   bbm.append({'time': t, 'value': row['boll_mid']})
        if is_valid(row.get('boll_lower')): bbl.append({'time': t, 'value': row['boll_lower']})

        # 副圖指標：只有勾選時才處理 (節省運算，且副圖本來就會重排)
        if show_vol:
            v = row['volume'] if is_valid(row['volume']) else 0
            color = COLOR_UP if row['close'] >= row['open'] else COLOR_DOWN
            vols.append({'time': t, 'value': v, 'color': color})

        if show_macd:
            if is_valid(row.get('macd_12_26_9')): macd_dif.append({'time': t, 'value': row['macd_12_26_9']})
            if is_valid(row.get('macds_12_26_9')): macd_dea.append({'time': t, 'value': row['macds_12_26_9']})
            if is_valid(row.get('macdh_12_26_9')): 
                h = row['macdh_12_26_9']
                macd_hist.append({'time': t, 'value': h, 'color': COLOR_UP if h > 0 else COLOR_DOWN})
        
        if show_kdj:
            if is_valid(row.get('stochk_14_3_3')): k_line.append({'time': t, 'value': row['stochk_14_3_3']})
            if is_valid(row.get('stochd_14_3_3')): d_line.append({'time': t, 'value': row['stochd_14_3_3']})
            
        if show_rsi and is_valid(row.get('rsi')): rsi_line.append({'time': t, 'value': row['rsi']})
        if show_obv and is_valid(row.get('obv')): obv_line.append({'time': t, 'value': row['obv']})
        if show_bias and is_valid(row.get('bias')): bias_line.append({'time': t, 'value': row['bias']})

    # --- 圖表配置 ---
    common_opts = {
        "layout": { "backgroundColor": "#FFFFFF", "textColor": "#333333" },
        "grid": { "vertLines": {"color": "#F0F0F0"}, "horzLines": {"color": "#F0F0F0"} },
        "rightPriceScale": { "borderColor": "#E0E0E0", "visible": True, "minimumWidth": 85 },
        "leftPriceScale": { "visible": False },
        "timeScale": { "borderColor": "#E0E0E0", "rightOffset": 5 },
        "handleScroll": { "mouseWheel": True, "pressedMouseMove": True },
        "handleScale": { "axisPressedMouseMove": True, "mouseWheel": True }
    }
    
    panes = []
    
    # --- 1. 主圖 (使用 visible 屬性控制顯示，不改變資料結構) ---
    series_main = [
        {"type": "Candlestick", "data": candles, "options": {"upColor": COLOR_UP, "downColor": COLOR_DOWN, "borderUpColor": COLOR_UP, "borderDownColor": COLOR_DOWN, "wickUpColor": COLOR_UP, "wickDownColor": COLOR_DOWN}}
    ]
    
    # MA 線永遠加入，但透過 visible: show_ma 控制可見度
    if ma5: series_main.append({"type": "Line", "data": ma5, "options": {"color": '#FFA500', "lineWidth": 1, "title": "EMA5", "visible": show_ma, "priceLineVisible": False, "lastValueVisible": False}})
    if ma10: series_main.append({"type": "Line", "data": ma10, "options": {"color": '#2196F3', "lineWidth": 1, "title": "EMA10", "visible": show_ma, "priceLineVisible": False, "lastValueVisible": False}})
    if ma20: series_main.append({"type": "Line", "data": ma20, "options": {"color": '#E040FB', "lineWidth": 1, "title": "EMA20", "visible": show_ma, "priceLineVisible": False, "lastValueVisible": False}})
    if ma60: series_main.append({"type": "Line", "data": ma60, "options": {"color": '#00E676', "lineWidth": 1, "title": "EMA60", "visible": show_ma, "priceLineVisible": False, "lastValueVisible": False}})
    
    # BOLL 線永遠加入，透過 visible: show_boll 控制
    if bbu: series_main.append({"type": "Line", "data": bbu, "options": {"color": "#2962FF", "lineWidth": 1, "lineStyle": 2, "title": "BBU", "visible": show_boll, "lastValueVisible": False, "priceLineVisible": False}})
    if bbm: series_main.append({"type": "Line", "data": bbm, "options": {"color": "#2962FF", "lineWidth": 1, "lineStyle": 2, "title": "MID", "visible": show_boll, "lastValueVisible": False, "priceLineVisible": False}})
    if bbl: series_main.append({"type": "Line", "data": bbl, "options": {"color": "#2962FF", "lineWidth": 1, "lineStyle": 2, "title": "BBL", "visible": show_boll, "lastValueVisible": False, "priceLineVisible": False}})
        
    panes.append({"chart": common_opts, "series": series_main, "height": 500})
    
    # --- 2. 副圖 (動態增減，必須重置) ---
    format_2f = {"type": "price", "precision": 2, "minMove": 0.01}
    
    if show_vol and vols:
        panes.append({"chart": common_opts, "series": [{"type": "Histogram", "data": vols, "options": {"priceFormat": {"type": "volume"}, "title": "VOL"}}], "height": 120})
        
    if show_macd and macd_dif:
        s_macd = [
            {"type": "Line", "data": macd_dif, "options": {"color": "#FFA500", "lineWidth": 1, "title": "DIF", "priceFormat": format_2f}},
            {"type": "Line", "data": macd_dea, "options": {"color": "#2196F3", "lineWidth": 1, "title": "DEA", "priceFormat": format_2f}},
            {"type": "Histogram", "data": macd_hist, "options": {"title": "MACD", "priceFormat": format_2f}}
        ]
        panes.append({"chart": common_opts, "series": s_macd, "height": 150})
        
    if show_kdj and k_line:
        s_kdj = [
            {"type": "Line", "data": k_line, "options": {"color": "#FFA500", "title": "K", "priceFormat": format_2f}},
            {"type": "Line", "data": d_line, "options": {"color": "#2196F3", "title": "D", "priceFormat": format_2f}}
        ]
        panes.append({"chart": common_opts, "series": s_kdj, "height": 120})
        
    if show_rsi and rsi_line:
        panes.append({"chart": common_opts, "series": [{"type": "Line", "data": rsi_line, "options": {"color": "#E040FB", "title": "RSI", "priceFormat": format_2f}}], "height": 120})
        
    if show_obv and obv_line:
        panes.append({"chart": common_opts, "series": [{"type": "Line", "data": obv_line, "options": {"color": "#FFA500", "title": "OBV", "priceFormat": {"type": "volume"}}}], "height": 120})

    if show_bias and bias_line:
        panes.append({"chart": common_opts, "series": [{"type": "Line", "data": bias_line, "options": {"color": "#607D8B", "title": "BIAS", "priceFormat": format_2f}}], "height": 120})

    # --- 關鍵 Key 邏輯 ---
    # 1. 基礎 Key：股票、週期、日期
    base_key = f"{ticker}_{interval_label}_{start_date}_{end_date}"
    
    # 2. 面板結構 Key：只包含會改變「副圖數量」的變數
    # 注意！這裡 【不包含】 show_ma 和 show_boll
    # 這樣當你切換主圖指標時，Key 不變，Streamlit 就不會重置圖表
    structure_key = f"{show_vol}_{show_macd}_{show_kdj}_{show_rsi}_{show_obv}_{show_bias}"
    
    st_key = f"desk_v80_{base_key}_{structure_key}"
    
    if len(candles) > 0:
        renderLightweightCharts(panes, key=st_key)
    else:
        st.warning("目前範圍無 K 線數據")
