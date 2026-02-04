import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from streamlit_lightweight_charts import renderLightweightCharts

# ---------------------------------------------------------
# 1. 頁面設定
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Futu Desktop Replica (UI++)")

# 注入 CSS：打造「質感按鈕」與「狀態回饋」
# 這裡我們覆寫了 stButton 的樣式，讓它看起來更像看盤軟體的快捷鍵
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 1rem; padding-left: 1rem; padding-right: 1rem;}
    h3 {margin-bottom: 0px;}
    .stRadio > div {flex-direction: row;} 
    div[data-testid="column"] {background-color: #FAFAFA; padding: 10px; border-radius: 5px;}
    div.stCheckbox {margin-bottom: -10px;}
    
    /* --- 按鈕質感優化核心 --- */
    div.stButton > button {
        width: 100%;
        border-radius: 20px; /* 圓角膠囊狀 */
        border: none;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.2s ease; /* 平滑過渡動畫 */
        padding: 0.25rem 0.5rem;
    }

    /* 未選中狀態 (Secondary) - 類似富途的淺灰底 */
    div.stButton > button[kind="secondary"] {
        background-color: #F0F2F5;
        color: #666666;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: #E1E4E8;
        color: #333333;
        border: none;
    }

    /* 選中狀態 (Primary) - 富途牛牛的經典藍/橘風格 */
    div.stButton > button[kind="primary"] {
        background-color: #2962FF; /* 專業深藍 */
        color: white;
        box-shadow: 0 2px 5px rgba(41, 98, 255, 0.3); /* 微微的陰影增加立體感 */
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #1E46BE;
        border: none;
    }
    /* ------------------------ */
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

        # 指標
        data['MA5'] = ta.ema(data[close_col], length=5)
        data['MA10'] = ta.ema(data[close_col], length=10)
        data['MA20'] = ta.ema(data[close_col], length=20)
        data['MA60'] = ta.ema(data[close_col], length=60)
        
        data['tp'] = (data['high'] + data['low'] + data[close_col]) / 3
        data['boll_mid'] = data['tp'].rolling(window=20).mean()
        data['boll_std'] = data['tp'].rolling(window=20).std()
        data['boll_upper'] = data['boll_mid'] + (2 * data['boll_std'])
        data['boll_lower'] = data['boll_mid'] - (2 * data['boll_std'])
        
        macd = ta.macd(data[close_col])
        if macd is not None: data = pd.concat([data, macd], axis=1)
        data['RSI'] = ta.rsi(data[close_col], length=14)
        stoch = ta.stoch(data['high'], data['low'], data[close_col])
        if stoch is not None: data = pd.concat([data, stoch], axis=1)
        data['OBV'] = ta.obv(data[close_col], data['volume'])
        data['BIAS'] = (data[close_col] - data['MA20']) / data['MA20'] * 100
        
        # 日期
        data = data.reset_index()
        data.columns = [str(col).lower() for col in data.columns]
        
        date_col = None
        for name in ['date', 'datetime', 'timestamp', 'index']:
            if name in data.columns:
                date_col = name; break
        if date_col is None:
            for col in data.columns:
                if pd.api.types.is_datetime64_any_dtype(data[col]):
                    date_col = col; break
        if date_col is None: return None
            
        data['date_obj'] = pd.to_datetime(data[date_col])
        data['time'] = data['date_obj'].astype('int64') // 10**9
            
        return data
    except Exception as e:
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

with col_tools:
    st.markdown("#### ⚙️ 指標")
    st.caption("主圖")
    show_ma = st.checkbox("MA (EMA)", value=True)
    show_boll = st.checkbox("BOLL", value=True)
    st.divider()
    st.caption("副圖")
    show_vol = st.checkbox("VOL 成交量", value=True)
    show_macd = st.checkbox("MACD", value=True)
    show_kdj = st.checkbox("KDJ", value=True)
    show_rsi = st.checkbox("RSI", value=True)
    show_obv = st.checkbox("OBV", value=False)
    show_bias = st.checkbox("BIAS", value=False)

with col_main:
    c_top1, c_top2 = st.columns([0.6, 0.4])
    with c_top1: st.subheader(f"{ticker} 走勢圖")
    with c_top2: interval_label = st.radio("週期", ["日K", "週K", "月K", "年K"], index=0, horizontal=True, label_visibility="collapsed")
    
    interval_map = {"日K": "1d", "週K": "1wk", "月K": "1mo", "年K": "1y"}
    full_df = get_data(ticker, period="max", interval=interval_map[interval_label])
    
    if full_df is None:
        st.error(f"無數據: {ticker}")
        st.stop()
        
    min_d, max_d = full_df['date_obj'].min().to_pydatetime(), full_df['date_obj'].max().to_pydatetime()
    
    # --- 【UI 質感升級】快捷區間選擇器 ---
    
    # 1. 初始化狀態：記錄哪個按鈕是「活躍 (Active)」的
    if 'active_btn' not in st.session_state:
        st.session_state['active_btn'] = '6m' # 預設 6個月
        
    if 'slider_range' not in st.session_state:
        default_start = max_d - timedelta(days=180)
        if default_start < min_d: default_start = min_d
        st.session_state['slider_range'] = (default_start, max_d)

    # 2. 定義按鈕邏輯
    def handle_btn_click(btn_key, months=0, years=0, ytd=False, is_max=False):
        # 更新活躍按鈕
        st.session_state['active_btn'] = btn_key
        
        # 更新時間
        end = max_d
        if is_max:
            start = min_d
        elif ytd:
            start = datetime(end.year, 1, 1)
            if start < min_d: start = min_d
        else:
            start = end - relativedelta(months=months, years=years)
            if start < min_d: start = min_d
        st.session_state['slider_range'] = (start, end)

    # 3. 渲染按鈕 (使用 columns 排版)
    # 我們根據 active_btn 來決定按鈕是 'primary' (深藍色/選中) 還是 'secondary' (灰色/未選中)
    btn_cols = st.columns(7)
    
    # 按鈕配置列表
    buttons = [
        {"label": "1月", "key": "1m", "m": 1, "y": 0, "ytd": False, "max": False},
        {"label": "3月", "key": "3m", "m": 3, "y": 0, "ytd": False, "max": False},
        {"label": "6月", "key": "6m", "m": 6, "y": 0, "ytd": False, "max": False},
        {"label": "1年", "key": "1y", "m": 0, "y": 1, "ytd": False, "max": False},
        {"label": "3年", "key": "3y", "m": 0, "y": 3, "ytd": False, "max": False},
        {"label": "今年", "key": "ytd", "m": 0, "y": 0, "ytd": True, "max": False},
        {"label": "最大", "key": "max", "m": 0, "y": 0, "ytd": False, "max": True},
    ]

    for i, btn in enumerate(buttons):
        with btn_cols[i]:
            # 判斷是否為當前活躍按鈕
            is_active = (st.session_state['active_btn'] == btn['key'])
            # 渲染按鈕
            if st.button(
                btn['label'], 
                key=f"btn_{btn['key']}", 
                type="primary" if is_active else "secondary", # 這裡控制顏色！
                use_container_width=True
            ):
                handle_btn_click(btn['key'], months=btn['m'], years=btn['y'], ytd=btn['ytd'], is_max=btn['max'])
                st.rerun() # 強制刷新以更新按鈕顏色

    # --- 雙向滑桿 ---
    # 如果使用者手動拖了滑桿，我們就把 active_btn 清空，表示「自定義模式」
    def on_slider_change():
        st.session_state['active_btn'] = None

    start_date, end_date = st.slider(
        "", 
        min_value=min_d, 
        max_value=max_d, 
        key='slider_range', 
        on_change=on_slider_change, # 偵測手動拖曳
        format="YYYY-MM-DD", 
        label_visibility="collapsed"
    )
    
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

        if show_vol:
            v = row['volume'] if is_valid(row['volume']) else 0
            color = COLOR_UP if row['close'] >= row['open'] else COLOR_DOWN
            vols.append({'time': t, 'value': v, 'color': color})
            
        if show_ma:
            if is_valid(row.get('ma5')): ma5.append({'time': t, 'value': row['ma5']})
            if is_valid(row.get('ma10')): ma10.append({'time': t, 'value': row['ma10']})
            if is_valid(row.get('ma20')): ma20.append({'time': t, 'value': row['ma20']})
            if is_valid(row.get('ma60')): ma60.append({'time': t, 'value': row['ma60']})
            
        if show_boll:
            if is_valid(row.get('boll_upper')): bbu.append({'time': t, 'value': row['boll_upper']})
            if is_valid(row.get('boll_mid')):   bbm.append({'time': t, 'value': row['boll_mid']})
            if is_valid(row.get('boll_lower')): bbl.append({'time': t, 'value': row['boll_lower']})

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
    
    format_2f = {"type": "price", "precision": 2, "minMove": 0.01}
    
    panes = []
    
    # 1. 主圖
    series_main = [
        {"type": "Candlestick", "data": candles, "options": {"upColor": COLOR_UP, "downColor": COLOR_DOWN, "borderUpColor": COLOR_UP, "borderDownColor": COLOR_DOWN, "wickUpColor": COLOR_UP, "wickDownColor": COLOR_DOWN}}
    ]
    
    if show_ma:
        if ma5: series_main.append({"type": "Line", "data": ma5, "options": {"color": '#FFA500', "lineWidth": 1, "title": "EMA5", "priceLineVisible": False, "lastValueVisible": False}})
        if ma10: series_main.append({"type": "Line", "data": ma10, "options": {"color": '#2196F3', "lineWidth": 1, "title": "EMA10", "priceLineVisible": False, "lastValueVisible": False}})
        if ma20: series_main.append({"type": "Line", "data": ma20, "options": {"color": '#E040FB', "lineWidth": 1, "title": "EMA20", "priceLineVisible": False, "lastValueVisible": False}})
        if ma60: series_main.append({"type": "Line", "data": ma60, "options": {"color": '#00E676', "lineWidth": 1, "title": "EMA60", "priceLineVisible": False, "lastValueVisible": False}})
    
    if show_boll:
        if bbu: series_main.append({"type": "Line", "data": bbu, "options": {"color": "#2962FF", "lineWidth": 1, "lineStyle": 2, "title": "BBU", "priceLineVisible": False, "lastValueVisible": False}})
        if bbm: series_main.append({"type": "Line", "data": bbm, "options": {"color": "#2962FF", "lineWidth": 1, "lineStyle": 2, "title": "MID", "priceLineVisible": False, "lastValueVisible": False}})
        if bbl: series_main.append({"type": "Line", "data": bbl, "options": {"color": "#2962FF", "lineWidth": 1, "lineStyle": 2, "title": "BBL", "priceLineVisible": False, "lastValueVisible": False}})
        
    panes.append({"chart": common_opts, "series": series_main, "height": 500})
    
    # 2. 副圖
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

    st_key = f"desk_v103_{ticker}_{interval_label}_{start_date}_{end_date}_{show_ma}_{show_boll}_{show_vol}_{show_macd}_{show_kdj}_{show_rsi}_{show_obv}_{show_bias}"
    
    if len(candles) > 0:
        renderLightweightCharts(panes, key=st_key)
    else:
        st.warning("目前範圍無 K 線數據")
