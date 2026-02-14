import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import streamlit.components.v1 as components

# ---------------------------------------------------------
# 1. 頁面設定
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Futu Desktop Replica (Final)")

st.markdown("""
<style>
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    h3 {margin-bottom: 0px;}
    div[data-testid="column"] {background-color: #FAFAFA; padding: 10px; border-radius: 5px;}
    
    div.stButton > button {
        width: 100%;
        border-radius: 20px;
        border: none;
        font-weight: 600;
        font-size: 14px;
        padding: 0.25rem 0.5rem;
    }
    div.stButton > button[kind="secondary"] {background-color: #F0F2F5; color: #666666;}
    div.stButton > button[kind="primary"] {background-color: #2962FF !important; color: white !important;}
    
    /* 策略儀表板樣式 */
    .strategy-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr); 
        gap: 10px;
        margin-bottom: 15px;
    }
    .strat-card {
        background-color: #fff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px 10px; 
        text-align: center;
        font-family: "Segoe UI", sans-serif;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .strat-active {
        background-color: #E8F5E9; /* 亮綠底 */
        border: 1px solid #4CAF50;
    }
    .strat-title { font-size: 14px; color: #666; font-weight: 600; margin-bottom: 5px; }
    .strat-status { font-size: 16px; font-weight: bold; color: #333; }
    
    .status-match { color: #2E7D32; } /* 符合條件 */
    .status-wait { color: #9E9E9E; }  /* 未符合 */
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. 介面控制 & 變數定義
# ---------------------------------------------------------
with st.sidebar:
    st.header("🔍 股票搜尋")
    market_mode = st.radio("市場", ["台股(市)", "台股(櫃)", "美股"], index=2, horizontal=True)
    raw_symbol = st.text_input("代碼", value="2330")
    
    if market_mode == "台股(市)": ticker = f"{raw_symbol}.TW" if not raw_symbol.upper().endswith(".TW") else raw_symbol
    elif market_mode == "台股(櫃)": ticker = f"{raw_symbol}.TWO" if not raw_symbol.upper().endswith(".TWO") else raw_symbol
    else: ticker = raw_symbol.upper()
    
    is_tw_stock = ticker.endswith('.TW') or ticker.endswith('.TWO')

# ---------------------------------------------------------
# 3. 資料層
# ---------------------------------------------------------
@st.cache_data(ttl=60)
def get_data(ticker, period="2y", interval="1d"):
    try:
        is_quarterly = (interval == "3mo")
        dl_interval = "1mo" if (interval == "1y" or is_quarterly) else interval
        
        data = yf.download(ticker, period=period, interval=dl_interval, progress=False)
        if data.empty: return None
        
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data.index = data.index.tz_localize(None)
        
        data.columns = [c.capitalize() for c in data.columns]
        
        if interval == "1y":
            data = data.resample('YE').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
        elif is_quarterly:
            data = data.resample('QE').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()

        data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
        data.columns = [str(col).lower() for col in data.columns]
        close_col = 'close' if 'close' in data.columns else 'adj close'
        if close_col not in data.columns: return None

        if ticker.endswith('.TW') or ticker.endswith('.TWO'):
            data['volume'] = data['volume'] / 1000

        # --- 指標計算 ---
        data['MA5'] = ta.sma(data[close_col], length=5)
        data['MA10'] = ta.sma(data[close_col], length=10)
        data['MA20'] = ta.sma(data[close_col], length=20)
        data['MA60'] = ta.sma(data[close_col], length=60)
        data['MA120'] = ta.sma(data[close_col], length=120)
        
        data['boll_mid'] = data[close_col].rolling(window=20).mean()
        data['boll_std'] = data[close_col].rolling(window=20).std()
        data['boll_upper'] = data['boll_mid'] + (2 * data['boll_std'])
        data['boll_lower'] = data['boll_mid'] - (2 * data['boll_std'])
        
        macd = ta.macd(data[close_col], fast=12, slow=26, signal=9)
        if macd is not None: data = pd.concat([data, macd], axis=1)
        
        low_list = data['low'].rolling(9, min_periods=1).min()
        high_list = data['high'].rolling(9, min_periods=1).max()
        rsv = (data[close_col] - low_list) / (high_list - low_list) * 100
        data['k'] = rsv.ewm(alpha=1/3, adjust=False).mean()
        data['d'] = data['k'].ewm(alpha=1/3, adjust=False).mean()
        data['j'] = 3 * data['k'] - 2 * data['d']

        data['RSI6'] = ta.rsi(data[close_col], length=6)
        data['RSI12'] = ta.rsi(data[close_col], length=12)
        data['RSI24'] = ta.rsi(data[close_col], length=24)

        sma6 = ta.sma(data[close_col], length=6)
        sma12 = ta.sma(data[close_col], length=12)
        sma24 = ta.sma(data[close_col], length=24)
        data['BIAS6'] = (data[close_col] - sma6) / sma6 * 100
        data['BIAS12'] = (data[close_col] - sma12) / sma12 * 100
        data['BIAS24'] = (data[close_col] - sma24) / sma24 * 100

        # ★ V109 修正：移除 data = data.tail(130).copy() 封印，釋放完整歷史資料

        # ★ 重算 OBV
        data['OBV'] = ta.obv(data[close_col], data['volume'])
        data['OBV_MA10'] = ta.sma(data['OBV'], length=10)
        
        data = data.reset_index()
        data.columns = [str(col).lower() for col in data.columns]
        
        date_col = None
        for name in ['date', 'datetime', 'timestamp', 'index']:
            if name in data.columns: date_col = name; break
        if date_col is None:
            for col in data.columns:
                if pd.api.types.is_datetime64_any_dtype(data[col]): date_col = col; break
        if date_col is None: return None
            
        data['date_obj'] = pd.to_datetime(data[date_col])
        data['time'] = data['date_obj'].astype('int64') // 10**9 
        data = data.sort_values('time')
        
        return data
    except Exception as e:
        print(f"Data Error: {e}")
        return None

# --- ★ 四大策略偵測邏輯 ---
def check_4_strategies(df):
    if len(df) < 30: return {}
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    results = {}
    
    # 1. 盤整後帶量突破
    past_20 = df.iloc[-21:-1]
    box_high = past_20['high'].max()
    box_low = past_20['low'].min()
    amp = (box_high - box_low) / box_low
    vol_ma5 = df['volume'].iloc[-6:-1].mean()
    if vol_ma5 == 0: vol_ma5 = 1
    
    cond1_box = amp < 0.15
    cond1_break = curr['close'] > box_high
    cond1_vol = curr['volume'] > (vol_ma5 * 2)
    
    if cond1_box and cond1_break and cond1_vol:
        results['S1'] = {'active': True, 'msg': '🚀 帶量突破'}
    elif not cond1_box:
        results['S1'] = {'active': False, 'msg': '波動過大'}
    else:
        results['S1'] = {'active': False, 'msg': '整理中'}

    # 2. 均線黃金交叉
    cond2_cross = (prev['ma20'] < prev['ma60']) and (curr['ma20'] > curr['ma60'])
    cond2_trend = curr['close'] > curr['ma120']
    
    if cond2_cross and cond2_trend:
        results['S2'] = {'active': True, 'msg': '🌟 黃金交叉'}
    elif curr['ma20'] > curr['ma60']:
        results['S2'] = {'active': False, 'msg': '多頭排列'}
    else:
        results['S2'] = {'active': False, 'msg': '空頭/整理'}

    # 3. 布林通道擠壓
    bw = (curr['boll_upper'] - curr['boll_lower']) / curr['boll_mid']
    cond3_squeeze = bw < 0.10
    cond3_break = curr['close'] > curr['boll_upper']
    
    if cond3_squeeze and cond3_break:
        results['S3'] = {'active': True, 'msg': '💥 擠壓噴出'}
    elif cond3_squeeze:
        results['S3'] = {'active': False, 'msg': '壓縮蓄力'}
    else:
        results['S3'] = {'active': False, 'msg': '通道張開'}

    # 4. KD 低檔黃金交叉
    cond4_low = curr['k'] < 20
    cond4_cross = (prev['k'] < prev['d']) and (curr['k'] > curr['d'])
    
    if cond4_low and cond4_cross:
        results['S4'] = {'active': True, 'msg': '🎣 低檔金叉'}
    elif curr['k'] < 20:
        results['S4'] = {'active': False, 'msg': '超賣鈍化'}
    else:
        results['S4'] = {'active': False, 'msg': '一般區間'}
    
    return results

col_main, col_tools = st.columns([0.85, 0.15])

with col_tools:
    st.markdown("#### ⚙️ 指標")
    st.caption("主圖")
    show_ma = st.checkbox("MA (SMA)", value=True)
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
    with c_top1: st.markdown(f"### {ticker} 走勢圖")
    with c_top2: interval_label = st.radio("週期", ["日K", "週K", "月K", "季K", "年K"], index=0, horizontal=True, label_visibility="collapsed")
    
    interval_map = {"日K": "1d", "週K": "1wk", "月K": "1mo", "季K": "3mo", "年K": "1y"}
    full_df = get_data(ticker, period="2y", interval=interval_map[interval_label])
    
    if full_df is None:
        st.error(f"無數據: {ticker}")
        st.stop()
    
    strats = check_4_strategies(full_df)
    if strats:
        s1 = strats['S1']
        s2 = strats['S2']
        s3 = strats['S3']
        s4 = strats['S4']
        
        st.markdown(f"""
        <div class="strategy-grid">
            <div class="strat-card {'strat-active' if s1['active'] else ''}">
                <div class="strat-title">1. 盤整帶量突破</div>
                <div class="strat-status { 'status-match' if s1['active'] else 'status-wait' }">{s1['msg']}</div>
            </div>
            <div class="strat-card {'strat-active' if s2['active'] else ''}">
                <div class="strat-title">2. 均線黃金交叉</div>
                <div class="strat-status { 'status-match' if s2['active'] else 'status-wait' }">{s2['msg']}</div>
            </div>
            <div class="strat-card {'strat-active' if s3['active'] else ''}">
                <div class="strat-title">3. 布林通道擠壓</div>
                <div class="strat-status { 'status-match' if s3['active'] else 'status-wait' }">{s3['msg']}</div>
            </div>
            <div class="strat-card {'strat-active' if s4['active'] else ''}">
                <div class="strat-title">4. KD 低檔黃金交叉</div>
                <div class="strat-status { 'status-match' if s4['active'] else 'status-wait' }">{s4['msg']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    min_d, max_d = full_df['date_obj'].min().to_pydatetime(), full_df['date_obj'].max().to_pydatetime()
    
    if 'active_btn' not in st.session_state: st.session_state['active_btn'] = '6m' 
    if 'slider_range' not in st.session_state:
        st.session_state['slider_range'] = (min_d, max_d)

    def handle_btn_click(btn_key, months=0, years=0, ytd=False, is_max=False):
        st.session_state['active_btn'] = btn_key
        end = max_d
        start = min_d 
        st.session_state['slider_range'] = (start, end)

    btn_cols = st.columns(7)
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
            is_active = (st.session_state['active_btn'] == btn['key'])
            if st.button(btn['label'], key=f"btn_{btn['key']}", type="primary" if is_active else "secondary", use_container_width=True):
                handle_btn_click(btn['key'], months=btn['m'], years=btn['y'], ytd=btn['ytd'], is_max=btn['max'])
                st.rerun()

    def on_slider_change(): st.session_state['active_btn'] = None
    start_date, end_date = st.slider("", min_value=min_d, max_value=max_d, key='slider_range', on_change=on_slider_change, format="YYYY-MM-DD", label_visibility="collapsed")
    
    df = full_df[(full_df['date_obj'] >= start_date) & (full_df['date_obj'] <= end_date)]
    if df.empty: st.stop()

    def to_json_list(df, cols):
        res = []
        df_clean = df.where(pd.notnull(df), None)
        for _, row in df_clean.iterrows():
            try:
                item = {'time': int(row['time'])}
                has_data = False
                for k, v in cols.items():
                    val = row.get(v)
                    if k in ['open','high','low','close'] and val is None:
                        has_data = False; break
                    item[k] = float(val) if val is not None else None
                    has_data = True
                if has_data: res.append(item)
            except: continue
        return json.dumps(res)

    candles_json = to_json_list(df, {'open':'open', 'high':'high', 'low':'low', 'close':'close'})
    
    vol_data_list = []
    if show_vol:
        for _, row in df.iterrows():
            try:
                v = row.get('volume')
                c = row.get('close')
                o = row.get('open')
                if pd.notnull(v) and pd.notnull(c) and pd.notnull(o):
                    color = '#FF5252' if c >= o else '#00B746'
                    vol_data_list.append({'time': int(row['time']), 'value': float(v), 'color': color})
                else:
                    vol_data_list.append({'time': int(row['time']), 'value': None}) 
            except: continue
    vol_json = json.dumps(vol_data_list)
    
    macd_data_list = []
    if show_macd:
        for _, row in df.iterrows():
            try:
                dif = row.get('macd_12_26_9')
                dea = row.get('macds_12_26_9')
                hist = row.get('macdh_12_26_9')
                item = {'time': int(row['time'])}
                if pd.notnull(dif) and pd.notnull(dea) and pd.notnull(hist):
                    item.update({'dif': float(dif), 'dea': float(dea), 'hist': float(hist), 'color': '#FF5252' if hist >= 0 else '#00B746'})
                else:
                    item.update({'dif': None, 'dea': None, 'hist': None})
                macd_data_list.append(item)
            except: continue
    macd_json = json.dumps(macd_data_list)
    
    ma_json = to_json_list(df, {'ma5':'ma5', 'ma10':'ma10', 'ma20':'ma20', 'ma60':'ma60'}) if show_ma else "[]"
    boll_json = to_json_list(df, {'up':'boll_upper', 'mid':'boll_mid', 'low':'boll_lower'}) if show_boll else "[]"
    kdj_json = to_json_list(df, {'k':'k', 'd':'d', 'j':'j'}) if show_kdj else "[]"
    rsi_json = to_json_list(df, {'rsi6':'rsi6', 'rsi12':'rsi12', 'rsi24':'rsi24'}) if show_rsi else "[]"
    
    obv_data_list = []
    if show_obv:
        for _, row in df.iterrows():
            item = {'time': int(row['time'])}
            val = row.get('obv')
            ma_val = row.get('obv_ma10')
            if pd.notnull(val): item['obv'] = float(val)
            else: item['obv'] = None
            if pd.notnull(ma_val): item['obv_ma'] = float(ma_val)
            else: item['obv_ma'] = None
            obv_data_list.append(item)
    obv_json = json.dumps(obv_data_list)
    
    bias_json = to_json_list(df, {'b6':'bias6', 'b12':'bias12', 'b24':'bias24'}) if show_bias else "[]"

    # ---------------------------------------------------------
    # 5. JavaScript (★ V109 繼承 V108 的格式分離邏輯)
    # ---------------------------------------------------------
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/lightweight-charts@3.8.0/dist/lightweight-charts.standalone.production.js"></script>
        <style>
            body {{ margin: 0; padding: 0; background-color: #ffffff; overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; }}
            
            .sub-chart {{
                background-color: #FFFFFF;
                background-image: linear-gradient(to right, #FAFAFA calc(100% - 60px), transparent calc(100% - 60px));
                background-size: 100% calc(100% - 30px);
                background-repeat: no-repeat;
                border-bottom: 1px solid #E0E0E0;
                margin-bottom: 10px;
            }}
            
            .chart-container {{ position: relative; width: 100%; }}
            
            .legend {{
                position: absolute; top: 10px; left: 10px; z-index: 100;
                font-size: 11px; 
                line-height: 16px; 
                font-weight: 500; pointer-events: none;
            }}
            .legend-small {{
                font-size: 11.5px; 
                line-height: 16px;
            }}
            
            .legend-row {{ display: flex; gap: 10px; margin-bottom: 2px; }}
            .legend-label {{ font-weight: bold; color: #333; margin-right: 5px; }}
            .legend-value {{ font-family: 'Consolas', 'Monaco', monospace; }}
        </style>
    </head>
    <body>
        <div id="main-chart" class="chart-container" style="height: 450px; border-bottom: 1px solid #E0E0E0; margin-bottom: 10px;">
            <div id="main-legend" class="legend"></div>
        </div>
        
        <div id="vol-chart" class="chart-container sub-chart" style="height: {'100px' if show_vol else '0px'}; display: {'block' if show_vol else 'none'};">
            <div id="vol-legend" class="legend legend-small"></div>
        </div>
        <div id="macd-chart" class="chart-container sub-chart" style="height: {'150px' if show_macd else '0px'}; display: {'block' if show_macd else 'none'};">
            <div id="macd-legend" class="legend"></div>
        </div>
        <div id="kdj-chart" class="chart-container sub-chart" style="height: {'120px' if show_kdj else '0px'}; display: {'block' if show_kdj else 'none'};">
            <div id="kdj-legend" class="legend"></div>
        </div>
        <div id="rsi-chart" class="chart-container sub-chart" style="height: {'120px' if show_rsi else '0px'}; display: {'block' if show_rsi else 'none'};">
            <div id="rsi-legend" class="legend"></div>
        </div>
        <div id="obv-chart" class="chart-container sub-chart" style="height: {'120px' if show_obv else '0px'}; display: {'block' if show_obv else 'none'};">
            <div id="obv-legend" class="legend legend-small"></div>
        </div>
        <div id="bias-chart" class="chart-container sub-chart" style="height: {'120px' if show_bias else '0px'}; display: {'block' if show_bias else 'none'};">
            <div id="bias-legend" class="legend"></div>
        </div>

        <script>
            try {{
                const candlesData = {candles_json};
                const maData = {ma_json};
                const bollData = {boll_json};
                const volData = {vol_json};
                const macdData = {macd_json};
                const kdjData = {kdj_json};
                const rsiData = {rsi_json};
                const obvData = {obv_json};
                const biasData = {bias_json};
                const isTW = {str(is_tw_stock).lower()};

                if (!candlesData || candlesData.length === 0) throw new Error("No Data");

                const FORCE_WIDTH = 60;
                
                const lineOpts = {{ lineWidth: 1, priceLineVisible: false, lastValueVisible: false }};
                const mainLayout = {{ backgroundColor: '#FFFFFF', textColor: '#333333', fontSize: 13.5 }};
                const indicatorLayout = {{ backgroundColor: 'transparent', textColor: '#333333', fontSize: 14 }};
                const volObvLayout = {{ backgroundColor: 'transparent', textColor: '#333333', fontSize: 11.5 }};
                const grid = {{ vertLines: {{ color: '#F0F0F0' }}, horzLines: {{ color: '#F0F0F0' }} }};
                const crosshair = {{ mode: LightweightCharts.CrosshairMode.Normal }};

                function getOpts(layout, scaleMargins) {{
                    return {{
                        layout: layout,
                        grid: grid,
                        rightPriceScale: {{ 
                            borderColor: '#E0E0E0', visible: true, minimumWidth: FORCE_WIDTH, scaleMargins: scaleMargins
                        }},
                        timeScale: {{ borderColor: '#E0E0E0', timeVisible: true, rightOffset: 5 }},
                        crosshair: crosshair,
                    }};
                }}

                // ★ FORMATTERS 
                // 1. Axis Formatter (座標軸)
                function fmtInt(val) {{ return Math.round(val).toString(); }} 
                function fmtBigInt(val) {{ 
                    let absVal = Math.abs(val);
                    if (absVal >= 100000000) return Math.round(val/100000000).toString() + '億';
                    if (absVal >= 10000) return Math.round(val/10000).toString() + '萬';
                    return Math.round(val).toString();
                }}

                // 2. Cursor (游標)
                function fmtDec2(val) {{ return val.toFixed(2); }} 
                function fmtDec3(val) {{ return val.toFixed(3); }} 
                function fmtBigDec3(val) {{ 
                    let absVal = Math.abs(val);
                    if (absVal >= 100000000) return (val/100000000).toFixed(3) + '億';
                    if (absVal >= 10000) return (val/10000).toFixed(3) + '萬';
                    return val.toFixed(3);
                }}

                // 3. Legend (左上角)
                function fmtLegendDec3(val) {{ return val.toFixed(3); }} 

                // ==========================================
                // 1. 主圖 Main (Axis:整數, Cursor:2位, Legend:3位)
                // ==========================================
                const mainChart = LightweightCharts.createChart(document.getElementById('main-chart'), {{
                    ...getOpts(mainLayout, {{ top: 0.1, bottom: 0.1 }}),
                    rightPriceScale: {{ 
                        visible: true, borderColor: '#E0E0E0', minimumWidth: FORCE_WIDTH, scaleMargins: {{ top: 0.1, bottom: 0.1 }},
                        tickMarkFormatter: (p) => fmtInt(p) 
                    }}
                }});
                
                const candleSeries = mainChart.addCandlestickSeries({{
                    upColor: '#FF5252', downColor: '#00B746', borderUpColor: '#FF5252', borderDownColor: '#00B746', wickUpColor: '#FF5252', wickDownColor: '#00B746',
                    priceFormat: {{ type: 'custom', formatter: (p) => fmtDec2(p) }} 
                }});
                candleSeries.setData(candlesData);

                if (maData.length > 0) {{
                    if (maData[0].ma5 !== undefined) {{ mainChart.addLineSeries({{ ...lineOpts, color: '#FFA500', title: 'MA(5)', priceFormat: {{ type: 'custom', formatter: fmtDec2 }} }}).setData(maData.map(d=>({{time:d.time, value:d.ma5}}))); }}
                    if (maData[0].ma10 !== undefined) {{ mainChart.addLineSeries({{ ...lineOpts, color: '#2196F3', title: 'MA(10)', priceFormat: {{ type: 'custom', formatter: fmtDec2 }} }}).setData(maData.map(d=>({{time:d.time, value:d.ma10}}))); }}
                    if (maData[0].ma20 !== undefined) {{ mainChart.addLineSeries({{ ...lineOpts, color: '#E040FB', title: 'MA(20)', priceFormat: {{ type: 'custom', formatter: fmtDec2 }} }}).setData(maData.map(d=>({{time:d.time, value:d.ma20}}))); }}
                    if (maData[0].ma60 !== undefined) {{ mainChart.addLineSeries({{ ...lineOpts, color: '#00E676', title: 'MA(60)', priceFormat: {{ type: 'custom', formatter: fmtDec2 }} }}).setData(maData.map(d=>({{time:d.time, value:d.ma60}}))); }}
                }}
                
                if (bollData.length > 0) {{
                    mainChart.addLineSeries({{ ...lineOpts, lineWidth: 1.5, color: '#FF4081', title: 'MID', priceFormat: {{ type: 'custom', formatter: fmtDec2 }} }}).setData(bollData.map(d=>({{time:d.time, value:d.mid}})));
                    mainChart.addLineSeries({{ ...lineOpts, color: '#FFD700', title: 'UP', priceFormat: {{ type: 'custom', formatter: fmtDec2 }} }}).setData(bollData.map(d=>({{time:d.time, value:d.up}})));
                    mainChart.addLineSeries({{ ...lineOpts, color: '#00E5FF', title: 'LOW', priceFormat: {{ type: 'custom', formatter: fmtDec2 }} }}).setData(bollData.map(d=>({{time:d.time, value:d.low}})));
                }}
                
                mainChart.priceScale('right').applyOptions({{ tickMarkFormatter: (p) => fmtInt(p) }});

                // ==========================================
                // 2. VOL Chart
                // ==========================================
                const volChartEl = document.getElementById('vol-chart');
                let volChart = null, volSeries = null;
                if (volChartEl.style.display !== 'none') {{
                    volChart = LightweightCharts.createChart(volChartEl, {{
                        layout: volObvLayout, grid: grid, crosshair: crosshair,
                        timeScale: {{ borderColor: '#E0E0E0', timeVisible: true, rightOffset: 5 }},
                        rightPriceScale: {{ 
                            borderColor: '#E0E0E0', visible: true, minimumWidth: FORCE_WIDTH, scaleMargins: {{top: 0.2, bottom: 0}},
                            tickMarkFormatter: (p) => fmtBigInt(p) 
                        }}
                    }});
                    
                    volSeries = volChart.addHistogramSeries({{ 
                        title: 'VOL', priceLineVisible: false,
                        priceFormat: {{ type: 'custom', formatter: (p) => fmtBigDec3(p) }} 
                    }});
                    volSeries.setData(volData);
                    
                    volChart.priceScale('right').applyOptions({{ tickMarkFormatter: (p) => fmtBigInt(p) }});
                }}

                // ==========================================
                // 3. 副圖們 
                // ==========================================
                function createSubChart(id) {{
                    const el = document.getElementById(id);
                    if (el.style.display === 'none') return null;
                    const chart = LightweightCharts.createChart(el, {{
                        layout: indicatorLayout, grid: grid, crosshair: crosshair,
                        timeScale: {{ borderColor: '#E0E0E0', timeVisible: true, rightOffset: 5 }},
                        rightPriceScale: {{ 
                            borderColor: '#E0E0E0', visible: true, minimumWidth: FORCE_WIDTH, scaleMargins: {{top: 0.1, bottom: 0.1}},
                            tickMarkFormatter: (p) => fmtInt(p) 
                        }}
                    }});
                    return chart;
                }}

                const macdChart = createSubChart('macd-chart');
                if (macdChart && macdData.length > 0) {{
                    macdChart.addLineSeries({{ ...lineOpts, color: '#E6A23C', priceFormat: {{type:'custom', formatter:fmtDec3}} }}).setData(macdData.map(d=>({{time:d.time, value:d.dif}})));
                    macdChart.addLineSeries({{ ...lineOpts, color: '#2196F3', priceFormat: {{type:'custom', formatter:fmtDec3}} }}).setData(macdData.map(d=>({{time:d.time, value:d.dea}})));
                    macdChart.addHistogramSeries({{ priceFormat: {{type:'custom', formatter:fmtDec3}} }}).setData(macdData.map(d=>({{time:d.time, value:d.hist, color:d.color}})));
                    macdChart.priceScale('right').applyOptions({{ tickMarkFormatter: (p) => fmtInt(p) }});
                }}

                const kdjChart = createSubChart('kdj-chart');
                if (kdjChart && kdjData.length > 0) {{
                    kdjChart.addLineSeries({{ ...lineOpts, color: '#E6A23C', priceFormat: {{type:'custom', formatter:fmtDec3}} }}).setData(kdjData.map(d=>({{time:d.time, value:d.k}})));
                    kdjChart.addLineSeries({{ ...lineOpts, color: '#2196F3', priceFormat: {{type:'custom', formatter:fmtDec3}} }}).setData(kdjData.map(d=>({{time:d.time, value:d.d}})));
                    kdjChart.addLineSeries({{ ...lineOpts, color: '#E040FB', priceFormat: {{type:'custom', formatter:fmtDec3}} }}).setData(kdjData.map(d=>({{time:d.time, value:d.j}})));
                    kdjChart.priceScale('right').applyOptions({{ tickMarkFormatter: (p) => fmtInt(p) }});
                }}

                const rsiChart = createSubChart('rsi-chart');
                if (rsiChart && rsiData.length > 0) {{
                    rsiChart.addLineSeries({{ ...lineOpts, color: '#E6A23C', priceFormat: {{type:'custom', formatter:fmtDec3}} }}).setData(rsiData.map(d=>({{time:d.time, value:d.rsi6}})));
                    rsiChart.addLineSeries({{ ...lineOpts, color: '#2196F3', priceFormat: {{type:'custom', formatter:fmtDec3}} }}).setData(rsiData.map(d=>({{time:d.time, value:d.rsi12}})));
                    rsiChart.addLineSeries({{ ...lineOpts, color: '#E040FB', priceFormat: {{type:'custom', formatter:fmtDec3}} }}).setData(rsiData.map(d=>({{time:d.time, value:d.rsi24}})));
                    rsiChart.priceScale('right').applyOptions({{ tickMarkFormatter: (p) => fmtInt(p) }});
                }}

                const biasChart = createSubChart('bias-chart');
                if (biasChart && biasData.length > 0) {{
                    biasChart.addLineSeries({{ ...lineOpts, color: '#2196F3', priceFormat: {{type:'custom', formatter:fmtDec3}} }}).setData(biasData.map(d=>({{time:d.time, value:d.b6}})));
                    biasChart.addLineSeries({{ ...lineOpts, color: '#E6A23C', priceFormat: {{type:'custom', formatter:fmtDec3}} }}).setData(biasData.map(d=>({{time:d.time, value:d.b12}})));
                    biasChart.addLineSeries({{ ...lineOpts, color: '#E040FB', priceFormat: {{type:'custom', formatter:fmtDec3}} }}).setData(biasData.map(d=>({{time:d.time, value:d.b24}})));
                    biasChart.priceScale('right').applyOptions({{ tickMarkFormatter: (p) => fmtInt(p) }});
                }}

                // ==========================================
                // 4. OBV Chart
                // ==========================================
                const obvChartEl = document.getElementById('obv-chart');
                let obvChart = null;
                if (obvChartEl.style.display !== 'none') {{
                    obvChart = LightweightCharts.createChart(obvChartEl, {{
                        layout: volObvLayout, grid: grid, crosshair: crosshair,
                        timeScale: {{ borderColor: '#E0E0E0', timeVisible: true, rightOffset: 5 }},
                        rightPriceScale: {{ 
                            borderColor: '#E0E0E0', visible: true, minimumWidth: FORCE_WIDTH, scaleMargins: {{top: 0.1, bottom: 0.1}},
                            tickMarkFormatter: (p) => fmtBigInt(p) 
                        }}
                    }});
                    
                    if (obvData.length > 0) {{
                        const s1 = obvChart.addLineSeries({{ ...lineOpts, color: '#FFD700', priceFormat: {{type:'custom', formatter: (p)=>fmtBigDec3(p)}} }});
                        s1.setData(obvData.map(d=>({{time:d.time, value:d.obv}})));
                        const s2 = obvChart.addLineSeries({{ ...lineOpts, color: '#29B6F6', priceFormat: {{type:'custom', formatter: (p)=>fmtBigDec3(p)}} }});
                        s2.setData(obvData.map(d=>({{time:d.time, value:d.obv_ma}})));
                    }}
                    obvChart.priceScale('right').applyOptions({{ tickMarkFormatter: (p) => fmtBigInt(p) }});
                }}

                const allCharts = [mainChart, volChart, macdChart, kdjChart, rsiChart, obvChart, biasChart].filter(c => c !== null);
                
                function updateLegends(param) {{
                    let t;
                    if (!param || !param.time) {{
                        if (candlesData.length > 0) t = candlesData[candlesData.length - 1].time;
                        else return;
                    }} else {{ t = param.time; }}

                    const mainLegendEl = document.getElementById('main-legend');
                    if (mainLegendEl && maData.length > 0) {{ const d = maData.find(x => x.time === t); if(d) {{ let h='<div class="legend-row"><span class="legend-label">MA(5,10,20,60)</span>'; if(d.ma5!=null)h+=`<span class="legend-value" style="color:#FFA500">MA5:${{fmtLegendDec3(d.ma5)}}</span> `; if(d.ma10!=null)h+=`<span class="legend-value" style="color:#2196F3">MA10:${{fmtLegendDec3(d.ma10)}}</span> `; if(d.ma20!=null)h+=`<span class="legend-value" style="color:#E040FB">MA20:${{fmtLegendDec3(d.ma20)}}</span> `; if(d.ma60!=null)h+=`<span class="legend-value" style="color:#00E676">MA60:${{fmtLegendDec3(d.ma60)}}</span>`; h+='</div>'; mainLegendEl.innerHTML=h; }} }}
                    if (mainLegendEl && bollData.length > 0) {{ const d = bollData.find(x => x.time === t); if(d) mainLegendEl.innerHTML += `<div class="legend-row"><span class="legend-label">BOLL(20,2)</span><span class="legend-value" style="color:#FF4081">MID:${{fmtLegendDec3(d.mid)}}</span><span class="legend-value" style="color:#FFD700">UP:${{fmtLegendDec3(d.up)}}</span><span class="legend-value" style="color:#00E5FF">LOW:${{fmtLegendDec3(d.low)}}</span></div>`; }}
                    
                    const volLegendEl = document.getElementById('vol-legend');
                    if (volLegendEl && volData.length > 0) {{
                        const d = volData.find(x => x.time === t);
                        if (d && d.value != null) {{
                            volLegendEl.innerHTML = `<div class="legend-row"><span class="legend-label">VOL</span><span class="legend-value" style="color: ${{d.color}}">VOL: ${{fmtBigDec3(d.value)}}</span></div>`;
                        }}
                    }}
                    
                    const macdLegendEl = document.getElementById('macd-legend');
                    if (macdLegendEl && macdData.length > 0) {{ const d = macdData.find(x => x.time === t); if(d && d.dif!=null) macdLegendEl.innerHTML=`<div class="legend-row"><span class="legend-label">MACD(12,26,9)</span><span class="legend-value" style="color:#E6A23C">DIF: ${{fmtLegendDec3(d.dif)}}</span><span class="legend-value" style="color:#2196F3">DEA: ${{fmtLegendDec3(d.dea)}}</span><span class="legend-value" style="color:#E040FB">MACD: ${{fmtLegendDec3(d.hist)}}</span></div>`; }}
                    
                    const kdjLegendEl = document.getElementById('kdj-legend');
                    if (kdjLegendEl && kdjData.length > 0) {{ const d = kdjData.find(x => x.time === t); if(d && d.k!=null) kdjLegendEl.innerHTML=`<div class="legend-row"><span class="legend-label">KDJ(9,3,3)</span><span class="legend-value" style="color:#E6A23C">K: ${{fmtLegendDec3(d.k)}}</span><span class="legend-value" style="color:#2196F3">D: ${{fmtLegendDec3(d.d)}}</span><span class="legend-value" style="color:#E040FB">J: ${{fmtLegendDec3(d.j)}}</span></div>`; }}
                    
                    const rsiLegendEl = document.getElementById('rsi-legend');
                    if (rsiLegendEl && rsiData.length > 0) {{ const d = rsiData.find(x => x.time === t); if(d) rsiLegendEl.innerHTML=`<div class="legend-row"><span class="legend-label">RSI(6,12,24)</span><span class="legend-value" style="color:#E6A23C">RSI6: ${{fmtLegendDec3(d.rsi6)}}</span><span class="legend-value" style="color:#2196F3">RSI12: ${{fmtLegendDec3(d.rsi12)}}</span><span class="legend-value" style="color:#E040FB">RSI24: ${{fmtLegendDec3(d.rsi24)}}</span></div>`; }}
                    
                    const obvLegendEl = document.getElementById('obv-legend');
                    if (obvLegendEl && obvData.length > 0) {{
                        const d = obvData.find(x => x.time === t);
                        if (d && d.obv != null) {{
                            obvLegendEl.innerHTML = `<div class="legend-row"><span class="legend-label">OBV(10)</span><span class="legend-value" style="color: #FFD700">OBV: ${{fmtBigDec3(d.obv)}}</span> <span class="legend-value" style="color: #29B6F6">MA10: ${{fmtBigDec3(d.obv_ma)}}</span></div>`;
                        }}
                    }}
                    
                    const biasLegendEl = document.getElementById('bias-legend');
                    if (biasLegendEl && biasData.length > 0) {{
                        const d = biasData.find(x => x.time === t);
                        if (d) {{
                            biasLegendEl.innerHTML = `<div class="legend-row"><span class="legend-label">BIAS(6,12,24)</span><span class="legend-value" style="color: #2196F3">BIAS1: ${{fmtLegendDec3(d.b6)}}</span><span class="legend-value" style="color: #E6A23C">BIAS2: ${{fmtLegendDec3(d.b12)}}</span><span class="legend-value" style="color: #E040FB">BIAS3: ${{fmtLegendDec3(d.b24)}}</span></div>`;
                        }}
                    }}
                }}

                allCharts.forEach(c => {{
                    c.priceScale('right').applyOptions({{ minimumWidth: FORCE_WIDTH }});
                    c.subscribeCrosshairMove(updateLegends);
                    c.timeScale().subscribeVisibleLogicalRangeChange(range => {{
                        if (range) allCharts.forEach(other => {{ if (other !== c) other.timeScale().setVisibleLogicalRange(range); }});
                    }});
                }});
                
                updateLegends(null); 

                window.addEventListener('resize', () => {{
                    allCharts.forEach(c => c.resize(document.body.clientWidth, c.options().height));
                }});

            }} catch (e) {{
                document.body.innerHTML = '<div style="color:red; padding:20px;">Chart Error: ' + e.message + '</div>';
            }}
        </script>
    </body>
    </html>
    """
    
    total_height = 460
    if show_vol: total_height += 100
    if show_macd: total_height += 150
    if show_kdj: total_height += 120
    if show_rsi: total_height += 120
    if show_obv: total_height += 120
    if show_bias: total_height += 120
    
    # ★ 緩衝高度
    total_height += 50

    components.html(html_code, height=total_height)
