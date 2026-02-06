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
# 3. 資料層 (長線 MACD + 短線 OBV)
# ---------------------------------------------------------
@st.cache_data(ttl=60)
def get_data(ticker, period="2y", interval="1d"):
    try:
        is_quarterly = (interval == "3mo")
        dl_interval = "1mo" if (interval == "1y" or is_quarterly) else interval
        
        # 1. 下載 2年 長資料 (確保 MACD/MA 準確)
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

        # 2. 計算長週期指標 (MA, BOLL, MACD, KDJ, RSI, BIAS)
        data['MA5'] = ta.sma(data[close_col], length=5)
        data['MA10'] = ta.sma(data[close_col], length=10)
        data['MA20'] = ta.sma(data[close_col], length=20)
        data['MA60'] = ta.sma(data[close_col], length=60)
        
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

        # 3. ★ 切割資料：保留最後半年 (130天)
        # 目的：讓 OBV 累加基期變短，數值變小
        data = data.tail(130).copy()

        # 4. ★ 重算 OBV
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
    # ★ 取得半年份 (130筆) 的資料
    full_df = get_data(ticker, period="2y", interval=interval_map[interval_label])
    
    if full_df is None:
        st.error(f"無數據: {ticker}")
        st.stop()
        
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
    # 5. JavaScript (★ 核心：V62架構 + 14.5px + 70px)
    # ---------------------------------------------------------
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/lightweight-charts@3.8.0/dist/lightweight-charts.standalone.production.js"></script>
        <style>
            body {{ margin: 0; padding: 0; background-color: #ffffff; overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; }}
            
            /* V62: 副圖區塊背景 (CSS 漸層：左灰右白) */
            .sub-chart {{
                background-color: #FFFFFF;
                background-image: linear-gradient(to right, #FAFAFA calc(100% - 70px), transparent calc(100% - 70px));
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

                const FORCE_WIDTH = 70;

                // 1. ★ 主圖: 字體 14.5px
                const mainLayout = {{ backgroundColor: '#FFFFFF', textColor: '#333333', fontSize: 14.5 }};
                
                // 2. 副圖: 透明, 字體 14/11.5
                const indicatorLayout = {{ backgroundColor: 'transparent', textColor: '#333333', fontSize: 14 }};
                const volObvLayout = {{ backgroundColor: 'transparent', textColor: '#333333', fontSize: 11.5 }};

                const grid = {{ vertLines: {{ color: '#F0F0F0' }}, horzLines: {{ color: '#F0F0F0' }} }};
                const crosshair = {{ mode: LightweightCharts.CrosshairMode.Normal }};

                function getOpts(layout, scaleMargins) {{
                    return {{
                        layout: layout,
                        grid: grid,
                        rightPriceScale: {{ 
                            borderColor: '#E0E0E0', 
                            visible: true,
                            minimumWidth: FORCE_WIDTH, 
                            scaleMargins: scaleMargins
                        }},
                        timeScale: {{ borderColor: '#E0E0E0', timeVisible: true, rightOffset: 5 }},
                        crosshair: crosshair,
                    }};
                }}

                function createChart(id, opts) {{
                    const el = document.getElementById(id);
                    if (el.style.display === 'none') return null;
                    return LightweightCharts.createChart(el, opts);
                }}

                function formatStandard(val) {{
                    if (val === undefined || val === null) return '-';
                    return val.toLocaleString('en-US', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }});
                }}

                function formatSmart(val) {{
                    if (val === undefined || val === null) return '-';
                    return parseFloat(val.toFixed(3)).toString();
                }}

                function formatNumber(val, decimals=2) {{
                    if (val === undefined || val === null) return '-';
                    return val.toLocaleString('en-US', {{ minimumFractionDigits: decimals, maximumFractionDigits: decimals }});
                }}

                function formatBigSmart(val) {{
                    if (val === undefined || val === null) return '-';
                    let absVal = Math.abs(val);
                    if (absVal >= 100000000) {{
                        return parseFloat((val / 100000000).toFixed(3)).toString() + '億';
                    }}
                    if (absVal >= 10000) {{
                        return parseFloat((val / 10000).toFixed(3)).toString() + '萬';
                    }}
                    return parseFloat(val.toFixed(3)).toString();
                }}

                function formatFixed3(val) {{
                    if (val === undefined || val === null) return '-';
                    return val.toFixed(3);
                }}
                
                function formatBigFixed3(val) {{
                    if (val === undefined || val === null) return '-';
                    let absVal = Math.abs(val);
                    if (absVal >= 100000000) return (val / 100000000).toFixed(3) + '億';
                    if (absVal >= 10000) return (val / 10000).toFixed(3) + '萬';
                    return val.toFixed(3);
                }}

                const mainChart = createChart('main-chart', {{
                    ...getOpts(mainLayout, {{ top: 0.1, bottom: 0.1 }}),
                    localization: {{ priceFormatter: (p) => formatStandard(p) }} 
                }});
                
                const volChart = createChart('vol-chart', {{
                    ...getOpts(volObvLayout, {{top: 0.2, bottom: 0}}),
                    localization: {{ priceFormatter: (p) => formatBigSmart(p) }}
                }});
                
                const macdChart = createChart('macd-chart', {{
                    ...getOpts(indicatorLayout, {{ top: 0.1, bottom: 0.1 }}),
                    localization: {{ priceFormatter: (p) => formatSmart(p) }}
                }});
                
                const kdjChart = createChart('kdj-chart', {{
                    ...getOpts(indicatorLayout, {{ top: 0.1, bottom: 0.1 }}),
                    localization: {{ priceFormatter: (p) => formatSmart(p) }}
                }});
                
                const rsiChart = createChart('rsi-chart', {{
                    ...getOpts(indicatorLayout, {{ top: 0.1, bottom: 0.1 }}),
                    localization: {{ priceFormatter: (p) => formatSmart(p) }}
                }});
                
                const obvChart = createChart('obv-chart', {{
                    ...getOpts(volObvLayout, {{ top: 0.1, bottom: 0.1 }}),
                    localization: {{ priceFormatter: (p) => formatNumber(p, 2) }}
                }});
                
                const biasChart = createChart('bias-chart', {{
                    ...getOpts(indicatorLayout, {{ top: 0.1, bottom: 0.1 }}),
                    localization: {{ priceFormatter: (p) => formatSmart(p) }}
                }});

                let volSeries, bollMidSeries, bollUpSeries, bollLowSeries, ma5Series, ma10Series, ma20Series, ma60Series;
                let rsi6Series, rsi12Series, rsi24Series;
                let bias6Series, bias12Series, bias24Series;
                let obvSeries, obvMaSeries;
                
                const lineOpts = {{ lineWidth: 1, priceLineVisible: false, lastValueVisible: false }};

                if (mainChart) {{
                    const candleSeries = mainChart.addCandlestickSeries({{
                        upColor: '#FF5252', downColor: '#00B746', borderUpColor: '#FF5252', borderDownColor: '#00B746', wickUpColor: '#FF5252', wickDownColor: '#00B746'
                    }});
                    candleSeries.setData(candlesData);

                    if (bollData.length > 0) {{
                        bollMidSeries = mainChart.addLineSeries({{ ...lineOpts, color: '#FF4081', title: 'MID' }}); 
                        bollUpSeries = mainChart.addLineSeries({{ ...lineOpts, color: '#FFD700', title: 'UPPER' }});
                        bollLowSeries = mainChart.addLineSeries({{ ...lineOpts, color: '#00E5FF', title: 'LOWER' }});
                        bollMidSeries.setData(bollData.map(d => ({{ time: d.time, value: d.mid }})));
                        bollUpSeries.setData(bollData.map(d => ({{ time: d.time, value: d.up }})));
                        bollLowSeries.setData(bollData.map(d => ({{ time: d.time, value: d.low }})));
                    }}

                    if (maData.length > 0) {{
                        const f = maData[0];
                        if (f.ma5 !== undefined) {{ ma5Series = mainChart.addLineSeries({{ ...lineOpts, color: '#FFA500', title: 'MA(5)' }}); ma5Series.setData(maData.map(d => ({{ time: d.time, value: d.ma5 }}))); }}
                        if (f.ma10 !== undefined) {{ ma10Series = mainChart.addLineSeries({{ ...lineOpts, color: '#2196F3', title: 'MA(10)' }}); ma10Series.setData(maData.map(d => ({{ time: d.time, value: d.ma10 }}))); }}
                        if (f.ma20 !== undefined) {{ ma20Series = mainChart.addLineSeries({{ ...lineOpts, color: '#E040FB', title: 'MA(20)' }}); ma20Series.setData(maData.map(d => ({{ time: d.time, value: d.ma20 }}))); }}
                        if (f.ma60 !== undefined) {{ ma60Series = mainChart.addLineSeries({{ ...lineOpts, color: '#00E676', title: 'MA(60)' }}); ma60Series.setData(maData.map(d => ({{ time: d.time, value: d.ma60 }}))); }}
                    }}
                }}
                
                if (volChart && volData.length > 0) {{ 
                    volSeries = volChart.addHistogramSeries({{ priceFormat: {{ type: 'volume' }}, title: 'VOL', priceLineVisible: false }});
                    volSeries.setData(volData); 
                }}
                if (macdChart && macdData.length > 0) {{
                    macdChart.addLineSeries({{ ...lineOpts, color: '#E6A23C', lineWidth: 1 }}).setData(macdData.map(d => ({{ time: d.time, value: d.dif }})));
                    macdChart.addLineSeries({{ ...lineOpts, color: '#2196F3', lineWidth: 1 }}).setData(macdData.map(d => ({{ time: d.time, value: d.dea }})));
                    macdChart.addHistogramSeries({{ priceLineVisible: false }}).setData(macdData.map(d => ({{ time: d.time, value: d.hist, color: d.color }})));
                }}
                if (kdjChart && kdjData.length > 0) {{
                    kdjChart.addLineSeries({{ ...lineOpts, color: '#E6A23C', lineWidth: 1 }}).setData(kdjData.map(d => ({{ time: d.time, value: d.k }})));
                    kdjChart.addLineSeries({{ ...lineOpts, color: '#2196F3', lineWidth: 1 }}).setData(kdjData.map(d => ({{ time: d.time, value: d.d }})));
                    kdjChart.addLineSeries({{ ...lineOpts, color: '#E040FB', lineWidth: 1 }}).setData(kdjData.map(d => ({{ time: d.time, value: d.j }})));
                }}
                if (rsiChart && rsiData.length > 0) {{
                    rsi6Series = rsiChart.addLineSeries({{ ...lineOpts, color: '#E6A23C', lineWidth: 1 }});
                    rsi12Series = rsiChart.addLineSeries({{ ...lineOpts, color: '#2196F3', lineWidth: 1 }});
                    rsi24Series = rsiChart.addLineSeries({{ ...lineOpts, color: '#E040FB', lineWidth: 1 }});
                    rsi6Series.setData(rsiData.map(d => ({{ time: d.time, value: d.rsi6 }})));
                    rsi12Series.setData(rsiData.map(d => ({{ time: d.time, value: d.rsi12 }})));
                    rsi24Series.setData(rsiData.map(d => ({{ time: d.time, value: d.rsi24 }})));
                }}
                
                if (obvChart && obvData.length > 0) {{ 
                    obvSeries = obvChart.addLineSeries({{ ...lineOpts, color: '#FFD700', lineWidth: 1 }});
                    obvMaSeries = obvChart.addLineSeries({{ ...lineOpts, color: '#29B6F6', lineWidth: 1 }});
                    obvSeries.setData(obvData.map(d => ({{ time: d.time, value: d.obv }}))); 
                    obvMaSeries.setData(obvData.map(d => ({{ time: d.time, value: d.obv_ma }}))); 
                }}
                
                if (biasChart && biasData.length > 0) {{
                    bias6Series = biasChart.addLineSeries({{ ...lineOpts, color: '#2196F3', lineWidth: 1 }});
                    bias12Series = biasChart.addLineSeries({{ ...lineOpts, color: '#E6A23C', lineWidth: 1 }});
                    bias24Series = biasChart.addLineSeries({{ ...lineOpts, color: '#E040FB', lineWidth: 1 }});
                    bias6Series.setData(biasData.map(d => ({{ time: d.time, value: d.b6 }})));
                    bias12Series.setData(biasData.map(d => ({{ time: d.time, value: d.b12 }})));
                    bias24Series.setData(biasData.map(d => ({{ time: d.time, value: d.b24 }})));
                }}

                const mainLegendEl = document.getElementById('main-legend');
                const volLegendEl = document.getElementById('vol-legend');
                const macdLegendEl = document.getElementById('macd-legend');
                const kdjLegendEl = document.getElementById('kdj-legend');
                const rsiLegendEl = document.getElementById('rsi-legend');
                const obvLegendEl = document.getElementById('obv-legend');
                const biasLegendEl = document.getElementById('bias-legend');

                function updateLegends(param) {{
                    let t;
                    if (!param || !param.time) {{
                        if (candlesData.length > 0) t = candlesData[candlesData.length - 1].time;
                        else return;
                    }} else {{
                        t = param.time;
                    }}

                    if (mainLegendEl && maData.length > 0) {{ const d = maData.find(x => x.time === t); if(d) {{ let h='<div class="legend-row"><span class="legend-label">MA(5,10,20,60)</span>'; if(d.ma5!=null)h+=`<span class="legend-value" style="color:#FFA500">MA5:${{d.ma5.toFixed(2)}}</span> `; if(d.ma10!=null)h+=`<span class="legend-value" style="color:#2196F3">MA10:${{d.ma10.toFixed(2)}}</span> `; if(d.ma20!=null)h+=`<span class="legend-value" style="color:#E040FB">MA20:${{d.ma20.toFixed(2)}}</span> `; if(d.ma60!=null)h+=`<span class="legend-value" style="color:#00E676">MA60:${{d.ma60.toFixed(2)}}</span>`; h+='</div>'; mainLegendEl.innerHTML=h; }} }}
                    if (mainLegendEl && bollData.length > 0) {{ const d = bollData.find(x => x.time === t); if(d) mainLegendEl.innerHTML += `<div class="legend-row"><span class="legend-label">BOLL(20,2)</span><span class="legend-value" style="color:#FF4081">MID:${{d.mid.toFixed(2)}}</span><span class="legend-value" style="color:#FFD700">UP:${{d.up!=null?d.up.toFixed(2):'-'}}</span><span class="legend-value" style="color:#00E5FF">LOW:${{d.low!=null?d.low.toFixed(2):'-'}}</span></div>`; }}
                    
                    if (volLegendEl && volData.length > 0) {{
                        const d = volData.find(x => x.time === t);
                        if (d && d.value != null) {{
                            volLegendEl.innerHTML = `<div class="legend-row"><span class="legend-label">VOL</span><span class="legend-value" style="color: ${{d.color}}">VOL: ${{formatBigFixed3(d.value)}}</span></div>`;
                        }}
                    }}
                    
                    if (macdLegendEl && macdData.length > 0) {{ const d = macdData.find(x => x.time === t); if(d && d.dif!=null) macdLegendEl.innerHTML=`<div class="legend-row"><span class="legend-label">MACD(12,26,9)</span><span class="legend-value" style="color:#E6A23C">DIF: ${{d.dif.toFixed(3)}}</span><span class="legend-value" style="color:#2196F3">DEA: ${{d.dea.toFixed(3)}}</span><span class="legend-value" style="color:#E040FB">MACD: ${{d.hist.toFixed(3)}}</span></div>`; }}
                    if (kdjLegendEl && kdjData.length > 0) {{ const d = kdjData.find(x => x.time === t); if(d && d.k!=null) kdjLegendEl.innerHTML=`<div class="legend-row"><span class="legend-label">KDJ(9,3,3)</span><span class="legend-value" style="color:#E6A23C">K: ${{d.k.toFixed(3)}}</span><span class="legend-value" style="color:#2196F3">D: ${{d.d.toFixed(3)}}</span><span class="legend-value" style="color:#E040FB">J: ${{d.j.toFixed(3)}}</span></div>`; }}
                    if (rsiLegendEl && rsiData.length > 0) {{ const d = rsiData.find(x => x.time === t); if(d) rsiLegendEl.innerHTML=`<div class="legend-row"><span class="legend-label">RSI(6,12,24)</span><span class="legend-value" style="color:#E6A23C">RSI6: ${{d.rsi6!=null?d.rsi6.toFixed(3):'-'}}</span><span class="legend-value" style="color:#2196F3">RSI12: ${{d.rsi12!=null?d.rsi12.toFixed(3):'-'}}</span><span class="legend-value" style="color:#E040FB">RSI24: ${{d.rsi24!=null?d.rsi24.toFixed(3):'-'}}</span></div>`; }}
                    
                    if (obvLegendEl && obvData.length > 0) {{
                        const d = obvData.find(x => x.time === t);
                        if (d && d.obv != null) {{
                            const obvVal = formatNumber(d.obv, 2);
                            const maVal = d.obv_ma ? formatNumber(d.obv_ma, 2) : '-';
                            obvLegendEl.innerHTML = `<div class="legend-row"><span class="legend-label">OBV(10)</span><span class="legend-value" style="color: #FFD700">OBV: ${{obvVal}}</span> <span class="legend-value" style="color: #29B6F6">MA10: ${{maVal}}</span></div>`;
                        }}
                    }}
                    
                    if (biasLegendEl && biasData.length > 0) {{
                        const d = biasData.find(x => x.time === t);
                        if (d) {{
                            biasLegendEl.innerHTML = `<div class="legend-row"><span class="legend-label">BIAS(6,12,24)</span><span class="legend-value" style="color: #2196F3">BIAS1: ${{d.b6!=null?d.b6.toFixed(3):'-'}}</span><span class="legend-value" style="color: #E6A23C">BIAS2: ${{d.b12!=null?d.b12.toFixed(3):'-'}}</span><span class="legend-value" style="color: #E040FB">BIAS3: ${{d.b24!=null?d.b24.toFixed(3):'-'}}</span></div>`;
                        }}
                    }}
                }}

                const allCharts = [mainChart, volChart, macdChart, kdjChart, rsiChart, obvChart, biasChart].filter(c => c !== null);
                
                allCharts.forEach(c => {{
                    // ★強制鎖定 70px (精準對齊，無白邊)
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
    
    # ★ 緩衝高度 (避免最後一圖被切)
    total_height += 60

    components.html(html_code, height=total_height)
