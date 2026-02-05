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
# 2. 資料層 (新增 KDJ 的 J 值計算)
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

        # --- 指標計算 ---
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
        
        # ★KDJ 計算 (J = 3K - 2D)
        stoch = ta.stoch(data['high'], data['low'], data[close_col])
        if stoch is not None: 
            data = pd.concat([data, stoch], axis=1)
            # pandas_ta 的欄位通常是 stochk_14_3_3, stochd_14_3_3
            # 我們這裡動態抓取欄位名稱
            k_col = [c for c in data.columns if c.startswith('stochk')][0]
            d_col = [c for c in data.columns if c.startswith('stochd')][0]
            data['k'] = data[k_col]
            data['d'] = data[d_col]
            data['j'] = 3 * data['k'] - 2 * data['d'] # 手動計算 J 值

        data['RSI'] = ta.rsi(data[close_col], length=14)
        data['OBV'] = ta.obv(data[close_col], data['volume'])
        data['BIAS'] = (data[close_col] - data['MA20']) / data['MA20'] * 100
        
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
    except: return None

# ---------------------------------------------------------
# 3. 介面控制
# ---------------------------------------------------------
with st.sidebar:
    st.header("🔍 股票搜尋")
    market_mode = st.radio("市場", ["台股(市)", "台股(櫃)", "美股"], index=2, horizontal=True)
    raw_symbol = st.text_input("代碼", value="MU")
    if market_mode == "台股(市)": ticker = f"{raw_symbol}.TW" if not raw_symbol.upper().endswith(".TW") else raw_symbol
    elif market_mode == "台股(櫃)": ticker = f"{raw_symbol}.TWO" if not raw_symbol.upper().endswith(".TWO") else raw_symbol
    else: ticker = raw_symbol.upper()

col_main, col_tools = st.columns([0.85, 0.15])

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
    with c_top1: st.markdown(f"### {ticker} 走勢圖")
    with c_top2: interval_label = st.radio("週期", ["日K", "週K", "月K", "季K", "年K"], index=0, horizontal=True, label_visibility="collapsed")
    
    interval_map = {"日K": "1d", "週K": "1wk", "月K": "1mo", "季K": "3mo", "年K": "1y"}
    full_df = get_data(ticker, period="max", interval=interval_map[interval_label])
    
    if full_df is None:
        st.error(f"無數據: {ticker}")
        st.stop()
        
    min_d, max_d = full_df['date_obj'].min().to_pydatetime(), full_df['date_obj'].max().to_pydatetime()
    
    if 'active_btn' not in st.session_state: st.session_state['active_btn'] = '6m'
    if 'slider_range' not in st.session_state:
        default_start = max_d - timedelta(days=180)
        if default_start < min_d: default_start = min_d
        st.session_state['slider_range'] = (default_start, max_d)

    def handle_btn_click(btn_key, months=0, years=0, ytd=False, is_max=False):
        st.session_state['active_btn'] = btn_key
        end = max_d
        if is_max: start = min_d
        elif ytd:
            start = datetime(end.year, 1, 1)
            if start < min_d: start = min_d
        else:
            start = end - relativedelta(months=months, years=years)
            if start < min_d: start = min_d
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

    # ---------------------------------------------------------
    # 4. JSON 序列化 (包含 KDJ 三線)
    # ---------------------------------------------------------
    def to_json_list(df, cols):
        res = []
        df_clean = df.where(pd.notnull(df), None)
        for _, row in df_clean.iterrows():
            try:
                item = {'time': int(row['time'])}
                valid = True
                for k, v in cols.items():
                    val = row.get(v)
                    if k in ['open','high','low','close'] and val is None: valid = False; break
                    item[k] = float(val) if val is not None else None
                if valid: res.append(item)
            except: continue
        return json.dumps(res)

    candles_json = to_json_list(df, {'open':'open', 'high':'high', 'low':'low', 'close':'close'})
    
    vol_data_list = []
    if show_vol:
        for _, row in df.iterrows():
            try:
                color = '#FF5252' if row['close'] >= row['open'] else '#00B746'
                vol_data_list.append({'time': int(row['time']), 'value': float(row['volume']), 'color': color})
            except: continue
    vol_json = json.dumps(vol_data_list)
    
    macd_data_list = []
    if show_macd:
        for _, row in df.iterrows():
            try:
                dif = row.get('macd_12_26_9')
                dea = row.get('macds_12_26_9')
                hist = row.get('macdh_12_26_9')
                if pd.notnull(dif) and pd.notnull(dea) and pd.notnull(hist):
                    color = '#FF5252' if hist >= 0 else '#00B746'
                    macd_data_list.append({'time': int(row['time']), 'dif': float(dif), 'dea': float(dea), 'hist': float(hist), 'color': color})
            except: continue
    macd_json = json.dumps(macd_data_list)
    
    ma_json = to_json_list(df, {'ma5':'ma5', 'ma10':'ma10', 'ma20':'ma20', 'ma60':'ma60'}) if show_ma else "[]"
    boll_json = to_json_list(df, {'up':'boll_upper', 'mid':'boll_mid', 'low':'boll_lower'}) if show_boll else "[]"
    
    # ★KDJ 數據包：包含 K, D, J 三線
    kdj_json = to_json_list(df, {'k':'k', 'd':'d', 'j':'j'}) if show_kdj else "[]"
    
    rsi_json = to_json_list(df, {'rsi':'rsi'}) if show_rsi else "[]"
    obv_json = to_json_list(df, {'obv':'obv'}) if show_obv else "[]"
    bias_json = to_json_list(df, {'bias':'bias'}) if show_bias else "[]"

    # ---------------------------------------------------------
    # 5. JavaScript (新增 KDJ 三線繪圖與 Legend)
    # ---------------------------------------------------------
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/lightweight-charts@3.8.0/dist/lightweight-charts.standalone.production.js"></script>
        <style>
            body {{ margin: 0; padding: 0; background-color: #ffffff; overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; }}
            .chart-container {{ position: relative; width: 100%; }}
            
            .legend {{
                position: absolute; top: 10px; left: 10px; z-index: 100;
                font-size: 12px; line-height: 18px; font-weight: 500; pointer-events: none;
            }}
            .legend-row {{ display: flex; gap: 10px; margin-bottom: 2px; }}
            .legend-label {{ font-weight: bold; color: #333; margin-right: 5px; }}
            .legend-value {{ font-family: 'Consolas', 'Monaco', monospace; }}
        </style>
    </head>
    <body>
        <div id="main-chart" class="chart-container" style="height: 450px;">
            <div id="main-legend" class="legend"></div>
        </div>
        
        <div id="vol-chart" class="chart-container" style="height: {'100px' if show_vol else '0px'}; display: {'block' if show_vol else 'none'};">
            <div id="vol-legend" class="legend"></div>
        </div>
        
        <div id="macd-chart" class="chart-container" style="height: {'150px' if show_macd else '0px'}; display: {'block' if show_macd else 'none'};">
            <div id="macd-legend" class="legend"></div>
        </div>
        
        <div id="kdj-chart" class="chart-container" style="height: {'120px' if show_kdj else '0px'}; display: {'block' if show_kdj else 'none'};">
            <div id="kdj-legend" class="legend"></div>
        </div>
        
        <div id="rsi-chart" class="chart-container" style="height: {'120px' if show_rsi else '0px'}; display: {'block' if show_rsi else 'none'};"></div>
        <div id="obv-chart" class="chart-container" style="height: {'120px' if show_obv else '0px'}; display: {'block' if show_obv else 'none'};"></div>
        <div id="bias-chart" class="chart-container" style="height: {'120px' if show_bias else '0px'}; display: {'block' if show_bias else 'none'};"></div>

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

                if (!candlesData || candlesData.length === 0) throw new Error("No Data");

                const chartOptions = {{
                    layout: {{ backgroundColor: '#FFFFFF', textColor: '#333333' }},
                    grid: {{ vertLines: {{ color: '#F0F0F0' }}, horzLines: {{ color: '#F0F0F0' }} }},
                    rightPriceScale: {{ borderColor: '#E0E0E0', scaleMargins: {{ top: 0.1, bottom: 0.1 }}, visible: true }},
                    timeScale: {{ borderColor: '#E0E0E0', timeVisible: true, rightOffset: 5 }},
                    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
                }};

                function createChart(id, opts) {{
                    const el = document.getElementById(id);
                    if (el.style.display === 'none') return null;
                    return LightweightCharts.createChart(el, opts);
                }}

                const mainChart = createChart('main-chart', chartOptions);
                const volChart = createChart('vol-chart', {{
                    ...chartOptions, 
                    rightPriceScale: {{ scaleMargins: {{top: 0.2, bottom: 0}}, visible: true }},
                    localization: {{ priceFormatter: (p) => (p / 10000).toFixed(0) + '萬' }}
                }});
                const macdChart = createChart('macd-chart', chartOptions);
                const kdjChart = createChart('kdj-chart', chartOptions);
                const rsiChart = createChart('rsi-chart', chartOptions);
                const obvChart = createChart('obv-chart', chartOptions);
                const biasChart = createChart('bias-chart', chartOptions);

                let volSeries, bollMidSeries, bollUpSeries, bollLowSeries, ma5Series, ma10Series, ma20Series, ma60Series;
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
                        if (f.ma5 !== null) {{ ma5Series = mainChart.addLineSeries({{ ...lineOpts, color: '#FFA500', title: 'EMA5' }}); ma5Series.setData(maData.map(d => ({{ time: d.time, value: d.ma5 }}))); }}
                        if (f.ma10 !== null) {{ ma10Series = mainChart.addLineSeries({{ ...lineOpts, color: '#2196F3', title: 'EMA10' }}); ma10Series.setData(maData.map(d => ({{ time: d.time, value: d.ma10 }}))); }}
                        if (f.ma20 !== null) {{ ma20Series = mainChart.addLineSeries({{ ...lineOpts, color: '#E040FB', title: 'EMA20' }}); ma20Series.setData(maData.map(d => ({{ time: d.time, value: d.ma20 }}))); }}
                        if (f.ma60 !== null) {{ ma60Series = mainChart.addLineSeries({{ ...lineOpts, color: '#00E676', title: 'EMA60' }}); ma60Series.setData(maData.map(d => ({{ time: d.time, value: d.ma60 }}))); }}
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
                
                // ★KDJ 繪圖 (依照截圖配色: K=橘, D=藍, J=紫)
                if (kdjChart && kdjData.length > 0) {{
                    // K Line: Orange
                    kdjChart.addLineSeries({{ ...lineOpts, color: '#E6A23C', lineWidth: 1 }}).setData(kdjData.map(d => ({{ time: d.time, value: d.k }})));
                    // D Line: Blue
                    kdjChart.addLineSeries({{ ...lineOpts, color: '#2196F3', lineWidth: 1 }}).setData(kdjData.map(d => ({{ time: d.time, value: d.d }})));
                    // J Line: Purple
                    kdjChart.addLineSeries({{ ...lineOpts, color: '#E040FB', lineWidth: 1 }}).setData(kdjData.map(d => ({{ time: d.time, value: d.j }})));
                }}

                if (rsiChart && rsiData.length > 0) {{ rsiChart.addLineSeries({{ ...lineOpts, color: '#E040FB' }}).setData(rsiData.map(d => ({{ time: d.time, value: d.rsi }}))); }}
                if (obvChart && obvData.length > 0) {{ obvChart.addLineSeries({{ ...lineOpts, color: '#FFA500', priceFormat: {{ type: 'volume' }} }}).setData(obvData.map(d => ({{ time: d.time, value: d.obv }}))); }}
                if (biasChart && biasData.length > 0) {{ biasChart.addLineSeries({{ ...lineOpts, color: '#607D8B' }}).setData(biasData.map(d => ({{ time: d.time, value: d.bias }}))); }}

                const mainLegendEl = document.getElementById('main-legend');
                const volLegendEl = document.getElementById('vol-legend');
                const macdLegendEl = document.getElementById('macd-legend');
                const kdjLegendEl = document.getElementById('kdj-legend');

                function updateLegends(param) {{
                    let t;
                    if (!param || !param.time) {{
                        if (candlesData.length > 0) t = candlesData[candlesData.length - 1].time;
                        else return;
                    }} else {{
                        t = param.time;
                    }}

                    // 1. Main Chart
                    let html = '';
                    if (bollData.length > 0) {{
                        const d = bollData.find(x => x.time === t);
                        if (d) {{
                            html += `<div class="legend-row"><span class="legend-label">BOLL</span><span class="legend-value" style="color:#FF4081">MID:${{d.mid.toFixed(2)}}</span><span class="legend-value" style="color:#FFD700">UP:${{d.up!=null?d.up.toFixed(2):'-'}}</span><span class="legend-value" style="color:#00E5FF">LOW:${{d.low!=null?d.low.toFixed(2):'-'}}</span></div>`;
                        }}
                    }}
                    if (maData.length > 0) {{
                        const d = maData.find(x => x.time === t);
                        if (d) {{
                            let maHtml = '<div class="legend-row"><span class="legend-label">EMA</span>';
                            if (d.ma5 != null) maHtml += `<span class="legend-value" style="color:#FFA500">EMA5:${{d.ma5.toFixed(2)}}</span> `;
                            if (d.ma10 != null) maHtml += `<span class="legend-value" style="color:#2196F3">EMA10:${{d.ma10.toFixed(2)}}</span> `;
                            if (d.ma20 != null) maHtml += `<span class="legend-value" style="color:#E040FB">EMA20:${{d.ma20.toFixed(2)}}</span> `;
                            if (d.ma60 != null) maHtml += `<span class="legend-value" style="color:#00E676">EMA60:${{d.ma60.toFixed(2)}}</span>`;
                            maHtml += '</div>';
                            html += maHtml;
                        }}
                    }}
                    if (mainLegendEl) mainLegendEl.innerHTML = html;

                    // 2. Vol Legend
                    if (volLegendEl && volData.length > 0) {{
                        const d = volData.find(x => x.time === t);
                        if (d) {{
                            const valInWan = (d.value / 10000).toFixed(2);
                            volLegendEl.innerHTML = `<div class="legend-row"><span class="legend-label">成交量</span><span class="legend-value" style="color: ${{d.color}}">VOL: ${{valInWan}}萬</span></div>`;
                        }}
                    }}

                    // 3. MACD Legend
                    if (macdLegendEl && macdData.length > 0) {{
                        const d = macdData.find(x => x.time === t);
                        if (d) {{
                            macdLegendEl.innerHTML = `<div class="legend-row"><span class="legend-label">MACD</span><span class="legend-value" style="color: #E6A23C">DIF: ${{d.dif.toFixed(3)}}</span><span class="legend-value" style="color: #2196F3">DEA: ${{d.dea.toFixed(3)}}</span><span class="legend-value" style="color: #E040FB">MACD: ${{d.hist.toFixed(3)}}</span></div>`;
                        }}
                    }}

                    // 4. ★KDJ Legend (新增)
                    if (kdjLegendEl && kdjData.length > 0) {{
                        const d = kdjData.find(x => x.time === t);
                        if (d) {{
                            kdjLegendEl.innerHTML = `<div class="legend-row"><span class="legend-label">KDJ</span><span class="legend-value" style="color: #E6A23C">K: ${{d.k.toFixed(3)}}</span><span class="legend-value" style="color: #2196F3">D: ${{d.d.toFixed(3)}}</span><span class="legend-value" style="color: #E040FB">J: ${{d.j.toFixed(3)}}</span></div>`;
                        }}
                    }}
                }}

                const allCharts = [mainChart, volChart, macdChart, kdjChart, rsiChart, obvChart, biasChart].filter(c => c !== null);
                allCharts.forEach(c => {{
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

    components.html(html_code, height=total_height)
