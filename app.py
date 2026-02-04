import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import streamlit.components.v1 as components

# ---------------------------------------------------------
# 1. 頁面設定
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Futu Desktop JS-Engine")

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
# 2. 資料層 (更加嚴格的數據格式化)
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

        # 強制刪除 OHLC 為空的行
        data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
        data.columns = [str(col).lower() for col in data.columns]
        
        close_col = 'close' if 'close' in data.columns else 'adj close'
        if close_col not in data.columns: return None

        # 指標計算
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
        
        # 日期處理
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
        data['time'] = data['date_obj'].astype('int64') // 10**9 # 轉秒
        
        # 【關鍵修復】確保時間排序，否則 JS 會崩潰
        data = data.sort_values('time')
            
        return data
    except: return None

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
    with c_top1: st.subheader(f"{ticker} 走勢圖")
    with c_top2: interval_label = st.radio("週期", ["日K", "週K", "月K", "年K"], index=0, horizontal=True, label_visibility="collapsed")
    
    interval_map = {"日K": "1d", "週K": "1wk", "月K": "1mo", "年K": "1y"}
    full_df = get_data(ticker, period="max", interval=interval_map[interval_label])
    
    if full_df is None:
        st.error("無數據")
        st.stop()

    # ---------------------------------------------------------
    # 4. 數據轉 JSON (修復 NaN 問題)
    # ---------------------------------------------------------
    def to_json_list(df, cols, time_col='time'):
        res = []
        # 將 NaN 替換為 None，這樣 json.dumps 會轉成 null，JS 才能讀
        df_clean = df.where(pd.notnull(df), None)
        
        for _, row in df_clean.iterrows():
            try:
                item = {'time': int(row[time_col])}
                valid = True
                for k, v in cols.items():
                    val = row.get(v)
                    # K線圖如果不完整，直接跳過這一天
                    if k in ['open', 'high', 'low', 'close'] and val is None:
                        valid = False; break
                    item[k] = val
                if valid: res.append(item)
            except: continue
        return json.dumps(res) # 這裡輸出的 null 是 JS 合法的

    # 準備數據
    candles_json = to_json_list(full_df, {'open':'open', 'high':'high', 'low':'low', 'close':'close'})
    vol_json = to_json_list(full_df, {'value':'volume'}) if show_vol else "[]"
    
    ma_json = "[]"
    if show_ma:
        ma_cols = {}
        for c in ['ma5', 'ma10', 'ma20', 'ma60']:
            if c in full_df.columns: ma_cols[c] = c
        ma_json = to_json_list(full_df, ma_cols)

    boll_json = "[]"
    if show_boll:
        boll_json = to_json_list(full_df, {'up':'boll_upper', 'mid':'boll_mid', 'low':'boll_lower'})

    macd_json = "[]"
    if show_macd:
        macd_json = to_json_list(full_df, {'dif':'macd_12_26_9', 'dea':'macds_12_26_9', 'hist':'macdh_12_26_9'})

    kdj_json = "[]"
    if show_kdj:
        kdj_json = to_json_list(full_df, {'k':'stochk_14_3_3', 'd':'stochd_14_3_3'})
        
    rsi_json = "[]"
    if show_rsi:
        rsi_json = to_json_list(full_df, {'rsi':'rsi'})

    obv_json = "[]"
    if show_obv:
        obv_json = to_json_list(full_df, {'obv':'obv'})

    bias_json = "[]"
    if show_bias:
        bias_json = to_json_list(full_df, {'bias':'bias'})

    # ---------------------------------------------------------
    # 5. JavaScript 引擎
    # ---------------------------------------------------------
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
        <style>
            body {{ margin: 0; padding: 0; background-color: #ffffff; overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }}
            .chart-container {{ position: relative; width: 100%; }}
        </style>
    </head>
    <body>
        <div id="main-chart" class="chart-container" style="height: 450px;"></div>
        <div id="vol-chart" class="chart-container" style="height: {'100px' if show_vol else '0px'}; display: {'block' if show_vol else 'none'};"></div>
        <div id="macd-chart" class="chart-container" style="height: {'150px' if show_macd else '0px'}; display: {'block' if show_macd else 'none'};"></div>
        <div id="kdj-chart" class="chart-container" style="height: {'120px' if show_kdj else '0px'}; display: {'block' if show_kdj else 'none'};"></div>
        <div id="rsi-chart" class="chart-container" style="height: {'120px' if show_rsi else '0px'}; display: {'block' if show_rsi else 'none'};"></div>
        <div id="obv-chart" class="chart-container" style="height: {'120px' if show_obv else '0px'}; display: {'block' if show_obv else 'none'};"></div>
        <div id="bias-chart" class="chart-container" style="height: {'120px' if show_bias else '0px'}; display: {'block' if show_bias else 'none'};"></div>

        <script>
            // 數據注入
            const candlesData = {candles_json};
            const volData = {vol_json};
            const maData = {ma_json};
            const bollData = {boll_json};
            const macdData = {macd_json};
            const kdjData = {kdj_json};
            const rsiData = {rsi_json};
            const obvData = {obv_json};
            const biasData = {bias_json};
            
            const ticker = "{ticker}";

            // 檢查數據是否為空，避免崩潰
            if (!candlesData || candlesData.length === 0) {{
                document.body.innerHTML = '<div style="padding:20px; color:red;">No Data Available for rendering</div>';
            }}

            const chartOptions = {{
                layout: {{ backgroundColor: '#FFFFFF', textColor: '#333333' }},
                grid: {{ vertLines: {{ color: '#F0F0F0' }}, horzLines: {{ color: '#F0F0F0' }} }},
                rightPriceScale: {{ borderColor: '#E0E0E0', scaleMargins: {{ top: 0.1, bottom: 0.1 }}, visible: true, minimumWidth: 80 }},
                timeScale: {{ borderColor: '#E0E0E0', timeVisible: true, rightOffset: 5 }},
                crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
            }};

            const charts = [];
            
            function createChart(id, opts) {{
                const el = document.getElementById(id);
                if (el.style.display === 'none') return null;
                const chart = LightweightCharts.createChart(el, opts);
                charts.push(chart);
                return chart;
            }}

            const mainChart = createChart('main-chart', chartOptions);
            const volChart = createChart('vol-chart', {{...chartOptions, rightPriceScale: {{...chartOptions.rightPriceScale, scaleMargins: {{top: 0.2, bottom: 0}}}}}}); 
            const macdChart = createChart('macd-chart', chartOptions);
            const kdjChart = createChart('kdj-chart', chartOptions);
            const rsiChart = createChart('rsi-chart', chartOptions);
            const obvChart = createChart('obv-chart', chartOptions);
            const biasChart = createChart('bias-chart', chartOptions);

            // --- 繪圖 ---
            if (mainChart) {{
                const candleSeries = mainChart.addCandlestickSeries({{
                    upColor: '#FF5252', downColor: '#00B746', borderUpColor: '#FF5252', borderDownColor: '#00B746', wickUpColor: '#FF5252', wickDownColor: '#00B746'
                }});
                candleSeries.setData(candlesData);

                if (maData.length > 0) {{
                    const first = maData[0];
                    if (first.ma5 !== undefined) mainChart.addLineSeries({{ color: '#FFA500', lineWidth: 1, title: 'EMA5' }}).setData(maData.map(d => ({{ time: d.time, value: d.ma5 }})));
                    if (first.ma10 !== undefined) mainChart.addLineSeries({{ color: '#2196F3', lineWidth: 1, title: 'EMA10' }}).setData(maData.map(d => ({{ time: d.time, value: d.ma10 }})));
                    if (first.ma20 !== undefined) mainChart.addLineSeries({{ color: '#E040FB', lineWidth: 1, title: 'EMA20' }}).setData(maData.map(d => ({{ time: d.time, value: d.ma20 }})));
                    if (first.ma60 !== undefined) mainChart.addLineSeries({{ color: '#00E676', lineWidth: 1, title: 'EMA60' }}).setData(maData.map(d => ({{ time: d.time, value: d.ma60 }})));
                }}

                if (bollData.length > 0) {{
                    mainChart.addLineSeries({{ color: '#2962FF', lineWidth: 1, lineStyle: 2, title: 'BBU' }}).setData(bollData.map(d => ({{ time: d.time, value: d.up }})));
                    mainChart.addLineSeries({{ color: '#2962FF', lineWidth: 1, lineStyle: 2, title: 'MID' }}).setData(bollData.map(d => ({{ time: d.time, value: d.mid }})));
                    mainChart.addLineSeries({{ color: '#2962FF', lineWidth: 1, lineStyle: 2, title: 'BBL' }}).setData(bollData.map(d => ({{ time: d.time, value: d.low }})));
                }}
            }}

            if (volChart && volData.length > 0) {{
                const volSeries = volChart.addHistogramSeries({{ priceFormat: {{ type: 'volume' }}, title: 'VOL' }});
                volSeries.setData(volData);
            }}

            if (macdChart && macdData.length > 0) {{
                macdChart.addLineSeries({{ color: '#FFA500', lineWidth: 1, title: 'DIF' }}).setData(macdData.map(d => ({{ time: d.time, value: d.dif }})));
                macdChart.addLineSeries({{ color: '#2196F3', lineWidth: 1, title: 'DEA' }}).setData(macdData.map(d => ({{ time: d.time, value: d.dea }})));
                const histSeries = macdChart.addHistogramSeries({{ title: 'MACD' }});
                histSeries.setData(macdData.map(d => ({{ time: d.time, value: d.hist, color: d.hist > 0 ? '#FF5252' : '#00B746' }})));
            }}

            if (kdjChart && kdjData.length > 0) {{
                kdjChart.addLineSeries({{ color: '#FFA500', title: 'K' }}).setData(kdjData.map(d => ({{ time: d.time, value: d.k }})));
                kdjChart.addLineSeries({{ color: '#2196F3', title: 'D' }}).setData(kdjData.map(d => ({{ time: d.time, value: d.d }})));
            }}

            if (rsiChart && rsiData.length > 0) {{
                rsiChart.addLineSeries({{ color: '#E040FB', title: 'RSI' }}).setData(rsiData.map(d => ({{ time: d.time, value: d.rsi }})));
            }}

            if (obvChart && obvData.length > 0) {{
                obvChart.addLineSeries({{ color: '#FFA500', title: 'OBV', priceFormat: {{ type: 'volume' }} }}).setData(obvData.map(d => ({{ time: d.time, value: d.obv }})));
            }}
            
            if (biasChart && biasData.length > 0) {{
                biasChart.addLineSeries({{ color: '#607D8B', title: 'BIAS' }}).setData(biasData.map(d => ({{ time: d.time, value: d.bias }})));
            }}

            // --- 同步與記憶 ---
            charts.forEach(c => {{
                c.timeScale().subscribeVisibleLogicalRangeChange(range => {{
                    if (range) {{
                        charts.forEach(other => {{ if (other !== c) other.timeScale().setVisibleLogicalRange(range); }});
                        localStorage.setItem('futu_mem_' + ticker, JSON.stringify(range));
                    }}
                }});
            }});

            const saved = localStorage.getItem('futu_mem_' + ticker);
            if (saved && mainChart) {{
                try {{ mainChart.timeScale().setVisibleLogicalRange(JSON.parse(saved)); }} catch(e){{}}
            }} else if (mainChart) {{
                mainChart.timeScale().fitContent();
            }}

            window.addEventListener('resize', () => {{
                charts.forEach(c => c.resize(document.body.clientWidth, c.options().height));
            }});
        </script>
    </body>
    </html>
    """
    
    # 計算總高度
    total_height = 460
    if show_vol: total_height += 100
    if show_macd: total_height += 150
    if show_kdj: total_height += 120
    if show_rsi: total_height += 120
    if show_obv: total_height += 120
    if show_bias: total_height += 120

    components.html(html_code, height=total_height)
