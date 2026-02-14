<style>
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    h3 {margin-bottom: 0px; color: #1A2A3A;}
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
    div.stButton > button[kind="primary"] {background-color: #1A365D !important; color: white !important;}
    
    /* 日式極簡風策略儀表板樣式 */
    .strategy-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr); 
        gap: 12px;
        margin-bottom: 20px;
    }
    .strat-card {
        background-color: #FCFBF9; /* 米白底色 */
        border: 1px solid #EBEBEB;
        border-radius: 12px; /* 粗圓角 */
        padding: 16px 10px; 
        text-align: center;
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        transition: all 0.3s ease;
    }
    /* 觸發成功的狀態 */
    .strat-active {
        background-color: #FFFFFF; 
        border: 1.5px solid #1A365D; /* 海軍藍邊框 */
        box-shadow: 0 4px 8px rgba(26, 54, 93, 0.08);
    }
    .strat-title { 
        font-size: 13px; 
        color: #1A365D; /* 海軍藍標題 */
        font-weight: 600; 
        margin-bottom: 8px; 
        letter-spacing: 0.5px;
    }
    .strat-status { 
        font-size: 15px; 
        font-weight: bold; 
    }
    
    .status-match { color: #D4AF37; } /* 芥末黃重點色 */
    .status-wait { color: #A0A0A0; font-weight: normal; }  /* 未符合時的灰黑色 */
</style>
