import streamlit as st
from stqdm import stqdm

from utils.get_tickers import get_tickers as gt
from analytics import StreamlitReport

st.set_page_config(layout="wide", page_title="QStats", page_icon=":chart:")

st.title("EES | QStats Vizualizer")
st.caption("Visualization App for Tools within QStats")

@st.cache_data
def pull():
    with open("data/eqetf_tickers.txt") as f:
        stocks = f.read().split("\n")

    with open("data/etf_tickers.txt") as f:
        etfs = f.read().split("\n")
    
    with open("data/forex_tickers.txt") as f:
        forex = f.read().split("\n")
    
    return stocks, etfs, forex

stock_tickers, etf_tickers, forex_tickers = pull()



cola, colb = st.columns([3, 2])
sectype = cola.radio("Supported Instrument Categories", 
                    ["Equities & ETFs", "Forex"],
                    horizontal=True
)

cola.info("Equities | NASDAQ, NYSE, AMEX")
stocks = cola.multiselect("Select", stock_tickers, 
                          default=["AAPL", "MSFT", "GOOGL", "META"], 
                          label_visibility="collapsed", disabled=(sectype != "Equities & ETFs"))

cola.info("ETFs | ~500 From Yahoo Finance")
etfs = cola.multiselect("Select", etf_tickers, default=["SPY"], 
                        label_visibility="collapsed", disabled=(sectype != "Equities & ETFs"))

cola.info("Forex | ~30 Pairs from Yahoo")
forex = cola.multiselect("Select", forex_tickers, default=forex_tickers[:3], 
                          label_visibility="collapsed", disabled=(sectype != "Forex"))

if sectype == "Equities & ETFs":
    assets = stocks + etfs
else:
    assets = forex

metric, time = colb.columns([2, 3])
comparison_metric = metric.selectbox("Asset Comparison Metric", ["Close Prices"])
depth = time.number_input(label="Depth of Historical Data (Years)", value=5.0, min_value=0.5, max_value=10.0, step=0.5)

metrics = {}
with colb.expander("Specify Report Features", expanded=True):
    a, b = st.columns(2)
    metrics["df_snapshot"] = a.toggle("DataFrame Snapshot", value=True)
    metrics["simple_plot"] = b.toggle("Simple Multi-Asset Price Graph", value=True)
    metrics["price_distro"] = a.toggle("Price Distributions (Hist)", value=True)
    metrics["simple_sma"] = b.toggle("Price vs. 30-Day SMA", value=True)
    metrics["asset_vol"] = a.toggle("Normalized Asset Price Volatilities", value=True)
    metrics["price_corr_map"] = b.toggle("Price Correlations Heatmap", value=True)
    metrics["price_corr_viz"] = a.toggle("Asset Price Pairplot (Correlations)", value=True)
    metrics["sma_corr_map"] = b.toggle("SMA Correlations Heatmap", value=True)
    metrics["monthly_returns"] = a.toggle("Monthly Returns Correlations Heatmap", value=True)
    expanded = b.toggle("Pre-Expanded Report Features", value=True)


submit = st.button("Generate Asset Report", use_container_width=True)
    
if submit and len(assets) == 1:
    st.warning("Select 2+ Securities or Use B-Type Analytics")
if submit:
    report = StreamlitReport(assets, depth, {})
    report.generate_report(metrics, expanded=expanded)
