import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_close_prices(symbol, depth, aslist = True):
    end_date = datetime.today()
    days = round(depth * 365)
    start_date = end_date - timedelta(days=days)
    
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)
    
    close_prices = data['Close']
    if aslist:
        return list(close_prices)
    return close_prices

def parse_asset_data(write, depth: float, assets: list):
    not_found = []
    data = pd.DataFrame()

    for ticker in args:
        try:
            prices = fetch_close_prices(ticker, depth)
            data[ticker] = prices
        except:
            not_found.append(ticker)
    
    write(f"Excluded Tickers: {', '.join(not_found)}")
    data.dropna(inplace=True)
    return data
