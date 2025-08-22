import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker="RELIANCE.NS", period="3mo", interval="1d"):
    """
    Fetch stock price data using Yahoo Finance.
    ticker: stock symbol (e.g., RELIANCE.NS for Reliance, AAPL for Apple)
    period: how much history (e.g., "3mo" = 3 months)
    interval: data frequency (e.g., "1d" = daily)
    """
    data = yf.download(ticker, period=period, interval=interval)
    data.reset_index(inplace=True)
    return data

if __name__ == "__main__":
    # Example: Reliance Industries (India NSE) - fetch 1 year of daily data
    df = fetch_stock_data("RELIANCE.NS", period="1y", interval="1d")
    print(df.head())   # show first 5 rows
    df.to_csv("reliance_stock.csv", index=False)
    print("✅ Saved Reliance stock data to reliance_stock.csv")

