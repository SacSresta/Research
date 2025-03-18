import yfinance as yf
import pandas as pd

def fetch_market_data(ticker, start_date, end_date):
    df = yf.download(ticker, auto_adjust=False, start=start_date, end=end_date)
    df = df.droplevel('Ticker', axis=1)
    return df.reset_index()



if __name__ == "__main__":
    ticker = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2020-12-31'
    df = fetch_market_data(ticker, start_date, end_date)
    print(df.head())  # Print the first few rows of the DataFrame