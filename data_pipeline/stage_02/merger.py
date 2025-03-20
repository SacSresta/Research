from data_pipeline.stage_01.historical_data import fetch_market_data
from data_pipeline.stage_01.news_collector import fetch_news_df
import pandas as pd
import argparse

def merge_data(symbol='AAPL', start_date='2025-01-01', end_date='2025-03-01'):
    news_df = fetch_news_df(symbol=symbol, start_date=start_date, end_date = end_date)
    market_df = fetch_market_data(ticker=symbol, start_date=start_date, end_date=end_date)
    
    # Convert dates to datetime format
    market_df['Date'] = pd.to_datetime(market_df['Date'])
    news_df['Date'] = pd.to_datetime(news_df['Date'])
    
    # Merge the dataframes
    merged_df = pd.merge(
        market_df,
        news_df,
        on='Date',
        how='inner'
    )

    return merged_df

def parse_args():

    parser = argparse.ArgumentParser(description="Fetch and merge market and news data")
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol (e.g., AAPL, MSFT)')
    parser.add_argument('--start_date', type=str, default='2025-01-01', help="Start date in 'YYYY-MM-DD' format")
    parser.add_argument('--end_date', type=str, default='2025-03-01', help="End date in 'YYYY-MM-DD' format")
    
    return parser.parse_args()

if __name__ == "__main__":
    try:

        args = parse_args()
        print(args)
        merged_df = merge_data(symbol=args.symbol, start_date=args.start_date, end_date=args.end_date)
        print(merged_df.tail(2))
        
    except Exception as e:
        print(f"An error occurred: {e}")