from data_pipeline.stage_01.historical_data import fetch_market_data
from data_pipeline.stage_01.news_collector import fetch_news_df
import pandas as pd
import argparse

def merge_data(symbol='AAPL', start_date='2025-01-01', end_date='2025-03-01',custom_keywords=None,false_positives=None):
    news_df = fetch_news_df(symbol=symbol, start_date=start_date, end_date = end_date,custom_keywords=custom_keywords,false_positives=false_positives)
    market_df = fetch_market_data(ticker=symbol, start_date=start_date, end_date=end_date)
    
    # Convert dates to datetime format
    market_df['Date'] = pd.to_datetime(market_df['Date'])
    news_df['Date'] = pd.to_datetime(news_df['Date'])
    
    # Merge the dataframes
    merged_df = pd.merge(
        market_df,
        news_df,
        on='Date',
        how='left'
    )

    merged_df['headline'] = merged_df['headline'].fillna("No headline available")
    print(merged_df.headline.isnull().sum())
    #merged_df.drop(columns=['Cleaned_Headlines','Ticker'], inplace=True)
    print("This is a market data shape",market_df.shape)
    print("This is a merged data", merged_df.shape)

    return merged_df

def parse_args():
    parser = argparse.ArgumentParser(description="Fetch and merge market and news data")
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol (e.g., AAPL, MSFT)')
    parser.add_argument('--start_date', type=str, default='2025-01-01', help="Start date in 'YYYY-MM-DD' format")
    parser.add_argument('--end_date', type=str, default='2025-03-01', help="End date in 'YYYY-MM-DD' format")
    parser.add_argument('--custom_keywords', type=str, nargs='+', help="List of custom keywords for filtering news")
    parser.add_argument('--false_positives', type=str, nargs='+', help="List of false positive keywords to exclude")

    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()

        # Convert keywords to lists (or None if not provided)
        custom_keywords = args.custom_keywords if args.custom_keywords else None
        false_positives = args.false_positives if args.false_positives else None

        merged_df = merge_data(symbol=args.symbol, start_date=args.start_date, end_date=args.end_date,
                               custom_keywords=custom_keywords, false_positives=false_positives)

        # Save merged data
        output_file = f"MERGED/merged_data_{args.symbol}_from_{args.start_date}_to_{args.end_date}.csv"
        merged_df.to_csv(output_file, index=False)
        print(f"✅ Merged data saved: {output_file}")

    except Exception as e:
        print(f"❌ An error occurred: {e}")