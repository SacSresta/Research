from data_pipeline.stage_01.historical_data import fetch_market_data
from data_pipeline.stage_01.news_collector import fetch_news_df
import pandas as pd

def merge_data(symbol='AAPL', start_date='2025-01-01', end_date='2025-03-17'):
    """
    Fetch and merge market data and news data for a given symbol and date range.
    
    Parameters:
    ----------
    symbol : str, default 'AAPL'
        The stock symbol to fetch data for
    start_date : str, default '2025-01-01'
        Start date for fetching data in 'YYYY-MM-DD' format
    end_date : str, default '2025-03-17'
        End date for fetching data in 'YYYY-MM-DD' format
        
    Returns:
    -------
    pandas.DataFrame
        Merged dataframe containing market and news data
    """
    news_df = fetch_news_df(symbol=symbol, start_date=start_date)
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
    merged_df.to_csv(f"Merged_{symbol}_from_{start_date}.csv")

    return merged_df

if __name__ == "__main__":
    # Example usage with parameters
    merged_df = merge_data(symbol='MSFT', start_date='2020-01-01')
    print(merged_df.head(2))
    
    # Use default parameters
    # merged_df = main()
    # print(merged_df.head(2))