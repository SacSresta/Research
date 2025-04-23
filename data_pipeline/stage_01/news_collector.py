
import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import json
import re



def get_historical_data(start_date='2020-01-01', symbol='AAPL', end_date=datetime.today().strftime("%Y-%m-%d")):
    def time_converter(time_str):
        """ Convert ISO 8601 timestamp to YYYY-MM-DD format. """
        if time_str.endswith('Z'):
            time_str = time_str[:-1] + '+00:00'
        try:
            timestamp_obj = datetime.fromisoformat(time_str)
        except ValueError:
            timestamp_obj = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%f%z")
        return timestamp_obj.strftime('%Y-%m-%d')

    json_collector = []

    while True:
        if start_date >= end_date:
            print("Reached the latest date. Stopping loop.")
            break

        # Adjust symbol if META and date is before 2021-10-28
        query_symbol = 'FB' if symbol == 'META' and start_date < '2021-10-28' else symbol

        url = f"https://data.alpaca.markets/v1beta1/news?start={start_date}&sort=asc&symbols={query_symbol}&limit=50"

        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": "PKY7ODAXYMWTX4JK7DVR",
            "APCA-API-SECRET-KEY": "yvQclRacekxCgwL0KX70SFkdbBAsVRvrBQnYaYGY"
        }

        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print("Error fetching data:", response.status_code)
            break

        data = response.json()

        if 'news' not in data or not data['news']:
            print("No more news data available. Stopping loop.")
            break

        json_collector.extend(data['news'])

        last_news_time = data['news'][-1]['updated_at']
        new_start_date = time_converter(last_news_time)

        if new_start_date != start_date:
            start_date = new_start_date
        else:
            new_start_date_dt = datetime.strptime(new_start_date, "%Y-%m-%d") + timedelta(days=1)
            start_date = new_start_date_dt.strftime("%Y-%m-%d")
    return json_collector


def preprocess_data(symbol, start_date,end_date):
    # Load the data
    json_data = get_historical_data(start_date=start_date,symbol=symbol,end_date =end_date)
    df = pd.DataFrame(json_data)
    df['Time'] = pd.DatetimeIndex(df['updated_at'])
    df['Ticker'] = symbol
    df =df.drop(columns=['updated_at','created_at','content','images','symbols','url','author','source','summary','id']).set_index(['Time'])

    return df

def fix_data(df):
    df["Time"] = pd.to_datetime(df.index)
    df["Date"] = df["Time"].dt.date  # Extract only the date part

    # Group by 'Ticker' and 'Date' and merge headlines
    df_grouped = df.groupby(["Ticker", "Date"])["headline"].apply(lambda x: " | ".join(x)).reset_index()

    return df_grouped

def clean_alpaca_news(
    df: pd.DataFrame, 
    ticker: str = "AAPL",
    custom_keywords: list = None,
    false_positives: list = None
) -> pd.DataFrame:

    # Initialize configurations
    ticker_keywords = [
        ticker.lower(), 'iphone', 'ipad', 'ios', 'mac', 'apple watch',
        'app store', 'tim cook', 'icloud', 'itunes'
    ]
    if custom_keywords:
        ticker_keywords += [kw.lower() for kw in custom_keywords]

    false_positives = false_positives or [
        'pineapple', 'apple fruit', 'apple pie', 'grapple', 'big apple'
    ]

    # Preprocessing pipeline
    def _clean_headline(text):
        text = str(text).lower()
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        text = re.sub(r'(\-\w+)\b', '', text)  # Remove source tags
        text = re.sub(r'\b(says|reports|according)\b.*$', '', text)
        return text.strip()
    
    processed = (
        df.loc[df['Ticker'] == ticker]
        .assign(headline=lambda x: x['headline'].str.split(' \| '))
        .explode('headline')
        .assign(cleaned=lambda x: x['headline'].apply(_clean_headline))
        .drop_duplicates(['Ticker', 'Date', 'cleaned'])
        .sort_values('Date')
    )

    # Relevance validation
    pos_pattern = r'\b(' + '|'.join(ticker_keywords) + r')\b'
    neg_pattern = r'\b(' + '|'.join(false_positives) + r')\b'
    
    processed['is_ticker'] = (
        processed['cleaned']
        .str.contains(pos_pattern, regex=True)
        .astype(int)
    ) & (
        ~processed['cleaned']
        .str.contains(neg_pattern, regex=True)
        .astype(int)
    )

    processed = processed[processed['is_ticker'] == 1]

    # Group and augment
    grouped = processed.groupby(['Ticker', 'Date']).agg(
        Headline_Count=('cleaned', 'count'),
        Headlines=('headline', list),
        Cleaned_Headlines=('cleaned', list),
        Mentioned_Tickers=('cleaned', 
            lambda x: list(set(re.findall(r'\b([A-Z]{2,4})\b', ' '.join(x))))
        )
    ).reset_index()

    return grouped.assign(
        First_Headline=lambda x: x['Headlines'].str[0],
        Date=lambda x: pd.to_datetime(x['Date'])
    )


def fetch_news_df(
    symbol, 
    start_date, 
    end_date=None, 
    custom_keywords=None, 
    false_positives=None
):
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    df = preprocess_data(symbol, start_date, end_date)

    df = fix_data(df)

    output_dir = 'MERGED'  
    output_file = f'merged_data_{symbol}_from_{start_date}_to_{end_date}.csv'  
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    df.to_csv(output_path, index=False)
    
    return df

if __name__ == "__main__":
    symbol = "TSLA"
    start_date = '2025-01-01'
    end_date = '2025-03-01'
    keywords = ['TSLA','tesla','elon','twitter','x']
    df = fetch_news_df(symbol, start_date, end_date, custom_keywords=keywords)  
    
    print(df.head(10))
    