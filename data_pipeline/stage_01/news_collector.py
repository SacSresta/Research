
import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import json




def get_historical_data(start_date = '2020-01-01', symbol='AAPL',end_date = datetime.today().strftime("%Y-%m-%d")):

    def time_converter(time_str):
        """ Convert ISO 8601 timestamp to YYYY-MM-DD format. """
        # Handle 'Z' timezone indicator by replacing it with +00:00
        if time_str.endswith('Z'):
            time_str = time_str[:-1] + '+00:00'
        
        try:
            # Try to parse with fromisoformat
            timestamp_obj = datetime.fromisoformat(time_str)
        except ValueError:
            # Alternative parsing if fromisoformat fails
            timestamp_obj = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%f%z")
        
        return timestamp_obj.strftime('%Y-%m-%d')

    start_date = start_date

    json_collector = []
    while True:
        end_date = end_date
        
        # Stop condition to prevent infinite loop
        if start_date >= end_date:
            print("Reached the latest date. Stopping loop.")
            break

        symbol = symbol
        url = f"https://data.alpaca.markets/v1beta1/news?start={start_date}&sort=asc&symbols={symbol}&limit=50"

        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": "PKY7ODAXYMWTX4JK7DVR",
            "APCA-API-SECRET-KEY": "yvQclRacekxCgwL0KX70SFkdbBAsVRvrBQnYaYGY"
        }

        response = requests.get(url, headers=headers)
        
        # Check if the response contains valid JSON data
        if response.status_code != 200:
            print("Error fetching data:", response.status_code)
            break
        
        data = response.json()
        
        # If no more news is returned, break the loop
        if 'news' not in data or not data['news']:
            print("No more news data available. Stopping loop.")
            break
        
        json_collector.extend(data['news'])
        
        # Extract last updated date from the latest fetched news
        last_news_time = data['news'][-1]['updated_at']
        new_start_date = time_converter(last_news_time)

        # Update the start_date more efficiently
        if new_start_date != start_date:
            start_date = new_start_date
            print(f"Same date as previous batch: {start_date}")
        else:
            # Add one day to the start date to avoid duplication
            new_start_date_dt = datetime.strptime(new_start_date, "%Y-%m-%d") + timedelta(days=1)
            start_date = new_start_date_dt.strftime("%Y-%m-%d")
        
        print("Fetching next batch starting from:", start_date)

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


def fetch_news_df(symbol, start_date, end_date = datetime.today().strftime("%Y-%m-%d")):
    df = preprocess_data(symbol, start_date,end_date)
    df = fix_data(df)
    output_dir = 'MERGED'  
    output_file = f'merged_data_{symbol}_from_{start_date}.csv'  
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    symbol = "TSLA"
    start_date = '2025-01-01'
    end_date = '2025-03-01'
    df = fetch_news_df(symbol, start_date, end_date)  # Fetch the data for the specified symbol
    print(df.head(10))
    