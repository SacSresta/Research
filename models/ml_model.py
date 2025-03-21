import pandas as pd
import numpy as np
np.NaN = np.nan
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns


def SSL(data, period):
    data['smaHigh'] = data['High'].rolling(window=period).mean()
    data['smaLow'] = data['Low'].rolling(window=period).mean()
    data['sslDown'] = data.apply(lambda row: row['smaHigh'] if row['Close'] < row['smaLow'] else row['smaLow'], axis=1)
    data['sslUp'] = data.apply(lambda row: row['smaLow'] if row['Close'] < row['smaLow'] else row['smaHigh'], axis=1)
    data['Trend'] = data.apply(lambda row: 1 if row['smaLow'] == row['sslDown'] else -1, axis=1)
    data['ssl_signal'] = (data['Trend'] != data['Trend'].shift(1)).astype(int)
    return data



if __name__ == "__main__":
    path = '/home/sacsresta/Documents/RESEARCH/Project/sentiment/merged_data_AAPL_from_2024-01-01_to_2025-01-01.csv'

    df = pd.read_csv(path)
    print(df)
    df.Date = pd.to_datetime(df.Date)
    df = df.drop(columns = ['Ticker','headline'])
    df = SSL(data=df.copy(), period=10)
    df['kama'] = ta.kama(df.Close, length=14)
    df['kama_signal'] = (df.Close > df.kama).astype(int)
    df['shifted_ssl_signal'] = df.Trend.shift(-1)
    df = df.dropna(ignore_index=True)
    print(df)