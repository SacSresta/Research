import pandas as pd
import numpy as np
np.NaN = np.nan
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


def SSL(data, period):
    data['smaHigh'] = data['High'].rolling(window=period).mean()
    data['smaLow'] = data['Low'].rolling(window=period).mean()
    data['sslDown'] = data.apply(lambda row: row['smaHigh'] if row['Close'] < row['smaLow'] else row['smaLow'], axis=1)
    data['sslUp'] = data.apply(lambda row: row['smaLow'] if row['Close'] < row['smaLow'] else row['smaHigh'], axis=1)
    data['Trend'] = data.apply(lambda row: 1 if row['smaLow'] == row['sslDown'] else -1, axis=1)
    data['ssl_signal'] = (data['Trend'] != data['Trend'].shift(1)).astype(int)
    return data




def lstm_df(df):
  X = df[['Close','sentiment_score','shifted_ssl_signal']]

  # Create lagged features for the target and close price
  X['shifted_ssl_prev'] = X['shifted_ssl_signal'].shift(1)  # Previous day's target
  X.dropna(inplace=True)
  return X

def create_data (look_back = 14, features_per_step = 3, data=None):
  
    # Initialize sequences
    look_back = 14  # 3-day window
    features_per_step = features_per_step  # Close, shifted_ssl_prev, sentiment

    X_train, y_train = [], []

    X = data
    for i in range(look_back + 1, len(X)):

        # Extract sequence for current sample (i-3, i-2, i-1)
        seq = []
        for j in range(look_back, 0, -1):
            # Timestep features:
            # [Close, shifted_ssl_prev (1-day lag), sentiment (0 for first two days)]
            close = X['Close'].iloc[i - j]

            ssl_prev = X['shifted_ssl_prev'].iloc[i - j - 1] if (i - j - 1) >= 0 else 0.0
            if j == 1:  # Current timestep (i-1 corresponding to "t" in input)
                sentiment = X['sentiment_score'].iloc[i - 1]
            else:
                sentiment = 0.0
            seq.append([close, ssl_prev, sentiment])
        X_train.append(seq)

        y_train.append(X['shifted_ssl_signal'].iloc[i].astype(int))


    return X_train,y_train
def convert_to_array(x,y):

  return np.array(x),np.array(y)

def scaling_func(x):
  # Reshape data for scaling (batch, sequence, features) → (batch*sequence, features)
  n_samples, n_timesteps, n_features = x.shape
  X_reshaped = x.reshape(-1, n_features)

  # Scale Close and sentiment (columns 0 and 2)
  scaler = StandardScaler()
  X_reshaped[:, [0, 2]] = scaler.fit_transform(X_reshaped[:, [0, 2]])

  # Reshape back to original structure
  X_scaled = X_reshaped.reshape(n_samples, n_timesteps, n_features)
  return X_scaled


def parse_args():

    parser = argparse.ArgumentParser(description="Machine Learning or Deep Learning Preprocessor")
    parser.add_argument('--ml', type=bool, default=False, help='it will give ml dataframe')

    
    return parser.parse_args()

def preprocess_data(data = None, look_back = 4 ,ml=False, split = 0.8, path = None):
    if path:
        df = pd.read_csv(path)
    else:
        df = data
    df.Date = pd.to_datetime(df.Date)
    df = df.drop(columns=["Ticker", "headline"])
    
    # Compute SSL indicator
    df = SSL(data=df.copy(), period=10)
    
    
    # Shift SSL signal for prediction alignment
    df["shifted_ssl_signal"] = df.Trend.shift(-1)
    
    # Drop NaN values and reset index
    df = df.dropna().reset_index(drop=True)

    if ml:  # ✅ No need to write `== True`
        return df
    else:  # ✅ Correct indentation
        data = lstm_df(df)
        x, y = create_data(data=data, look_back=look_back)

        X, y = convert_to_array(x, y)
        X_scaled = scaling_func(X)

        train_size = int(split * len(X_scaled))
        X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]

        y_train, y_test = y[:train_size].reshape(-1, 1), y[train_size:].reshape(-1, 1)

        return X_train,y_train,X_test,y_test

   

if __name__ == "__main__":
    args = parse_args()
    
    X_train,y_train,X_test,y_test = preprocess_data(path='/home/sacsresta/Documents/RESEARCH/Project/sentiment/merged_data_AAPL_from_2024-01-01_to_2025-01-01.csv')
    print(X_train)