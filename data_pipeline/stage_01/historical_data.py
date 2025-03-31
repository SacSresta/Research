import yfinance as yf
import pandas as pd
import numpy as np
np.NaN = np.nan
import pandas_ta as ta

def add_technical_indicators(df):
    """Add common technical indicators to a DataFrame using pandas_ta"""
    
    # Set Date as index (required for time-based calculations)
    if 'Date' in df.columns:
        df = df.set_index('Date')

    # Momentum Indicators
    df.ta.rsi(length=14, close='Close', append=True)  
    df.ta.macd(fast=12, slow=26, signal=9, close='Close', append=True) 

    # Trend Indicators
    df.ta.adx(length=14, high='High', low='Low', close='Close', append=True)  
    df.ta.ema(length=20, close='Close', append=True) 

    # Volatility Indicators
    df.ta.bbands(length=20, std=2, close='Close', append=True) 

    # Volume Indicators
    df.ta.obv(close='Close', volume='Volume', append=True)  

    # Oscillators
    df.ta.stoch(high='High', low='Low', close='Close', append=True)  

    # Optional: Cleanup intermediate columns
    df.drop(columns=['BBM_20_2.0'], errors='ignore', inplace=True) 
    
    return df.reset_index()  

def supertrend(df, factor=3, atr_period=10):
    # Calculate ATR
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR - using Wilder's smoothing method
    atr = tr.copy()
    for i in range(1, len(atr)):
        if not np.isnan(tr.iloc[i]) and not np.isnan(atr.iloc[i-1]):
            atr.iloc[i] = (atr.iloc[i-1] * (atr_period - 1) + tr.iloc[i]) / atr_period
    
    # Calculate midpoint and bands
    hl2 = (high + low) / 2
    
    # Basic upper and lower bands
    upper_band_basic = hl2 + (factor * atr)
    lower_band_basic = hl2 - (factor * atr)
    
    # Final upper and lower bands with the trend logic
    upper_band = upper_band_basic.copy()
    lower_band = lower_band_basic.copy()
    
    # Initialize supertrend and direction
    st = pd.Series(0.0, index=df.index)
    direction = pd.Series(0, index=df.index)
    
    # Set initial values
    if len(df) > 0:
        direction.iloc[0] = -1  # Start with downtrend (-1 for down)
        st.iloc[0] = upper_band.iloc[0]  # Start with upper band
    
    # Apply the recursive logic
    for i in range(1, len(df)):
        # Skip calculations on NaN values
        if np.isnan(atr.iloc[i]):
            continue
            
        # Adjust upper band
        if (upper_band_basic.iloc[i] < upper_band.iloc[i-1]) or (close.iloc[i-1] > upper_band.iloc[i-1]):
            upper_band.iloc[i] = upper_band_basic.iloc[i]
        else:
            upper_band.iloc[i] = upper_band.iloc[i-1]
            
        # Adjust lower band
        if (lower_band_basic.iloc[i] > lower_band.iloc[i-1]) or (close.iloc[i-1] < lower_band.iloc[i-1]):
            lower_band.iloc[i] = lower_band_basic.iloc[i]
        else:
            lower_band.iloc[i] = lower_band.iloc[i-1]
        
        # Determine trend direction
        if close.iloc[i] > upper_band.iloc[i-1]:
            direction.iloc[i] = 1  # Uptrend
        elif close.iloc[i] < lower_band.iloc[i-1]:
            direction.iloc[i] = -1  # Downtrend
        else:
            direction.iloc[i] = direction.iloc[i-1]  # No change
            
        # Set supertrend value
        if direction.iloc[i] == 1:
            st.iloc[i] = lower_band.iloc[i]
        else:
            st.iloc[i] = upper_band.iloc[i]
    
    # Add calculated values to DataFrame
    df['Supertrend'] = st
    df['Direction'] = direction
    df['UpperBand'] = upper_band
    df['LowerBand'] = lower_band
    df['Uptrend'] = np.where(direction == 1, st, np.nan)
    df['Downtrend'] = np.where(direction == -1, st, np.nan)
    
    return df



def fetch_market_data(ticker, start_date, end_date):
    df = yf.download(ticker, auto_adjust=False, start=start_date, end=end_date)
    df = df.droplevel('Ticker', axis=1)
    df.reset_index(inplace=True)
    df = supertrend(df , factor=3, atr_period=10)
    return df





if __name__ == "__main__":
    ticker = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2020-12-31'
    df = fetch_market_data(ticker, start_date, end_date)
    print(df.head())  # Print the first few rows of the DataFrame