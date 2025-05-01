from alpaca.data.historical import NewsClient
from alpaca.data.requests import StockBarsRequest
import os
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
from alpaca.data.historical import  StockHistoricalDataClient,NewsClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.common.enums import Sort
from datetime import datetime
from data_pipeline.stage_01.historical_data import add_technical_indicators,supertrend
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
from alpaca.data.historical import NewsClient
from alpaca.data.requests import NewsRequest
import time
import joblib
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import OrderRequest,GetAssetsRequest,GetCalendarRequest,StopLossRequest, LimitOrderRequest,CancelOrderResponse,StopOrderRequest,GetOrdersRequest
from alpaca.trading.enums import OrderSide,OrderType,TimeInForce,OrderStatus

load_dotenv()

api = os.getenv('ALPACA_API_KEY')
secret = os.getenv('ALPACA_SECRET_KEY')

def create_order_buy(symbol,qty = 10,api = api,secret = secret):

    tcliet = TradingClient(api,secret)

    params = OrderRequest(symbol=symbol,qty=qty,side=OrderSide.BUY,type=OrderType.MARKET, time_in_force=TimeInForce.DAY)
    tcliet.submit_order(params)

def get_status():
    tcliet = TradingClient(api,secret)
    return tcliet.get_all_positions()
def get_sentiment_class(headline):
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    inputs = tokenizer(headline, max_length=512, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().numpy()[0]
    label = model.config.id2label
    predicted_class = label[int(torch.argmax(outputs.logits).numpy())]
    

    return predicted_class


def create_data(data, max_lag = 15):
  max_lag = max_lag  

  for i in range(1, max_lag + 1):
    data[f'Close_lag{i}'] = data['Close'].shift(i)


  return data

def get_data(symbol ='AAPL', api = os.getenv('ALPACA_API_KEY') ,secret = os.getenv('ALPACA_SECRET_KEY')):
    # Get the current date and time
    now = datetime.now()

    # Format the datetime object into YYYYMMDD format
    formatted_date = now.strftime('%Y%m%d')

    # Convert the formatted string back into a datetime object
    formatted_datetime = datetime.strptime(formatted_date, '%Y%m%d')

    client = StockHistoricalDataClient(api, secret)
    params = StockBarsRequest(
        symbol_or_symbols=symbol,timeframe=TimeFrame.Day, start=datetime(2024,6, 1), end=formatted_datetime, limit=100,sort=Sort.DESC,feed='iex'
        
    )

    data = client.get_stock_bars(params)
    data = dict(data)
    check = []
    for i in data['data'][symbol]:
        check.append(dict(i))
    df = pd.DataFrame(check)
    df.sort_values(by='timestamp',inplace=True)
    # Convert 'timestamp' to datetime and extract only the date
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.date

    # Rename columns
    df.rename(columns={
        'timestamp': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume':'Volume'
    }, inplace=True)

    # Keep only the required columns
    df = df[['Date', 'Open', 'High', 'Low', 'Close','Volume']]
    
    df = supertrend(df , factor=3, atr_period=10)
    df = add_technical_indicators(df)
    df = df.drop(columns=['Supertrend','UpperBand', 'LowerBand', 'Uptrend',
        'Downtrend','Direction'])
    df.dropna(inplace=True)
    df.set_index('Date', inplace=True)
    

    return df

def get_news(df):
    nclient = NewsClient(api,secret, raw_data=True)
    params = NewsRequest(symbol_or_symbols = 'AAPL', limit=50)
    data = pd.DataFrame(nclient.get_news(params)['news'])
    data['Time'] = pd.DatetimeIndex(data['updated_at'])
    data =data.drop(columns=['updated_at','created_at','content','images','symbols','url','author','source','summary','id'])
    data["Date"] = data["Time"].dt.date
    data = data.groupby("Date")["headline"].apply(lambda x: " | ".join(x)).reset_index()
    data.set_index('Date',inplace=True)
    df.loc[df.index[-1:], 'sentiment_class'] = get_sentiment_class(data['headline'].iloc[0]) 

    return df


if __name__ == "__main__":
    starting = time.time()
    base_dir = os.getcwd()
    models_dir = os.path.join(base_dir, 'saved_models')
    artifacts_dir = os.path.join(base_dir, 'artifact')
    for filename in os.listdir(models_dir):
        if filename.endswith('.pkl'):
            start = time.time()
            ticker = filename.split('_')[0]
            lag = filename.split('_')[2]
            print(ticker,lag)
            model = joblib.load(os.path.join(models_dir,filename))
            df = get_data(ticker,)
            df = create_data(df,max_lag=int(lag))
            df = get_news(df)
            scaler_path = os.path.join(artifacts_dir, f'scaler_{ticker}_{lag}.pkl')
            encoder_path = os.path.join(artifacts_dir, f'encoder_{ticker}_{lag}.pkl')
            scaler = joblib.load(scaler_path)
            encoder = joblib.load(encoder_path)
            X_test = df.iloc[-1:].copy()
            X_test['sentiment_class'] = encoder.transform(X_test['sentiment_class'])
            X_test = X_test[scaler.feature_names_in_]
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            print(y_pred)
            print(ticker)
            open_position = dict(get_status()[0])['symbol']
            print(open_position)
            if open_position == ticker:
                pass
            elif open_position != ticker:
                if y_pred == 1:
                    print(f"Buying Signal Received for {ticker}")
                    create_order_buy(ticker)
            end = time.time()
            final_time = end - start
            print(final_time)
            time.sleep(5)

    ending = time.time()
    print(ending - starting)