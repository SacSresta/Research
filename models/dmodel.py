# lstm_trading_model.py
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from backtesting import Strategy, Backtest
import tensorflow.keras.backend as K

def prepare_data(data, lookback=60):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data.iloc[i-lookback:i, :].values)
        y.append(data.iloc[i]['shifted_direction'])
    scaler = MinMaxScaler(feature_range=(0, 1))
    return X, y, scaler

def build_lstm_model(input_shape):
    K.clear_session()
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def give_path(path):
    df = pd.read_csv(path)
    df.drop(columns=['Cleaned_Headlines','Ticker'], inplace=True, errors='ignore')
    df['headline'] = df['headline'].str.replace(r'[\[\]\'\"]', '', regex=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['shifted_direction'] = df['Direction'].shift(-1)
    df.dropna(subset=['shifted_direction'],inplace = True)
    df['shifted_direction'] = df['shifted_direction'].astype(int)
    dl_df = df[['Close', 'High', 'Low', 'Open', 'Volume','sentiment_score','shifted_direction']]
    return dl_df

def split_data(X, y):
    split = int(0.8 * len(X))
    X_array = np.array(X)
    y_array = np.array(y)
    X_train, X_test = X_array[:split], X_array[split:]
    y_train, y_test = y_array[:split], y_array[split:]
    return X_train, X_test, y_train, y_test, split

def scaling_func(X_train, X_test, y_train, y_test):
    n_samples_train, n_timesteps, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(n_samples_train * n_timesteps, n_features)
    
    X_scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaler.fit(X_train_reshaped)
    
    X_train_scaled_reshaped = X_scaler.transform(X_train_reshaped)
    X_train_scaled = X_train_scaled_reshaped.reshape(n_samples_train, n_timesteps, n_features)
    
    n_samples_test = X_test.shape[0]
    X_test_reshaped = X_test.reshape(n_samples_test * n_timesteps, n_features)
    X_test_scaled_reshaped = X_scaler.transform(X_test_reshaped)
    X_test_scaled = X_test_scaled_reshaped.reshape(n_samples_test, n_timesteps, n_features)
    
    y_scaler = MinMaxScaler(feature_range=(0, 1)) 
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler

def plot_loss(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def compare_results(lstm_stats, actual_stats):
    comparison = pd.DataFrame({
        'LSTM Model': [
            lstm_stats['Return [%]'],
            lstm_stats['Max. Drawdown [%]'],
            lstm_stats['# Trades'],
            lstm_stats['Win Rate [%]'],
            lstm_stats['Profit Factor'],
            lstm_stats['Avg. Trade [%]'],
            lstm_stats['Sharpe Ratio'],
            lstm_stats['Buy & Hold Return [%]']
        ],
        'Actual Direction': [
            actual_stats['Return [%]'],
            actual_stats['Max. Drawdown [%]'],
            actual_stats['# Trades'],
            actual_stats['Win Rate [%]'],
            actual_stats['Profit Factor'],
            actual_stats['Avg. Trade [%]'],
            actual_stats['Sharpe Ratio'],
            actual_stats['Buy & Hold Return [%]']
        ]
    }, index=[
        'Return [%]', 
        'Max Drawdown [%]', 
        '# Trades',
        'Win Rate [%]',
        'Profit Factor',
        'Avg. Trade [%]',
        'Sharpe Ratio',
        'Buy & Hold Return [%]'
    ])
    
    return comparison

def run_backtest(backtest_df, signals, strategy_class):
    """Run backtest with a given strategy class and signals"""
    bt = Backtest(backtest_df, strategy_class, cash=10000, commission=0.001)
    stats = bt.run()
    return bt, stats

class LSTMPredictionStrategy(Strategy):
    def init(self):
        # Store signals in the class
        self.signals = self.data.signals
        
    def next(self):
        if self.signals[-1] == 1:
            if not self.position:
                self.buy()
        elif self.signals[-1] == 0:
            if self.position:
                self.position.close()

class ActualDirectionStrategy(Strategy):
    def init(self):
        # Store signals in the class
        self.signals = self.data.signals
        
    def next(self):
        if self.signals[-1] == 1:
            if not self.position:
                self.buy()
        elif self.signals[-1] == -1:
            if self.position:
                self.position.close()

def run_lstm_analysis(file_path, lookback=60, epochs=5, batch_size=32, show_plots=False):
    """Run full LSTM analysis on a single file and return results"""
    # Load and prepare data
    dl_df = give_path(path=file_path)
    
    # Prepare sequences
    X, y, _ = prepare_data(dl_df, lookback=lookback)
    
    # Split data
    X_train, X_test, y_train, y_test, split = split_data(X, y)
    
    # Get dimensions
    n_samples_train, n_timesteps, n_features = X_train.shape
    
    # Scale data
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler = scaling_func(
        X_train, X_test, y_train, y_test)
    
    # Build and train model
    input_shape = (n_timesteps, n_features)
    model = build_lstm_model(input_shape)
    history = model.fit(
        X_train_scaled, y_train_scaled, 
        epochs=epochs, batch_size=batch_size, 
        validation_split=0.3, verbose=1
    )
    
    # Plot training history if requested
    if show_plots:
        plot_loss(history)
    
    # Make predictions
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_pred_binary = np.where(y_pred > 0.5, 1, 0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_binary)
    
    # Prepare backtest data
    backtest_df = dl_df[lookback+split:].copy()  # Create a copy to avoid modifying original
    
    # Add signals to the dataframe for backtesting
    backtest_df['signals'] = y_pred_binary.reshape(-1)
    
    # Run LSTM model backtest
    lstm_bt, lstm_stats = run_backtest(backtest_df, y_pred_binary.reshape(-1), LSTMPredictionStrategy)
    
    # Add actual signals for comparison
    backtest_df['signals'] = backtest_df['shifted_direction'].values
    
    # Run actual direction backtest
    actual_bt, actual_stats = run_backtest(backtest_df, backtest_df['shifted_direction'].values, ActualDirectionStrategy)
    
    # Generate comparison
    comparison = compare_results(lstm_stats, actual_stats)
    
    # Return all results
    results = {
        'ticker': file_path.split('_')[-3],  # Extract ticker from filename
        'model_accuracy': history.history['val_accuracy'][-1],
        'confusion_matrix': cm,
        'lstm_stats': lstm_stats,
        'actual_stats': actual_stats,
        'comparison': comparison
    }
    
    return results