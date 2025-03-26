from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix

from backtesting import Strategy
from backtesting.lib import crossover
from backtesting import Backtest

# Add at the beginning of your script
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



def build_lstm_model(input_shape):
    K.clear_session()
    model = Sequential()
    
    # Stacked LSTM layers with dropout and kernel initialization
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape, 
                   kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))  # Reduce overfitting
    
    model.add(LSTM(64, return_sequences=True, 
                   kernel_regularizer=l2(1e-4)))  # L2 regularization
    model.add(Dropout(0.2))
    
    model.add(LSTM(32))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(1e-4)))
    
    # Optimizer with learning rate tuning
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, 
                  loss='binary_crossentropy', 
                  metrics=['accuracy', 'Precision', 'Recall'])
    return model

def plot_loss(history):
        # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Model Loss')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    plt.show()


def give_path(path):
    df = pd.read_csv(path)
    
    df['headline'] = df['headline'].str.replace(r'[\[\]\'\"]', '', regex=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date',inplace=True)
    df['shifted_direction'] = df['Direction'].shift(-1)
    df.dropna(subset=['shifted_direction'],inplace = True)
    df['shifted_direction'] = df['shifted_direction'].astype(int)
    dl_df = df[['Close', 'High', 'Low', 'Open', 'Volume','sentiment_score','Direction','shifted_direction']]
    return dl_df

def prepare_data(data, lookback=60):
    # Create sequences from data (without scaling yet)
    X, y = [], []
    for i in range(lookback, len(data)):
        # Get sequence of lookback timesteps for all features
        X.append(data.iloc[i-lookback:i, :-2].values)
        
        # Target is the Close price at current timestep (not fixed index 1)
        y.append(data.iloc[i]['shifted_direction'])
    

    
    # Create scaler (but don't apply it yet)
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    return X, y, scaler

def final_preprocessing(lookback,data):
    X, y, scaler = prepare_data(dl_df, lookback=lookback)
    return X,y
    


def split_data(X,y):
    split = int(0.8 * len(X))
    # Convert to numpy arrays
    X_array = np.array(X)
    y_array = np.array(y)
    X_train, X_test = X_array[:split], X_array[split:]
    y_train, y_test = y_array[:split], y_array[split:]
    return X_train, X_test, y_train, y_test, split

def scaling_func(X_train, X_test, y_train, y_test):
    # X shape is (samples, timesteps, features)
    n_samples_train, n_timesteps, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(n_samples_train * n_timesteps, n_features)

    # Fit scaler on training data only
    X_scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaler.fit(X_train_reshaped)

    # Scale training data
    X_train_scaled_reshaped = X_scaler.transform(X_train_reshaped)
    X_train_scaled = X_train_scaled_reshaped.reshape(n_samples_train, n_timesteps, n_features)

    # Scale test data using the same scaler (no fitting on test data)
    n_samples_test = X_test.shape[0]
    X_test_reshaped = X_test.reshape(n_samples_test * n_timesteps, n_features)
    X_test_scaled_reshaped = X_scaler.transform(X_test_reshaped)
    X_test_scaled = X_test_scaled_reshaped.reshape(n_samples_test, n_timesteps, n_features)

    # Scale target values
    y_scaler = MinMaxScaler(feature_range=(0, 1)) 
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,y_scaler


# Function to compare backtest results
def compare_results(lstm_stats, actual_stats,ticker):
    # Create a DataFrame to compare key metrics
    comparison = pd.DataFrame({
        f'LSTM Model{ticker}': [
            lstm_stats['Return [%]'],
            lstm_stats['Max. Drawdown [%]'],
            lstm_stats['# Trades'],
            lstm_stats['Win Rate [%]'],
            lstm_stats['Profit Factor'],
            lstm_stats['Avg. Trade [%]'],
            lstm_stats['Sharpe Ratio'],
            lstm_stats['Buy & Hold Return [%]'],
        ],
        f'Actual Direction{ticker}': [
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

# Function to plot equity curves side by side
def plot_equity_comparison(lstm_backtest, actual_backtest):
    plt.figure(figsize=(15, 6))
    
    # Plot both equity curves
    plt.plot(lstm_backtest._equity_curve, label='LSTM Model Strategy')
    plt.plot(actual_backtest._equity_curve, label='Actual Direction Strategy')
    
    plt.title('Equity Curve Comparison')
    plt.xlabel('Trading Days')
    plt.ylabel('Equity ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def get_lstm_stats(backtest_y_pred,backtest_df):
    def get_signal():
        return backtest_y_pred
    
    class LSTMPredictionStrategy(Strategy):
        def init(self):
            self.signal = self.I(get_signal)

        def next(self):
            if self.signal == 1:
                if not self.position:
                    self.buy()
            elif self.signal == 0:
                if self.position:
                    self.position.close()

    # Run backtests
    lstm_bt = Backtest(backtest_df, LSTMPredictionStrategy, cash=100000, commission=0.001)
    lstm_stats = lstm_bt.run()
    return lstm_stats


def get_actual_stats(backtest_df):
    print(backtest_df.columns)
    def get_signal():
        return backtest_df['Direction'].values
    class ActualDirectionStrategy(Strategy):
        def init(self):
            self.signal = self.I(get_signal)

        def next(self):
            if self.signal == 1:
                if not self.position:
                    self.buy()
            elif self.signal == -1:
                if self.position:
                    self.position.close()

    actual_bt = Backtest(backtest_df, ActualDirectionStrategy, cash=100000, commission=0.001)
    actual_stats = actual_bt.run()

    return actual_stats

if __name__ == "__main__":
   

    dl_df = give_path(path='/home/sacsresta/Documents/RESEARCH/Project/sentiment/merged_data_META_from_2015-01-01_to_2025-03-01.csv')


    lookback = 60

    # Prepare sequences without scaling
    X, y, scaler = prepare_data(dl_df, lookback=lookback)


    X_train, X_test, y_train, y_test,split = split_data(X,y)

    n_samples_train, n_timesteps, n_features = X_train.shape

    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,y_scaler = scaling_func( X_train, X_test, y_train, y_test)

    input_shape = (n_timesteps, n_features)
    model = build_lstm_model(input_shape)
    history = model.fit(X_train_scaled, y_train_scaled, epochs=10, batch_size=32, validation_split=0.3, verbose=1)
    plot_loss(history)

    # Make predictions and inverse transform to get actual values
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_pred = np.where(y_pred > 0.5, 1, 0)

    cm = confusion_matrix(y_test_scaled, y_pred)
    print(cm)
    backtest_y_pred = y_pred.reshape(1,-1)[0]


    backtest_df = dl_df[lookback+split:]
    print(backtest_df.columns)

    def get_signal():
        return backtest_y_pred
    
    class LSTMPredictionStrategy(Strategy):
        def init(self):
            self.signal = self.I(get_signal)

        def next(self):
            if self.signal == 1:
                if not self.position:
                    self.buy()
            elif self.signal == 0:
                if self.position:
                    self.position.close()

    # Run backtests
    lstm_bt = Backtest(backtest_df, LSTMPredictionStrategy, cash=10000, commission=0.001)
    lstm_stats = lstm_bt.run()

    def get_signal():
        return backtest_df['Direction'].values
    print(get_signal())
    class ActualDirectionStrategy(Strategy):
        def init(self):
            self.signal = self.I(get_signal)

        def next(self):
            if self.signal == 1:
                if not self.position:
                    self.buy()
            elif self.signal == -1:
                if self.position:
                    self.position.close()

    actual_bt = Backtest(backtest_df, ActualDirectionStrategy, cash=10000, commission=0.001)
    actual_stats = actual_bt.run()
    # Print comparison
    print("\n=== LSTM Model Performance ===")
    print(lstm_stats[["Return [%]", "Max. Drawdown [%]", "# Trades", "Win Rate [%]", "Profit Factor", "Sharpe Ratio","Buy & Hold Return [%]"]])
    
    print("\n=== Actual Direction Performance ===")
    print(actual_stats[["Return [%]", "Max. Drawdown [%]", "# Trades", "Win Rate [%]", "Profit Factor", "Sharpe Ratio","Buy & Hold Return [%]"]])
    
    # Generate and display comparison table
    comparison = compare_results(lstm_stats, actual_stats)
    print("\n=== Side-by-Side Comparison ===")
    print(comparison)
    