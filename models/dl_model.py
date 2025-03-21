import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout,GRU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics import  classification_report, confusion_matrix
from models.data_preprocessor import preprocess_data
import numpy as np


def train_model(X_train,y_train):

    K.clear_session()
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[-2], X_train.shape[-1])),
        Dropout(0.2),
        LSTM(32, return_sequences=True),
        LSTM(16, return_sequences=False),
        Dense(8, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='tanh')  # Changed to sigmoid
    ])


    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='mse',
        metrics=[
            'accuracy',
        ]
    )

    model.summary()


    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    class_weight = {-1: 2, 1: 1}  # Prioritize minimizing missed sell signals

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.3,
        callbacks=[early_stop],
        class_weight=class_weight,
        verbose=1
    )

    return model, history

def evaluate_model(X_test,y_test,model):
    y_pred = model.predict(X_test)
    # Convert predictions to signals (-1/1)
    y_pred_signal = np.where(y_pred >= 0, 1, -1)
    test_accuracy = np.mean(y_pred_signal == y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(classification_report(y_test, y_pred_signal))
    print(confusion_matrix(y_test, y_pred_signal))


if __name__ == "__main__":
    path = '/home/sacsresta/Documents/RESEARCH/Project/sentiment/merged_data_AAPL_from_2024-01-01_to_2025-01-01.csv'
    X_train,y_train,X_test,y_test = preprocess_data(path = path)
    print(X_train)
    model, history = train_model(X_train,y_train)
    evaluate_model(X_test,y_test,model)