from models.dl_model import prepare_data, build_lstm_model, plot_loss, give_path,final_preprocessing, split_data,scaling_func,compare_results,plot_equity_comparison,get_lstm_stats,get_actual_stats
import os
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import numpy as np
import pandas as pd

def run(lookback,path, ticker,epochs):
    df = give_path(path)
    X, y, scaler = prepare_data(df, lookback=lookback)
    X_train, X_test, y_train, y_test,split = split_data(X,y)
    n_samples_train, n_timesteps, n_features = X_train.shape
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,y_scaler = scaling_func( X_train, X_test, y_train, y_test)
    input_shape = (n_timesteps, n_features)
    model = build_lstm_model(input_shape)
    history = model.fit(X_train_scaled, y_train_scaled, epochs=epochs, batch_size=32, validation_split=0.3, verbose=1)
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    cm = confusion_matrix(y_test_scaled, y_pred)
    print("Confusion_matrix",cm)
    print("Accuracy", accuracy_score(y_test_scaled,y_pred))
    print("Classification Report", classification_report(y_test_scaled,y_pred))
    backtest_y_pred = y_pred.reshape(1,-1)[0]
    backtest_df = df[lookback+split:]
    print(backtest_df.iloc[0])
    lstm_stats = get_lstm_stats(backtest_y_pred,backtest_df)
    actual_stats = get_actual_stats(backtest_df)


    print(f"\n=== LSTM Model Performance for {ticker}===")

    print(lstm_stats)
    
    print(f"\n=== Actual Direction Performance for {ticker} ===")
    print(actual_stats)
    
    # Generate and display comparison table
    comparison = compare_results(lstm_stats, actual_stats,ticker=ticker)
    print("\n=== Side-by-Side Comparison ===")
    print(comparison)
    return comparison
    


if __name__ == "__main__":
    print(os.getcwd())
    cur_dir = os.getcwd()
    cur_dir = os.path.join(cur_dir,'sentiment')
    results_dir = os.path.join(cur_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    summary = []
    for path in os.listdir(cur_dir):
        print(path)
        if path.endswith(".csv"):
            path = os.path.join(cur_dir,path)
            ticker = path.split('_')[2]
            print("*******************************8")
            print(f"Processing for {ticker}")
            comparison = run(path=path, lookback=20, ticker=ticker,epochs = 20)
            print(type(comparison))

        summary.append(comparison)

    summary_df = pd.concat(summary, axis=1, join="outer")
    print(summary_df)
    summary_df.to_csv("summary.csv", index=False)  # Save as CSV

        
        

