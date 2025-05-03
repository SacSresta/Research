import joblib
import os

for index, row in new_df.iterrows():
    ticker = row['Ticker_x']
    lag = row['lag']
    conf = row['Configuration']
    
    # Construct the path to the data file
    path = rf'C:\Users\sachi\Documents\Researchcode\Conferance_sentiment_categorical\merged_data_{ticker}_from_2015-01-01_to_2025-03-01.csv'
    
    # Load the data
    df = get_data(path, ind=True)

    # Preprocess the data
    X_train, X_test, y_train, y_test, encoder = preprocess_data_by_date(df, max_lag=lag)
    X_train_scaled, X_test_scaled, scaler = scaling_function(X_train, X_test)

    # Convert the string to a callable model
    model_string = conf
    model = eval(model_string)

    # Print the model to verify
    print(model)
    
    # Fit the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Print the risk backtest result
    print(risk_backtest(y_pred, X_test))

    # Define the model save path
    model_save_path = rf'C:\Users\sachi\Documents\Researchcode\saved_models\{ticker}_lag_{lag}_model.pkl'
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Save the trained model
    joblib.dump(model, model_save_path)
    
    print(f"Model saved to {model_save_path}")
