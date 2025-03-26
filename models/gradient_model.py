from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def run():
    # Load and prepare data
    data = pd.read_csv('/home/sacsresta/Documents/RESEARCH/Project/sentiment/merged_data_META_from_2015-01-01_to_2025-03-01.csv')
    print(data.columns)
    data.set_index('Date', inplace=True)
    data['shifted_direction'] = data['Direction'].shift(-1)
    data.dropna(subset=['shifted_direction'], inplace=True)
    data['shifted_direction'] = data['shifted_direction'].astype(int)

    # Feature selection
    X = data.drop(columns = ['Adj Close',
       'Direction', 'UpperBand', 'LowerBand', 'Uptrend',
       'Downtrend', 'Ticker', 'headline','shifted_direction','Supertrend'])
    y = data['shifted_direction']
    print(X.columns)

    # Setup time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'subsample': [0.8, 1.0]
    }
    
    # Initialize GridSearchCV with time series split
    grid_search = GridSearchCV(
        estimator=GradientBoostingClassifier(random_state=42),
        param_grid=param_grid,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1  # Use all available cores
    )
    
    # Perform grid search
    grid_search.fit(X_scaled, y)
    
    # Print best parameters
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score: {:.4f}".format(grid_search.best_score_))
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Initialize collection variables for final evaluation
    fold_accuracy = []
    all_predictions = []
    all_actuals = []

    # Evaluate using time series cross-validation
    for train_index, test_index in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train the best model
        best_model.fit(X_train, y_train)

        # Make predictions
        y_pred = best_model.predict(X_test)

        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred)
        fold_accuracy.append(acc)
        
        # Store predictions and actual values
        all_predictions.extend(y_pred)
        all_actuals.extend(y_test.values)
        
        print(f"Fold accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    print(f"\nAverage accuracy across folds: {np.mean(fold_accuracy):.4f}")

    # Print the overall classification report
    print("\nOverall Classification Report:")
    print(classification_report(all_actuals, all_predictions))
    
    # Feature importance
    feature_importance = best_model.feature_importances_
    feature_names = X.columns
    
    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(importance_df)


if __name__ == "__main__":
    run()