import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt



def run():
    # Load and prepare data
    data = pd.read_csv('/home/sacsresta/Documents/RESEARCH/Project/sentiment/merged_data_AAPL_from_2015-01-01_to_2025-03-01.csv')
    print(data.columns)
    data.set_index('Date', inplace=True)
    data['shifted_direction'] = data['Direction'].shift(-1)
    data.dropna(subset=['shifted_direction'], inplace=True)
    data['shifted_direction'] = data['shifted_direction'].astype(int)

    # Feature selection
    X = data.drop(columns = ['Adj Close',
       'Supertrend', 'Direction', 'UpperBand', 'LowerBand', 'Uptrend',
       'Downtrend', 'Ticker', 'headline','shifted_direction','sentiment_score'])
    y = data['shifted_direction']
    print(X.columns)
 


    tscv = TimeSeriesSplit(n_splits=3)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define models to compare
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),

        'LightGBM': lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, random_state=42)
    }

    # Store results for each model
    model_results = {}

    # Run cross-validation for each model
    for model_name, model in models.items():
        print(f"\n==== Training {model_name} ====")

        fold_accuracy = []
        all_predictions = []
        all_actuals = []

        for fold, (train_index, test_index) in enumerate(tscv.split(X_scaled)):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Train the model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate accuracy
            acc = accuracy_score(y_test, y_pred)
            print(f"Classification report for {fold}", classification_report(y_test, y_pred))
            fold_accuracy.append(acc)

            # Store predictions and actual values
            all_predictions.extend(y_pred)
            all_actuals.extend(y_test.values)

            print(f"Fold {fold+1} accuracy: {acc:.4f}")

        # Calculate average accuracy
        avg_accuracy = np.mean(fold_accuracy)
        print(f"Average accuracy: {avg_accuracy:.4f}")

        # Print classification report for all folds combined
        print("\nClassification Report:")
        print(classification_report(all_actuals, all_predictions))

        # Store results
        model_results[model_name] = {
            'accuracy': avg_accuracy,
            'predictions': all_predictions,
            'actuals': all_actuals
        }

    # Compare model performances
    accuracies = {name: results['accuracy'] for name, results in model_results.items()}
    best_model_name = max(accuracies, key=accuracies.get)

    print(f"\n==== Model Comparison ====")
    for model_name, accuracy in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name}: {accuracy:.4f}")

    print(f"\nBest model: {best_model_name} with accuracy {accuracies[best_model_name]:.4f}")

    # Visualize model comparison
    plt.figure(figsize=(12, 6))
    model_names = list(accuracies.keys())
    model_accs = list(accuracies.values())

    # Sort by accuracy
    sorted_indices = np.argsort(model_accs)[::-1]
    model_names = [model_names[i] for i in sorted_indices]
    model_accs = [model_accs[i] for i in sorted_indices]

    plt.bar(model_names, model_accs, color='skyblue')
    plt.axhline(y=np.mean(model_accs), color='r', linestyle='-', label='Average')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(min(model_accs) - 0.05, max(model_accs) + 0.05)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.legend()
    plt.show()



if __name__ == '__main__':
    run()