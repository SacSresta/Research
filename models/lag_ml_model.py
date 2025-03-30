import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
import pickle
import joblib
import os

def classifier_models():
    # Updated classifier dictionary
    classifiers = {
        'Logistic Regression': LogisticRegression(C=10, max_iter=1000, solver='liblinear'),
        'Random Forest': RandomForestClassifier(max_depth=10, min_samples_split=10, n_estimators=50),
        'XGBoost': XGBClassifier(learning_rate=0.01, max_depth=3, 
                         n_estimators=100, eval_metric='logloss', 
                         n_jobs=1),  # Prevents joblib conflicts
        'SVM': SVC(C=10, kernel='linear', probability=True),
        'Naive Bayes': BernoulliNB(),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3),  # Uses decision trees [[7]]
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),  # Instance-based learning
        'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_split=10),  # Base tree model [[7]]
        'AdaBoost': AdaBoostClassifier(n_estimators=50, learning_rate=0.5),  # Boosting ensemble
        'SGD Classifier': SGDClassifier(loss='log_loss', alpha=0.001),  # Linear model with SGD [[10]]
        'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000),  # Neural network
        'LGBMClassifier' :LGBMClassifier(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=5, 
        boosting_type='gbdt'
    )
    }
    return classifiers

def get_param_grids():
    # Define hyperparameter grids
    param_grids = {
        'Logistic Regression': {
            'C': [0.1, 1, 10], 
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        },
        'Random Forest': {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 10],
            'max_features': ['sqrt', 'log2']
        },

        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        },
        'Naive Bayes': {
            'alpha': [0.1, 0.5, 1.0]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'min_samples_split': [2, 10]
        },
        'K-Nearest Neighbors': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree'],
            'p': [1, 2]
        },
        'Decision Tree': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 10],
            'max_features': [None, 'sqrt']
        },
        'AdaBoost': {
            'n_estimators': [50, 100],
            'learning_rate': [0.1, 0.5, 1.0]
        },
        'SGD Classifier': {
            'alpha': [0.0001, 0.001],
            'loss': ['log_loss', 'modified_huber'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'learning_rate': ['constant', 'optimal']
        },
        'MLP': {
            'hidden_layer_sizes': [(100,), (50,50)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'learning_rate_init': [0.001, 0.01],
            'alpha': [0.0001, 0.001]
        }
    }
    return param_grids

def grid_optimize_model(classifiers,X_train_scaled, y_train,param_grids):

    
    # Dictionary to store best models
    best_models = {}

    for name, clf in classifiers.items():
        print(f"Optimizing {name}...")
        
        if name in param_grids:
            grid_search = GridSearchCV(clf, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train_scaled, y_train)
            best_models[name] = grid_search.best_estimator_  # Save best model
            print(f"Best parameters for {name}: {grid_search.best_params_}")
        else:
            clf.fit(X_train_scaled, y_train)
            best_models[name] = clf
    print(f"No hyperparameters to tune for {name}")
    return best_models



def random_optimize_model(classifiers, X_train_scaled, y_train,param_grids):

    best_models = {}

    for name, clf in classifiers.items():
        print(f"Optimizing {name}...")
        
        if name in param_grids:
            search = RandomizedSearchCV(clf, param_distributions=param_grids[name], 
                                        n_iter=10, cv=5, scoring='accuracy', 
                                        n_jobs=4, verbose=2, random_state=42)
            search.fit(X_train_scaled, y_train)
            best_models[name] = search.best_estimator_
            print(f"Best parameters for {name}: {search.best_params_}")
        else:
            clf.fit(X_train_scaled, y_train)
            best_models[name] = clf
            print(f"No hyperparameters to tune for {name}")

    return best_models


def create_data(X, max_lag = 60):
  # Define the number of lags you want
  max_lag = max_lag  # For example, to create lags 1 through 5

  # Create lagged features in a loop
  for i in range(1, max_lag + 1):
      X[f'Close_lag{i}'] = X['Close'].shift(i)

  # Define X and y (drop the first two rows with NaN lags)
  y = X['shifted_direction'].iloc[max_lag:].astype(int)
  X = X.drop(columns = ['shifted_direction']).iloc[max_lag:]
  return X,y

def get_data(path=r'C:\Users\sachi\Documents\Researchcode\sentiment\merged_data_AAPL_from_2015-01-01_to_2025-03-01.csv'):
    df = pd.read_csv(path)
    df['shifted_direction'] = df['Direction'].shift(-1)
    df = df.drop(columns=['Supertrend','UpperBand', 'LowerBand', 'Uptrend',
        'Downtrend', 'headline','Adj Close'])
    df['Date'] = pd.to_datetime(df['Date'])
    df.dropna(inplace=True)
    return df

def preprocess_data(df,max_lag=60):
    X = df[['Date', 'Close', 'High', 'Low', 'Open', 'Volume', 
        'sentiment_score', 'shifted_direction']].copy()
    X,y = create_data(X, max_lag=max_lag)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    X.set_index('Date',inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = False)
    return X_train,X_test,y_train,y_test

def scaling_function(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled,X_test_scaled

def train_model(classifiers,X_train_scaled,X_test_scaled,y_train,y_test):
    # Create list to store results
    results = []
    y_pred_dict = {}

    # Train-test loop
    for name, clf in classifiers.items():
        # Train model
        clf.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = clf.predict(X_test_scaled)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Store results
        results.append((name, accuracy, cm))

        # Display results
        print(f"\n{name} Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print(report)
        y_pred_dict[name] = y_pred
    df = pd.DataFrame(results,columns=['Model', 'Accuracy', 'Confusion Matrix'])
    return df,y_pred_dict  

def main():
    df = get_data(path=r'C:\Users\sachi\Documents\Researchcode\sentiment\merged_data_AAPL_from_2015-01-01_to_2025-03-01.csv')
    X_train,X_test,y_train,y_test = preprocess_data(df)
    print(X_train)
    X_train_scaled,X_test_scaled = scaling_function(X_train,X_test)
    print(X_train_scaled.shape,X_test_scaled.shape,y_train.shape,y_test.shape)
    classifier = classifier_models()
    os.makedirs('saved_models', exist_ok=True)
    results = train_model(classifiers=classifier, X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled,y_train=y_train,y_test=y_test)
    print(results)
    results.to_csv('model_summary_no_opt.csv', mode='w',
               index=True)
    best_base = results.loc[results['Accuracy'].idxmax(), 'Model']
    joblib.dump(classifier[best_base], f"saved_models/best_base_{best_base.replace(' ', '_')}.pkl")
    param_grids = get_param_grids()
    print("Working on optimizing with GridSearchCV")
    best_models = grid_optimize_model(classifier,X_train_scaled, y_train,param_grids=param_grids)
    results,y_pred_dict = train_model(classifiers=best_models, X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled,y_train=y_train,y_test=y_test)
    print(results)

    results.to_csv('model_summary_grid_opt.csv', mode='w',
               index=True)
    best_grid = results.loc[results['Accuracy'].idxmax(), 'Model']
    joblib.dump(best_models[best_grid], f"saved_models/best_grid_{best_grid.replace(' ', '_')}.pkl")
    print('-----------------------------')  
    print("Working on optimizing with RandomSearchCV")
    random_best_models = random_optimize_model(classifier,X_train_scaled, y_train,param_grids=param_grids)
    results = train_model(classifiers=random_best_models, X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled,y_train=y_train,y_test=y_test)
    print(results)
    results.to_csv('model_summary_rand_opt.csv', mode='w',
               index=True)
    best_random = results.loc[results['Accuracy'].idxmax(), 'Model']
    joblib.dump(random_best_models[best_random], f"saved_models/best_random_{best_random.replace(' ', '_')}.pkl")
    print('--------------------------')





if __name__ == "__main__":
    main()