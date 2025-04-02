from models.lag_ml_model import scaling_function,preprocess_data,classifier_models,train_model,grid_optimize_model,random_optimize_model,get_param_grids,get_data
from backtesting import Backtest,Strategy
import pandas as pd
import os
def backtest(y_pred, X_test):
    def get_signal():
        return y_pred
    class MyStrategy(Strategy):
        def init(self):
            self.signal = self.I(get_signal)
        def next(self):
            if self.signal[-1] == 1:
                if not self.position:
                    self.buy()
            elif self.signal[-1] == 0:
                if self.position:
                    self.position.close()

    bt = Backtest(X_test,MyStrategy,cash=10000)
    stats = bt.run()
    return stats,bt


def backtest_loop(y_pred_dict,X_test):
    stats_l=[]
    bt_collection = {}
    for name,y_pred in y_pred_dict.items():
        print(f"Backtesting for {name}")
        stats,bt = backtest(y_pred,X_test)
        bt_collection[name] = bt
        stats['Model'] = name
        stats_l.append(stats)

    normal_df = pd.DataFrame(stats_l)

    return normal_df,bt_collection

def normal_model(X_train_scaled,X_test_scaled,y_train,y_test):
    # load data
    classifier = classifier_models()
    results,y_pred_dict = train_model(classifiers=classifier, X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled,y_train=y_train,y_test=y_test)
    
    return results,y_pred_dict

def grid_model(X_train_scaled,X_test_scaled,y_train,y_test):
    # load data
    classifier = classifier_models()
    param_grids = get_param_grids()
    print("Working on optimizing with GridSearchCV")
    best_models = grid_optimize_model(classifier,X_train_scaled, y_train,param_grids=param_grids)
    results,y_pred_dict = train_model(classifiers=best_models, X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled,y_train=y_train,y_test=y_test)

    return results,y_pred_dict

def random_model(X_train_scaled,X_test_scaled,y_train,y_test):
    classifier = classifier_models()
    param_grids = get_param_grids()
    print("Working on optimizing with RandomSearchCV")
    random_best_models = random_optimize_model(classifier,X_train_scaled, y_train,param_grids=param_grids)
    results,y_pred_dict = train_model(classifiers=random_best_models, X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled,y_train=y_train,y_test=y_test)
    return results,y_pred_dict




if __name__ == "__main__":
    dir = r'C:\Users\sachi\Documents\Researchcode\sentiment'
    saving_path = 'strategy_name'
    for filepath in os.listdir(dir):
        path = os.path.join(dir,filepath)
        if filepath.endswith('.csv'):
            print(f"Working on {filepath}")
            ticker = filepath.split('_')[2]
            df = get_data(path,ind=True)
            X_train,X_test,y_train,y_test = preprocess_data(df,max_lag=60)
            X_train_scaled,X_test_scaled = scaling_function(X_train,X_test)
            #No Optimation Model
            results,y_pred_dict = normal_model(X_train_scaled,X_test_scaled,y_train,y_test)
            normal_df,bt_collection = backtest_loop(y_pred_dict,X_test)
            saving_dir = os.path.join(dir, f'{saving_path}/normal') 
            os.makedirs(saving_dir,exist_ok=True)
            print("Saving Backtesting Result")
            normal_merge = pd.merge(results,normal_df, how='inner')
            normal_merge.to_csv(os.path.join(saving_dir, f'{ticker}_normal_returns_accuracy.csv'))
            print("Normal Results and Returns Saved")
            #GridSearchCV
            results,y_pred_dict = grid_model(X_train_scaled,X_test_scaled,y_train,y_test)
            grid_df,bt_collection = backtest_loop(y_pred_dict,X_test)
            saving_dir = os.path.join(dir, f'{saving_path}/grid')
            os.makedirs(saving_dir,exist_ok=True)
            print("Saving Backtesting Result")
            grid_merge = pd.merge(results,grid_df, how='inner')
            grid_merge.to_csv(os.path.join(saving_dir, f'{ticker}_grid_returns_accuracy.csv'))
            print("Grid Optimization Completed")
            #RandomSearchCV
            results,y_pred_dict = random_model(X_train_scaled,X_test_scaled,y_train,y_test)
            random_df,bt_collection= backtest_loop(y_pred_dict,X_test)
            saving_dir = os.path.join(dir, f'{saving_path}/random')
            os.makedirs(saving_dir,exist_ok=True)
            print("Saving Backtesting Result")
            random_merge = pd.merge(results,random_df, how='inner')
            random_merge.to_csv(os.path.join(saving_dir,f'{ticker}_random_returns_accuracy.csv'))
            print("Randomization Completed")
            actual,_ = backtest(y_test,X_test)
            actual['Model'] = 'Actual'
            combined_df = pd.concat([normal_merge,grid_merge,random_merge,actual], axis=0, keys=['normal','grid','random','actual'])
            combined_df.to_csv(os.path.join(saving_dir,f'{ticker}_combined_returns_accuracy.csv'))




