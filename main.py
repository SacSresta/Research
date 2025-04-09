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

def risk_backtest(y_pred, X_test, risk=0.20):
    def get_signal():
        return y_pred

    class MyStrategy(Strategy):
        # Risk settings:
        RISK_PER_TRADE = 1  # This will now be the percentage of equity to invest
        STOP_LOSS_PCT = risk  # 20% stop-loss from entry price

        def init(self):
            self.signal = self.I(get_signal)

        def next(self):
            price = self.data.Close[-1]
            current_equity = self.equity
            
            if self.signal[-1] == 1:
                if not self.position:
                    # Calculate shares based on risk percentage of total equity
                    investment_amount = current_equity * self.RISK_PER_TRADE
                    shares = int(investment_amount / price)
                    stop_loss = price * (1 - self.STOP_LOSS_PCT)

                    # Place a buy order with a protective stop-loss at 20% below entry
                    self.buy(size=shares, sl=stop_loss)
            else:
                # If signal == 0 and we have an open position, close it
                if self.position:
                    self.position.close()

    bt = Backtest(X_test, MyStrategy, cash=10000)
    stats = bt.run()
    return stats, bt


def risk_backtest_loop(y_pred_dict,X_test,risk = 0.20):
    stats_l=[]
    bt_collection = {}
    for name,y_pred in y_pred_dict.items():
        print(f"Backtesting for {name}")
        stats,bt = risk_backtest(y_pred,X_test,risk = risk)
        bt_collection[name] = bt
        stats['Model'] = name
        stats_l.append(stats)

    normal_df = pd.DataFrame(stats_l)

    return normal_df,bt_collection
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


def normal_run(lag=60):
    dir = r'C:\Users\sachi\Documents\Researchcode\sentiment'

    combined_collector = {}
    for filepath in os.listdir(dir):
        path = os.path.join(dir,filepath)
        if filepath.endswith('.csv'):
            print(f"Working on {filepath}")
            saving_path = f'strategy_{lag}'
            ticker = filepath.split('_')[2]
            print(f"Loading Data for {ticker}, testing on number of lag {lag}, saving path is {saving_path}")
            df = get_data(path,ind=True)
            X_train,X_test,y_train,y_test = preprocess_data(df,max_lag=lag)
            X_train_scaled,X_test_scaled = scaling_function(X_train,X_test)
            #No Optimation Model
            results,y_pred_dict = normal_model(X_train_scaled,X_test_scaled,y_train,y_test)
            normal_df,bt_collection = backtest_loop(y_pred_dict,X_test)
            risk_df,bt_collection = risk_backtest_loop(y_pred_dict,X_test)
            actual,_ = backtest(y_test,X_test)
            actual['Model'] = 'Actual'
            actual = pd.DataFrame(actual).T
            saving_dir = os.path.join(dir, f'{saving_path}/normal') 
            os.makedirs(saving_dir,exist_ok=True)
            print("Saving Backtesting Result")
            normal_merge = pd.merge(results,normal_df, how='inner')
            normal_merge = pd.concat([normal_merge,actual],axis=0)
            risk_merge = pd.merge(results,risk_df, how='inner')
            risk_merge = pd.concat([risk_merge,actual],axis=0)
            normal_merge = pd.merge(normal_merge,risk_merge, on='Model')
            normal_merge.to_csv(os.path.join(saving_dir, f'{ticker}_normal_returns_accuracy.csv'))
            print("Normal Results and Returns Saved")
            combined_collector[ticker] = normal_merge

    return combined_collector,ticker
def grid_run(lag=60):
    dir = r'C:\Users\sachi\Documents\Researchcode\sentiment'

    combined_collector = {}
    for filepath in os.listdir(dir):
        path = os.path.join(dir,filepath)
        if filepath.endswith('.csv'):
            print(f"Working on {filepath}")
            saving_path = f'strategy_{lag}'
            ticker = filepath.split('_')[2]
            print(f"Loading Data for {ticker}, testing on number of lag {lag}, saving path is {saving_path}")
            df = get_data(path,ind=True)
            X_train,X_test,y_train,y_test = preprocess_data(df,max_lag=lag)
            X_train_scaled,X_test_scaled = scaling_function(X_train,X_test)
            #GridSearchCV
            results,y_pred_dict = grid_model(X_train_scaled,X_test_scaled,y_train,y_test)
            grid_df,bt_collection = backtest_loop(y_pred_dict,X_test)
            risk_df,bt_collection = risk_backtest_loop(y_pred_dict,X_test)
            actual,_ = backtest(y_test,X_test)
            actual['Model'] = 'Actual'
            actual = pd.DataFrame(actual).T
            saving_dir = os.path.join(dir, f'{saving_path}/grid')
            os.makedirs(saving_dir,exist_ok=True)
            print("Saving Backtesting Result")
            grid_merge = pd.merge(results,grid_df, how='inner')
            grid_merge = pd.concat([grid_merge,actual],axis=0)
            risk_merge = pd.merge(results,risk_df, how='inner')
            risk_merge = pd.concat([risk_merge,actual],axis=0)
            grid_merge = pd.merge(grid_merge,risk_merge, on='Model')
            grid_merge.to_csv(os.path.join(saving_dir, f'{ticker}_grid_returns_accuracy.csv'))
            print("Grid Optimization Completed")
            print("Normal Results and Returns Saved")
            combined_collector[ticker] = grid_merge

    return combined_collector,ticker
def random_run(lag=60):
    dir = r'C:\Users\sachi\Documents\Researchcode\sentiment'

    combined_collector = {}
    for filepath in os.listdir(dir):
        path = os.path.join(dir,filepath)
        if filepath.endswith('.csv'):
            print(f"Working on {filepath}")
            saving_path = f'strategy_{lag}'
            ticker = filepath.split('_')[2]
            print(f"Loading Data for {ticker}, testing on number of lag {lag}, saving path is {saving_path}")
            df = get_data(path,ind=True)
            X_train,X_test,y_train,y_test = preprocess_data(df,max_lag=lag)
            X_train_scaled,X_test_scaled = scaling_function(X_train,X_test)
            results,y_pred_dict = random_model(X_train_scaled,X_test_scaled,y_train,y_test)
            random_df,bt_collection= backtest_loop(y_pred_dict,X_test)
            risk_df,bt_collection = risk_backtest_loop(y_pred_dict,X_test)
            actual,_ = backtest(y_test,X_test)
            actual['Model'] = 'Actual'
            actual = pd.DataFrame(actual).T
            saving_dir = os.path.join(dir, f'{saving_path}/random')
            os.makedirs(saving_dir,exist_ok=True)
            print("Saving Backtesting Result")
            random_merge = pd.merge(results,random_df, how='inner')
            random_merge = pd.concat([random_merge,actual],axis=0)
            risk_merge = pd.merge(results,risk_df, how='inner')
            risk_merge = pd.concat([risk_merge,actual],axis=0)
            random_merge = pd.merge(random_merge,risk_merge, on='Model')
            random_merge.to_csv(os.path.join(saving_dir,f'{ticker}_random_returns_accuracy.csv'))
            print("Normal Results and Returns Saved")

            combined_collector[ticker] = random_merge

    return combined_collector,ticker


def run(lag=60):
    dir = r'C:\Users\sachi\Documents\Researchcode\sentiment'

    combined_collector = []
    for filepath in os.listdir(dir):
        path = os.path.join(dir,filepath)
        if filepath.endswith('.csv'):
            print(f"Working on {filepath}")
            saving_path = f'strategy_{lag}'
            ticker = filepath.split('_')[2]
            print(f"Loading Data for {ticker}, testing on number of lag {lag}, saving path is {saving_path}")
            df = get_data(path,ind=True)
            X_train,X_test,y_train,y_test = preprocess_data(df,max_lag=lag)
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
            actual = pd.DataFrame(actual).T
            saving_dir = os.path.join(dir, f'{saving_path}/combined')
            os.makedirs(saving_dir,exist_ok=True)
            print("Saving Backtesting Result")
            combined_df = pd.concat([normal_merge,grid_merge,random_merge,actual], axis=0, keys=['normal','grid','random','actual'])
            #combined_df.to_csv(os.path.join(saving_dir,f'{ticker}_combined_returns_accuracy.csv'))
            combined_collector.append(combined_df)
    
    return combined_collector,ticker


    



if __name__ == "__main__":

    output_dir = os.path.join(os.getcwd(), 'master_combined_risk')
    os.makedirs(output_dir, exist_ok=True)

    for lag in range(0,60,5):
        combined_collector, ticker = normal_run(lag)
        df = pd.concat(combined_collector.values(),axis=0, keys=list(combined_collector.keys()))
        df.to_csv(os.path.join(output_dir,f'master_combined_df_{lag}_normal.csv'))

        combined_collector, ticker = grid_run(lag)
        df = pd.concat(combined_collector.values(),axis=0, keys=list(combined_collector.keys()))
        df.to_csv(os.path.join(output_dir,f'master_combined_df_{lag}_grid.csv'))
        
        combined_collector, ticker = random_run(lag)
        df = pd.concat(combined_collector.values(),axis=0, keys=list(combined_collector.keys()))
        df.to_csv(os.path.join(output_dir,f'master_combined_df_{lag}_random.csv'))        




