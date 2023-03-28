import optuna
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

## Reading the data
data = pd.read_csv('Data/W23P1_training_new.csv')

## Defining the input and target variables
variables = ['distance', 'duration', 'haversine', 'time_estimate', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 
             'dropoff_latitude', 'dropoff_other', 'change_borough', 'LGA']

X = data.drop(columns = ['uid', 'fare_amount', 'pickup_datetime', 
                         'passenger_count', 'distance', 'duration', 'haversine', 'same_lat', 'same_long', 'same_coord', 'pickup_longitude_rounded', 
                         'pickup_latitude_rounded', 'dropoff_longitude_rounded', 'dropoff_latitude_rounded', 'same_lat_rounded', 'same_long_rounded', 
                         'same_coord_rounded', 'pickup_date', 'pickup_day', 'pickup_hour', 'pickup_day_of_week', 'Friday', 'Monday', 'Saturday', 'Sunday', 
                         'Thursday', 'Tuesday', 'Wednesday', 'weekend', 'morning', 'rush_hour', 'overnight', 'holiday', 'pickup_LGA', 'dropoff_LGA', 'LGA', 
                         'pickup_JFK', 'dropoff_JFK', 'JFK', 'pickup_EWR', 'dropoff_EWR', 'EWR', 'pickup_airport', 'dropoff_airport', 'airport', 'pickup_borough', 
                         'dropoff_borough', 'change_borough', 'pickup_bronx', 'pickup_brooklyn', 'pickup_manhattan', 'pickup_other', 'pickup_queens', 
                         'pickup_staten_island', 'dropoff_bronx', 'dropoff_brooklyn', 'dropoff_manhattan', 'dropoff_other', 'dropoff_queens', 
                         'dropoff_staten_island'])
Y = data['fare_amount']

## Splitting the data into train and validation sets
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = 0.75)

## Defining Optuna objective functions
def rf_reg_objective(trial):

    ## Defining the XGBoost hyper-parameter grid
    rf_param_grid = {'n_estimators': trial.suggest_int('n_estimators', 100, 1000, 50),
                     'max_depth': trial.suggest_int('max_depth', 3, 12), 
                     'min_samples_split': trial.suggest_int('min_samples_split', 2, 20), 
                     'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20), 
                     'random_state': trial.suggest_int('random_state', 1, 500),
                     'max_features': trial.suggest_categorical('max_features', ['sqrt', None])
                    }
    
    ## Building the model
    rf_md = RandomForestRegressor(**rf_param_grid, n_jobs = -1, criterion = 'squared_error').fit(X_train, Y_train)
    
    ## Predicting on the test data-frame
    rf_md_preds = rf_md.predict(X_validation)
    
    ## Evaluating model performance on the test set
    rf_md_mse = mean_squared_error(Y_validation, rf_md_preds, squared = False)
    
    return rf_md_mse

def hist_reg_objective(trial):

    ## Defining the XGBoost hyper-parameter grid
    hist_param_grid = {'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step = 0.01),
                       'max_iter': trial.suggest_int('n_estimators', 100, 1000, 50),
                       'max_depth': trial.suggest_int('max_depth', 3, 12), 
                       'l2_regularization': trial.suggest_float('l2_regularization', 0, 0.1, step = 0.002),
                       'random_state': trial.suggest_int('random_state', 1, 500),
                      }
    
    ## Building the model
    hist_md = HistGradientBoostingRegressor(**hist_param_grid, loss = 'squared_error', early_stopping = True).fit(X_train, Y_train)
    
    ## Predicting on the test data-frame
    hist_md_preds = hist_md.predict(X_validation)
    
    ## Evaluating model performance on the test set
    hist_md_mse = mean_squared_error(Y_validation, hist_md_preds, squared = False)
    
    return hist_md_mse

def xgb_reg_objective(trial):

    ## Defining the XGBoost hyper-parameter grid
    xgboost_param_grid = {'n_estimators': trial.suggest_int('n_estimators', 100, 1000, 50), 
                          'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step = 0.01), 
                          'max_depth': trial.suggest_int('max_depth', 3, 12), 
                          'gamma': trial.suggest_float('gamma', 0, 0.3, step = 0.05), 
                          'min_child_weight': trial.suggest_int('min_child_weight', 1, 20), 
                          'subsample': trial.suggest_float('subsample', 0.6, 1, step = 0.05), 
                          'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1, step = 0.05),
                          'seed': trial.suggest_int('seed', 1, 1000)
                         }
    
    ## Building the model
    xgb_md = XGBRegressor(**xgboost_param_grid, n_jobs = -1, booster = 'gbtree', tree_method = 'hist').fit(X_train, Y_train)
    
    ## Predicting on the test data-frame
    xgb_md_preds = xgb_md.predict(X_validation)
    
    ## Evaluating model performance on the test set
    xgb_md_mse = mean_squared_error(Y_validation, xgb_md_preds, squared = False)
    
    return xgb_md_mse

def lgbm_reg_objective(trial):
    
    ## Defining the LGB hyper-parameter grid
    LGB_param_grid = {'n_estimators': trial.suggest_int('n_estimators', 100, 1000, 50),
                      'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step = 0.01),
                      'num_leaves': trial.suggest_int('num_leaves', 5, 40, step = 1),
                      'max_depth': trial.suggest_int('max_depth', 3, 12),
                      'subsample': trial.suggest_float('subsample', 0.6, 1, step = 0.05), 
                      'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1, step = 0.05),
                      'random_state': trial.suggest_int('random_state', 1, 1000),
                     }
                     
    ## Building the LightGBM model
    model = LGBMRegressor(**LGB_param_grid, n_jobs = -1, boosting_type = 'dart', objective = 'rmse', verbosity = -1).fit(X_train, Y_train)
        
    ## Predicting on the test data-frame
    lgbm_md_preds = model.predict(X_validation)
    
    ## Evaluating model performance on the test set
    lgbm_md_mse = mean_squared_error(Y_validation, lgbm_md_preds, squared = False)
    
    return lgbm_md_mse



## Starting RandomForest
## ----
## Creating a study object and to optimize the home objective function
study_rf = optuna.create_study(direction = 'minimize')
study_rf.optimize(rf_reg_objective, n_trials = 100)

## Starting HistGradientBoosting
## ----
## Creating a study object and to optimize the home objective function
study_hist = optuna.create_study(direction = 'minimize')
study_hist.optimize(hist_reg_objective, n_trials = 500)

## Starting XGBoost
## ----
## Creating a study object and to optimize the home objective function
study_xgb = optuna.create_study(direction = 'minimize')
study_xgb.optimize(xgb_reg_objective, n_trials = 500)

## Starting LightGBM
## ----
## Creating a study object and to optimize the home objective function
study_lgbm = optuna.create_study(direction = 'minimize')
study_lgbm.optimize(lgbm_reg_objective, n_trials = 500)

## Printing best hyper-parameter set
print('Random Forest: \n', study_rf.best_trial.params)
print(study_rf.best_trial.value)

## Printing best hyper-parameter set
print('HistGB: \n', study_hist.best_trial.params)
print(study_hist.best_trial.value)

## Printing best hyper-parameter set
print('\nXGBoost: \n', study_xgb.best_trial.params)
print(study_xgb.best_trial.value)

## Printing best hyper-parameter set
print('\nLightGBM: \n', study_lgbm.best_trial.params)
print(study_lgbm.best_trial.value)