import numpy as np
import pandas as pd

import optuna
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

## Defining optuna objective functions
class rf_objective:

    def __init__(self, seed):
        self.seed = seed

    def __call__(self, trial):
        
        params = dict(criterion = 'log_loss',
                      n_estimators = trial.suggest_int('n_estimators', 100, 1500, step = 100),
                      max_depth = trial.suggest_int('max_depth', 3, 12, step = 1),
                      min_samples_split = trial.suggest_int('min_samples_split', 5, 100, step = 5),
                      min_samples_leaf = trial.suggest_int('min_samples_leaf', 5, 100, step = 5))
        scores = []
        skf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = self.seed)
        for train_idx, valid_idx in skf.split(X, Y):

            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            Y_train , Y_valid = Y.iloc[train_idx] , Y.iloc[valid_idx]

            model = RandomForestClassifier(**params).fit(X_train, Y_train)

            preds_valid = model.predict_proba(X_valid)
            scores.append(log_loss(Y_valid, preds_valid))
        return np.mean(scores)
                                   
                                   
class xgb_objective:

    def __init__(self, seed):
        self.seed = seed

    def __call__(self, trial):
        
        params = dict(objective = 'multi:softprob',
                      eval_metric = 'mlogloss',
                      n_estimators = trial.suggest_int('n_estimators', 300, 1500, step = 100),
                      learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, step = 0.01),
                      max_depth = trial.suggest_int('max_depth', 3, 12, step = 1),
                      gamma = trial.suggest_float('reg_alpha', 0, 100, step = 10),
                      min_child_weight = trial.suggest_int('min_child_weight', 0, 200, step = 10),
                      subsample = trial.suggest_float('subsample', 0.6, 1, step = 0.05), 
                      colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1, step = 0.05))
        scores = []
        skf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = self.seed)
        for train_idx, valid_idx in skf.split(X, Y):

            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            Y_train , Y_valid = Y.iloc[train_idx] , Y.iloc[valid_idx]

            model = XGBClassifier(**params).fit(X_train, Y_train)

            preds_valid = model.predict_proba(X_valid)
            scores.append(log_loss(Y_valid, preds_valid))
        return np.mean(scores)
                

class lgbm_objective:

    def __init__(self, seed):
        self.seed = seed

    def __call__(self, trial):
        
        params = dict(objective = 'multiclass',
                      metric = 'multi_logloss',
                      n_estimators = trial.suggest_int('n_estimators', 300, 1500, step = 100),
                      learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, step = 0.01),
                      max_depth = trial.suggest_int('max_depth', 3, 12, step = 1),
                      reg_alpha = trial.suggest_float('reg_alpha', 0.1, 10, log = True),
                      reg_lambda = trial.suggest_float('reg_lambda', 0.1, 10, log = True),
                      num_leaves = trial.suggest_int('num_leaves', 11, 101, step = 5),
                      subsample = trial.suggest_float('subsample', 0.4, 1, step = 0.05),
                      colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1, step = 0.05))
        scores = []
        skf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = self.seed)
        for train_idx, valid_idx in skf.split(X, Y):

            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            Y_train , Y_valid = Y.iloc[train_idx] , Y.iloc[valid_idx]

            model = LGBMClassifier(**params).fit(X_train, Y_train)

            preds_valid = model.predict_proba(X_valid)
            scores.append(log_loss(Y_valid, preds_valid))
        return np.mean(scores)
                                   
class hist_objective:

    def __init__(self, seed):
        self.seed = seed

    def __call__(self, trial):
        
        params = dict(max_iter = trial.suggest_int('max_iter', 300, 1000, step = 100),
                      learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, step = 0.01),
                      max_depth = trial.suggest_int('max_depth', 3, 12, step = 1),
                      l2_regularization = trial.suggest_float('l2_regularization', 0.1, 10))
        scores = []
        skf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = self.seed)
        for train_idx, valid_idx in skf.split(X, Y):

            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            Y_train , Y_valid = Y.iloc[train_idx] , Y.iloc[valid_idx]

            model = HistGradientBoostingClassifier(**params).fit(X_train, Y_train)

            preds_valid = model.predict_proba(X_valid)
            scores.append(log_loss(Y_valid, preds_valid))
        return np.mean(scores)
    
## Defining SEED and Trials
SEED = 42
N_TRIALS = 100

## Defining input and target variables
## Defining input and target variables
training = pd.read_csv('Data/Training_hp.csv')
X = training.drop(columns = ['interest_level'])
Y = training['interest_level']

## Executing the optimization
study_rf = optuna.create_study(direction = 'minimize')
study_rf.optimize(rf_objective(SEED), n_trials = N_TRIALS)

study_xgb = optuna.create_study(direction = 'minimize')
study_xgb.optimize(xgb_objective(SEED), n_trials = N_TRIALS)

study_lgbm = optuna.create_study(direction = 'minimize')
study_lgbm.optimize(lgbm_objective(SEED), n_trials = N_TRIALS)

study_hist = optuna.create_study(direction = 'minimize')
study_hist.optimize(hist_objective(SEED), n_trials = N_TRIALS)

print(study_rf.best_trial.params)
print(study_rf.best_trial.value)

print(study_xgb.best_trial.params)
print(study_xgb.best_trial.value)

print(study_lgbm.best_trial.params)
print(study_lgbm.best_trial.value)

print(study_hist.best_trial.params)
print(study_hist.best_trial.value)