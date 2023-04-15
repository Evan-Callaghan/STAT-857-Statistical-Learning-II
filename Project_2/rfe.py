import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


def flat_list(my_list):
    
    ## Defining list to store results
    out_list = list()
    for i in my_list:
        out_list += i
    return out_list

def RF_RFE_rep_cross_val(X, Y, numb_folds, max_features, numb_reps):
    
    ## Defining list to store results
    RFE_rep_results = list()
    for i in range(2, max_features):
        RFE_rep_results.append(RF_rep_cross_val(X, Y, numb_folds, i, numb_reps))
        print('Features -->', i) ## Sanity check
    return RFE_rep_results

def RF_rep_cross_val(X, Y, numb_folds, numb_features, numb_reps):
    
    ## Defining the list to store results
    rep_results = list()
    for i in range(0, numb_reps):
        rep_results.append(RF_cross_val(X, Y, numb_folds, numb_features))
    return flat_list(rep_results)

def RF_cross_val(X, Y, numb_folds, numb_features):
    
    ## Defining list to store results
    results = list()
    
    ## Defining the number of folds
    kf = StratifiedKFold(n_splits = numb_folds, shuffle = True, random_state = 42)
    
    for train_index, test_index in kf.split(X, Y):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        ## Running RFE with i features
        RF_rfe = RFE(estimator = RandomForestClassifier(n_estimators = 100, max_depth = 5), 
                     n_features_to_select = numb_features).fit(X_train, Y_train)
        
        ## Variables to be considered
        to_select = X_train.columns[RF_rfe.support_]
        to_select_list.append(RF_rfe.support_)
        
        ## Building the Random Forest model
        X_train_md = X_train[to_select]
        X_test_md = X_test[to_select]
        
        RF_md = RandomForestClassifier(n_estimators = 500, max_depth = 5).fit(X_train_md, Y_train)
        
        ## Predicting on the test data-frame and storing RMSE
        results.append(log_loss(Y_test, RF_md.predict_proba(X_test_md)))

    return results

## Defining list to store results
to_select_list = list()

## Defining input and target variables
training = pd.read_csv('Data/Training_RFE.csv'); print('-- Data read --')
X = training.drop(columns = ['interest_level']); Y = training['interest_level']

## Running RFE to estimate number of features to be selected
RFE_numb_features = RF_RFE_rep_cross_val(X, Y, numb_folds = 5, max_features = 101, numb_reps = 1)


## Identifying features
features = pd.DataFrame(to_select_list)
features.columns = X.columns
feature_selections = 100 * features.apply(np.sum, axis = 0) / features.shape[0]
feature_selections = pd.DataFrame(feature_selections).reset_index(drop = False)

## Model performance given the number of variables
feature_performance = pd.DataFrame(RFE_numb_features)
feature_performance.columns = [['Split_1', 'Split_2', 'Split_3']]
feature_performance['Mean'] = feature_performance.apply(np.mean, axis = 1)
feature_performance['Num_features'] = feature_performance.index + 2

print(feature_performance)
feature_performance.to_csv('performance.csv', index = False)

feature_selections = feature_selections.sort_values(0, ascending = False)
print(feature_selections)
feature_selections.to_csv('features.csv', index = False)
