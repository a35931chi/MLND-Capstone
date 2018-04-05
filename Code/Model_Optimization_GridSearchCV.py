#this capstone project is written in python IDLE 3.5.3
#this modeling portion leverages GridSearchCV to find the optimal hyperparamters for each modeling algorithms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import gmtime, strftime, time

from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
import xgboost as xgb
import lightgbm as lgb

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split, ShuffleSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

from Feature_Engineering import data_wrangle

import warnings
warnings.filterwarnings('ignore')

def rmse(prediction, yval): #this method calculates the metrics
    return np.sqrt(mean_squared_error(prediction, yval))

#the following methods represents each modeling algorithms. They follow the same setup:
#step 1 - define hyperparameters and their ranges to optimize
#step 2 - define modeling algorithms
#step 3 - run through GridSearchCV to get the best hyperparameters
#step 4 - record duration, train and test metrics.
def DTR_GSCV(Xtrain, Xval, ytrain, yval):
    '''
    no PCA:
    Decision Tree GridSearchCV:  06 Nov 2017 22:14:28
    Time algo takes: 0.627 seconds
    Train score: 0.1760 (1.46%)
    Test error: 0.1870 (1.56%)
    DecisionTreeRegressor(criterion='mse', max_depth=5, max_features=None,
               max_leaf_nodes=None, min_impurity_split=1e-07,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, presort=False, random_state=None,
               splitter='best')

    PCA:
    Decision Tree GridSearchCV:  08 Nov 2017 16:48:16
    Time algo takes: 0.216 seconds
    Train score: 0.2648 (2.20%)
    Test error: 0.2657 (2.21%)
    DecisionTreeRegressor(criterion='mse', max_depth=4, max_features=None,
               max_leaf_nodes=None, min_impurity_split=1e-07,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, presort=False, random_state=None,
               splitter='best')
    '''
    print('Decision Tree GridSearchCV: ', strftime('%d %b %Y %H:%M:%S', gmtime()))
    
    params = {'max_depth': [4]}
    
    cv_sets = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
    regressor = DecisionTreeRegressor()
    t0 = time()
    grid = GridSearchCV(estimator = regressor, param_grid = params,
                        scoring = 'neg_mean_squared_error', cv = cv_sets)
    
    grid = grid.fit(Xtrain, ytrain)
    test_score = rmse(grid.predict(Xval), yval)
    print('Time algo takes: {:.3f} seconds'.format(time() - t0))
    print('Train score: {:.4f} ({:.2f}%)'.format(np.sqrt(-grid.best_score_), np.sqrt(-grid.best_score_) / np.mean(ytrain) * 100))
    print('Test error: {:.4f} ({:.2f}%)'.format(test_score, test_score / np.mean(yval) * 100))
    
    print(grid.best_estimator_)
    #print(grid.cv_results_)
    pass

def RFR_GSCV(Xtrain, Xval, ytrain, yval):
    '''
    no PCA:
    Random Forest GridSearchCV:  07 Nov 2017 01:09:00
    Time algo takes: 1068.360 seconds
    Train score: 0.1195 (0.99%)
    Test error: 0.1434 (1.19%)
    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=22,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_split=1e-07, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               n_estimators=28, n_jobs=1, oob_score=False, random_state=None,
               verbose=0, warm_start=False)
    PCA:
    Random Forest GridSearchCV:  08 Nov 2017 16:51:32
    Time algo takes: 375.928 seconds
    Train score: 0.1956 (1.63%)
    Test error: 0.1833 (1.53%)
    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=29,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_split=1e-07, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               n_estimators=19, n_jobs=1, oob_score=False, random_state=None,
               verbose=0, warm_start=False)
    '''
    print('Random Forest GridSearchCV: ', strftime('%d %b %Y %H:%M:%S', gmtime()))

    params = {'max_depth': [29],
              'n_estimators': [19],
              'bootstrap': [True]}

    cv_sets = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
    regressor = RandomForestRegressor()
    t0 = time()
    grid = GridSearchCV(estimator = regressor, param_grid = params,
                        scoring = 'neg_mean_squared_error', cv = cv_sets)

    grid = grid.fit(Xtrain, ytrain)

    test_score = rmse(grid.predict(Xval), yval)
    print('Time algo takes: {:.3f} seconds'.format(time() - t0))
    print('Train score: {:.4f} ({:.2f}%)'.format(np.sqrt(-grid.best_score_), np.sqrt(-grid.best_score_) / np.mean(ytrain) * 100))
    print('Test error: {:.4f} ({:.2f}%)'.format(test_score, test_score / np.mean(yval) * 100))
    
    print(grid.best_estimator_)
    #print(grid.cv_results_)
    pass

def ABR_GSCV(Xtrain, Xval, ytrain, yval):
    '''
    no PCA:
    AdaBoost GridSearchCV:  07 Nov 2017 02:07:07
    Time algo takes: 803.174 seconds
    Train score: 0.1661 (1.38%)
    Test error: 0.1741 (1.45%)
    AdaBoostRegressor(base_estimator=None, learning_rate=5.5, loss='exponential',
             n_estimators=54, random_state=None)

    PCA:
    AdaBoost GridSearchCV:  08 Nov 2017 16:59:50
    Time algo takes: 461.843 seconds
    Train score: 0.2096 (1.74%)
    Test error: 0.2079 (1.73%)
    AdaBoostRegressor(base_estimator=None, learning_rate=5.5, loss='exponential',
             n_estimators=56, random_state=None)
    '''
    print('AdaBoost GridSearchCV: ', strftime('%d %b %Y %H:%M:%S', gmtime()))

    params = {'learning_rate': [5.5],
              'n_estimators': [56],
              'loss': ['exponential']}
    
    cv_sets = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
    regressor = AdaBoostRegressor()
    t0 = time()
    grid = GridSearchCV(estimator = regressor, param_grid = params,
                        scoring = 'neg_mean_squared_error', cv = cv_sets)
    
    grid = grid.fit(Xtrain, ytrain)

    test_score = rmse(grid.predict(Xval), yval)
    print('Time algo takes: {:.3f} seconds'.format(time() - t0))
    print('Train score: {:.4f} ({:.2f}%)'.format(np.sqrt(-grid.best_score_), np.sqrt(-grid.best_score_) / np.mean(ytrain) * 100))
    print('Test error: {:.4f} ({:.2f}%)'.format(test_score, test_score / np.mean(yval) * 100))
    
    print(grid.best_estimator_)
    #print(grid.cv_results_)
    pass

def GBR_GSCV(Xtrain, Xval, ytrain, yval):
    '''
    params = {'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05, 0.055, 0.06, 0.065,
                                    0.07, 0.075, 0.08, 0.085, 0.09, 0.1, 0.2, 0.3],
                  'n_estimators': range(2000, 6000, 100),
                  'max_depth': range(1,5)}
    no PCA:
    GradientBoost GridSearchCV:  07 Nov 2017 20:55:43
    Time algo takes: 2134.165 seconds
    Train score: 0.1178 (0.98%)
    Test error: 0.1154 (0.96%)
    GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.12, loss='ls', max_depth=4, max_features=None,
                 max_leaf_nodes=None, min_impurity_split=1e-07,
                 min_samples_leaf=1, min_samples_split=2,
                 min_weight_fraction_leaf=0.0, n_estimators=300,
                 presort='auto', random_state=None, subsample=1.0, verbose=0,
                 warm_start=False)

    PCA:
    GradientBoost GridSearchCV:  08 Nov 2017 18:29:55
    Time algo takes: 103.385 seconds
    Train score: 0.1973 (1.64%)
    Test error: 0.1795 (1.49%)
    GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.105, loss='ls', max_depth=4,
                 max_features=None, max_leaf_nodes=None,
                 min_impurity_split=1e-07, min_samples_leaf=1,
                 min_samples_split=2, min_weight_fraction_leaf=0.0,
                 n_estimators=80, presort='auto', random_state=None,
                 subsample=1.0, verbose=0, warm_start=False)    
    
    '''
    print('GradientBoost GridSearchCV: ', strftime('%d %b %Y %H:%M:%S', gmtime()))
    params = {'learning_rate': [0.105],
              'max_depth': [4],
              'n_estimators': [80]}
    
    cv_sets = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
    regressor = GradientBoostingRegressor()
    t0 = time()
    grid = GridSearchCV(estimator = regressor, param_grid = params,
                        scoring = 'neg_mean_squared_error', cv = cv_sets)
    
    grid = grid.fit(Xtrain, ytrain)

    test_score = rmse(grid.predict(Xval), yval)
    print('Time algo takes: {:.3f} seconds'.format(time() - t0))
    print('Train score: {:.4f} ({:.2f}%)'.format(np.sqrt(-grid.best_score_), np.sqrt(-grid.best_score_) / np.mean(ytrain) * 100))
    print('Test error: {:.4f} ({:.2f}%)'.format(test_score, test_score / np.mean(yval) * 100))
    
    print(grid.best_estimator_)
    #print(grid.cv_results_)
    pass

def KNNR_GSCV(Xtrain, Xval, ytrain, yval):
    '''
    no PCA:
    KNeighbors GridSearchCV:  07 Nov 2017 21:40:49
    Time algo takes: 83.039 seconds
    Train score: 0.2524 (2.10%)
    Test error: 0.2580 (2.15%)
    KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
              metric_params=None, n_jobs=1, n_neighbors=20, p=2,
              weights='uniform')

    PCA:
    KNeighbors GridSearchCV:  08 Nov 2017 18:37:32
    Time algo takes: 8.716 seconds
    Train score: 0.2523 (2.10%)
    Test error: 0.2581 (2.15%)
    KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
              metric_params=None, n_jobs=1, n_neighbors=20, p=2,
              weights='uniform')
    '''
    print('KNeighbors GridSearchCV: ', strftime('%d %b %Y %H:%M:%S', gmtime()))
    params = {'algorithm': ['auto'], 
              'n_neighbors': [20]}
    
    cv_sets = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
    regressor = KNeighborsRegressor()
    t0 = time()
    grid = GridSearchCV(estimator = regressor, param_grid = params,
                        scoring = 'neg_mean_squared_error', cv = cv_sets)
    
    grid = grid.fit(Xtrain, ytrain)

    test_score = rmse(grid.predict(Xval), yval)
    print('Time algo takes: {:.3f} seconds'.format(time() - t0))
    print('Train score: {:.4f} ({:.2f}%)'.format(np.sqrt(-grid.best_score_), np.sqrt(-grid.best_score_) / np.mean(ytrain) * 100))
    print('Test error: {:.4f} ({:.2f}%)'.format(test_score, test_score / np.mean(yval) * 100))
    
    print(grid.best_estimator_)
    #print(grid.cv_results_)
    pass


def MLPR_GSCV(Xtrain, Xval, ytrain, yval): #algorithms = , n_neighbors =
    '''
    params = {'hidden_layer_sizes': [(200,), (100,), (75,), (50,), (25,), (10,), (5,),
                               (100, 100), (100, 75), (100, 50), (100, 25), (100, 10), (100, 5),
                               (100, 100, 75), (100, 100, 50), (100, 100, 25), (100, 100, 10), (100, 100, 5),
                               (100, 75, 75), (100, 75, 50), (100, 75, 25), (100, 75, 10), (100, 75, 5)],
                  'max_iter': range(100, 2000, 100)}
    no PCA:
    MLP GridSearchCV:  08 Nov 2017 05:11:12
    Time algo takes: 10859.238 seconds
    Train score: 0.4370 (3.64%)
    Test error: 0.5648 (4.70%)
    MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(3000, 2000), learning_rate='constant',
           learning_rate_init=0.001, max_iter=2200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=None,
           shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
           verbose=False, warm_start=False)

    PCA:
    MLP GridSearchCV:  09 Nov 2017 02:35:49
    Time algo takes: 289.899 seconds
    Train score: 1.6060 (13.36%)
    Test error: 1.6080 (13.38%)
    MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(5, 5), learning_rate='constant',
           learning_rate_init=0.001, max_iter=2900, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=None,
           shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
           verbose=False, warm_start=False)
    '''
    print('MLP GridSearchCV: ', strftime('%d %b %Y %H:%M:%S', gmtime()))
    params = {'hidden_layer_sizes': [(5, 5)],
              'max_iter': [3000]}
    
    cv_sets = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
    regressor = MLPRegressor()
    t0 = time()
    grid = GridSearchCV(estimator = regressor, param_grid = params,
                        scoring = 'neg_mean_squared_error', cv = cv_sets)
    
    grid = grid.fit(Xtrain, ytrain)

    test_score = rmse(grid.predict(Xval), yval)
    print('Time algo takes: {:.3f} seconds'.format(time() - t0))
    print('Train score: {:.4f} ({:.2f}%)'.format(np.sqrt(-grid.best_score_), np.sqrt(-grid.best_score_) / np.mean(ytrain) * 100))
    print('Test error: {:.4f} ({:.2f}%)'.format(test_score, test_score / np.mean(yval) * 100))
    
    print(grid.best_estimator_)
    #print(grid.cv_results_)
    pass

def KerasR_GSCV(Xtrain, Xval, ytrain, yval): #algorithms = , n_neighbors =
    '''
    no PCA:
    Time algo takes: 9863.695 seconds (102 seconds)
    Train score: 0.4704 (3.91%)
    Test error: 7949.4912 (66011.74%)
    <keras.wrappers.scikit_learn.KerasRegressor object at 0x000001CF61C05B38>
    {'optimizer': 'rmsprop', 'batch_size': 25, 'epochs': 100}
    -0.221298281843
    Keras GridSearchCV:  12 Nov 2017 22:45:17

    PCA:
    Time algo takes: 3572.027 seconds (37 seconds)
    Train score: 0.5168 (4.30%)
    Test error: 0.4066 (3.38%)
    <keras.wrappers.scikit_learn.KerasRegressor object at 0x000001CF88E84E80>
    {'optimizer': 'adam', 'batch_size': 25, 'epochs': 1000}
    -0.267122648185
    '''
    def keras1(optimizer):
        model = Sequential()
        model.add(Dense(2000, input_dim = Xtrain.shape[1], activation = 'relu'))
        model.add(Dense(1))
        model.compile(loss = 'mean_squared_error', metrics = ['mse'], optimizer = optimizer)
        return model

    print('Keras GridSearchCV: ', strftime('%d %b %Y %H:%M:%S', gmtime()))
    params = {'batch_size': [25, 30, 50],
              'epochs': [100, 500, 1000],
              'optimizer': ['adam', 'rmsprop']}
    
    cv_sets = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
    regressor = KerasRegressor(build_fn = keras1, verbose = False)
    t0 = time()
    grid = GridSearchCV(estimator = regressor, param_grid = params,
                        scoring = 'neg_mean_squared_error', cv = cv_sets)

    grid = grid.fit(Xtrain.as_matrix(), ytrain.as_matrix())

    test_score = rmse(grid.predict(Xval.as_matrix()), yval.as_matrix())
    print('Time algo takes: {:.3f} seconds'.format(time() - t0))
    print('Train score: {:.4f} ({:.2f}%)'.format(np.sqrt(-grid.best_score_), np.sqrt(-grid.best_score_) / np.mean(ytrain) * 100))
    print('Test error: {:.4f} ({:.2f}%)'.format(test_score, test_score / np.mean(yval) * 100))
    
    print(grid.best_estimator_)
    print(grid.best_params_)
    print(grid.best_score_)
    #print(grid.best_index_)
    #print(grid.cv_results_)
    pass


def Lasso_GSCV(Xtrain, Xval, ytrain, yval):
    '''
    no PCA:
    Time algo takes: 0.078 seconds
    Train score: 0.1118 (0.93%)
    Test error: 0.1017 (0.84%)
    Lasso(alpha=0.0006, copy_X=True, fit_intercept=True, max_iter=100,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)

    PCA:
    Lasso GridSearchCV:  13 Nov 2017 16:30:18
    Time algo takes: 0.016 seconds
    Train score: 0.1857 (1.55%)
    Test error: 0.1801 (1.50%)
    Lasso(alpha=0.0005, copy_X=True, fit_intercept=True, max_iter=100,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)
    '''
    print('Lasso GridSearchCV: ', strftime('%d %b %Y %H:%M:%S', gmtime()))
    params = {'alpha': [0.0005],
              'max_iter': [100]}
    
    cv_sets = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
    t0 = time()

    scaler = RobustScaler()
    Xtrainscaled = scaler.fit_transform(Xtrain)
    Xvalscaled = scaler.transform(Xval)

    regressor = Lasso()
    grid = GridSearchCV(estimator = regressor, param_grid = params,
                        scoring = 'neg_mean_squared_error', cv = cv_sets)
    
    grid = grid.fit(Xtrainscaled, ytrain)
    
    test_score = rmse(grid.predict(Xvalscaled), yval)
    print('Time algo takes: {:.3f} seconds'.format(time() - t0))
    print('Train score: {:.4f} ({:.2f}%)'.format(np.sqrt(-grid.best_score_), np.sqrt(-grid.best_score_) / np.mean(ytrain) * 100))
    print('Test error: {:.4f} ({:.2f}%)'.format(test_score, test_score / np.mean(yval) * 100))
    
    print(grid.best_estimator_)
    #print(grid.cv_results_)
    pass


def ENetR_GSCV(Xtrain, Xval, ytrain, yval):
    '''            
    no PCA:
    ENet GridSearchCV:  13 Nov 2017 08:18:24
    Time algo takes: 5.485 seconds
    Train score: 0.1115 (0.93%)
    Test error: 0.1044 (0.87%)
    ElasticNet(alpha=0.013, copy_X=True, fit_intercept=True, l1_ratio=1e-07,
          max_iter=1000, normalize=False, positive=False, precompute=False,
          random_state=None, selection='cyclic', tol=0.0001, warm_start=False)
    PCA:
    ENet GridSearchCV:  13 Nov 2017 08:29:32
    Time algo takes: 0.141 seconds
    Train score: 0.1857 (1.55%)
    Test error: 0.1800 (1.49%)
    ElasticNet(alpha=1.6e-05, copy_X=True, fit_intercept=True, l1_ratio=0.09,
          max_iter=1000, normalize=False, positive=False, precompute=False,
          random_state=None, selection='cyclic', tol=0.0001, warm_start=False)
      
    '''
    print('ENet GridSearchCV: ', strftime('%d %b %Y %H:%M:%S', gmtime()))
    params = {'alpha': [0.00001, 0.000012, 0.000013, 0.000014, 0.000015, 0.000016],
              'l1_ratio': [1, 5, 9]}
    
    cv_sets = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
    
    t0 = time()
    
    scaler = RobustScaler()
    Xtrainscaled = scaler.fit_transform(Xtrain)
    Xvalscaled = scaler.transform(Xval)
    
    regressor = ElasticNet()
    
    grid = GridSearchCV(estimator = regressor, param_grid = params,
                        scoring = 'neg_mean_squared_error', cv = cv_sets)
    
    grid = grid.fit(Xtrainscaled, ytrain)

    test_score = rmse(grid.predict(Xvalscaled), yval)
    print('Time algo takes: {:.3f} seconds'.format(time() - t0))
    print('Train score: {:.4f} ({:.2f}%)'.format(np.sqrt(-grid.best_score_), np.sqrt(-grid.best_score_) / np.mean(ytrain) * 100))
    print('Test error: {:.4f} ({:.2f}%)'.format(test_score, test_score / np.mean(yval) * 100))
    
    print(grid.best_estimator_)
    #print(grid.cv_results_)
    pass


def XGBR_GSCV(Xtrain, Xval, ytrain, yval):
    '''
    params = {'reg_alpha': [0.00001, 0.000025, 0.00005,0.000075,
                        0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007,
                        0.0008, 0.0009, 0.001, 0.0025, 0.005],
              'n_estimators': range(1500, 5000, 100),
              'learning_rate': [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065,
                                0.07, 0.075, 0.08, 0.085, 0.09, 0.1],
              'max_depth': range(1, 10)}
    no PCA:
    XGBoost GridSearchCV:  08 Nov 2017 16:50:58
    Time algo takes: 4296.251 seconds
    Train score: 0.1153 (0.96%)
    Test error: 0.1087 (0.90%)
    XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=1, gamma=0,
           learning_rate=0.08, max_delta_step=0, max_depth=4,
           min_child_weight=1, missing=None, n_estimators=300, nthread=-1,
           objective='reg:linear', reg_alpha=0.0002, reg_lambda=1,
           scale_pos_weight=1, seed=0, silent=True, subsample=1)


    PCA:
    XGBoost GridSearchCV:  08 Nov 2017 20:24:08
    Time algo takes: 861.214 seconds
    Train score: 0.1980 (1.65%)
    Test error: 0.1847 (1.54%)
    XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=1, gamma=0,
           learning_rate=0.09, max_delta_step=0, max_depth=2,
           min_child_weight=1, missing=None, n_estimators=150, nthread=-1,
           objective='reg:linear', reg_alpha=0.0001, reg_lambda=1,
           scale_pos_weight=1, seed=0, silent=True, subsample=1)
    '''
    print('XGBoost GridSearchCV: ', strftime('%d %b %Y %H:%M:%S', gmtime()))
    params = {'learning_rate': [0.09],
              'max_depth': [2],
              'n_estimators': [150],
              'reg_alpha': [0.0001]}
    
    cv_sets = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
    regressor = xgb.XGBRegressor()
    t0 = time()
    grid = GridSearchCV(estimator = regressor, param_grid = params,
                        scoring = 'neg_mean_squared_error', cv = cv_sets)
    
    grid = grid.fit(Xtrain, ytrain)

    test_score = rmse(grid.predict(Xval), yval)
    print('Time algo takes: {:.3f} seconds'.format(time() - t0))
    print('Train score: {:.4f} ({:.2f}%)'.format(np.sqrt(-grid.best_score_), np.sqrt(-grid.best_score_) / np.mean(ytrain) * 100))
    print('Test error: {:.4f} ({:.2f}%)'.format(test_score, test_score / np.mean(yval) * 100))
    
    print(grid.best_estimator_)
    #print(grid.cv_results_)
    pass

def Lasso_Robust(Xtrain, Xval, ytrain, yval):
    '''
    Best Model:
    Lasso(alpha=0.0006, copy_X=True, fit_intercept=True, max_iter=100,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)
    '''
    num_cols = ['LotFrontage', 'LotArea', 'TotalSF', 'OverallQual',
                'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
                'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
                'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
                'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
                'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                'PoolArea', 'MiscVal']
    Xtrain.reset_index(inplace = True)
    Xval.reset_index(inplace = True)
    Xtrain.drop(['index'], axis = 1, inplace = True)
    Xval.drop(['index'], axis = 1, inplace = True)
    
    ytrain.reset_index(drop = True, inplace = True)
    yval.reset_index(drop = True, inplace = True)
    
    n_folds = 10
    
    if True: #try different folds
        regressor = make_pipeline(RobustScaler(), Lasso(alpha = 0.0006, max_iter = 100))
        kf = KFold(n_folds, shuffle = True)
        rmse = np.sqrt(-cross_val_score(regressor, Xtrain.values, ytrain, scoring = 'neg_mean_squared_error', cv = kf))
        plt.plot(rmse)
        plt.xlabel('Kth Fold')
        plt.ylabel('RMSE')
        plt.title('Kth Fold vs. RMSE')
        plt.axhline(np.mean(rmse), linestyle = ':', color = 'r', label = 'mean RMSE')
        plt.legend()
        plt.tight_layout()
        plt.savefig('RMSE for each KFold.png')
        plt.show()
        
    if True: #try different random states
        mean = []
        std = []
        for i in range(20):
            regressor = make_pipeline(RobustScaler(), Lasso(alpha = 0.0006, max_iter = 100))
            kf = KFold(n_folds, shuffle = True)
            rmse = np.sqrt(-cross_val_score(regressor, Xtrain.values, ytrain, scoring = 'neg_mean_squared_error', cv = kf))
            mean.append(np.mean(rmse))
            std.append(np.std(rmse))
        fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (10, 5))
        ax1.plot(mean)
        ax1.axhline(np.mean(mean), linestyle = ':', color = 'r')
        ax1.set_xlabel('#th iteration')
        ax1.set_ylabel('mean RMSE')
        ax1.set_title('20 K-Fold results: mean RMSE')
        ax2.plot(std)
        ax2.axhline(np.mean(std), linestyle = ':', color = 'r')
        ax2.set_xlabel('#th iteration')
        ax2.set_ylabel('std RMSE')
        ax2.set_title('20 K-Fold results: std RMSE')
        plt.tight_layout()
        plt.savefig('avg RMSE for diff random state.png')
        plt.show()

    #try small changes to the dataset
    if True: 
        #deletion observations points
        yidx = Xtrain.shape[0]
        error = []
        fracs = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2]

        for frac in fracs: 
            Xtrain_copy = pd.DataFrame.copy(Xtrain)
            ytrain_copy = pd.DataFrame.copy(ytrain)
            cidx = np.random.choice(yidx, round(0.05 * yidx))
            Xtrain_copy.drop(Xtrain_copy.index[cidx], inplace = True)
            ytrain_copy.drop(ytrain_copy.index[cidx], inplace = True)

            scaler = RobustScaler()
            Xtrainscaled = scaler.fit_transform(Xtrain_copy)
            Xvalscaled = scaler.transform(Xval)

            regressor = Lasso(alpha = 0.0006, copy_X = True, fit_intercept = True,
                              max_iter = 100, normalize = False, positive = False,
                              precompute = False, random_state = None, selection = 'cyclic',
                              tol = 0.0001, warm_start = False)
    
            regressor.fit(Xtrainscaled, ytrain_copy)
            test_score = np.sqrt(mean_squared_error(regressor.predict(Xvalscaled), yval))
            error.append(test_score)

            print('{:.0f}% data deleted, Test error: {:.4f} ({:.2f}%)'.format(frac * 100, test_score, test_score / np.mean(yval) * 100))
        plt.plot(fracs, error)
        plt.annotate('{:.4f}, {:.2f}% of target'.format(test_score, test_score / np.mean(yval) * 100),
                xy=(frac, test_score), xytext=(30, -20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
        #plt.ylim(ymin = 0)
        plt.axhline(0.1017, linestyle = ':', color = 'r', label = 'benchmark')
        plt.title('Test RMSE vs. % data removed')
        plt.xlabel('% data removed')
        plt.ylabel('Test RMSE')
        plt.legend()
        plt.tight_layout()
        plt.savefig('test RMSE vs % data removed.png')
        plt.show()
        
    if True:
        #select some observations and mutiply some points by 10
        yidx = Xtrain.shape[0]
        error_m = []
        error_d = []
        fracs = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2]

        for frac in fracs: 
            Xtrain_copy = pd.DataFrame.copy(Xtrain)

            cidx = np.random.choice(yidx, round(frac * yidx))
            Xtrain_copy.ix[cidx , num_cols] = Xtrain_copy.ix[cidx , num_cols] * 10
            
            scaler = RobustScaler()
            Xtrainscaled = scaler.fit_transform(Xtrain_copy)
            Xvalscaled = scaler.transform(Xval)

            regressor = Lasso(alpha = 0.0006, copy_X = True, fit_intercept = True,
                              max_iter = 100, normalize = False, positive = False,
                              precompute = False, random_state = None, selection = 'cyclic',
                              tol = 0.0001, warm_start = False)
    
            regressor.fit(Xtrainscaled, ytrain)
            test_scorem = np.sqrt(mean_squared_error(regressor.predict(Xvalscaled), yval))
            error_m.append(test_scorem)
            print('{:.0f}% data scaled up, Test error: {:.4f} ({:.2f}%)'.format(frac * 100, test_scorem, test_scorem / np.mean(yval) * 100))

        for frac in fracs: 
            Xtrain_copy = pd.DataFrame.copy(Xtrain)

            cidx = np.random.choice(yidx, round(frac * yidx))
            Xtrain_copy.ix[cidx , num_cols] = Xtrain_copy.ix[cidx , num_cols] / 10
            
            scaler = RobustScaler()
            Xtrainscaled = scaler.fit_transform(Xtrain_copy)
            Xvalscaled = scaler.transform(Xval)

            regressor = Lasso(alpha = 0.0006, copy_X = True, fit_intercept = True,
                              max_iter = 100, normalize = False, positive = False,
                              precompute = False, random_state = None, selection = 'cyclic',
                              tol = 0.0001, warm_start = False)
    
            regressor.fit(Xtrainscaled, ytrain)
            test_scored = np.sqrt(mean_squared_error(regressor.predict(Xvalscaled), yval))
            error_d.append(test_scored)
            print('{:.0f}% data scaled up, Test error: {:.4f} ({:.2f}%)'.format(frac * 100, test_scored, test_scored / np.mean(yval) * 100))
            
        plt.plot(fracs, error_m, label = 'some data scaled up')
        plt.plot(fracs, error_d, label = 'some data scaled down')
        plt.annotate('{:.4f}, {:.2f}% of target'.format(test_scorem, test_scorem / np.mean(yval) * 100),
                xy=(frac, test_scorem), xytext=(30, -20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
        plt.annotate('{:.4f}, {:.2f}% of target'.format(test_scored, test_scored / np.mean(yval) * 100),
                xy=(frac, test_scored), xytext=(30, -20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
        
        #plt.ylim(ymin = 0)
        plt.axhline(0.1017, linestyle = ':', color = 'r', label = 'benchmark')
        plt.title('Test RMSE vs. % data altered')
        plt.xlabel('% data altered')
        plt.ylabel('Test RMSE')
        plt.legend()
        plt.tight_layout()
        plt.savefig('test RMSE vs % data scaled.png')
        plt.show()

    pass

def Lasso_Residual_plot(Xtrain, Xval, ytrain, yval):
    
    regressor = Lasso(alpha = 0.0006, copy_X = True, fit_intercept = True,
                      max_iter = 100, normalize = False, positive = False,
                      precompute = False, random_state = None,
                      selection = 'cyclic', tol = 0.0001, warm_start = False)

    scaler = RobustScaler()
    Xtrainscaled = scaler.fit_transform(Xtrain)
    Xvalscaled = scaler.transform(Xval)

    regressor = regressor.fit(Xtrainscaled, ytrain)

    ytrain_residual = regressor.predict(Xtrainscaled) - ytrain
    ytrain_residual.rename('Residual', inplace = True)
    yval_residual = regressor.predict(Xvalscaled) - yval
    yval_residual.rename('Residual', inplace = True)
    df_ytrain = pd.concat([ytrain, ytrain_residual], axis = 1)
    
    df_ytrain['data'] = 'train'
    df_yval = pd.concat([yval, yval_residual], axis = 1)
    df_yval['data'] = 'val'
    df = pd.concat([df_ytrain, df_yval], axis = 0)
    
    # Plot the residuals after fitting lasso model
    sns.lmplot(data = df, x = 'SalePrice', y = 'Residual', col = 'data', hue = 'data')
    plt.suptitle('Lasso Regression - Residual plot for Train data, train corr: {:.3f}, val corr: {:.3f}'.format(df_ytrain.corr().as_matrix()[0, 1], df_yval.corr().as_matrix()[0, 1]))
    plt.tight_layout(rect = [0, 0.03, 1, 0.95])  
    plt.savefig('Lasso Regression Residual Plot.png')
    plt.show()

    pass


if __name__ == '__main__':
    #gather data
    Xtrain, Xtest, y, XtrainPCA, XtestPCA = data_wrangle()
    #split data into train/test datasets
    Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, y, test_size = 0.2, random_state = 0)
    XtrainPCA, XvalPCA, ytrainPCA, yvalPCA = train_test_split(XtrainPCA, y, test_size = 0.2, random_state = 0)

    #I use the following code to test the combination of PCA and nonPCA datasets vs. modeling algorithms
    
    #DTR_GSCV(Xtrain, Xval, ytrain, yval)
    #DTR_GSCV(XtrainPCA, XvalPCA, ytrainPCA, yvalPCA)
    
    #RFR_GSCV(Xtrain, Xval, ytrain, yval)
    #RFR_GSCV(XtrainPCA, XvalPCA, ytrainPCA, yvalPCA)
    
    #ABR_GSCV(Xtrain, Xval, ytrain, yval)
    #ABR_GSCV(XtrainPCA, XvalPCA, ytrainPCA, yvalPCA)

    #GBR_GSCV(Xtrain, Xval, ytrain, yval)
    #GBR_GSCV(XtrainPCA, XvalPCA, ytrainPCA, yvalPCA)
    
    #KNNR_GSCV(Xtrain, Xval, ytrain, yval)
    #KNNR_GSCV(XtrainPCA, XvalPCA, ytrainPCA, yvalPCA)
    
    #MLPR_GSCV(Xtrain, Xval, ytrain, yval)
    #MLPR_GSCV(XtrainPCA, XvalPCA, ytrainPCA, yvalPCA)

    #KerasR_GSCV(Xtrain, Xval, ytrain, yval)
    #KerasR_GSCV(XtrainPCA, XvalPCA, ytrainPCA, yvalPCA)
    
    #Lasso_GSCV(Xtrain, Xval, ytrain, yval)
    #Lasso_GSCV(XtrainPCA, XvalPCA, ytrainPCA, yvalPCA)
    
    #ENetR_GSCV(Xtrain, Xval, ytrain, yval)
    #ENetR_GSCV(XtrainPCA, XvalPCA, ytrainPCA, yvalPCA)
    
    #XGBR_GSCV(Xtrain, Xval, ytrain, yval)
    #XGBR_GSCV(XtrainPCA, XvalPCA, ytrainPCA, yvalPCA)
    
    Lasso_Robust(Xtrain, Xval, ytrain, yval)
    
    #Lasso_Residual_plot(Xtrain, Xval, ytrain, yval)
