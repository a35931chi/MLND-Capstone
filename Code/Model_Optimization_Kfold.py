#this capstone project is written in python IDLE 3.5.3
#this is code to get a sense of how the models will perform and produce visualization on how error would move with hyperparameters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

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
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

from Feature_Engineering import data_wrangle

def rmse_cv(model, Xtrain, ytrain): #calculate RMSE with cross validation
    #Validation function
    n_folds = 5
    kf = KFold(n_folds, shuffle = True, random_state = 42).get_n_splits(Xtrain.values)
    rmse = np.sqrt(-cross_val_score(model, Xtrain.values, ytrain, scoring = 'neg_mean_squared_error', cv = kf))
    return rmse

def rmse(prediction, yval): #calculate RMSE
    return np.sqrt(mean_squared_error(prediction, yval))

#I wrote this method to produce visualization on error vs. hyperparameter
#this method also works with categories
def seek_hyperpram_plot(avg_score, avg_std, test_scores, ranges, avgytrain, avgytest, labels):
    best_idx = avg_score.index(np.min(avg_score))

    print('{} optimized: {}'.format(labels[2], ranges[best_idx]))
    print('Best scores: train {:.4f} and test {:.4f}'.format(avg_score[best_idx], test_scores[best_idx]))
    print('As % of avg target: train {:.2f}% and test {:.2f}%'.format(avg_score[best_idx] / avgytrain * 100,
                                                                      test_scores[best_idx] / avgytest * 100))
        
    if type(ranges[0]) not in (int, float):
        xlabel = [str(e) for e in ranges]
        tempxlabel = range(len(xlabel))
        fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (10, 5))
        ax1.plot(tempxlabel, avg_score, label = 'train')
        ax1.plot(tempxlabel, test_scores, label = 'test')
        ax1.scatter(tempxlabel[best_idx], avg_score[best_idx], color = 'r')
        ax1.set_xticks(tempxlabel)
        ax1.set_xticklabels(xlabel, rotation = 90)
        ax1.set_title(labels[0]+ ': ' + labels[1] + ' vs. avg rmse')
        ax1.set_xlabel(labels[1])
        ax1.set_ylabel('avg rsme')
        ax1.text(max(ax1.get_xlim()), max(ax1.get_ylim()),  
                 'best rmse {:.4f} @ {}: {}'.format(avg_score[best_idx], labels[1], xlabel[best_idx]),
                 verticalalignment = 'top', horizontalalignment = 'right', fontsize = 10)
        ax1.legend(loc = 'best')
        
        ax2.plot(tempxlabel, avg_std)
        ax2.scatter(tempxlabel[best_idx], avg_std[best_idx], color = 'r')
        ax2.set_xticks(tempxlabel)
        ax2.set_xticklabels(xlabel, rotation = 90)
        ax2.set_title(labels[0]+ ': ' + labels[1] + ' vs. std rmse')
        ax2.set_xlabel(labels[1])
        ax2.set_ylabel('std rsme')
        ax2.text(max(ax2.get_xlim()), max(ax2.get_ylim()), 
                 'std rmse {:.4f} @ {}: {}'.format(avg_std[best_idx], labels[1], xlabel[best_idx]),
                 verticalalignment = 'top', horizontalalignment = 'right', fontsize = 10)
        plt.tight_layout()
        plt.savefig(labels[2] + '.png')
        plt.show()
    else:
        fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (10, 5))
        ax1.plot(ranges, avg_score, label = 'train')
        ax1.plot(ranges, test_scores, label = 'test')
        ax1.scatter(ranges[best_idx], avg_score[best_idx], color = 'r')
        ax1.set_title(labels[0]+ ': ' + labels[1] + ' vs. avg rmse')
        ax1.set_xlabel(labels[1])
        ax1.set_ylabel('avg rsme')
        ax1.text(max(ax1.get_xlim()), max(ax1.get_ylim()),  
                 'best rmse {:.4f} @ {}: {}'.format(avg_score[best_idx], labels[1], ranges[best_idx]),
                 verticalalignment = 'top', horizontalalignment = 'right', fontsize = 10)
        ax1.legend(loc = 'best')
        
        ax2.plot(ranges, avg_std)
        ax2.scatter(ranges[best_idx], avg_std[best_idx], color = 'r')
        ax2.set_title(labels[0]+ ': ' + labels[1] + ' vs. std rmse')
        ax2.set_xlabel(labels[1])
        ax2.set_ylabel('std rsme')
        ax2.text(max(ax2.get_xlim()), max(ax2.get_ylim()), 
                 'std rmse {:.4f} @ {}: {}'.format(avg_std[best_idx], labels[1], ranges[best_idx]),
                 verticalalignment = 'top', horizontalalignment = 'right', fontsize = 10)
        plt.tight_layout()
        plt.savefig(labels[2] + '.png')
        plt.show()
    
    pass

#the following sections goes into each individual modeling algos
#the general setups are similar.
#step 1 - define range of hyper parameters
#step 2 - perform cross validation on each individual parameter values, record erro
#step 3 - record time it takes to complete evaluation
#step 4 - output visualization, illustration error vs. range of parameters
def LassoR(Xtrain, Xval, ytrain, yval):
    alphas = [0.00001, 0.000025, 0.00005,0.000075,
              0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007,
              0.0008, 0.0009, 0.001, 0.0025, 0.005]
    avg_build_scores = []
    std_build_scores = []
    test_scores = []
    timer = []
    
    for alpha in alphas:
        t0 = time.time()
        lasso = make_pipeline(RobustScaler(), Lasso(alpha = alpha))
        build_score = rmse_cv(lasso, Xtrain, ytrain)
        lasso.fit(Xtrain, ytrain)
        test_score = rmse(lasso.predict(Xval), yval)
        avg_build_scores.append(build_score.mean())
        std_build_scores.append(build_score.std())
        test_scores.append(test_score)    
        timer.append(time.time() - t0)

    print('Time algo takes: {:.3f} seconds'.format(np.mean(timer)))
    seek_hyperpram_plot(avg_build_scores, std_build_scores, test_scores, alphas,
                        ytrain.mean(), yval.mean(), ['Lasso','alpha','Lasso seek alpha'])
    pass

def ENetR(Xtrain, Xval, ytrain, yval): #alpha = 0.0005, l1_ratio = 0.9
    alphas = [0.00001, 0.000025, 0.00005,0.000075,
              0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
              0.001, 0.0025, 0.005]
    l1_ratios = [0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69,
                 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79,
                 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89,
                 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    
    avg_build_scores = []
    std_build_scores = []
    test_scores = []
    timer = []

    for alpha in alphas:
        l1_ratio = 0.9
        t0 = time.time()
        ENet = make_pipeline(RobustScaler(), ElasticNet(alpha = alpha, l1_ratio = l1_ratio))
        build_score = rmse_cv(ENet, Xtrain, ytrain)
        ENet.fit(Xtrain, ytrain)
        test_score = rmse(ENet.predict(Xval), yval)
        avg_build_scores.append(build_score.mean())
        std_build_scores.append(build_score.std())
        test_scores.append(test_score)
        timer.append(time.time() - t0)
        
    print('Time algo takes: {:.3f} seconds'.format(np.mean(timer)))
    seek_hyperpram_plot(avg_build_scores, std_build_scores, test_scores, alphas,
                        ytrain.mean(), yval.mean(), ['ENet','alpha','ENet seek alpha'])

    alpha = alphas[avg_build_scores.index(np.min(avg_build_scores))]

    avg_build_scores = []
    std_build_scores = []
    test_scores = []
    timer = []

    for l1_ratio in l1_ratios:
        t0 = time.time()
        ENet = make_pipeline(RobustScaler(), ElasticNet(alpha = alpha, l1_ratio = l1_ratio))
        build_score = rmse_cv(ENet, Xtrain, ytrain)
        ENet.fit(Xtrain, ytrain)
        test_score = rmse(ENet.predict(Xval), yval)
        avg_build_scores.append(build_score.mean())
        std_build_scores.append(build_score.std())
        test_scores.append(test_score)
        timer.append(time.time() - t0)
        
    print('Time algo takes: {:.3f} seconds'.format(np.mean(timer)))
    seek_hyperpram_plot(avg_build_scores, std_build_scores, test_scores, l1_ratios,
                        ytrain.mean(), yval.mean(), ['ENet','l1_ratio','ENet seek l1_ratio'])
    pass
    
def DTR(Xtrain, Xval, ytrain, yval): 
    max_depths = range(1,10)
    
    avg_build_scores = []
    std_build_scores = []
    test_scores = []
    timer = []
    
    for max_depth in max_depths:
        t0 = time.time()
        decision = DecisionTreeRegressor(max_depth = max_depth)
        build_score = rmse_cv(decision, Xtrain, ytrain)
        decision.fit(Xtrain, ytrain)
        test_score = rmse(decision.predict(Xval), yval)
        avg_build_scores.append(build_score.mean())
        std_build_scores.append(build_score.std())
        test_scores.append(test_score)
        timer.append(time.time() - t0)

    print('Time algo takes: {:.3f} seconds'.format(np.mean(timer)))
    seek_hyperpram_plot(avg_build_scores, std_build_scores, test_scores, max_depths,
                        ytrain.mean(), yval.mean(), ['DecisionTree','max_depth','DecisionTree seek max_depth'])
    pass

def RFR(Xtrain, Xval, ytrain, yval): 
    max_depths = range(1,51)
    n_estimatorss = range(1,61)

    avg_build_scores = []
    std_build_scores = []
    test_scores = []
    timer = []
    
    for max_depth in max_depths:
        #max_depth = 30
        n_estimators = 35
        t0 = time.time()
        forest = RandomForestRegressor(max_depth = max_depth,
                                       n_estimators = n_estimators)
        build_score = rmse_cv(forest, Xtrain, ytrain)
        forest.fit(Xtrain, ytrain)
        test_score = rmse(forest.predict(Xval), yval)
        avg_build_scores.append(build_score.mean())
        std_build_scores.append(build_score.std())
        test_scores.append(test_score)
        timer.append(time.time() - t0)

    print('Time algo takes: {:.3f} seconds'.format(np.mean(timer)))
    seek_hyperpram_plot(avg_build_scores, std_build_scores, test_scores, max_depths,
                        ytrain.mean(), yval.mean(), ['RandomForest','max_depth','RandomForest seek max_depth'])

    max_depth = max_depths[avg_build_scores.index(np.min(avg_build_scores))]

    avg_build_scores = []
    std_build_scores = []
    test_scores = []
    timer = []
    
    for n_estimators in n_estimatorss:
        #n_estimators = 35
        t0 = time.time()
        forest = RandomForestRegressor(max_depth = max_depth, n_estimators = n_estimators)
        build_score = rmse_cv(forest, Xtrain, ytrain)
        forest.fit(Xtrain, ytrain)
        test_score = rmse(forest.predict(Xval), yval)
        avg_build_scores.append(build_score.mean())
        std_build_scores.append(build_score.std())
        test_scores.append(test_score)
        timer.append(time.time() - t0)

    print('Time algo takes: {:.3f} seconds'.format(np.mean(timer)))
    seek_hyperpram_plot(avg_build_scores, std_build_scores, test_scores, n_estimatorss,
                        ytrain.mean(), yval.mean(), ['RandomForest','n_estimators','RandomForest seek n_estimators'])
    pass

def ABR(Xtrain, Xval, ytrain, yval): #loss = 'exponential', learning_rate = 4, n_estimators = 60
    losses = ['linear', 'square', 'exponential']
    learning_rates = [0.01, 0.025, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                      1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 10]
    n_estimatorss = range(10,71)

    avg_build_scores = []
    std_build_scores = []
    test_scores = []
    timer = []
    
    for loss in losses:
        n_estimators = 60
        t0 = time.time()
        AdaBoost = AdaBoostRegressor(loss = loss)
        build_score = rmse_cv(AdaBoost, Xtrain, ytrain)
        AdaBoost.fit(Xtrain, ytrain)
        test_score = rmse(AdaBoost.predict(Xval), yval)
        avg_build_scores.append(build_score.mean())
        std_build_scores.append(build_score.std())
        test_scores.append(test_score)
        timer.append(time.time() - t0)

    print('Time algo takes: {:.3f} seconds'.format(np.mean(timer)))
    seek_hyperpram_plot(avg_build_scores, std_build_scores, test_scores, losses,
                        ytrain.mean(), yval.mean(), ['AdaBoost','loss','AdaBoost seek loss'])

    loss = losses[avg_build_scores.index(np.min(avg_build_scores))]
    
    avg_build_scores = []
    std_build_scores = []
    test_scores = []
    timer = []
    
    for learning_rate in learning_rates:
        n_estimators = 60
        t0 = time.time()
        AdaBoost = AdaBoostRegressor(loss = loss, learning_rate = learning_rate)
        build_score = rmse_cv(AdaBoost, Xtrain, ytrain)
        AdaBoost.fit(Xtrain, ytrain)
        test_score = rmse(AdaBoost.predict(Xval), yval)
        avg_build_scores.append(build_score.mean())
        std_build_scores.append(build_score.std())
        test_scores.append(test_score)
        timer.append(time.time() - t0)

    print('Time algo takes: {:.3f} seconds'.format(np.mean(timer)))
    seek_hyperpram_plot(avg_build_scores, std_build_scores, test_scores, learning_rates,
                        ytrain.mean(), yval.mean(), ['AdaBoost','learning_rate','AdaBoost seek learning_rate'])

    learning_rate = learning_rates[avg_build_scores.index(np.min(avg_build_scores))]
    
    avg_build_scores = []
    std_build_scores = []
    test_scores = []
    timer = []
    
    for n_estimators in n_estimatorss:
        t0 = time.time()
        AdaBoost = AdaBoostRegressor(loss = loss, learning_rate = learning_rate, n_estimators = n_estimators)
        build_score = rmse_cv(AdaBoost, Xtrain, ytrain)
        AdaBoost.fit(Xtrain, ytrain)
        test_score = rmse(AdaBoost.predict(Xval), yval)
        avg_build_scores.append(build_score.mean())
        std_build_scores.append(build_score.std())
        test_scores.append(test_score)
        timer.append(time.time() - t0)

    print('Time algo takes: {:.3f} seconds'.format(np.mean(timer)))
    seek_hyperpram_plot(avg_build_scores, std_build_scores, test_scores, n_estimatorss,
                        ytrain.mean(), yval.mean(), ['AdaBoost','n_estimators','AdaBoost seek n_estimators'])

    pass

def GBR(Xtrain, Xval, ytrain, yval): #max_depth = 5
    learning_rates = [0.01, 0.02, 0.03, 0.04, 0.05, 0.055, 0.06, 0.065,
                      0.07, 0.075, 0.08, 0.085, 0.09, 0.1, 0.2, 0.3]
    n_estimatorss = range(2000, 6000, 100)
    max_depths = range(1,5)
    
    avg_build_scores = []
    std_build_scores = []
    test_scores = []
    timer = []
    
    for max_depth in max_depths:
        t0 = time.time()
        loss = 'huber'
        GBoost = GradientBoostingRegressor(loss = loss, n_estimators = 3000,
                                           learning_rate = 0.05, max_depth = max_depth)
        build_score = rmse_cv(GBoost, Xtrain, ytrain)
        GBoost.fit(Xtrain, ytrain)
        test_score = rmse(GBoost.predict(Xval), yval)
        avg_build_scores.append(build_score.mean())
        std_build_scores.append(build_score.std())
        test_scores.append(test_score)
        timer.append(time.time() - t0)

    print('Time algo takes: {:.3f} seconds'.format(np.mean(timer)))
    seek_hyperpram_plot(avg_build_scores, std_build_scores, test_scores, max_depths,
                        ytrain.mean(), yval.mean(), ['GradientBoost','max_depth','GradientBoost seek max_depth'])

    max_depth = max_depths[avg_build_scores.index(np.min(avg_build_scores))]
    avg_build_scores = []
    std_build_scores = []
    test_scores = []
    timer = []
    
    for learning_rate in learning_rates:
        t0 = time.time()
        loss = 'huber'
        GBoost = GradientBoostingRegressor(loss = loss, n_estimators = 3000,
                                           learning_rate = learning_rate,
                                           max_depth = max_depth)
        build_score = rmse_cv(GBoost, Xtrain, ytrain)
        GBoost.fit(Xtrain, ytrain)
        test_score = rmse(GBoost.predict(Xval), yval)
        avg_build_scores.append(build_score.mean())
        std_build_scores.append(build_score.std())
        test_scores.append(test_score)
        timer.append(time.time() - t0)

    print('Time algo takes: {:.3f} seconds'.format(np.mean(timer)))
    seek_hyperpram_plot(avg_build_scores, std_build_scores, test_scores, learning_rates,
                        ytrain.mean(), yval.mean(), ['GradientBoost','learning_rate','GradientBoost seek learning_rate'])

    learning_rate = learning_rates[avg_build_scores.index(np.min(avg_build_scores))]
    avg_build_scores = []
    std_build_scores = []
    test_scores = []
    timer = []
    
    for n_estimators in n_estimatorss:
        t0 = time.time()
        loss = 'huber'
        GBoost = GradientBoostingRegressor(loss = loss, n_estimators = n_estimators,
                                           learning_rate = learning_rate,
                                           max_depth = max_depth)
        build_score = rmse_cv(GBoost, Xtrain, ytrain)
        GBoost.fit(Xtrain, ytrain)
        test_score = rmse(GBoost.predict(Xval), yval)
        avg_build_scores.append(build_score.mean())
        std_build_scores.append(build_score.std())
        test_scores.append(test_score)
        timer.append(time.time() - t0)

    print('Time algo takes: {:.3f} seconds'.format(np.mean(timer)))
    seek_hyperpram_plot(avg_build_scores, std_build_scores, test_scores, n_estimatorss,
                        ytrain.mean(), yval.mean(), ['GradientBoost','n_estimators','GradientBoost seek n_estimators'])
    pass

def KNNR(Xtrain, Xval, ytrain, yval): #algorithms = , n_neighbors = 
    algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
    n_neighborss = range(1,51)

    avg_build_scores = []
    std_build_scores = []
    test_scores = []
    timer = []
    
    for algorithm in algorithms:
        n_neighbors = 13
        t0 = time.time()
        KNR = KNeighborsRegressor(algorithm = algorithm, n_neighbors = n_neighbors)
        build_score = rmse_cv(KNR, Xtrain, ytrain)
        KNR.fit(Xtrain, ytrain)
        test_score = rmse(KNR.predict(Xval), yval)
        avg_build_scores.append(build_score.mean())
        std_build_scores.append(build_score.std())
        test_scores.append(test_score)
        timer.append(time.time() - t0)
    #print(avg_build_scores)

    print('Time algo takes: {:.3f} seconds'.format(np.mean(timer)))
    seek_hyperpram_plot(avg_build_scores, std_build_scores, test_scores, algorithms,
                        ytrain.mean(), yval.mean(), ['KNeighbors','algorithm','KNeighbors seek algorithm'])

    algorithm = algorithms[avg_build_scores.index(np.min(avg_build_scores))]

    avg_build_scores = []
    std_build_scores = []
    test_scores = []
    timer = []
    
    for n_neighbors in n_neighborss:
        t0 = time.time()
        KNR = KNeighborsRegressor(algorithm = algorithm, n_neighbors = n_neighbors)
        build_score = rmse_cv(KNR, Xtrain, ytrain)
        KNR.fit(Xtrain, ytrain)
        test_score = rmse(KNR.predict(Xval), yval)
        avg_build_scores.append(build_score.mean())
        std_build_scores.append(build_score.std())
        test_scores.append(test_score)
        timer.append(time.time() - t0)
    #print(avg_build_scores)

    print('Time algo takes: {:.3f} seconds'.format(np.mean(timer)))
    seek_hyperpram_plot(avg_build_scores, std_build_scores, test_scores, n_neighborss,
                        ytrain.mean(), yval.mean(), ['KNeighbors','n_neighbors','KNeighbors seek n_neighbors'])

    pass

def MLPR(Xtrain, Xval, ytrain, yval): #algorithms = , n_neighbors =
    
    hidden_layer_sizess = [(200,), (100,), (75,), (50,), (25,), (10,), (5,), 
                           (100, 100), (100, 75), (100, 50), (100, 25), (100, 10), (100, 5),
                           (100, 100, 75), (100, 100, 50), (100, 100, 25), (100, 100, 10), (100, 100, 5),
                           (100, 75, 75), (100, 75, 50), (100, 75, 25), (100, 75, 10), (100, 75, 5)]
    max_iters = range(100, 2000, 50)
    
    avg_build_scores = []
    std_build_scores = []
    test_scores = []
    timer = []
    
    for hidden_layer_sizes in hidden_layer_sizess:
        t0 = time.time()
        MLP = MLPRegressor(activation  = 'relu', hidden_layer_sizes = hidden_layer_sizes,
                           solver = 'lbfgs', learning_rate = 'adaptive')
        build_score = rmse_cv(MLP.predict(Xtrain), ytrain)
        MLP.fit(Xtrain, ytrain)
        test_score = rmse(MLP, Xval, yval)
        avg_build_scores.append(build_score.mean())
        std_build_scores.append(build_score.std())
        test_scores.append(test_score)
        timer.append(time.time() - t0)

    print('Time algo takes: {:.3f} seconds'.format(np.mean(timer)))
    seek_hyperpram_plot(avg_build_scores, std_build_scores, test_scores, hidden_layer_sizess,
                        ytrain.mean(), yval.mean(), ['MLPRegressor','hidden_layer_sizes','MLPRegressor seek hidden_layer_sizes'])

    hidden_layer_sizes = hidden_layer_sizess[avg_build_scores.index(np.min(avg_build_scores))]

    avg_build_scores = []
    std_build_scores = []
    test_scores = []
    timer = []
    
    for max_iter in max_iters:
        t0 = time.time()
        MLP = MLPRegressor(activation  = 'relu', hidden_layer_sizes = hidden_layer_sizes,
                           solver = 'lbfgs', learning_rate = 'adaptive',
                           max_iter = max_iter)
        build_score = rmse_cv(MLP, Xtrain, ytrain)
        MLP.fit(Xtrain, ytrain)
        test_score = rmse(MLP.predict(Xval), yval)
        avg_build_scores.append(build_score.mean())
        std_build_scores.append(build_score.std())
        test_scores.append(test_score)
        timer.append(time.time() - t0)

    print('Time algo takes: {:.3f} seconds'.format(np.mean(timer)))
    seek_hyperpram_plot(avg_build_scores, std_build_scores, test_scores, max_iters,
                        ytrain.mean(), yval.mean(), ['MLPRegressor','max_iter','MLPRegressor seek max_iter'])

    pass

def KerasR(Xtrain, Xval, ytrain, yval): #algorithms = , n_neighbors =

    def keras_model1a():
        model = Sequential()
        model.add(Dense(2000, input_dim = Xtrain.shape[1], activation = 'relu'))
        model.add(Dense(1))
        model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['mse'])
        return model

    def keras_model1b():
        model = Sequential()
        model.add(Dense(1500, input_dim = Xtrain.shape[1], activation = 'relu'))
        model.add(Dense(1))
        model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['mse'])
        return model

    def keras_model1c():
        model = Sequential()
        model.add(Dense(1000, input_dim = Xtrain.shape[1], activation = 'relu'))
        model.add(Dense(1))
        model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['mse'])
        return model

    def keras_model2a():
        model = Sequential()
        model.add(Dense(2000, input_dim = Xtrain.shape[1], activation = 'relu'))
        model.add(Dense(2000, activation = 'relu'))
        model.add(Dense(1))
        model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['mse'])
        return model
    
    def keras_model2b():
        model = Sequential()
        model.add(Dense(2000, input_dim = Xtrain.shape[1], activation = 'relu'))
        model.add(Dense(1500, activation = 'relu'))
        model.add(Dense(1))
        model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['mse'])
        return model
    
    def keras_model2c():
        model = Sequential()
        model.add(Dense(2000, input_dim = Xtrain.shape[1], activation = 'relu'))
        model.add(Dense(1000, activation = 'relu'))
        model.add(Dense(1))
        model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['mse'])
        return model

    def keras_model2d():
        model = Sequential()
        model.add(Dense(2000, input_dim = Xtrain.shape[1], activation = 'relu'))
        model.add(Dense(500, activation = 'relu'))
        model.add(Dense(1))
        model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['mse'])
        return model

    def keras_model2e():
        model = Sequential()
        model.add(Dense(2000, input_dim = Xtrain.shape[1], activation = 'relu'))
        model.add(Dense(250, activation = 'relu'))
        model.add(Dense(1))
        model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['mse'])
        return model

    def keras_model3a():
        model = Sequential()
        model.add(Dense(2000, input_dim = Xtrain.shape[1], activation = 'relu'))
        model.add(Dense(1500, activation = 'relu'))
        model.add(Dense(1000, activation = 'relu'))
        model.add(Dense(1))
        model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['mse'])
        return model

    def keras_model3b():
        model = Sequential()
        model.add(Dense(2000, input_dim = Xtrain.shape[1], activation = 'relu'))
        model.add(Dense(1500, activation = 'relu'))
        model.add(Dense(500, activation = 'relu'))
        model.add(Dense(1))
        model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['mse'])
        return model

    def keras_model4a():
        model = Sequential()
        model.add(Dense(2000, input_dim = Xtrain.shape[1], activation = 'relu'))
        model.add(Dense(1500, activation = 'relu'))
        model.add(Dense(1000, activation = 'relu'))
        model.add(Dense(500, activation = 'relu'))
        model.add(Dense(1))
        model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['mse'])
        return model

    def keras_model4b():
        model = Sequential()
        model.add(Dense(2000, input_dim = Xtrain.shape[1], activation = 'relu'))
        model.add(Dense(1500, activation = 'relu'))
        model.add(Dense(1000, activation = 'relu'))
        model.add(Dense(250, activation = 'relu'))
        model.add(Dense(1))
        model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['mse'])
        return model

    layouts = [(2000,), (1500,), (1000,),
               (2000, 2000), (2000, 1500), (2000, 1000), (2000, 500), (2000, 250),
               (2000, 1500, 1000), (2000, 1500, 500),
               (2000, 1500, 1000, 500), (2000, 1500, 1000, 250)]

    
    models = [keras_model1a, keras_model1b, keras_model1c,
              keras_model2a, keras_model2b, keras_model2c, keras_model2d, keras_model2e,
              keras_model3a, keras_model3b,
              keras_model4a, keras_model4b]
    
    avg_build_scores = []
    std_build_scores = []
    test_scores = []
    timer = []
    
    for model in models:
        t0 = time.time()
        estimator  = KerasRegressor(build_fn = model, verbose = False)
        build_score = rmse_cv(estimator, Xtrain, ytrain)
        estimator.fit(Xtrain.as_matrix(), ytrain.as_matrix())
        test_score = rmse(estimator.predict(Xval.as_matrix()), yval)
        avg_build_scores.append(build_score.mean())
        std_build_scores.append(build_score.std())
        test_scores.append(test_score)
        timer.append(time.time() - t0)

    print('Time algo takes: {:.3f} seconds'.format(np.mean(timer)))

    seek_hyperpram_plot(avg_build_scores, std_build_scores, test_scores, layouts,
                        ytrain.mean(), yval.mean(), ['KerasRegressor','layouts','KerasRegressor seek layouts'])

    model = models[avg_build_scores.index(np.min(avg_build_scores))]
    batch_sizes = range(5, 100, 5)
    
    avg_build_scores = []
    std_build_scores = []
    test_scores = []
    timer = []
    
    for batch_size in batch_sizes:
        t0 = time.time()
        estimator  = KerasRegressor(build_fn = model, verbose = False, batch_size = batch_size)
        build_score = rmse_cv(estimator, Xtrain, ytrain)
        estimator.fit(Xtrain.as_matrix(), ytrain.as_matrix())
        test_score = rmse(estimator.predict(Xval.as_matrix()), yval)
        avg_build_scores.append(build_score.mean())
        std_build_scores.append(build_score.std())
        test_scores.append(test_score)
        timer.append(time.time() - t0)

    print('Time algo takes: {:.3f} seconds'.format(np.mean(timer)))

    seek_hyperpram_plot(avg_build_scores, std_build_scores, test_scores, batch_sizes,
                        ytrain.mean(), yval.mean(), ['KerasRegressor','batch_size','KerasRegressor seek batch_size'])

    batch_size = batch_sizes[avg_build_scores.index(np.min(avg_build_scores))]
    
    nb_epochs = range(200, 3000, 100)

    avg_build_scores = []
    std_build_scores = []
    test_scores = []
    timer = []
    
    for nb_epoch in nb_epochs:
        t0 = time.time()
        estimator  = KerasRegressor(build_fn = model, verbose = False,
                                    batch_size = batch_size, nb_epoch = nb_epoch)
        build_score = rmse_cv(estimator, Xtrain, ytrain)
        estimator.fit(Xtrain.as_matrix(), ytrain.as_matrix())
        test_score = rmse(estimator.predict(Xval.as_matrix()), yval)
        avg_build_scores.append(build_score.mean())
        std_build_scores.append(build_score.std())
        test_scores.append(test_score)
        timer.append(time.time() - t0)

    print('Time algo takes: {:.3f} seconds'.format(np.mean(timer)))

    seek_hyperpram_plot(avg_build_scores, std_build_scores, test_scores, nb_epochs,
                        ytrain.mean(), yval.mean(), ['KerasRegressor','nb_epoch','KerasRegressor seek nb_epoch'])
    
    pass

def XGBR(Xtrain, Xval, ytrain, yval):
    alphas = [0.00001, 0.000025, 0.00005,0.000075,
              0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007,
              0.0008, 0.0009, 0.001, 0.0025, 0.005]
    
    n_estimatorss = range(1500, 5000, 100)
    learning_rates = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065,
                      0.07, 0.075, 0.08, 0.085, 0.09, 0.1]
    max_depths = range(1, 10)
    
    avg_build_scores = []
    std_build_scores = []
    test_scores = []
    timer = []

    for n_estimators in n_estimatorss:
        t0 = time.time()
        model_xgb = xgb.XGBRegressor(colsample_bytree = 0.4603, gamma = 0.0468,
                                     min_child_weight = 1.7817, n_estimators = n_estimators,
                                     reg_alpha = 0.4640, reg_lambda = 0.8571,
                                     subsample = 0.5213, silent = 1, nthread = -1)
        
        build_score = rmse_cv(model_xgb, Xtrain, ytrain)
        model_xgb.fit(Xtrain, ytrain)
        test_score = rmse(model_xgb.predict(Xval), yval)
        avg_build_scores.append(build_score.mean())
        std_build_scores.append(build_score.std())
        test_scores.append(test_score)    
        timer.append(time.time() - t0)

    print('Time algo takes: {:.3f} seconds'.format(np.mean(timer)))
    seek_hyperpram_plot(avg_build_scores, std_build_scores, test_scores, n_estimatorss,
                        ytrain.mean(), yval.mean(), ['XGBoost','n_estimators','XGBoost seek n_estimators'])

    n_estimators = n_estimatorss[avg_build_scores.index(np.min(avg_build_scores))]
    
    avg_build_scores = []
    std_build_scores = []
    test_scores = []
    timer = []
    
    for learning_rate in learning_rates:
        t0 = time.time()
        model_xgb = xgb.XGBRegressor(colsample_bytree = 0.4603, gamma = 0.0468,
                                     learning_rate = learning_rate, n_estimators = n_estimators,
                                     min_child_weight = 1.7817, 
                                     reg_alpha = 0.4640, reg_lambda = 0.8571,
                                     subsample = 0.5213, silent = 1, nthread = -1)
        
        build_score = rmse_cv(model_xgb, Xtrain, ytrain)
        model_xgb.fit(Xtrain, ytrain)
        test_score = rmse(model_xgb.predict(Xval), yval)
        avg_build_scores.append(build_score.mean())
        std_build_scores.append(build_score.std())
        test_scores.append(test_score)    
        timer.append(time.time() - t0)

    print('Time algo takes: {:.3f} seconds'.format(np.mean(timer)))
    seek_hyperpram_plot(avg_build_scores, std_build_scores, test_scores, learning_rates,
                        ytrain.mean(), yval.mean(), ['XGBoost','learning_rate','XGBoost seek learning_rate'])

    learning_rate = learning_rates[avg_build_scores.index(np.min(avg_build_scores))]

    avg_build_scores = []
    std_build_scores = []
    test_scores = []
    timer = []
    
    for max_depth in max_depths:
        t0 = time.time()
        model_xgb = xgb.XGBRegressor(colsample_bytree = 0.4603, gamma = 0.0468,
                                     learning_rate = learning_rate, max_depth = max_depth,
                                     n_estimators = n_estimators, min_child_weight = 1.7817, 
                                     reg_alpha = 0.4640, reg_lambda = 0.8571,
                                     subsample = 0.5213, silent = 1, nthread = -1)
        
        build_score = rmse_cv(model_xgb, Xtrain, ytrain)
        model_xgb.fit(Xtrain, ytrain)
        test_score = rmse(model_xgb.predict(Xval), yval)
        avg_build_scores.append(build_score.mean())
        std_build_scores.append(build_score.std())
        test_scores.append(test_score)    
        timer.append(time.time() - t0)

    print('Time algo takes: {:.3f} seconds'.format(np.mean(timer)))
    seek_hyperpram_plot(avg_build_scores, std_build_scores, test_scores, max_depths,
                        ytrain.mean(), yval.mean(), ['XGBoost','max_depth','XGBoost seek max_depth'])

    max_depth = max_depths[avg_build_scores.index(np.min(avg_build_scores))]

    pass


if __name__ == '__main__':
    #gather data
    Xtrain, Xtest, ytrain, XtrainPCA, XtestPCA = data_wrangle()
    Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain, test_size = 0.2)

    #Here I try each model to get visualization
    
    #DTR(Xtrain, Xval, ytrain, yval)
    #RFR(Xtrain, Xval, ytrain, yval)
    #ABR(Xtrain, Xval, ytrain, yval)
    #GBR(Xtrain, Xval, ytrain, yval)
    #KNNR(Xtrain, Xval, ytrain, yval)
    #MLPR(Xtrain, Xval, ytrain, yval)
    #KerasR(Xtrain, Xval, ytrain, yval)
    #XGBR(Xtrain, Xval, ytrain, yval)
    #LassoR(Xtrain, Xval, ytrain, yval)
    #ENetR(Xtrain, Xval, ytrain, yval)
    
