Introduction
This is the final project for Udacity's MLND. Analysis done in python IDLE 3.5.3. I attempted to predict SalePrice for the Ames housing dataset. 

Details
Please refer to the capstone proposal.pdf and capstone report.pdf for the complete project and report details.

References:
Utility Libraries:
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import visuals as vs #from one of the MLND project. Was useful in visualizing PCA
from time import gmtime, strftime, time
import warnings

Statistical Libraries:
from scipy import stats
from scipy.special import boxcox1p
from scipy.stats import norm, skew 

Modeling Libraries:
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
import xgboost as xgb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split, ShuffleSplit, GridSearchCV
from sklearn.metrics import mean_squared_error

Data Gathering Library:
from Feature_Engineering import data_wrangle

Please refer to:
feature engineering: /Code/Feature_Engineering.py
main model optimization & selection:  /Code/Model Optimization GridSearchCV.py
visualizations: /Visualization
project proposal: capstone proposal.pdf
project report: capstone project report.pdf
data dictionary: Features.xlsx
model summary: Model Selection.xlsx# MLND
