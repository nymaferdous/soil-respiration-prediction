
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from matplotlib import pyplot
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
import seaborn as sns
import numpy as np
from sys import exit
from matplotlib import rcParams
from os import makedirs

# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Dense, Dropout
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from numpy import dstack
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
from numpy.random import randn
from matplotlib import pyplot
import os
import matplotlib.pyplot as plt


# data = pd.read_csv("Data_for_ML_crop_residues_assessment.csv")
from ctgan import CTGAN
from ctgan import load_demo
# from ctgan import CTGANSynthesizer
#
# def get_dataset():
#     # X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=1)
#     X1 = pd.get_dummies(data, columns=["Code", "muname", "areasymbol", "areaname", "Crop_Rotations"])
#     X = X1[['Sand(%)','Crop_Rotations_CSW', 'Crop_Rotations_CS', 'Corn Yield(bu/Ha)',
#             'Soybean Yield(bu/Ha)', 'Slope(%)', 'SlopeLength(m)','T_factor', 'K_factor', 'Silt(%)', 'Clay(%)', 'Organic matter(%)', 'Rainfall_Erosivity', 'SCI','Organic Matter factor']]
#     # y = data['Soil erosion factor']
#     # y = data['Organic Matter factor']
#     return X
# y = data['Organic Matter factor']
# X= get_dataset()
# print(X.shape)
# from sklearn.model_selection import train_test_split
# import xgboost
# from sklearn.preprocessing import StandardScaler
#
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# X = scaler.transform(X)
# discrete_columns = [
# 'Sand(%)', 'Soybean Yield(bu/Ha)', 'Slope(%)', 'SlopeLength(m)','T_factor', 'K_factor', 'Silt(%)', 'Clay(%)', 'Organic matter(%)', 'Rainfall_Erosivity', 'SCI','Organic Matter factor'
# ]
from sdv.tabular import CTGAN
from sdv.tabular import CopulaGAN
from ctgan import load_demo

real_data = load_demo()
print(real_data.head())

# Names of the columns that are discrete
discrete_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'income'
]

ctgan = CTGAN(epochs=10)
model = CopulaGAN()
# ctgan.fit(real_data)
model.fit(real_data)

# Create synthetic data
# synthetic_data = ctgan.sample(1000)
new_data = model.sample(num_rows=200)
# model.save('my_model.pkl')
# loaded = GaussianCopula.load('my_model.pkl')
