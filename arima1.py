# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import seaborn as sns

# read_file = pd.read_excel("oakridge_n20.xlsx")
# read_file.to_csv("Test.csv")

from pandas import read_csv
# from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.ar_model import AutoReg
df = pd.DataFrame(pd.read_csv("Test.csv",header=0,encoding='utf-8'))
# print(df.head())
# timeserie=df.copy()
df['Yearc']=pd.to_datetime(df[['year', 'month', 'day']])
# timeserie.index.name = 'index'
# df['week_day']=df['Yearc'].dt.dayofweek
# print(df['week_day'])
# print(timeserie['Month'])
# df2 = pd.DataFrame().assign(Year=df['Yearc'], dsrfclitwwest=df['dsrfclitwwest'])
df2 = pd.DataFrame().assign(Year=df['Yearc'],NIT=df['N2OFlux'])
# print(df2.head())

df2.index = df2['Year']
del df2['Year']

X = df2.values

size = int(len(X) * 0.80)
train, test = X[0:size], X[size:len(df2)]
history = [x for x in train]
predictions = list()
# walk-forward validation
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	# model = AutoReg(train, lags=20)
	model_fit = model.fit()
	# print('Coefficients: %s' % model_fit.params)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	# predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# mse1 = metrics.mean_squared_error(y_train, predictions1)
r2 = r2_score(test, predictions)
mape = mean_absolute_percentage_error(test, predictions)
# r2_train = metrics.r2_score(y_train, predictions1)
print("MAPE",mape)
# print("MAE Train",mae1)
# print("R2 Train",r2_train)
print("R2",r2)
#
# plot forecasts against actual outcomes
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
#