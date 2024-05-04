# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import pandas as pd
# import statsmodels.api as sm
import matplotlib.pyplot as plt
# from statsmodels.tsa.stattools import adfuller
import seaborn as sns
from darts import TimeSeries

read_file = pd.read_excel("oakridge_n20.xlsx")

# Write the dataframe object
# into csv file
read_file.to_csv("Test.csv",index=None,header=True)

# read csv file and convert
# into a dataframe object
df = pd.DataFrame(pd.read_csv("Test.csv"))
print(df.head())
df['Yearc']=pd.to_datetime(df[['year', 'month', 'day']])
df2 = pd.DataFrame().assign(Year=df['Yearc'], N2OFlux=df['N2OFlux'])
df2['Year'] = [d.year for d in df2.Year]
fig, axes = plt.subplots(dpi= 80)
sns.boxplot(x='Year', y='N2OFlux', data=df2, ax=axes)
axes.set_title('Year-wise Box Plot\n(The Trend)', fontsize=18)
plt.show()
series = TimeSeries.from_dataframe(df, "Yearc", "N2OFlux")
# ndf=pd.concat([df['N2OFlux'],df1], axis=1, ignore_index=True)
# ndf.columns=["N2OFlux","Month"]
# print(ndf.head())
from matplotlib import pyplot
df2.index = df2['Year']
del df2['Year']

print(df2.head())

# df2.plot()
# pyplot.show()

import matplotlib.pyplot as plt
import seaborn as sns

# sns.lineplot(df2)
# plt.ylabel('N2OFlux')
# plt.show()

# rolling_mean = df2.rolling(7).mean()
# rolling_std = df2.rolling(7).std()
# plt.plot(df2, color="blue",label="Original N2OFlux")
# plt.plot(rolling_mean, color="red", label="Rolling Mean N2OFlux")
# plt.plot(rolling_std, color="black", label = "Rolling Standard Deviation N2OFlux")
# plt.title("Passenger Time Series, Rolling Mean, Standard Deviation")
# plt.legend(loc="best")
# plt.show()


# from pandas.plotting import autocorrelation_plot
# autocorrelation_plot(df2)
# pyplot.show()
# result = adfuller(df2, autolag="AIC")
# output_df = pd.DataFrame({"Values":[result[0],result[1],result[2],result[3], result[4]['1%'], result[4]['5%'], result[4]['10%']]  , "Metric":["Test Statistics","p-value","No. of lags used","Number of observations used",
#                                                         "critical value (1%)", "critical value (5%)", "critical value (10%)"]})
# print(output_df)
#
# print('ADF Statistic: {}'.format(result[0]))
#
# autocorrelation_lag1 = df2['N2OFlux'].autocorr(lag=1)
# print("One Month Lag: ", autocorrelation_lag1)
#
# autocorrelation_lag3 = df2['N2OFlux'].autocorr(lag=3)
# print("Three Month Lag: ", autocorrelation_lag3)
#
# autocorrelation_lag6 = df2['N2OFlux'].autocorr(lag=6)
# print("Six Month Lag: ", autocorrelation_lag6)
#
# autocorrelation_lag9 = df2['N2OFlux'].autocorr(lag=9)
# print("Nine Month Lag: ", autocorrelation_lag9)
#
# # from statsmodels.tsa.seasonal import seasonal_decompose
# # decompose = seasonal_decompose(df2['N2OFlux'],model='additive', period=7)
# # decompose.plot()
# # plt.show()
#
# from statsmodels.tsa.arima.model import ARIMA
from pandas import DataFrame

# model = ARIMA(df2, order=(5,1,0))
# model_fit = model.fit()
# # summary of fit model
# print(model_fit.summary())
# # line plot of residuals
# residuals = DataFrame(model_fit.resid)
# residuals.plot()
# pyplot.show()
# density plot of residuals
# residuals.plot(kind='kde')
# pyplot.show()
# # summary stats of residuals
# print(residuals.describe())

from pandas import read_csv
# from pandas import datetime
from matplotlib import pyplot
# from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_squared_error
from math import sqrt
# from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from darts.metrics import mape,rmse
# load dataset
# def parser(x):
# 	return datetime.strptime('190'+x, '%Y-%m')
# series = read_csv('shampoo-sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)

# df2.index = df2.index.to_period('M')
# df2.set_index('Any', inplace=True)
df2.index.name = 'year'
# split into train and test sets
X = series
# df2['N2OFlux_log'] = np.sqrt(df2['N2OFlux'])
# df2['N2OFlux_log_diff'] = df2['N2OFlux_log'] - df2['N2OFlux_log'].shift(1)
# df2['N2OFlux_log_diff'].dropna().plot()
# print(df2.head())
# from pandas.plotting import lag_plot
# lag_plot(df2)
# pyplot.show()

# X = np.sqrt(df2['N2OFlux_log_diff'].values)
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(series)]
# print(len(train))
# print(len(test))
history = [x for x in train]
predictions = list()
from darts.models import NBEATSModel

# model = ExponentialSmoothing()
model_nbeats = NBEATSModel(
    input_chunk_length=50,
    output_chunk_length=365,
    generic_architecture=False,
    num_blocks=3,
    num_layers=4,
    layer_widths=512,
    n_epochs=500,
    nr_epochs_val_period=1,
    batch_size=800,
    model_name="nbeats_interpretable_run",
)
# pred_series = model_nbeats.historical_forecasts(
#     series,
#     start=pd.Timestamp("20170901"),
#     forecast_horizon=7,
#     stride=5,
#     retrain=False,
#     verbose=True,
# )
# model.fit(train)
model_nbeats.fit(train,val_series=test, verbose=True)
# pred_series = model_nbeats.historical_forecasts(
#     series,
#     start=pd.Timestamp("20170901"),
#     forecast_horizon=7,
#     stride=5,
#     retrain=False,
#     verbose=True,
# )
# display_forecast(pred_series, series, "7 day", start_date=pd.Timestamp("20170901"))
# prediction = model.predict(len(test), num_samples=1000)
prediction = model_nbeats.predict(n=360, series = train)
# pred_error = mape(test[:360],prediction)
#error_follower = mape(val_follower[:36],pred_follower)
# print("this is the error for the closing price:",pred_error )
import matplotlib.pyplot as plt
series.plot()
prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
plt.legend()
plt.show()
print(test)
# walk-forward validation
# for t in range(len(test)):
# 	# model = ARIMA(history, order=(5,1,0))
# 	# model = AutoReg(train, lags=20)
# 	model_fit = model.fit()
# 	# print('Coefficients: %s' % model_fit.params)
# 	output = model_fit.forecast()
# 	yhat = output[0]
# 	predictions.append(yhat)
# 	# predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
# 	obs = test[t]
# 	history.append(obs)
# 	print('predicted=%f, expected=%f' % (yhat, obs))
# # evaluate forecasts
print('The MAPE is: {:.2f}, with theta = {}.'.format(rmse(series, prediction)))
# rmse = sqrt(mean_squared_error(test.univariate_component(0), prediction))
# print('Test RMSE: %.3f' % rmse)
# # plot forecasts against actual outcomes
# pyplot.plot(test)
# pyplot.plot(predictions, color='red')
# pyplot.show()
# #
# df2['Yearc'] = df.index
# train = df[df['Date'] < pd.to_datetime("1960-08", format='%Y-%m')]
# train['train'] = train['#Passengers']
# del train['Date']
# del train['#Passengers']
# test = df[df['Date'] >= pd.to_datetime("1960-08", format='%Y-%m')]
# del test['Date']
# test['test'] = test['#Passengers']
# del test['#Passengers']
# plt.plot(train, color = "black")
# plt.plot(test, color = "red")
# plt.title("Train/Test split for Passenger Data")
# plt.ylabel("Passenger Number")
# plt.xlabel('Year-Month')
# sns.set()
# plt.show()
# print('p-value: {}'.format(result[1]))
# print('Critical Values:')
# for key, value in result[4].items():
#     print('\t{}: {}'.format(key, value))

# df_log = np.log(ndf)
# plt.plot(df_log)
# def get_stationarity(timeseries):
#     # rolling statistics
#     rolling_mean = timeseries.rolling(window=12).mean()
#     rolling_std = timeseries.rolling(window=12).std()
#
#     # rolling statistics plot
#     original = plt.plot(timeseries, color='blue', label='Original')
#     mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
#     std = plt.plot(rolling_std, color='black', label='Rolling Std')
#     plt.legend(loc='best')
#     plt.title('Rolling Mean & Standard Deviation')
#     plt.show(block=False)
#
#     # Dickey–Fuller test:
#     result = adfuller(timeseries['N2OFlux'])
#     print('ADF Statistic: {}'.format(result[0]))
#     print('p-value: {}'.format(result[1]))
#     print('Critical Values:')
#     for key, value in result[4].items():
#         print('\t{}: {}'.format(key, value))


# rolling_mean_exp_decay = df_log.ewm(halflife=12, min_periods=0, adjust=True).mean()
# df_log_exp_decay = df_log - rolling_mean_exp_decay
# df_log_exp_decay.dropna(inplace=True)
# get_stationarity(df_log_exp_decay)