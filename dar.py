# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import pandas as pd
# import statsmodels.api as sm
import matplotlib.pyplot as plt
# from statsmodels.tsa.stattools import adfuller
import seaborn as sns
import sklearn.ensemble
from darts import TimeSeries
import torch

# read_file = pd.read_excel("oakridgeu_n20.xlsx")
# read_file1 = pd.read_excel("oakridge_NIT.xlsx")
# read_file2 = pd.read_excel("oakridge_n20.xlsx")
# read_file3 = pd.read_excel("oakridge_co.xlsx")
# read_file = pd.read_excel("oakridgeufour.xlsx")
# read_file5 = pd.read_excel("Saha1.xlsx")
# read_file6 = pd.read_excel("oakridge_co.xlsx")
# Write the dataframe object
# into csv file
# read_file.to_csv("oakridgeufour.csv",index=None,header=True)
# read_file1.to_csv("NIT.csv",index=None,header=True)
# read_file2.to_csv("n20.csv",index=None,header=True)
# read_file3.to_csv("oakridgeu_CO.csv",index=None,header=True)
# read_file4.to_csv("OKfour.csv",index=None,header=True)
# read_file5.to_csv("Saha1.csv",index=None,header=True)
# read_file6.to_csv("OKco.csv",index=None,header=True)

# read csv file and convert
# into a dataframe object
# df = pd.DataFrame(pd.read_csv("oakridge_highwest_soilp.csv"))
df=pd.DataFrame(pd.read_csv("dc_sipHN.csv"))

# f, ax = plt.subplots(figsize=(10, 8))
# corr = df.corr()
# sns.heatmap(corr, annot=True,  fmt= '.2f')
# plt.show()

# # df['yearc']=pd.to_datetime(df[['year', 'month', 'day']])
# df.index=df['yearc']
df.index=df['time']

# select all features
# features_all =df[['yearc','stemp','ATemp','N2OFlux','NOFlux','NIT','CH4','Precipitation','sresp','msresp','slitresp']]
# features_all =df[['yearc','stemp','ATemp','pdsrfclit','Precipitation ','pdsmnrl']]
features_all =df[['time','stemp','snow','melt','drain','sublim']]

# plot a correlation of all features
# correlation matrix
sns.set(font_scale=2)
f,ax=plt.subplots(figsize=(15,10))
sns.heatmap(features_all.corr(), annot=True, cmap='coolwarm', fmt = ".2f", center=0, vmin=-1, vmax=1)
plt.title('Correlation between features', fontsize=25, weight='bold' )
plt.show()

sns.set(font_scale=1)

from statsmodels.tsa.seasonal import STL

# stl = STL(df.N2OFlux, seasonal=13)
# ##stl = STL(dta.co2, period=12, seasonal_deg=0, trend_deg=1, low_pass_deg=1, robust=True)
# res = stl.fit()
# fig = res.plot()
# plt.show()

# trend = res.trend
# seasonal = res.seasonal
# residual = res.resid
# df = pd.concat([res.trend, res.seasonal, res.resid,df.yearc,df.stemp,df.ATemp, df.Precipitation], 1)
# print(df)
# df=df.loc[~df.index.duplicated(), :]

# df =df[df['N2OFlux'] !=0]
# df.index=df['year']
# print(df.head())
# df.index = pd.to_datetime(df.index)
# df1 = pd.DataFrame(pd.read_csv("NIT.csv"))
# df1['Year'] = pd.to_datetime(df1[['year', 'month', 'day']])
# #
# df2 = pd.DataFrame(pd.read_csv("n20.csv"))
# df2['Year'] = pd.to_datetime(df2[['year', 'month', 'day']])
# #
# df3 = pd.DataFrame(pd.read_csv("noflux.csv"))
# df3['Year'] = pd.to_datetime(df3[['year', 'month', 'day']])

# df= pd.DataFrame(pd.read_csv("Saha1.csv"))

# df2 = pd.DataFrame().assign(Year=df['Yearc'], N2OFlux=df['N2OFlux'])
# df2['Year'] = [d.year for d in df2.Year]
# fig, axes = plt.subplots(dpi= 80)
# sns.boxplot(x='Year', y='N2OFlux', data=df2, ax=axes)
# axes.set_title('Year-wise Box Plot\n(The Trend)', fontsize=18)
# plt.show()
from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
from darts.dataprocessing.transformers import Scaler
# df['lCH4']= df['CH4'].apply(lambda x : np.log(x))
# df['ldiffCH4']= df['lCH4'] - df['lCH4'].shift()


# df['dsrfclitwwest'] = scaler.fit_transform(df['dsrfclitwwest'].to_numpy().reshape(-1, 1))
# series = TimeSeries.from_dataframe(df, "yearc", "N2OFlux",fill_missing_dates=True, freq=None)
# series_CH4 = TimeSeries.from_dataframe(df, "yearc", "CH4")
series_CO = TimeSeries.from_dataframe(df, "yearc", "dsoilresp")
# series_n20 = TimeSeries.from_dataframe(df,"yearc","NOFlux")
# series_n20 = TimeSeries.from_dataframe(df,"yearc","trend")
# series_n20s = TimeSeries.from_dataframe(df,"yearc","season")q
# series_n20r = TimeSeries.from_dataframe(df,"yearc","resid")
# series_noflux = TimeSeries.from_dataframe(df, "yearc", "NOFlux")
# series_deloi = TimeSeries.from_dataframe(df, "yearc", "deloi")
series_stemp = TimeSeries.from_dataframe(df, "yearc", "stemp")
series_atemp = TimeSeries.from_dataframe(df, "yearc", "ATemp")
series_sresp = TimeSeries.from_dataframe(df, "yearc", "pdsrfclit")
# series_msresp = TimeSeries.from_dataframe(df, "yearc", "Precipitation")
series_slitresp = TimeSeries.from_dataframe(df, "yearc", "pdsmnrl")
# series_noflux = TimeSeries.from_dataframe(df, "yearc", "NOFlux")
series_prec= TimeSeries.from_dataframe(df, "yearc", "Precipitation ")
# series_maxt=TimeSeries.from_dataframe(df, "yearc", "maxt")
# series_mint=TimeSeries.from_dataframe(df, "yearc", "mint")
# print(df.index.duplicated())
#
#_df = _df[c_quantity]
# df.set_index(df.year, inplace=True)
# df = df.resample('D').sum().fillna(0)
# # df['year'] = pd.date_range(start='9/2/2012', end = '11/15/2013', freq='B')
# series_sahaN20=TimeSeries.from_dataframe(df, "year", "N2O")

# scaler.fit("Yearc","Ch4")
from darts import concatenate

# series = concatenate([series_CH4, series_NIT], axis=1)

# series = series_CH4.stack(series_prec)

# ndf=pd.concat([df['N2OFlux'],df1], axis=1, ignore_index=True)
# ndf.columns=["N2OFlux","Month"]
# print(ndf.head())
from matplotlib import pyplot
# df2.index = df2['Year']
# del df2['Year']
#
# print(df2.head())

# df2.plot()
# pyplot.show()
# import matplotlib.pyplot as plt
# corr_matrix = df2.corr().abs()
# sns.heatmap(corr_matrix,annot=True,fmt='.2f')
# plt.show()

# Select upper triangle of correlation matrix


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
from darts.metrics import mape,rmse,smape,r2_score,mase,rmsle,mae
# load dataset
# def parser(x):
# 	return datetime.strptime('190'+x, '%Y-%m')
# series = read_csv('shampoo-sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)

# df2.index = df2.index.to_period('M')
# df2.set_index('Any', inplace=True)
# df2.index.name = 'year'
# split into train and test sets
# X = series
# df2['N2OFlux_log'] = np.sqrt(df2['N2OFlux'])
# df2['N2OFlux_log_diff'] = df2['N2OFlux_log'] - df2['N2OFlux_log'].shift(1)
# df2['N2OFlux_log_diff'].dropna().plot()
# print(df2.head())
# from pandas.plotting import lag_plot
# lag_plot(df2)
# pyplot.show()
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler
# scaler_data = Scaler()


# X = np.sqrt(df2['N2OFlux_log_diff'].values)
# size = int(len(X) * 0.66)
# train, test = X[0:size], X[size:len(series)]
# # print(len(train))
# # print(len(test))
# history = [x for x in train]
# train,  test = data_series_scaled[:-36], data_series_scaled[-36:]
# prepare data for standardization
# values = series.values
# values = values.reshape((len(values), 1))
# train the standardization
# scaler = StandardScaler()
# scaler = scaler.fit(values)

# scaled_CH4 =scaler_data.fit_transform(series_CH4)
# scaled_NIT =scaler_data.fit_transform(series_NIT)
# scaled_N20 =scaler_data.fit_transform(series_n20s)
# scaled_N20 =scaler_data.fit_transform(series_n20)
# scaled_deloi =scaler_data.fit_transform(series_deloi)
# scaled_CO =scaler_data.fit_transform(series_CO)
# # scaled_noflux =scaler_data.fit_transform(series_noflux)
# scaled_prec =scaler_data.fit_transform(series_prec)
# scaled_stemp =scaler_data.fit_transform(series_stemp)
# scaled_atemp =scaler_data.fit_transform(series_atemp)
# scaled_sresp =scaler_data.fit_transform(series_sresp)
# scaled_msresp =scaler_data.fit_transform(series_msresp)
# scaled_slitresp =scaler_data.fit_transform(series_slitresp)
# scaled_series=scaler.fit_transform(series)

# temp_cov=series_maxt.stack(series_mint)
temp_cov=series_stemp.stack(series_atemp)
series_cov=temp_cov.stack(series_prec)
series_cov1= series_cov.stack(series_sresp)
series_cov2= series_cov1.stack(series_stemp)
series_fcov= series_cov2.stack(series_slitresp)

scaler_CH4,scaler_NIT, scaler_N20,scaler_NOFLUX,scaler_series, scaler_prec, scaler_sresp,scaler_msresp, scaler_slitresp= Scaler(), Scaler(), Scaler(), Scaler(), Scaler(), Scaler(), Scaler(), Scaler(), Scaler()
scaler_sahaN20=Scaler()
scaler_stemp=Scaler()
scaler_deloi=Scaler()
scaler_CO=Scaler()
sclaer_sresp=Scaler()
sclaer_msresp=Scaler()

scaler_temp=Scaler()
scaler_cov, scaler_cov1, scaler_fcov=Scaler(),Scaler(), Scaler()


# scaled_CH4 =scaler_CH4.fit_transform(series_CH4)
# scaled_NIT =scaler_NIT.fit_transform(series_NIT)
# scaled_N20 =scaler_N20.fit_transform(series_n20)
scaled_sresp=scaler_sresp.fit_transform(series_sresp)
scaled_stemp=scaler_stemp.fit_transform(series_stemp)
scaled_slitresp=scaler_slitresp.fit_transform(series_slitresp)
# scaled_deloi =scaler_deloi.fit_transform(series_deloi)
scaled_CO =scaler_CO.fit_transform(series_CO)
# scaled_noflux =scaler_NOFLUX.fit_transform(series_noflux)
scaled_prec =scaler_prec.fit_transform(series_prec)
scaled_temp =scaler_temp.fit_transform(temp_cov)
scaled_cov =scaler_cov.fit_transform(series_cov)
scaled_cov1 =scaler_cov1.fit_transform(series_cov1)
scaled_fcov=scaler_fcov.fit_transform(series_fcov)
# scaled_sahaN20=scaler_sahaN20.fit_transform(series_sahaN20)


# train_CH4, test_CH4 = scaled_CH4[:-1000], scaled_CH4[-1000:]
# train_NIT, test_NIT = scaled_NIT[:-1000], scaled_NIT[-1000:]
# size = int(len(series_n20) * 0.60)
# train_N20, test_N20 = scaled_N20[:-1000], scaled_N20[-1000:]
# train_deloi, test_deloi = scaled_deloi[0:size], scaled_deloi[size:len(df)]
# train_N20, test_N20 = scaled_N20[:-1500], scaled_N20[-1500:]
# train_CO,test_CO=series_CO[:-1000],scaled_CO[-1000:]
train_CO, test_CO = scaled_CO[:-1900], scaled_CO[-1900:]
# train_noflux, test_noflux = scaled_noflux[:-1000], scaled_noflux[-1000:]
# train_sresp, test_sresp = scaled_sresp[:-1000], scaled_sresp[-1000:]
# train_msresp, test_msresp = scaled_msresp[:-1000], scaled_msresp[-1000:]
# train_slitresp, test_slitresp = scaled_slitresp[:-1000], scaled_slitresp[-1000:]
# # train_prec, test_prec = scaled_prec[:-1000], scaled_prec[-1000:]
# train_temp,test_temp = scaled_temp[:-1000],scaled_temp[-1000:]
# train_cov,test_cov = scaled_cov[:-1000],scaled_cov[-1000:]
# train_sahaN20,test_sahaN20 = scaled_sahaN20[:-1000],scaled_sahaN20[-1000:]
# print(len(train_CH4))
# train_series, test_series = scaled_series[:-1000], scaled_series[-1000:]
# predictions = list()
from darts.models import NBEATSModel,RandomForest, LightGBMModel, XGBModel, NHiTSModel, KalmanForecaster,TCNModel,Prophet,RegressionEnsembleModel,NaiveEnsembleModel, AutoARIMA,XGBModel,ARIMA,VARIMA, ExponentialSmoothing,NaiveSeasonal,NaiveDrift,BlockRNNModel,RandomForest,RNNModel,LightGBMModel
from darts.utils.likelihood_models import LaplaceLikelihood
from darts.models.forecasting.tft_model import TFTModel

# model = TCNModel(
#     input_chunk_length=24,
#     output_chunk_length=12,
#     random_state=42,
#     n_epochs=10
#     # likelihood=LaplaceLikelihood(),
# )
# # # model = ExponentialSmoothing()
#
# model_nbeats = KalmanForecaster(
#     dim_x=1
# )

# pred_series = model_nbeats.historical_forecasts(
#     series,
#     start=pd.Timestamp("20170901"),
#     forecast_horizon=7,
#     stride=5,
#     retrain=False,
#     verbose=True,
# )
# model.fit(train)

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
# predictions = list()
# for t in range(len(test)):
	# model = ARIMA(history, order=(5,1,0))
	# model = AutoReg(train, lags=20)
# encoders = {"datetime_attribute": {"past": ["month", "year"]}, "transformer": Scaler()}



def eval_model(model):
    model.fit(train_CO,past_covariates=scaled_fcov)
    # model.fit(train_CO,past_covariates=scaled_fcov)
    # # model.fit(train_CO)
    from keras.utils.vis_utils import plot_model
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    # model.fit(train_CH4)


    # model.fit(train_CH4)
    # model.fit(train_series)
    # predictions = model.predict(len(test_N20),past_covariates=scaled_cov)
    # predictions = model.predict(n=14, series=test_CH4, past_covariates=train_prec)
    # predictions = model.predict(n=1000,past_covariates=scaled_cov)-
    # predictions = model.predict(len(test_CO),past_covariates=scaled_fcov)
    # predictions = model.predict(len(test_slitresp),future_covariates=scaled_fcov)
    # predictions_unscaled=scaler_CO.inverse_transform(predictions)
    predictions = model.predict(len(test_CO),past_covariates=scaled_fcov)
    # predictions = model.predict(n=36)
    print(predictions)
    # predval= predictions[['N2OFlux']].values()
    test_N20_unscaled= scaler_CO.inverse_transform(test_CO)
    # print(model.summary())
    # print('model {} obtains MAPE: {:.2f}%'.format(model, smape(test_N20_unscaled, predictions_unscaled)))
    # print('model {} obtains MAE: {:.2f}%'.format(model, mae(test_N20_unscaled, predictions_unscaled)))
    # print('model {} obtains RMSE: {:.2f}%'.format(model, rmse(test_N20_unscaled, predictions_unscaled)))
    # print('model {} obtains R2: {:.2f}%'.format(model, r2_score(test_N20_unscaled, predictions_unscaled)))

    print('model {} obtains MAPE: {:.2f}%'.format(model, smape(test_N20_unscaled, predictions)))
    print('model {} obtains MAE: {:.2f}%'.format(model, mae(test_N20_unscaled, predictions)))
    print('model {} obtains RMSE: {:.2f}%'.format(model, rmse(test_N20_unscaled, predictions)))
    print('model {} obtains R2: {:.2f}%'.format(model, r2_score(test_N20_unscaled, predictions)))


    plt.figure(figsize=(10, 6))
    # test_CH4.plot(label="Real",color="blue")
    test_CO.plot(label="Actual",color="blue")
    predictions.plot(label="Predicted",color="red")
    plt.legend()
    plt.show()

# eval_model(model = NBEATSModel(
#         input_chunk_length=24,
#         output_chunk_length=12,
#         generic_architecture=False,
#         num_blocks=5,
#         num_layers=10,
#         layer_widths=512,
#         activation='ReLU',
#         optimizer_kwargs={'lr': 1e-3},
#         n_epochs=100,
#         nr_epochs_val_period=1,
#         batch_size=800,
#         model_name="nbeats_interpretable_run",
#     ))
# model = Prophet()
# eval_model(
#                                                         TCNModel(
#                                                             input_chunk_length=24,
#                                                             output_chunk_length=12,
#                                                             random_state=42,
#                                                             n_epochs=100
#                                                             # likelihood=LaplaceLikelihood(),
#                                                         ),
#                                                         NBEATSModel(
#                                                                 input_chunk_length=10,
#                                                                 output_chunk_length=7,
#                                                                 generic_architecture=False,
#                                                                 num_blocks=3,
#                                                                 num_layers=2,
#                                                                 layer_widths=512,
#                                                                 optimizer_kwargs={'lr': 1e-4},
#                                                                 n_epochs=100,
#                                                                 nr_epochs_val_period=1,
#                                                                 batch_size=800,
#                                                                 model_name="nbeats_interpretable_run",
#                                                             ),
#                                                 TFTModel(input_chunk_length=10, output_chunk_length=7, hidden_size=16, lstm_layers=1, num_attention_heads=4, full_attention=False, feed_forward='GatedResidualNetwork', dropout=0.1,n_epochs=100,optimizer_kwargs={'lr': 1e-4},
# #                            # lr_scheduler_cls={
# #                            #     "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
# #                            #     "reduce_on_plateau": True,
# #                            #     # val_checkpoint_on is val_loss passed in as checkpoint_on
# #                            #     "monitor": "val_checkpoint_on",
# #                            #     "patience": 5,
# #                            #     "mode": "min",
# #                            #     "factor": 0.1,
# #                            #     "verbose": True,
# #                            #     "min_lr": 1e-8,
# #                            # },
#             hidden_continuous_size=8, categorical_embedding_sizes=None, add_relative_index=True, loss_fn=None, likelihood=None, norm_type='LayerNorm', use_static_covariates=True))
#
# #
#                                                     ],
#                                 regression_train_n_points=100))
# # model_fit=model_nbeats.fit(train)
# predictions=model_nbeats.predict(n=36)

# model_fit=model.fit([train_CH4,train_NIT])
# model_fit=model.fit(train_CH4)
# # predictions=model.predict(n=36,series=train_CH4)
# predictions=model.predict(len(test_CH4))


# eval_model(model=RandomForest(lags=[-5]))
# eval_model(model = TCNModel.gridsearch(parameters=
#     input_chunk_length=24,
#     output_chunk_length=12,
#     random_state=42,
#     kernel_size=3,
#     num_filters=3,
#     n_epochs=100
#
#     # likelihood=LaplaceLikelihood(),
# ))
#
# eval_model(model = TCNModel(
#     input_chunk_length=24,
#     output_chunk_length=12,
#     kernel_size=3,
#     dilation_base=4,
#     n_epochs=100,
#
#     # likelihood=LaplaceLikelihood()
# ))
# # if gridsearch == True:
#         parameters = {
#             "input_chunk_length": [16, 32],
#             "output_chunk_length": [1],
#             "num_stacks": [16, 30],
#             "num_blocks": [1, 2, 3, 5, 10],
#             "num_layers": [2, 3, 4],
#             "layer_widths": [256, 512, 1024],
#             "n_epochs": [20],
#             "nr_epochs_val_period": [1],
#             "batch_size": [128, 256, 512, 1024],
#             "random_state": [0],
#         }
# eval_model(model = RNNModel(
#     model="LSTM",
#     hidden_dim=40,
#     dropout=0.3,
#     batch_size=16,
#     n_epochs=100,
#     optimizer_kwargs={"lr": 1e-3},
#     model_name="Air_RNN",
#     log_tensorboard=True,
#     n_rnn_layers=5,
#     random_state=42,
#     training_length=20,
#     input_chunk_length=14,
#     output_chunk_length=12,
#     force_reset=True,
#     save_checkpoints=True,
# ))
# eval_model(LightGBMModel(lags_past_covariates=[-5,-4,-3],lags=[-5,-4,-3,-2,-1]))
# eval_model(model=Prophet())
# eval_model(model=RandomForest(lags_past_covariates=[-5,-4,-3],lags=[-5,-4,-3,-2,-1]))
# eval_model(model=sklearn.ensemble.HistGradientBoostingRegressor())
# # print("pred",pred)
# eval_model(LightGBMModel(lags=[-5,-4,-3,-2],lags_past_covariates=[-5,-4,-3,-2]))
# eval_model(XGBModel(lags=[-5,-4,-3],lags_past_covariates=[-1,-2,-3,-4,-5]))
# eval_model(model= BlockRNNModel(input_chunk_length=14,model = 'LSTM',dropout=0.2,random_state = 42,output_chunk_length = 12,n_epochs=100))

# eval_model(AutoARIMA())
# eval_model(ARIMA())
# eval_model(VARIMA())
# eval_model(NaiveSeasonal())
# eval_model(NaiveDrift())
# eval_model(Theta())
# eval_model(model = KalmanForecaster(dim_x=4))


# eval_model(model= TFTModel(input_chunk_length=10, output_chunk_length=7, hidden_size=16, lstm_layers=4, num_attention_heads=4, full_attention=False, feed_forward='GatedResidualNetwork', dropout=0.1,n_epochs=100,optimizer_kwargs={'lr': 1e-3},
#                            # lr_scheduler_cls={
#                            #     "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
#                            #     "reduce_on_plateau": True,
#                            #     # val_checkpoint_on is val_loss passed in as checkpoint_on
#                            #     "monitor": "val_checkpoint_on",
#                            #     "patience": 5,
#                            #     "mode": "min",a
#                            #     "factor": 0.1,
#                            #     "verbose": True,
#                            #     "min_lr": 1e-8,
#                            # },
#             hidden_continuous_size=8, categorical_embedding_sizes=None, add_relative_index=True, loss_fn=None, likelihood=None, norm_type='LayerNorm', use_static_covariates=False))

# eval_model(model = TCNModel(
#     input_chunk_length=14,
#     output_chunk_length=7,
#     random_state=0,
#     n_epochs=100,
#     dilation_base=3,
#     dropout=0.1,
#     weight_norm=False,
#     optimizer_kwargs={"lr": 0.0001},
#     kernel_size=5,
#     num_filters=5,
#
#     likelihood=LaplaceLikelihood()
# ))

#
# avg= np.mean( np.array([ pred, pred1 ]), axis=0 )
# print("Avg", avg)
# models=   [RandomForest(lags=5,lags_past_covariates=[-5,-4,-3,-2,-1]),XGBModel(lags=5, lags_past_covariates=[-5,-4, -3,-2,-1])]
# models=   [RandomForest(lags=5,lags_past_covariates=[-5,-4,-3,-2,-1]),Prophet()]
# eval_model(model= NaiveEnsembleModel(models=models))
# eval_model(model = RegressionEnsembleModel(
#                                 forecasting_models=[
#                                                         RandomForest(lags=5,lags_past_covariates=[-5,-4,-3,-2,-1]),
#                                                         # XGBModel(lags=5, lags_past_covariates=[-5,-4, -3,-2,-1])
#                                                         # Prophet()
#                                                         # AutoARIMA(),
#                                                         # TCNModel(
#                                                         #     input_chunk_length=24,
#                                                         #     output_chunk_length=12,
#                                                         #     random_state=42,
#                                                         #     n_epochs=100
#                                                         #     # likelihood=LaplaceLikelihood(),
#                                                         # )
#                                                         TCNModel(
#                                                             input_chunk_length=24,
#                                                             output_chunk_length=12,
#                                                             random_state=42,
#                                                             n_epochs=100
#                                                             # likelihood=LaplaceLikelihood(),
#                                                         ),
#                                                         # TFTModel(input_chunk_length=10, output_chunk_length=7, hidden_size=16, lstm_layers=1, num_attention_heads=4, full_attention=False, feed_forward='GatedResidualNetwork', dropout=0.1,
#                                                         # n_epochs=100, hidden_continuous_size=8, categorical_embedding_sizes=None, add_relative_index=True, loss_fn=None, likelihood=None, norm_type='LayerNorm', use_static_covariates=True),
#                                                         # NBEATSModel(
#                                                         #         input_chunk_length=10,
#                                                         #         output_chunk_length=7,
#                                                         #         generic_architecture=False,
#                                                         #         num_blocks=3,
#                                                         #         num_layers=2,
#                                                         #         layer_widths=512,
#                                                         #         optimizer_kwargs={'lr': 1e-4},
#                                                         #         n_epochs=100,
#                                                         #         nr_epochs_val_period=1,
#                                                         #         # callbacks=[ca],
#                                                         #         batch_size=800,
#                                                         #         model_name="nbeats_interpretable_run",
#                                                         #     )
#
#                                                     ],
#                                 regression_train_n_points=8000))
from keras.callbacks import Callback
from math import pi
from math import cos
from math import floor
from keras import backend
# class CosineAnnealingLearningRateSchedule(Callback):
# 	# constructor
# 	def __init__(self, n_epochs, n_cycles, lrate_max, verbose=0):
# 		self.epochs = n_epochs
# 		self.cycles = n_cycles
# 		self.lr_max = lrate_max
# 		self.lrates = list()
#
# 	# calculate learning rate for an epoch
# 	def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):
# 		epochs_per_cycle = floor(n_epochs/n_cycles)
# 		cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
# 		return lrate_max/2 * (cos(cos_inner) + 1)
#
# 	# calculate and set learning rate at the start of the epoch
# 	def on_epoch_begin(self, epoch, logs=None):
# 		# calculate learning rate
# 		lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_max)
# 		# set learning rate
# 		backend.set_value(self.model.optimizer.lr, lr)
# 		# log value
# 		self.lrates.append(lr)

# n_epochs = 400
# n_cycles = n_epochs / 50
# ca = CosineAnnealingLearningRateSchedule(n_epochs, n_cycles, 0.01)

# eval_model(model = NBEATSModel(
#         input_chunk_length=24,
#         output_chunk_length=12,
#         generic_architecture=False,
#         num_blocks=4,
#         num_layers=3,
#         layer_widths=512,
#         optimizer_kwargs={'lr': 1e-3},
#         # lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
#         optimizer_cls = torch.optim.Adam,
#         n_epochs=100,
#         nr_epochs_val_period=1,
#         batch_size=400,
#         model_name="nbeats_interpretable_run",
#     ))
# eval_model(model = NHiTSModel(
#         input_chunk_length=24,
#         output_chunk_length=12,
#         num_blocks=3,
#         num_layers=4,
#         layer_widths=512,
#         optimizer_kwargs={'lr': 1e-3},
#         # lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
#         optimizer_cls = torch.optim.Adam,
#         n_epochs=100,
#         nr_epochs_val_period=1,
#         batch_size=800,
#         model_name="nbeats_interpretable_run",
#     ))
# print(data_series_scaled)

# output = model_fit.forecast()
# yhat = output[0]
# predictions.append(yhat)
# # predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
# obs = test[t]
# history.append(obs)
# print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
# print("Mape = {:.2f}%".format(mape(data_series_scaled, predictions)))
# print("RMSE = {:.2f}%".format(rmse(data_series_scaled, predictions)))

# print("Mape = {:.2f}%".format(mape(test_CH4, predictions)))


# plot forecasts against actual outcomes
# pyplot.plot(test)
# pyplot.plot(predictions, color='red')
# pyplot.show()
# prediction = model_nbeats.predict(n=360, series = test)
# pred_error = mape(test[:360],prediction)
#error_follower = mape(val_follower[:36],pred_follower)
# print("this is the error for the closing price:",pred_error )
# import matplotlib.pyplot as plt
# series.plot()
# prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
# plt.legend()
# plt.show()
# print(test)
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
# print('The MAPE is: {:.2f}.'.format(r2_score(test[:360], prediction)))

# rmse = sqrt(mean_squared_error(test[:360], prediction))
# print('Test RMSE: %.3f' % rmse)
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
# import matplotlib.pyplot as plt
# # predictions= scaler.inverse_transform(predictions)
# series.plot(label="actual")
# predictions.plot(label="forecast")
# plt.legend()
# plt.show()

# pred_CH4= scaler_CH4.inverse_transform(predictions)


