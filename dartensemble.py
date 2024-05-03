import time

## Importing Libraries
import sys
import numbers
import time
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from functools import reduce

import pmdarima as pmd
import statsmodels.api as sm
from scipy.stats import normaltest

from darts import TimeSeries
from darts.models import (
    NaiveSeasonal,
    NaiveDrift,
    Prophet,
    ExponentialSmoothing,
    ARIMA,
    AutoARIMA,
    Theta,
    RegressionEnsembleModel)

from darts.metrics import mape, mase, mae, mse, ope, r2_score, rmse, rmsle
from darts.utils.statistics import check_seasonality, plot_acf, plot_residuals_analysis
from darts.dataprocessing.transformers.boxcox import BoxCox
from darts.utils.utils import ModelMode
read_file = pd.read_excel("oakridge_CH4.xlsx")
read_file.to_csv("Test.csv",index=None,header=True)
df = pd.DataFrame(pd.read_csv("Test.csv"))
print(df.head())
df['Yearc']=pd.to_datetime(df[['year', 'month', 'day']])

series = TimeSeries.from_dataframe(df, "Yearc", "CH4")
plt.figure(100, figsize=(12, 5))
series.plot()
plt.show()

from darts.dataprocessing.transformers import Scaler
scaler = Scaler()
series_scaled=scaler.fit_transform(series)
train,  val = series_scaled[:-1000], series_scaled[-1000:]

ALPHA = 0.05
TRACE = False                 # print also the suboptimal models while SARIMA tuning process is running
MSEAS = 12

for m in range(2, 25):
    is_seasonal, mseas = check_seasonality(series, m=m, alpha=ALPHA)
    if is_seasonal:
        break

print("seasonal? " + str(is_seasonal))
if is_seasonal:
    print('There is seasonality of order {}.'.format(mseas))
# if isinstance(TRAIN, numbers.Number):
#     split_at = TRAIN
# else:
#     split_at = pd.Timestamp(TRAIN)
# train, val = series.split_before(split_at)

plt.figure(101, figsize=(12, 5))
train.plot(label='training')
val.plot(label='validation')
plt.legend()
plt.show()


def accuracy_metrics(act, forecast, resid, t_start):
    sr = resid.pd_series()
    sa = act.pd_series()
    res_r2 = r2_score(act, forecast)
    n_act = len(act)
    res_mape = mape(act, forecast)
    res_mae = mae(act, forecast)
    res_r2 = r2_score(act, forecast)
    res_rmse = rmse(act, forecast)
    res_rmsle = rmsle(act, forecast)
    res_pe = sr / sa
    res_rmspe = np.sqrt(np.sum(res_pe ** 2) / n_act)  # root mean square percentage error

    res_time = time.perf_counter() - t_start

    res_mean = np.mean(sr)
    res_std = np.std(sr)  # std error of the model = std deviation of the noise
    res_se = res_std / np.sqrt(n_act)  # std error in estimating the mean
    res_sefc = np.sqrt(res_std + res_se ** 2)  # std error of the forecast

    res_accuracy = {"MAPE": res_mape, "RMSE": res_rmse, "-R squared": -res_r2,
                    "se": res_sefc, "time": res_time}
    return res_accuracy


def eval_model(model):
    t_start = time.perf_counter()
    strmodel = str(model)[:30]
    print("beginning: " + strmodel)

    # fit the model and compute predictions
    n_val = len(val)
    res = model.fit(train)
    forecast = model.predict(n_val)

    # for naive forecast, concatenate seasonal fc with drift fc
    if model == m_naive:
        if is_seasonal:
            fc_drift = forecast
            modelS = NaiveSeasonal(K=mseas)
            modelS.fit(train)
            fc_seas = modelS.predict(len(val))
            forecast = fc_drift + fc_seas - train.last_value()

    resid = forecast - val
    res_accuracy = accuracy_metrics(val, forecast, resid, t_start)

    results = [forecast, res_accuracy]

    print("completed: " + strmodel + ":" + str(res_accuracy["time"]) + " sec")
    return results
# prepare Naive forecaster

m_naive = NaiveDrift()
print("model:", m_naive)

# prepare Exponential Smoothing forecaster

if is_seasonal:
    #m_expon = ExponentialSmoothing(seasonal_periods=mseas)
    m_expon = ExponentialSmoothing( trend=ModelMode.MULTIPLICATIVE,
                                    damped=False,
                                    seasonal=ModelMode.MULTIPLICATIVE,
                                    seasonal_periods=mseas)
else:
    m_expon = ExponentialSmoothing()

print("model:", m_expon)

m_prophet = Prophet()    #frequency=mseas)
print("model:", m_prophet)

y = np.asarray(series.pd_series())
# get order of first differencing: the higher of KPSS and ADF test results
n_kpss = pmd.arima.ndiffs(y, alpha=ALPHA, test='kpss', max_d=2)
n_adf = pmd.arima.ndiffs(y, alpha=ALPHA, test='adf', max_d=2)
n_diff = max(n_adf, n_kpss)

# get order of seasonal differencing: the higher of OCSB and CH test results
n_ocsb = pmd.arima.OCSBTest(m=max(4,mseas)).estimate_seasonal_differencing_term(y)
n_ch = pmd.arima.CHTest(m=max(4,mseas)).estimate_seasonal_differencing_term(y)
ns_diff = max(n_ocsb, n_ch, is_seasonal * 1)

# set up the ARIMA forecaster
m_arima = AutoARIMA(
    start_p=1, d=n_diff, start_q=1,
    max_p=4, max_d=n_diff, max_q=4,
    start_P=0, D=ns_diff, start_Q=0, m=max(4,mseas), seasonal=is_seasonal,
    max_P=3, max_D=1, max_Q=3,
    max_order=5,                       # p+q+p+Q <= max_order
    stationary=False,
    information_criterion="bic", alpha=ALPHA,
    test="kpss", seasonal_test="ocsb",
    stepwise=True,
    suppress_warnings=True, error_action="trace", trace=TRACE, with_intercept="auto")
print("model:", m_arima)

# prepare Theta forecaster

# # search space for best theta value: check 100 alternatives
# thetas = 2 - np.linspace(-10, 10, 100)
#
# # initialize search
# best_mape = float('inf')
# best_theta = 0
# # search for best theta among 50 values, as measured by MAPE
# for theta in thetas:
#     model = Theta(theta)
#     res = model.fit(train)
#     pred_theta = model.predict(len(val))
#     res_mape = mape(val, pred_theta)
#
#     if res_mape < best_mape:
#         best_mape = res_mape
#         best_theta = theta
#
# m_theta = Theta(best_theta)   # best theta model among 100
# print("model:", m_theta)

models = [
    # m_theta,
    m_arima,
    m_naive,
    m_prophet]
model_predictions = [eval_model(model) for model in models]
df_acc = pd.DataFrame.from_dict(model_predictions[0][1], orient="index")
df_acc.columns = [str(models[1])]

# for i, m in enumerate(models):
#     if i > 0:
#         df = pd.DataFrame.from_dict(model_predictions[i][1], orient="index")
#         df.columns = [str(m)]
#         df_acc = pd.concat([df_acc, df], axis=1)
#     i +=1
#
# pd.set_option("display.precision",3)
# plot the forecasts

# pairs = math.ceil(len(models) / 2)  # how many rows of charts
# fig, ax = plt.subplots(pairs, 2, figsize=(20, 5 * pairs))
# ax = ax.ravel()

# for i, m in enumerate(models):
#     series.plot(label="actual", ax=ax[i])
#     model_predictions[i][0].plot(label="prediction: " + str(m), ax=ax[i])
#
#     mape_model = model_predictions[i][1]["MAPE"]
#     time_model = model_predictions[i][1]["time"]
#     ax[i].set_title("\n\n" + str(m)[:30] + ": MAPE {:.1f}%".format(mape_model) + " - time {:.1f}sec".format(time_model))
#
#     ax[i].set_xlabel("")
#     ax[i].legend()
# plt.show()
act = val

resL = {}
resN = {}
for i, m in enumerate(models):
    pred = model_predictions[i][0]
    resid = pred - act
    sr = resid.pd_series()

    resL[str(m)] = sm.stats.acorr_ljungbox(sr, lags=[5], return_df=False)[1][0]
    resN[str(m)] = normaltest(sr)[1]

print("\nLjung-Box test for white-noise residuals: p-value > alpha?")
[print(key, ":", value) for key, value in resL.items()]

print("\ntest for normality of residuals: p-value > alpha?")
[print(key, ":", value) for key, value in resN.items()]
# act = val
# df_desc = pd.DataFrame()
#
# for i,m in enumerate(models):
#         pred = model_predictions[i][0]
#         resid = pred - act
#
#         df_desc = pd.concat([df_desc, resid.describe()], axis=1)
#
#         plot_residuals_analysis(resid)
#         plt.title(str(m))
# plt.show()

def ensemble_eval(train, val, models):
    t_start =  time.perf_counter()
    n_train = 50                # use 50 observation to train the ensemble model
    n_val = len(val)            # forecast as many periods as are in the valuation dataset


    # compute predictions
    # ensemble_model = RegressionEnsembleModel(forecasting_models=models, regression_train_n_points=n_train)
    # new Jan 022: RegressionEnsembleModel class no longer accepts as argument the list "models" of previously instantiated models
    # instead, explicitly list each of the forecast methods
    ensemble_model = RegressionEnsembleModel(
                                forecasting_models=[
                                                        Prophet(),
                                                        # AutoARIMA(),
                                                        NaiveDrift(),
                                                    ],
                                regression_train_n_points=12)



    ensemble_model.fit(train)
    forecast = ensemble_model.predict(n_val)
    resid = forecast - val


    res_accuracy = accuracy_metrics(val, forecast, resid, t_start)


    # plot the ensemble forecast
    series.plot(label="actual")
    forecast.plot(label="Ensemble forecast")
    plt.title("RMSE = {:.2f}%".format(res_accuracy["RMSE"]))
    plt.legend()
    plt.show()


    results = [forecast, res_accuracy]
    return results
#
col_heads = ["ARIMA", "Naive", "Prophet", "avg", "Ensemble"]
models2 = models

# run the ensemble forecast
print("Ensemble of all 5 forecasters:")
res_ensemble = ensemble_eval(train, val, models2)

resid = res_ensemble[0] - val
sr = resid.pd_series()
plot_residuals_analysis(resid)
plt.title("Emsemble forecast")
plt.show()

resL = sm.stats.acorr_ljungbox(sr, lags=[5], return_df=False)[1][0]
resN = normaltest(sr)[1]

print("\nLjung-Box test for white-noise residuals: p-value > alpha?")
print(resL)

print("\ntest for normality of residuals: p-value > alpha?")
print(resN)

pairs = math.ceil(len(models)/2)                    # how many rows of charts
fig, ax = plt.subplots(pairs, 2, figsize=(20, 5 * pairs))
ax = ax.ravel()

for i,m in enumerate(models):
        series.plot(label="actual", ax=ax[i])
        model_predictions[i][0].plot(label="prediction: "+str(m), ax=ax[i])
        rmse_model =  model_predictions[i][1]["RMSE"]
        time_model =  model_predictions[i][1]["time"]
        ax[i].set_title("\n\n" + str(m) + ": RMSE {:.0f}".format(rmse_model) + " - time {:.1f}sec".format(time_model))
        ax[i].set_xlabel("")
        ax[i].legend()


# add the ensemble:
series.plot(label="actual", ax=ax[i+1])
res_ensemble[0].plot(label="prediction: Ensemble", ax=ax[i+1])
rmse_model =  res_ensemble[1]["RMSE"]
time_model =  res_ensemble[1]["time"]
ax[i+1].set_title("\n\n Ensemble: RMSE {:.0f}".format(rmse_model) + " - time {:.1f}sec".format(time_model))
ax[i+1].set_xlabel("")
ax[i+1].legend()
plt.show()