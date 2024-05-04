import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers,layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers.convolutional import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten,Bidirectional
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import tensorflow

df = pd.DataFrame(pd.read_csv("Test.csv"))
df['Yearc']=pd.to_datetime(df[['year', 'month', 'day']])
print(df.head())
df=df.drop(columns=['day', 'month','year','Day of the year, 1-366',' Maximum temperature for day, degrees C','Minimum temperature for day, degrees C', 'Precipitation for day, centimeters'])

# df2 = pd.DataFrame().assign(Year=df['Yearc'], N2OFlux=df['N2OFlux'])
# print(df2)
# df2['Year'] = [d.year for d in df2.Year]
# df2 = pd.DataFrame().assign(Year=df['Yearc'], N2OFlux=df['N2OFlux'])
# print(df2)
# lag_size = (test['date'].max().date() - train['date'].max().date()).days
# print('Max date from train set: %s' % train['date'].max().date())
# print('Max date from test set: %s' % test['date'].max().date())
# print('Forecast lag size', lag_size)
# fig, axes = plt.subplots(dpi= 80)
# sns.boxplot(x='Year', y='N2OFlux', data=df2, ax=axes)
# axes.set_title('Year-wise Box Plot\n(The Trend)', fontsize=18)
# plt.show()
X = df
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(df)]
# Set seeds to make the experiment more reproducible.
# from tensorflow import set_random_seed
from numpy.random import seed
tensorflow.random.set_seed(1)
# set_random_seed(1)
seed(1)
print('Min date from train set: %s' % train['Yearc'].min().date())
print('Max date from train set: %s' % train['Yearc'].max().date())
lag_size = (test['Yearc'].max().date() - train['Yearc'].max().date()).days
# print('Max date from train set: %s' % train['Yearc'].max().date())
# print('Max date from test set: %s' % test['Yearc'].max().date())
# print('Forecast lag size', lag_size)

def series_to_supervised(data, window=1, lag=1, dropnan=True):
    cols, names = list(), list()
    # Input sequence (t-n, ... t-1)
    for i in range(window, 0, -1):
        cols.append(data.shift(i))
        names += [('%s(t-%d)' % (col, i)) for col in data.columns]
    # Current timestep (t=0)
    cols.append(data)
    names += [('%s(t)' % (col)) for col in data.columns]
    # Target timestep (t=lag)
    cols.append(data.shift(-lag))
    names += [('%s(t+%d)' % (col, lag)) for col in data.columns]
    # Put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


series = [df['N2OFlux'].values]
# series = series.astype('float32')

from sklearn.preprocessing import MinMaxScaler



# scaler = MinMaxScaler(feature_range=(0, 1))
# series = scaler.fit_transform(series)


window = 29
lag = lag_size
series = series_to_supervised(df.drop('Yearc', axis=1), window=window, lag=lag)


# last_store = 'store(t-%d)' % window
# series = series[(series['store(t)'] == series[last_store])]
# series = series[(series['N2OFlux(t)'] == series[last_N2OFlux])]

# columns_to_drop = [('%s(t+%d)' % (col, lag)) for col in ['N2OFlux']]
# for i in range(window, 0, -1):
#     columns_to_drop += [('%s(t-%d)' % (col, i)) for col in ['N2OFlux']]
# series.drop(columns_to_drop, axis=1, inplace=True)
# series.drop(['N2OFlux(t)'], axis=1, inplace=True)
# print(series.head())

labels_col = 'N2OFlux(t+%d)' % lag_size
labels = series[labels_col]
series = series.drop(labels_col, axis=1)



X_train, X_valid, Y_train, Y_valid = train_test_split(series, labels.values, test_size=0.4, random_state=0)
print('Train set shape', X_train.shape)
print('Validation set shape', X_valid.shape)
print(Y_train[0])


epochs = 40
batch = 256
lr = 0.0003
adam = optimizers.Adam(lr)

X_train_series = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_valid_series = X_valid.values.reshape((X_valid.shape[0], X_valid.shape[1], 1))
#
# X_train_series = X_train.values.reshape(X_train.shape[1])

# print('Train set shape', X_train_series.shape)
# print('Validation set shape', X_valid_series.shape)
#
# model_lstm = Sequential()
# model_lstm.add(Bidirectional(LSTM(50, activation='relu', input_shape=(X_train_series.shape[1], X_train_series.shape[2]))))
# model_lstm.add(Dense(1))
# model_lstm.compile(loss='mse', optimizer=adam)
# model_lstm.summary()

# lstm_history = model_lstm.fit(X_train_series, Y_train, validation_data=(X_valid_series, Y_valid), epochs=epochs, verbose=2)
# model_lstm.summary()
# trainPredict = model_lstm.predict(X_train_series )
# testPredict = model_lstm.predict(X_valid_series)
# print('Train rmse:', np.sqrt(mean_squared_error(Y_train, trainPredict)))
# print('Validation rmse:', np.sqrt(mean_squared_error(Y_valid, testPredict)))
# J=8
# Q=12
# X_train_series = Scattering1D(J, Q=Q)(X_train_series)
# X_valid_series = Scattering1D(J, Q=Q)(X_valid_series)
subsequences = 2
timesteps = X_train_series.shape[1]//subsequences
X_train_series_sub = X_train_series.reshape((X_train_series.shape[0], subsequences, timesteps, 1))
X_valid_series_sub = X_valid_series.reshape((X_valid_series.shape[0], subsequences, timesteps, 1))
print('Train set shape', X_train_series_sub.shape)
print('Validation set shape', X_valid_series_sub.shape)
#
model_cnn_lstm = Sequential()
model_cnn_lstm.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, X_train_series_sub.shape[2], X_train_series_sub.shape[3])))
model_cnn_lstm.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model_cnn_lstm.add(TimeDistributed(Flatten()))
model_cnn_lstm.add(LSTM(50, activation='relu'))
model_cnn_lstm.add(Dense(1))
model_cnn_lstm.compile(loss='mse', optimizer=adam)

# model_cnn_lstm = Sequential()
# model_cnn_lstm.add(TimeDistributed(MaxPooling1D(pool_size=2)))
# model_cnn_lstm.add(TimeDistributed(Flatten()))
# model_cnn_lstm.add(LSTM(50, activation='relu'))
# model_cnn_lstm.add(Dense(1))
# model_cnn_lstm.compile(loss='mse', optimizer=adam)
#
cnn_lstm_history = model_cnn_lstm.fit(X_train_series_sub, Y_train, validation_data=(X_valid_series_sub, Y_valid), epochs=epochs, verbose=2)
cnn_lstm_train_pred = model_cnn_lstm.predict(X_train_series_sub)
cnn_lstm_valid_pred = model_cnn_lstm.predict(X_valid_series_sub)
print('Train rmse:', np.sqrt(mean_squared_error(Y_train, cnn_lstm_train_pred)))
print('Validation rmse:', np.sqrt(mean_squared_error(Y_valid, cnn_lstm_valid_pred)))


# invert predictions
# trainPredict = scaler.inverse_transform(cnn_lstm_train_pred)
# trainY = scaler.inverse_transform([Y_train])
# testPredict = scaler.inverse_transform(cnn_lstm_valid_pred)
# testY = scaler.inverse_transform([Y_valid])
# trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))
# look_back = 3
# # shift train predictions for plotting
# trainPredictPlot = np.empty_like(series)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# # shift test predictions for plotting
# testPredictPlot = np.empty_like(series)
# testPredictPlot[:, :] = np.nan
# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(series)-1, :] = testPredict
# # plot baseline and predictions
# plt.plot(scaler.inverse_transform(series))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()
