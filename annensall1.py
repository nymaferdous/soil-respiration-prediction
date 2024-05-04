#MLP for time series forecasting

from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt

dataframe = read_csv('CH4.csv', engine='python')
dataset = dataframe.values

train_size = int(dataset.shape[0] * 0.67)
train_df, test_df = dataset[:train_size, :], dataset[train_size:, :]

#converting multidimensional array to single dim
# data = dataset.flatten()
# raw_seq = data.tolist()

train_data = train_df.flatten()
train_raw_seq = train_data.tolist()

test_data = test_df.flatten()
test_raw_seq = test_data.tolist()



#split a univariate sequence into samples
def split_sequence(sequence, n_steps):
 X, y = list(), list()
 for i in range(len(sequence)):
  # find the end of this pattern
  end_ix = i + n_steps
  # check if we are beyond the sequence
  if end_ix > len(sequence)-1:
   break
  #input and output sample
  seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
  X.append(seq_x)
  y.append(seq_y)
 return array(X), array(y)

from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_percentage_error

#choose a number of time steps
n_steps = 6
#split into samples
# X, y = split_sequence(raw_seq, n_steps)
X_train, y_train = split_sequence(train_raw_seq, n_steps)
X_test, y_test = split_sequence(test_raw_seq, n_steps)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

# X_train_s=scaler.fit_transform(train_X)
# X_test_s=scaler.fit_transform(test_X)
# y_train_s=scaler.fit_transform(train_y)
# y_test_s=scaler.fit_transform(test_y)
# for i in range(5):
#     print(train_X[i], train_y[i])

#define model
model = Sequential()
model.add(Dense(150, activation='relu', input_dim=n_steps))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse',metrics =['accuracy'])

#fit model
model.fit(X_train, y_train, epochs=2, verbose=1)
# model.fit(X_test, y_test, epochs=10, verbose=1)

# trainPredict = model.predict(X_train_s)

testPredict = model.predict(X_test)

# print('Train rmse:', np.sqrt(mean_squared_error(train_y, trainPredict)))
print('Test rmse:', np.sqrt(mean_squared_error(y_test, testPredict)))

# print('Train mape:', mean_absolute_percentage_error(train_y, trainPredict))
print('Test mape:', mean_absolute_percentage_error(y_test, testPredict))

# inv_y_test=scaler.inverse_transform(y_test)
# inv_y_pred=scaler.inverse_transform(testPredict)

# print('Train r2:', r2_score(train_y, trainPredict))
print('Test r2:', r2_score(y_test, testPredict))

fig=plt.figure(figsize=(15,10))
plt.plot(test_df,color='b',label="Real")
plt.plot(testPredict,color='r',label="predicted")
plt.legend()
plt.show()

# trainScore = np.sqrt(mean_squared_error(train_y[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = np.sqrt(mean_squared_error(test_y[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))

# train_score = model.evaluate(train_X, train_y)
# test_score = model.evaluate(test_X, test_y)
# print('Train score: {} MSE'.format(train_score))
# print('Test score: {} MSE'.format(test_score))
# print(X_test.shape)

# plt.figure(figsize=(12, 8))
# train_prediction = model.predict(X_train)
# train_stamp = np.arange(n_steps, n_steps + X_train.shape[0])
# test_prediction = model.predict(X_test)
# test_stamp = np.arange(2 * n_steps + X_test.shape[1], len(dataset))
# # plt.plot(dataset, label='true values')
# # plt.plot(train_stamp, train_prediction, label='train prediction')
# plt.plot(test_stamp, test_prediction, label = 'test_prediction')
# plt.legend()
# plt.show()

# trainPredictPlot = np.empty_like(dataset)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[n_steps:len(trainPredict)+n_steps, :] = trainPredict
# # shift test predictions for plotting
# testPredictPlot = np.empty_like(dataset)
# testPredictPlot[:, :] = np.nan
# testPredictPlot[len(trainPredict)+(n_steps*2)+1:len(dataset)-1, :] = testPredict
# # plot baseline and predictions
# # plt.plot(scaler.inverse_transform(dataset))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()

#prediction for 156thday
# x_input = array([13758429,14377942,15046478])
# x_input = x_input.reshape((1, n_steps))
# yhat = model.predict(x_input, verbose=0)
# print(yhat)

#prediction for 157th day
# x_input = array([14377942,15046478,15733087])
# x_input = x_input.reshape((1, n_steps))
# yhat = model.predict(x_input, verbose=1)
# print(yhat)

#prediction for 158th day
# x_input = array([15046478,15733087,16455969])
# x_input = x_input.reshape((1, n_steps))
# yhat = model.predict(x_input, verbose=1)
# print(yhat)

# generate predictions for training
# trainPredict = model.predict(X_train)
# testPredict = model.predict(X_test)
# # shift train predictions for plotting
# trainPredictPlot = np.empty_like(dataset)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[train_raw_seq:len(trainPredict)+n_steps, :] = trainPredict
# # shift test predictions for plotting
# testPredictPlot = np.empty_like(dataset)
# testPredictPlot[:, :] = np.nan
# testPredictPlot[len(trainPredict)+(n_steps*2)+1:len(dataset)-1, :] = testPredict
# # plot baseline and predictions
# plt.plot(dataset)
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()