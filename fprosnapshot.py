import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import lightgbm as ltb
from sklearn.neural_network import MLPRegressor
df = pd.DataFrame(pd.read_csv("oakridgeufour.csv",header=0,encoding='utf-8'))
df=df[['ATemp','msresp','sresp','stemp','slitresp','Precipitation','year','month','day','CO2resp']]
# df=df[['ATemp','msresp','sresp','stemp','slitresp','Precipitation','CH4','NIT','N2OFlux','NOFlux', 'CO2resp','day','month','year']]


from sklearn.ensemble import IsolationForest

##data Visualization
numerical_column= ["ATemp","msresp","sresp","stemp","slitresp","Precipitation",'CH4','NIT','N2OFlux','NOFlux','CO2resp']
import seaborn as sns

# plt.figure(figsize=(18,11))
# for i,j in zip(range(1, 11),numerical_column):
#     plt.subplot(3, 4, i)
#     sns.boxplot(data=df, x=j)
#     # sns.set_theme()
#     # plt.title('{}'.format(j))
# plt.show()

def drop_outliers_IQR(df):

   q1=df.quantile(0.25)

   q3=df.quantile(0.75)

   IQR=q3-q1

   not_outliers = df[~((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]

   outliers_dropped = not_outliers.dropna().reset_index()

   return outliers_dropped



# df=df[['ATemp','msresp','sresp','stemp','slitresp','Precipitation','CO2resp','day','month','year']]
# df=df[['ATemp','msresp','stemp','Precipitation','year','month','day']]
df['datetime']=pd.to_datetime(df[['year', 'month', 'day']])
# df_monthly=df.resample('M').sum()
from keras.models import load_model
# load models from file
def load_all_models(n_models):
	all_models = list()
	for i in range(n_models):
		# define filename for this ensemble
		filename = 'snapshot_model_' + str(i + 1) + '.h5'
		# load model from file
		model = load_model(filename)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models
import numpy
from numpy import array
from numpy import argmax
from sklearn.metrics import accuracy_score
# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, testX):
	# make predictions
	yhats = [model.predict(testX) for model in members]
	yhats = array(yhats)
	# sum across ensemble members
	summed = numpy.sum(yhats, axis=0)
	# argmax across classes
	result = argmax(summed, axis=1)
	return result

# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, n_members, testX, testy):
	# select a subset of members
	subset = members[:n_members]
	# make prediction
	yhat = ensemble_predictions(subset, testX)
	# calculate accuracy
	return accuracy_score(testy, yhat)
def prophet_features(df, horizon=1000):
    # temp_df=df_monthly.reset_index()
    temp_df = df.reset_index()
    temp_df = temp_df[['datetime', 'CO2resp']]
    print(df['datetime'])
    temp_df.rename(columns={'datetime': 'ds', 'CO2resp': 'y'}, inplace=True)

    # take last week of the dataset for validation
    train, test = temp_df.iloc[:-horizon, :], temp_df.iloc[-horizon:, :]

    # define prophet model
    m = Prophet(
        growth='linear',
        seasonality_mode='additive',
        interval_width=0.75,
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False
    )
    import matplotlib.pyplot as plt
    # m.add_regressor('maxt',standardize=False)
    # m.add_regressor('mint', standardize=False)
    # m.add_regressor('Precipitation', standardize=False)
    from neuralprophet import NeuralProphet
    # m = NeuralProphet()

    from neuralprophet import NeuralProphet
    # train prophet model
    m.fit(train)

    # extract features from data using prophet to predict train set
    predictions_train = m.predict(train.drop('y', axis=1))
    print(predictions_train)
    # extract features from data using prophet to predict test set
    predictions_test = m.predict(test.drop('y', axis=1))

    # merge train and test predictions
    predictions = pd.concat([predictions_train, predictions_test], axis=0)
    m.plot_components(predictions_test)
    plt.show()

    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # sorted(zip(clf.feature_importances_, X.columns), reverse=True)
    # feature_imp = pd.DataFrame(sorted(zip(m., X.columns)), columns=['Value', 'Feature'])
    #
    # plt.figure(figsize=(20, 10))
    # sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    # plt.title('LightGBM Features (avg over folds)')
    # plt.tight_layout()
    # plt.show()
    # plt.savefig('lgbm_importances-01.png')
    # # fig1 = m.plot(predictions)
    # # plt.show()
    # from prophet.plot import plot_plotly, plot_components_plotly
    #
    # plot_plotly(m, predictions)
    # plt.show()

    return predictions,predictions_test

from keras.callbacks import Callback
from keras.optimizers import SGD
from matplotlib import pyplot
from keras import backend
from math import pi
from math import cos
from math import floor
class SnapshotEnsemble(Callback):
	# constructor
	def __init__(self, n_epochs, n_cycles, lrate_max, verbose=0):
		self.epochs = n_epochs
		self.cycles = n_cycles
		self.lr_max = lrate_max
		self.lrates = list()

	# calculate learning rate for epoch
	def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):
		epochs_per_cycle = floor(n_epochs/n_cycles)
		cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
		return lrate_max/2 * (cos(cos_inner) + 1)


	# calculate and set learning rate at the start of the epoch
	def on_epoch_begin(self, epoch, logs={}):
		# calculate learning rate
		lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_max)
		# set learning rate
		backend.set_value(self.model.optimizer.lr, lr)
		# log value
		self.lrates.append(lr)

	# save models at the end of each cycle
	def on_epoch_end(self, epoch, logs={}):
		# check if we can save model
		epochs_per_cycle = floor(self.epochs / self.cycles)
		if epoch != 0 and (epoch + 1) % epochs_per_cycle == 0:
			# save model to file
			filename = "snapshot_model_%d.h5" % int((epoch + 1) / epochs_per_cycle)
			self.model.save(filename)
			print('>saved snapshot %s, epoch %d' % (filename, epoch))




def train_time_series_with_folds_autoreg_prophet_features(df, horizon=1000, lags=[1,2,3,4,5]):
    # create a dataframe with all the new features created with Prophet
    new_prophet_features,prophet_test_features = prophet_features(df, horizon=horizon)
    # new_prophet_features, prophet_test_features = prophet_features(df_monthly, horizon=horizon)
    # df.reset_index(inplace=True)
    # df_monthly.reset_index(inplace=True)
    from sklearn.preprocessing import StandardScaler

    # merge the Prophet features dataframe with the our first dataframe
    df = pd.merge(df, new_prophet_features, left_on=['datetime'], right_on=['ds'], how='inner')
    print(df.columns)

    # df = pd.merge(df_monthly, new_prophet_features, left_on=['datetime'], right_on=['ds'], how='inner')
    # print(df.columns)
    # df = pd.merge(df_monthly, new_prophet_features, how='inner')

    df = pd.merge(df, new_prophet_features,how='inner')



    # scaler = StandardScaler()
    # df = scaler.fit_transform(df_series)
    # print(df)

    # df.drop('ds', axis=1, inplace=True)
    df.set_index('datetime', inplace=True)
    # print (len(df.columns))

    # for lag in lags:
    #     df[f'yhat_lag_{lag}'] = df['yhat'].shift(lag)

    # df_temp= df.loc[:, df.columns != df['index']]
    # print(df_temp)
    # df= drop_outliers_IQR(df)


    df.dropna(axis=0, how='any')
    # print(df)

    df = df.fillna(0)
    df = df.apply(lambda x: pd.factorize(x)[0])

    X = df.drop('CO2resp', axis=1)
    y = df['CO2resp']
    print("X column",X.columns)



    import numpy as np
    import seaborn as sns
    corr_matrix = df.corr().abs()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f')
    plt.show()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.5)]
    # print(X.columns)
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_regression
    # fs = SelectKBest(score_func=f_regression, k=15)
    # X = fs.fit_transform(X, y)
    # # Drop features
    # X.drop(to_drop, axis=1, inplace=True)

    import numpy as np
    # take last week of the dataset for validation
    X_train, X_test = X.iloc[:-horizon, :], X.iloc[-horizon:, :]
    y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]
    # X_train, X_test = X.loc[:-horizon, :], X.iloc[-horizon:, :]
    # y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]

    # iso = IsolationForest(contamination=0.1)
    # yhat = iso.fit_predict(X_train)
    #
    # mask = yhat != -1
    # X_train, y_train = X_train[mask, :], y_train[mask]
    # # summarize the shape of the updated training dataset
    # print(X_train.shape, y_train.shape)

    # from sklearn.preprocessing import StandardScaler
    from merf import MERF
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test=scaler.fit_transform(X_test)
    # X_train, X_test = X[:-horizon, :], X[-horizon:, :]
    # y_train, y_test = y[:-horizon], y[-horizon:]
    from sklearn.decomposition import PCA

    # pca = PCA(n_components=20)
    #
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)
    from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
    # mtf = MarkovTransitionField()
    # img = mtf.fit_transform(X_train)
    # plt.imshow(img[1])
    # plt.set_cmap("gist_rainbow")
    # plt.show()

    # define LightGBM model, train it and make predictions
    from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
    from sklearn.linear_model import ElasticNet
    from xgboost import XGBRegressor
    import catboost as cb
    from keras.models import Sequential, load_model
    from matplotlib import pyplot
    from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
    from sklearn import metrics
    from sklearn.metrics import mean_absolute_error
    from bayes_opt import BayesianOptimization
    from sklearn.metrics import make_scorer, mean_squared_error
    # my_scorer = make_scorer(rms, greater_is_better=False)

    from sklearn.model_selection import GridSearchCV, cross_val_score

    # Converting the max_depth and n_estimator values from float to int
    from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor
    from sklearn.gaussian_process import GaussianProcessClassifier,GaussianProcessRegressor
    from sklearn.linear_model import LinearRegression
    from classifier import CascadeForestRegressor
    # model=GradientBoostingRegressor(min_samples_split = 500,max_depth=8,max_features='sqrt',subsample=0.8)
    # model=GaussianProcessRegressor()
    # model = AdaBoostRegressor()
    # model = ltb.LGBMRegressor(random_state=42)
    from sklearn.feature_selection import SelectFromModel
    # model = ltb.LGBMRegressor(random_state=42)
    import shap
    # model=Prophet()
    # model= RandomForestRegressor(n_estimators=40)
    # model=LinearRegression()
    # model=ElasticNet()
    sample_size = X_train.shape[0]  # number of samples in train set
    time_steps = X_train.shape[1]
    # train_data_reshaped = X_train.values.reshape(sample_size, time_steps, 1)
    # test_data_reshaped = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
    # n_timesteps = train_data_reshaped.shape[1]  # 13
    # n_features = train_data_reshaped.shape[2]
    # model1 = Sequential(name="model1_conv1D")
    # # model.add(keras.layers.Input(shape=(n_timesteps, 16)))
    # model1.add(Conv1D(64, 2, activation="relu", input_shape=(29, 1)))
    # model1.add(Conv1D(filters=32, kernel_size=7, activation='relu', name="Conv1D_1"))
    # # model.add(Dropout(0.5))
    # model1.add(Conv1D(filters=16, kernel_size=3, activation='relu', name="Conv1D_2"))
    # # model.add(MaxPooling1D(pool_size=2, name="MaxPooling2D"))
    # model1.add(Conv1D(filters=32, kernel_size=2, activation='relu', name="Conv1D_4"))
    # model1.add(Conv1D(filters=32, kernel_size=4, activation='relu', name="Conv1D_5"))
    # # model.add(Conv1D(filters=32, kernel_size=4, activation='relu', name="Conv1D_6"))
    # # model1.add(MaxPooling1D(pool_size=2, name="MaxPooling1D"))
    # model1.add(Flatten())
    # # model.add(Dense(64, activation='relu', name="Dense_1"))
    # model1.add(Dense(1, name="Dense_2"))
    # model1.summary()
    # model=XGBRegressor()
    # model=cb.CatBoostRegressor()
    # model=RandomForestRegressor(random_state=0)
    # model = CascadeForestRegressor(random_state=1)
    # model=MLPRegressor(hidden_layer_sizes=(20, 20,30), alpha=1e-8, random_state=1, max_iter=300, warm_start=True,
    #              solver='adam', verbose=10, tol=1e-8, learning_rate_init=.001, activation='relu')
    # # model=MERF()
    # # model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    # define model
    from keras.optimizers import SGD
    model = Sequential()
    model.add(Dense(20, input_dim=26, activation='relu'))
    model.add(Dense(20))
    model.add(Dense(30))
    model.add(Dense(1, activation='relu'))
    # opt = (lr=0.01, momentum=0.9)
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    n_epochs = 200
    n_cycles = n_epochs / 10
    ca = SnapshotEnsemble(n_epochs, n_cycles, 0.001)
    # fit model
    history= model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=n_epochs, verbose=0, callbacks=[ca])
    explainer = shap.Explainer(model.predict, X_test)
    shap_values = explainer(X_test)

    # shap.summary_plot(shap_values)
    # or
    shap.plots.beeswarm(shap_values)
    plt.show()
    _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    # learning curves of model accuracy
    # pyplot.plot(history.history['loss'], label='train')
    # pyplot.plot(history.history['val_loss'], label='test')
    # pyplot.legend()
    # pyplot.show()
    # history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, verbose=0)
    #
    # model.fit(X_train, y_train)
    # # model1.fit(X_train, y_train)
    # # model.fit(train_data_reshaped, y_train)
    predictions = model.predict(X_test)
    # # predictions=model.predict(X_test.loc[["2003","2004","2005"]])
    # print(predictions.shape)
    # print(y_test.shape)
    from numpy import mean
    from numpy import std
    from matplotlib import pyplot

    members = load_all_models(10)
    print('Loaded %d models' % len(members))
    # reverse loaded models so we build the ensemble with the last models first
    members = list(reversed(members))
    # evaluate different numbers of ensembles on hold out set
    single_scores, ensemble_scores = list(), list()
    for i in range(1, len(members) + 1):
        # evaluate model with i members
        ensemble_score = evaluate_n_members(members, i, X_test, y_test)
        # evaluate the i'th model standalone
        # testy_enc = to_categorical(testy)
        _, single_score = members[i - 1].evaluate(X_test, y_test, verbose=0)
        # summarize this step
        print('> %d: single=%.3f, ensemble=%.3f' % (i, single_score, ensemble_score))
        ensemble_scores.append(ensemble_score)
        single_scores.append(single_score)
    # summarize average accuracy of a single final model
    print('Accuracy %.3f (%.3f)' % (mean(single_scores), std(single_scores)))
    # plot score vs number of ensemble members
    x_axis = [i for i in range(1, len(members) + 1)]
    pyplot.plot(x_axis, single_scores, marker='o', linestyle='None')
    pyplot.plot(x_axis, ensemble_scores, marker='o')
    pyplot.show()

    pyplot.plot(ca.lrates)
    pyplot.show()

    import numpy as np
    # calculate MAE
    from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,r2_score
    predictions=predictions.flatten()

    # y_test=y_test.loc[["2003","2004","2005"]]
    mae = np.round(mean_absolute_error(y_test, predictions), 3)
    mape = np.round(mean_absolute_percentage_error(y_test, predictions), 3)
    from math import sqrt
    rmse=sqrt(mean_squared_error(y_test,predictions))
    r2=r2_score(y_test,predictions)
    print("MAE: ",mae)
    print("MAPE: ", mape)
    print("RMSE:",rmse)
    print("R2:", r2)


    # plot reality vs prediction for the last week of the dataset
    fig = plt.figure(figsize=(16, 6))
    # plt.title(f'Real vs Prediction - MAE {mae}', fontsize=20)
    plt.plot(y_test, color='red')
    plt.plot(pd.Series(predictions, index=y_test.index), color='green')
    # plt.plot(predictions, y_test.index)
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Co2resp', fontsize=16)
    plt.legend(labels=['Real', 'Prediction'], fontsize=16)
    plt.grid()
    plt.show()

    # plt.plot(df.index, df['CO2resp'], )
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # # test_CH4.plot(label="Real",color="blue")
    # y_test.plot(label="Actual", color="blue")
    # predictions.plot(label="Predicted", color="red")
    # plt.legend()
    # plt.show()

train_time_series_with_folds_autoreg_prophet_features(df, horizon=1000,lags=[1,2,3,4,5])