import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import lightgbm as ltb
from sklearn.neural_network import MLPRegressor
import pandas as pd
import shap

# inputExcelFile ="oakridge_highwest_soilp.xlsx"
#
# excelFile = pd.read_excel (inputExcelFile)
#
# excelFile.to_csv ("oakridge_highwest_soilp.csv", index = None, header=True)

data = pd.read_csv("oakridge_orloweast_soilp.csv")

# df = pd.DataFrame(pd.read_csv("oakridge_orloweast_soilp.csv",header=0,encoding='utf-8'))
# df=data[['atemp''sresp','stemp','slitresp','Precipitation','CH4','year','month','day','CO2resp']]
# df=df[['ATemp','msresp','sresp','stemp','slitresp','Precipitation','CH4','NIT','N2OFlux','NOFlux', 'CO2resp','day','month','year']]
df=data[['atemp','stemp','pdsmnrl','pdsrfclit','dsoilresp']]
# df1=data[['day','month','year','atemp','stemp','prec','pdsmnrl','pdsrfclit']]
# df=data[['pdsoilresp','normalized_pdsoilresp','nsresp']]
# df=df[['ATemp','stemp','Precipitation ','sresp','slitrsp','sminrlrsp','day','month','year']]
import matplotlib.pyplot as mpl
mpl.rcParams['font.size'] = 20
# df1.hist()
# plt.show()


import seaborn as sns
# data.nsresp.plot.density(color='green')
# data.normalized_pdsoilresp.plot.density(color='blue')
# data.pdsoilresp.plot.density(color='magenta')
# plt.title('Density plot ')
# for column in df1:
#     sns.boxplot(x=column, data=df1, whis=1)
#     plt.tight_layout()
#     # plt.show()
# plt.show()
# df2=data[['atemp','stemp','prec','pdsmnrl','pdsrfclit']]
# fig, ax = plt.subplots()
# ax.violinplot(df2, showmeans=False, showmedians=False)
# ax.set_xlabel('Input Features')
# ax.set_ylabel('Value')
# xticklabels = ['atemp', 'stemp','prec','pdsmnrl','pdsrfclit']
# ax.set_xticks([1,2,3,4,5])
# ax.set_xticklabels(xticklabels)
# ax.yaxis.grid(True)
# plt.show()
# plt.hist(df['pdsoilresp'],alpha=0.4,
#          label='Predicted')
#
# plt.hist(data['ndsoilresp'],alpha=0.4,
#          label='Actual')
#
# plt.hist(data['nsresp'],alpha=0.4,
#          label='Observed')
#
# plt.legend(loc='upper right')
# plt.title('Overlapping')
# plt.show()

from sklearn.ensemble import IsolationForest

##data Visualization
# numerical_column= ["ATemp","msresp","sresp","stemp","slitresp","Precipitation",'CH4','NIT','N2OFlux','NOFlux','CO2resp']
import seaborn as sns

# plt.figure(figsize=(18,11))
# for i,j in zip(range(1, 11),numerical_column):
#     plt.subplot(3, 4, i)
#     sns.boxplot(data=df, x=j)
#     # sns.set_theme()
#     # plt.title('{}'.format(j))
# plt.show()

# def drop_outliers_IQR(df):
#
#    q1=df.quantile(0.25)
#
#    q3=df.quantile(0.75)
#
#    IQR=q3-q1
#
#    not_outliers = df[~((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]
#
#    outliers_dropped = not_outliers.dropna().reset_index()
#
#    return outliers_dropped



# df=df[['ATemp','msresp','sresp','stemp','slitresp','Precipitation','CO2resp','day','month','year']]
# df=df[['ATemp','msresp','stemp','Precipitation','year','month','day']]
data['datetime']=pd.to_datetime(data[['year', 'month', 'day']])
# df['datetime']=pd.to_datetime(data[['year', 'month', 'day']])
# df_monthly=df.resample('M').sum()
def prophet_features(df, horizon=100):
    # temp_df=df_monthly.reset_index()
    temp_df = df.reset_index()
    temp_df = temp_df[['datetime', 'dsoilresp']]
    # print(df['datetime'])
    temp_df.rename(columns={'datetime': 'ds', 'dsoilresp': 'y'}, inplace=True)


    # take last week of the dataset for validation
    train, test = temp_df.iloc[:-horizon, :], temp_df.iloc[-horizon:, :]
    print(test)

    # define prophet model
    m = Prophet(
        growth='linear',
        seasonality_mode='additive',
        interval_width=0.95,
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
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
    # print(predictions_train)
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


def train_time_series_with_folds_autoreg_prophet_features(df, horizon=1900, lags=[1,2,3,4,5]):
    # create a dataframe with all the new features created with Prophet
    new_prophet_features,prophet_test_features = prophet_features(df, horizon=horizon)
    # print(df.columns)
    # print(new_prophet_features.columns)
    # new_prophet_features, prophet_test_features = prophet_features(df_monthly, horizon=horizon)
    # df.reset_index(inplace=True)
    # df_monthly.reset_index(inplace=True)
    from sklearn.preprocessing import StandardScaler

    # merge the Prophet features dataframe with the our first dataframe
    df = pd.merge(df, new_prophet_features, left_on=['datetime'], right_on=['ds'], how='inner')
    # print(df.columns)

    # df = pd.merge(df_monthly, new_prophet_features, left_on=['datetime'], right_on=['ds'], how='inner')
    # print(df.columns)
    # df = pd.merge(df_monthly, new_prophet_features, how='inner')

    # df = pd.merge(df, new_prophet_features, how='inner')



    # scaler = StandardScaler()
    # df = scaler.fit_transform(df_series)
    # print(df)

    # df.drop('ds', axis=1, inplace=True)
    df.set_index('datetime', inplace=True)
    # print (len(df.columns))
    #
    for lag in lags:
        df[f'yhat_lag_{lag}'] = df['yhat'].shift(lag)

    # df_temp= df.loc[:, df.columns != df['index']]
    # print(df_temp)
    # df= drop_outliers_IQR(df)


    df.dropna(axis=0, how='any')
    # print(df)

    df = df.fillna(0)
    df = df.apply(lambda x: pd.factorize(x)[0])
    # X= df.drop('additive_terms_lower','multiplicative_terms')
    X = df.drop('dsoilresp', axis=1)
    # X = X[[
    #     'stemp', 'atemp', 'prec', 'pdsrfclit', 'pdsmnrl',
    #     'ds', 'trend', 'yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper',
    #     'additive_terms', 'additive_terms_lower', 'additive_terms_upper',
    #     'yhat']]
    # print(df)
    print(X.columns)
    # print(X['yhat'])
    y = df['dsoilresp']
    # print("X column",X.columns)
    # print(df)


    # import numpy as np
    # import seaborn as sns
    #
    # sns.set(font_scale=1.4)
    # corr_matrix = df[['maxt','mint','atemp','stemp','prec','dsoilresp']].corr().abs()
    # data_wide = df.pivot(columns='Yearc',
    #                      values='sresp')
    # data_wide.head()
    # data_wide.plot.density(figsize=(7, 7),
    #                        linewidth=4)

    # plt.xlabel("life_Exp")
    # plt.show()

    # corr_matrix=df.corr().abs()
    # Drop features
    # sns.heatmap(corr_matrix, annot=True, fmt='.2f')
    # plt.show()
    # upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # # Find features with correlation greater than 0.95
    # to_drop = [column for column in upper.columns if any(upper[column] > 0.5)]
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
    # from classifier import CascadeForestRegressor
    # # model=GradientBoostingRegressor(min_samples_split = 500,max_depth=8,max_features='sqrt',subsample=0.8)
    # model=GaussianProcessRegressor()
    # model = AdaBoostRegressor()
    # model = ltb.LGBMRegressor(random_state=42)
    from sklearn.feature_selection import SelectFromModel
    # model = ltb.LGBMRegressor(random_state=42)

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
    # model1.add(Conv1D(64, 2, activation="relu", input_shape=(21, 1)))
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
    model=MLPRegressor(hidden_layer_sizes=(150, 100,20), alpha=1e-8, random_state=1, max_iter=300, warm_start=True,
                 solver='adam', verbose=10, tol=1e-8, learning_rate_init=.001, activation='relu')
    # model=MERF()
    # model1.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    #
    # model1.fit(X_train, y_train)
    model.fit(X_train, y_train)
    # print()
    # model.fit(train_data_reshaped, y_train)
    predictions = model.predict(X_test)
    print(predictions)
    predictions1=model.predict(X_train)
    # print(df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
    # predictions=model.predict(X_test.loc[["2003","2004","2005"]])
    # print(predictions.shape)
    # print(y_test.shape)


    import numpy as np
    # calculate MAE
    from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,r2_score
    predictions=predictions.flatten()
    predictions1 = predictions1.flatten()
    # X_test=X_test.flatten()
    CSV = pd.DataFrame({
        "Prediction": y_test,

    })
    # CSV = pd.DataFrame({
    #     "Prediction": predictions,
    #
    # })
    CSV1=pd.DataFrame({"Y_train":y_train})
    # CSV2=pd.DataFrame({"Ytrain":y_train})
    # CSV2.to_csv("Ytrain.csv", index=False)
    # CSV3=pd.DataFrame({"pytrain":predictions1})
    # CSV = pd.DataFrame({
    #     "Prediction": predictions
    # })

    CSV.to_csv("ftest.csv", index=False)
    CSV1.to_csv("ftrain.csv", index=False)
    # CSV3.to_csv("fprediction_train.csv", index=False)
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
    fig = plt.figure(figsize=(10, 6))
    # plt.rcParams.update({
    #     # "figure.facecolor": (1.0, 0.0, 0.0, 0.3),  # red   with alpha = 30%
    #     # "axes.facecolor": (0.0, 1.0, 0.0, 0.5),  # green with alpha = 50%
    #     "axes.facecolor": (0.0, 0.0, 1.0, 0.2),  # blue  with alpha = 20%
    # })
    # plt.title(f'Real vs Prediction - MAE {mae}', fontsize=20)
    ax=plt.axes()
    # ax.set_facecolor((0.0, 0.0, 1.0, 0.01))
    # ax.set_alpha(0.8)
    plt.plot(y_test, color='red')
    plt.plot(pd.Series(predictions, index=y_test.index), color='blue')
    # plt.plot(predictions, y_test.index)
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('CO2flux', fontsize=16)
    plt.legend(labels=['Actual', 'Predicted'], fontsize=16)
    # plt.grid()
    plt.show()

    # feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    # feat_importances.nlargest(20).plot(kind='barh')
    # plt.show()

    # explainer = shap.Explainer(model.predict, X_test)
    # shap_values = explainer(X_test)
    # #
    # shap.summary_plot(shap_values)
    # # # # or
    # # # # shap.plots.beeswarm(shap_values)
    # plt.show()

    # feature_importance = model.feature_importances_
    # sorted_idx = np.argsort(feature_importance)
    # fig = plt.figure(figsize=(12, 6))
    # plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    # plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
    # plt.title('Feature Importance')

    # plt.plot(df.index, df['CO2resp'], )
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # # test_CH4.plot(label="Real",color="blue")
    # y_test.plot(label="Actual", color="blue")
    # predictions.plot(label="Predicted", color="red")
    # plt.legend()
    # plt.show()

train_time_series_with_folds_autoreg_prophet_features(data, horizon=1900,lags=[1,2,3,4,5])