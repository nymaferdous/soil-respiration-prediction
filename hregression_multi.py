import numpy as np
# manipulating data via DataFrames, 2-D tabular, column-oriented data structure
import pandas as pd
from pandas import read_excel
from sklearn.manifold import TSNE
my_sheet = 'FinalAnalysis_forManuscript'
file_name = 'Data_for_ML_crop_residues_assessment.xlsx'
from math import sqrt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
# producing plots and other 2D data visualizations. Use plotly if you want interactive graphs
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
# statistical visualizations (a wrapper around Matplotlib)
import seaborn as sns
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from statistics import mean
from deepforest import CascadeForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn import linear_model
import joblib
from gradient_free_optimizers import HillClimbingOptimizer,LipschitzOptimizer,EvolutionStrategyOptimizer
from swarmlib import fireflyalgorithm,FireflyProblem
from sklearn.multioutput import MultiOutputRegressor

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
warnings.filterwarnings("ignore")
sns.set(style="white", color_codes=True)

# Assign the csv data to a DataFrame
# data = pd.read_csv("dc_sip.csv")

data = pd.DataFrame(pd.read_excel("modelC.xlsx"))
# df = read_excel("Data_for_ML_crop_residues_assessment.xlsx", sheet_name = 'FinalAnalysis_forManuscript')
# print(df.head())

# # Plot a histogram of SepalLength Frequency on Species (matplotlib)
# Iris_setosa = data[data["Species"] == "Iris-setosa"]
# Iris_versicolor = data[data["Species"] == "Iris-versicolor"]
# Iris_virginica = data[data["Species"] == "Iris-virginica"]
#
# Iris_setosa["SepalLengthCm"].plot.hist(alpha=0.5,color='blue',bins=50) # Setting the opacity(alpha value) & setting the bar width((bins value)
# Iris_versicolor["SepalLengthCm"].plot.hist(alpha=0.5,color='green',bins=50)
# Iris_virginica["SepalLengthCm"].plot.hist(alpha=0.5,color='red',bins=50)
# plt.legend(['Iris-setosa','Iris_versicolor','Iris-virginica'])
# plt.xlabel('SepalLengthCm')
# plt.ylabel('Frequency')
# plt.title('SepalLength on Species')
# plt.show()
#
# from sklearn.preprocessing import LabelEncoder
#
# labelencoder = LabelEncoder()
# data["Species"] = labelencoder.fit_transform(data["Species"])
# # data["Species"]
# # Construct a dataframe from a dictionary
# species = pd.DataFrame({'Species': ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']})
# X1=pd.get_dummies(data, columns=["Code","muname","areasymbol","areaname","Crop Rotations"])
# X = X1[['Removal Rate','Sand(%)','K_factor','Crop Rotations_CSW','Crop Rotations_CS','Corn Yield(bu/Ha)','Soybean Yield(bu/Ha)','Slope(%)','SlopeLength(m)',
# 'T_factor','K_factor','Sand(%)','Silt(%)','Clay(%)','Organic matter(%)','Rainfall_Erosivity']]

X = data[['day','month','year']]

# y = data['totsysc']
y=data['aglivc']
# y = data[['no_totsysc', 'w_totsysc','HN_totsysc','N_totsysc']]


class Stats:

    def __init__(self, X, y, model):
        self.data = X
        self.target = y
        self.model = model
        ## degrees of freedom population dep. variable variance
        self._dft = X.shape[0] - 1
        ## degrees of freedom population error variance
        self._dfe = X.shape[0] - X.shape[1] - 1

    def sse(self):
        '''returns sum of squared errors (model vs actual)'''
        squared_errors = (self.target - self.model.predict(self.data)) ** 2
        return np.sum(squared_errors)

    def sst(self):
        '''returns total sum of squared errors (actual vs avg(actual))'''
        avg_y = np.mean(self.target)
        squared_errors = (self.target - avg_y) ** 2
        return np.sum(squared_errors)

    def r_squared(self):
        '''returns calculated value of r^2'''
        return 1 - self.sse() / self.sst()

    def adj_r_squared(self):
        '''returns calculated value of adjusted r^2'''
        return 1 - (self.sse() / self._dfe) / (self.sst() / self._dft)

def pretty_print_stats(stats_obj):
    '''returns report of statistics for a given model object'''
    items = ( ('sse:', stats_obj.sse()), ('sst:', stats_obj.sst()),
             ('r^2:', stats_obj.r_squared()), ('adj_r^2:', stats_obj.adj_r_squared()) )
    for item in items:
        print('{0:8} {1:.4f}'.format(item[0], item[1]))


# X1 = data[['Code', 'areasymbol']].values
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(categorical_features=[0])
# ohe.fit_transform(X1).toarray()
# data["Code"] = data["Code"].astype('category')
# data["Code"] = data["Code"].cat.codes
# print(data.head())

#
# # Sample the train data set while holding out 20% for testing (evaluating) the classifier
from sklearn.model_selection import train_test_split
from sklearn import metrics,model_selection

from sklearn.metrics import accuracy_score
# X_train =X[X['time'].isin(1990,2050)]
# X_test = X[~X['time'].isin(1990,2050)]
# y_train=y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True)

from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# pca = PCA(n_components=14)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))

from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
# scaler.fit(y_train)
#y_train = scaler.transform(y_train)
# kernel = DotProduct()
# model=LinearRegression()


# def model(para):
#     gbc = RandomForestRegressor(
#         n_estimators=para["n_estimators"],
#         max_depth=para["max_depth"],
#         min_samples_split=para["min_samples_split"],
#         min_samples_leaf=para["min_samples_leaf"],
#     )
#     # gbc=CascadeForestRegressor(n_trees=para["n_trees"],
#     #                                max_depth=para["max_depth"],
#     #                                min_samples_split=para["min_samples_split"],
#     #                                min_samples_leaf=para["min_samples_leaf"]
#     #                            )
#
#
#     gbc.fit(X_train, y_train)
#     # model.summary()
#     #
#     # joblib.dump(model, "rf_model.pkl")
#     # predictions1= model.predict(X_train)
#     # model1 = joblib.load('rf_model.pkl')
#     predictions = gbc.predict(X_test)
#     mae = metrics.mean_absolute_error(y_test, predictions)
#     print("MAE", mae)
#     print("RMSE", sqrt(metrics.mean_squared_error(y_test, predictions)))
#     r2 = metrics.r2_score(y_test, predictions)
#     print("R2", r2)
#     # scores = cross_val_score(gbc, X, y, cv=3)
#
#     return mae
#     return scores.mean()

# search_space = {
#     "n_trees": np.arange(40, 150, 2),
#     "max_depth": np.arange(2, 120, 1),
#     "min_samples_split": np.arange(2, 12, 1),
#     "min_samples_leaf": np.arange(1, 12, 1),
# }

# opt = LipschitzOptimizer(search_space)
# opt.search(model, n_iter=50)
# ndim_problem = 3
# problem = {'fitness_function': model,  # cost function
#            'ndim_problem': ndim_problem,  # dimension
#            'lower_boundary': -5.0*np.ones((ndim_problem,)),  # search boundary
#            'upper_boundary': 5.0*np.ones((ndim_problem,))}

# 3. Run one or more black-box optimizers on the given optimization problem:
#   here we choose LM-MA-ES owing to its low complexity and metric-learning ability for LSO
#   https://pypop.readthedocs.io/en/latest/es/lmmaes.html
# from pypop7.optimizers.es.lmmaes import LMMAES
# # define all the necessary algorithm options (which differ among different optimizers)
# options = {'fitness_threshold': 1e-10,  # terminate when the best-so-far fitness is lower than this threshold
#            'max_runtime': 3600,  # 1 hours (terminate when the actual runtime exceeds it)
#            'seed_rng': 0,  # seed of random number generation (which must be explicitly set for repeatability)
#            'x': 4.0*np.ones((ndim_problem,)),  # initial mean of search (mutation/sampling) distribution
#            'sigma': 3.0,  # initial global step-size of search distribution (not necessarily optimal)
#            'verbose': 500}
# lmmaes = LMMAES(problem, options)  # initialize the optimizer
# results = lmmaes.optimize()
# model=CascadeForestRegressor()
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ExpSineSquared

# seasonal_kernel = (
#     2.0**2
#     * RBF(length_scale=100.0)
#     * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds="fixed")
# )
# from sklearn.gaussian_process.kernels import RationalQuadratic
#
# irregularities_kernel = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
# from sklearn.gaussian_process.kernels import WhiteKernel
#
# noise_kernel = 0.1**2 * RBF(length_scale=0.1) + WhiteKernel(
#     noise_level=0.1**2, noise_level_bounds=(1e-5, 1e5)
# )
# long_term_trend_kernel = 50.0**2 * RBF(length_scale=50.0)
# kernel = (
#     long_term_trend_kernel + seasonal_kernel + irregularities_kernel + noise_kernel
# )
# # model=XGBRegressor()
# model=linear_model.Ridge()
# model= linear_model.BayesianRidge()
# model = GradientBoostingRegressor()
# model = GridSearchCV(svm.SVR(),param_grid,refit=True,verbose=2)
# model=KNeighborsRegressor()
model=RandomForestRegressor()
# model =  MultiOutputRegressor(RandomForestRegressor())
# model=DecisionTreeRegressor()
# model=SVR()
# model = MLPRegressor(hidden_layer_sizes=(20,30,60), alpha=1e-8, random_state=1, max_iter=300, warm_start=True,solver='adam', verbose=10, tol=1e-8,
#                     learning_rate_init=.001,activation='relu')
# model = GaussianProcessRegressor(kernel=kernel).fit(X, y)

model.fit(X_train,y_train)
# model.summary()
#
# joblib.dump(model, "rf_model.pkl")
# predictions1= model.predict(X_train)
# model1 = joblib.load('rf_model.pkl')
predictions= model.predict(X_test)
# print(X_test.size)
# print(y_test.size)


from sklearn.metrics import mean_squared_error

# writer = pd.ExcelWriter('soilcarbon.xlsx',  engine='xlsxwriter')

# CSV4 = pd.DataFrame({
#     "time": X_test['time'],
#     # "month":X_test['month'],
#     # "year":X_test['year'],
#     # "dsrfclit": X_test["dsrfclit"],
#     "Actual": y_test,
#     "Predicted":predictions
# })
# CSV4.to_excel(writer, sheet_name='Sheet1')
# writer.close()
## residuals
# residuals = y_test - predictions
# max_error = max(residuals) if abs(max(residuals)) > abs(min(residuals)) else min(residuals)
# max_idx = list(residuals).index(max(residuals)) if abs(max(residuals)) > abs(min(residuals)) else list(residuals).index(min(residuals))
# max_true, max_pred = y_test[max_idx], predictions[max_idx]
# print("Max Error:", "{:,.0f}".format(max_error))

# fig, ax = plt.subplots()
# ax.scatter(predictions, y_test, edgecolors=(0, 0, 1))
# ax.plot([y_test.min(), y_test.max()], [predictions.min(), predictions.max()], 'r--', lw=2)
# ax.set_xlabel('Actual')
# ax.set_ylabel('Predicted')
# plt.show()

import seaborn as sns
# plt.figure(figsize=(5, 7))


# ax = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
# sns.distplot(predictions, hist=False, color="b", label="Fitted Values" , ax=ax)
#
#
# plt.title('Actual vs Fitted Values for NEE')
#
#
# plt.show()
# plt.close()
# X_grid = np.arange(min(X_value), max(X_value), 0.01)
# X_grid = X_grid.reshape((len(X_grid), 1))
# plt.scatter(X_test, y_test, color = 'red')
# plt.scatter(X_test, predictions, color = 'green')
# plt.title('Random Forest Regression')
# plt.xlabel('Temperature')
# plt.ylabel('Revenue')
# plt.show()
print(np.average(predictions))
print(np.average(y_test.to_numpy()))
mae = metrics.mean_absolute_error(y_test, predictions)
# mae1 = metrics.mean_absolute_error(y_train, predictions1)
print("RMSE",sqrt(mean_squared_error(y_test, predictions)))
# print("RMSE Train",sqrt(mean_squared_error(y_train, predictions1)))
mse = metrics.mean_squared_error(y_test, predictions)
# mse1 = metrics.mean_squared_error(y_train, predictions1)
r2 = metrics.r2_score(y_test, predictions)
# r2_train = metrics.r2_score(y_train, predictions1)
print("MAE",mae)
# print("MAE Train",mae1)
# print("R2 Train",r2_train)
print("R2",r2)
# #
print(y_test)
print(predictions.shape)
# CSV4 = pd.DataFrame({
#     "day":X_test['day'],
#     "month":X_test['month'],
#     "year":X_test['year'],
#     # "dsmnrl": X_test["dsrfclit"],
#     # "Observed": y_test['no_totsysc'],
#     "no_Predicted":predictions[:,0],
#     "W_Predicted":predictions[:,1],
#     "HN_Predicted":predictions[:,2],
#     "N_Predicted": predictions[:, 3],
# })
# writer = pd.ExcelWriter('aglivc_harvard.xlsx', engine='xlsxwriter')
# CSV4.to_excel(writer, sheet_name='Sheet1')
# writer.save()


# print("MSE",mse)
# print("MSE train",mse1)

# stats = Stats(X_train, y_train, model)
# pretty_print_stats(stats)


