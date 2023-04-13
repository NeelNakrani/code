from sklearn.svm import SVR
from sklearn.pipeline import  make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def train_SVR(x_train, y_train):
    svr = SVR(kernel='rbf')
    svr.fit(x_train, y_train)
    return svr


def train_MLPRegressor(x_train, y_train):
    mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, learning_rate='adaptive')
    mlp.fit(x_train, y_train)
    return mlp


def train_random_forest(x_train, y_train):
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(x_train, y_train)
    return rf
