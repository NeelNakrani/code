from typing import Any

import numpy as np
from numpy import ndarray

from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

default_models = [
    SVR(kernel='rbf'),
    MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, learning_rate='adaptive'),
    RandomForestRegressor(n_estimators=100),
    GradientBoostingRegressor(n_estimators=100)
]


class SMLE:
    weights: ndarray | Any

    def __int__(self, base_models: default_models, no_tscv_split: 4, alpha: 0.1):
        self.models = base_models
        self.no_tscv_split = no_tscv_split

        # to keep track of errors for each model in each iteration for training phase
        self.errors = np.zeros((len(self.models), self.no_tscv_split))

        self.weights = np.zeros((len(self.models),))

        self.alpha = alpha

    def fit(self, x, y):
        tscv = TimeSeriesSplit(n_splits=self.no_tscv_split)

        for i, model in enumerate(self.models):
            for j, (train_idx, val_idx) in enumerate(tscv.split(x)):
                x_train, y_train = x[train_idx], y[train_idx]
                x_val, y_val = x[val_idx], y[val_idx]
                model.fit(x_train, y_train)
                y_pred = model.predict(x_val)
                self.errors[i, j] = mean_squared_error(y_val, y_pred)

        # # normalize errors
        # self.errors = self.errors / np.sum(self.errors, axis=0)

        # soft selection algorithm
        weights = np.zeros((len(self.models),))
        # initially generate weights like [1, 0, 0, ...]
        weights[0] = 1
        for j in range(1, self.no_tscv_split):
            weights = self._update_weights(weights, j)
        self.weights = weights

    def _update_weights(self, weights, j):
        grad = np.zeros_like(weights)
        for i in range(len(self.models)):
            grad[i] = np.sum(self.errors[i, j-1:j+1] * weights[i])
        weights = weights - self.alpha * grad
        weights[weights < 0] = 0
        return weights

    def predict(self, x):
        y_pred = np.zeros((len(x),))
        for i, model in enumerate(self.models):
            y_pred += self.weights[i] * model.predict(x)
        return np.array(y_pred).T
