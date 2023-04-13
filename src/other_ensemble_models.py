from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from smle import default_models


def bagging_regressor(x_train, y_train):
    bagging_model = BaggingRegressor(base_estimator=LinearRegression(), n_estimators=4, random_state=42)
    bagging_model.fit(x_train, y_train)
    return bagging_model


def gradient_boosting_model(x_train, y_train):
    boosting_model = GradientBoostingRegressor(n_estimators=4, random_state=42)
    boosting_model.fit(x_train, y_train)
    return boosting_model


def votting_regressor(x_train, y_train):
    votting_reg = VotingRegressor(estimators=[("svr", default_models[0]), ("MLPR", default_models[1]), ("rf", default_models[2]), ("GBR", default_models[3])])
    votting_reg.fit(x_train, y_train)
    return votting_reg


def train_ensemble_with_grid_search_cv(x_train, y_train):
    tscv = TimeSeriesSplit(n_splits=5)
    base_model = RandomForestRegressor(random_state=42)
    ensemble_model = BaggingRegressor(estimator=base_model, random_state=42)
    # Hyperparameter tuning
    param_grid = {
        'estimator__n_estimators': [10, 50, 100],
        'base_estimator__max_depth': [None, 10, 20],
        'n_estimators': [10, 20],
        'max_samples': [0.5, 1.0],
        'max_features': [0.5, 1.0],
    }

    grid_search = GridSearchCV(ensemble_model, param_grid, scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1)
    grid_search.fit(x_train, y_train)

    # Best parameters
    best_params = grid_search.best_params_
    print("Best Parameters: ", best_params)

    # Train and evaluate the final ensemble model
    best_ensemble_model = grid_search.best_estimator_
    best_ensemble_model.fit(x_train, y_train)
    return best_ensemble_model


# y_pred = best_ensemble_model.predict(X)
# mse = mean_squared_error(y, y_pred)
# print("Mean Squared Error: ", mse)
