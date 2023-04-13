import pandas as pd
from src.preprocessing import Preprocessor
from src.smle import SMLE, default_models, models_with_standard_scaler
from sklearn.metrics import mean_squared_error
from other_ensemble_models import train_ensemble_with_grid_search_cv, bagging_regressor, gradient_boosting_model, votting_regressor
from simple_models import train_SVR, train_MLPRegressor, train_random_forest
import warnings

def preprocess_data():
    raw_data = pd.read_csv('../data/GlobalLandTemperaturesByCity.csv')
    preprocessor = Preprocessor(raw_data)
    data = preprocessor.data_clean_up()
    return preprocessor.train_test_split_and_add_lags(lags=4)


def train_model(model, x, y):
    print("model training started for: ", model)
    model.fit(x, y)
    return model


def predict(model, x_test, y_test):
    preds = model.predict(x_test)
    print("RMSE of model ", model, " is : ", mean_squared_error(y_test, preds, squared=False))


def main():
    print("Preprocessing started: ")
    x_train, x_test, y_train, y_test = preprocess_data()
    model = train_model(SMLE(base_models=models_with_standard_scaler, no_tscv_split=40, alpha=0.01), x_train, y_train)
    predict(model, x_test, y_test)

    svr = train_SVR(x_train, y_train)
    predict(svr, x_test, y_test)

    mlp = train_MLPRegressor(x_train, y_train)
    predict(mlp, x_test, y_test)

    rf = train_random_forest(x_train, y_train)
    predict(rf, x_test, y_test)
    # smle_preds = model.predict(x_test)
    # print("MSE of SMLE: ", mean_squared_error(y_true=y_test, y_pred=smle_preds))
    bagging_model = bagging_regressor(x_train, y_train)
    predict(bagging_model, x_test, y_test)

    votting_model = votting_regressor(x_train, y_train)
    predict(votting_model, x_test, y_test)

    gb_model = gradient_boosting_model(x_train, y_train)
    predict(gb_model, x_test, y_test)

    optimized_bagging_model = train_ensemble_with_grid_search_cv(x_train, y_train)
    predict(optimized_bagging_model, x_test, y_test)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
