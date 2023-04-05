import pandas as pd
from src.preprocessing import Preprocessor
from src.smle import SMLE
from sklearn.metrics import mean_squared_error


def preprocess_data():
    raw_data = pd.read_csv('../data/GlobalLandTemperaturesByCity.csv')
    preprocessor = Preprocessor(raw_data)
    data = preprocessor.data_clean_up()
    return preprocessor.train_test_split_and_add_lags()


def train_model(x, y):
    model = SMLE()
    model.fit(x, y)
    return model


def main():
    x_train, x_test, y_train, y_test = preprocess_data()
    model = train_model(x_train, y_train)
    predictions = model.predict(x_test)
    print(mean_squared_error(y_true=y_test, y_pred=predictions))


if __name__ == '__main__':
    main()
