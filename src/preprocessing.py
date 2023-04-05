import numpy as np
import pandas as pd

TEMPERATURE_COLUMN = 'AverageTemperature'


def to_float(string):
    try:
        return float(string)
    except ValueError:
        return float('nan')


def make_groups(df, key):
    groups = {}
    for name, group in df.groupby(key):
        groups[name] = group
    return groups


def make_lags(data, lags):
    return pd.concat({f'lag_{i}': data[TEMPERATURE_COLUMN].shift(i) for i in range(1, lags + 1)}, axis=1)


class Preprocessor:

    def __init__(self, data):
        self.data = data

    def data_clean_up(self):
        self.data['date'] = pd.to_datetime(self.data['dt'], errors='coerce')
        self.data.drop('dt', axis=1)

        self.data[TEMPERATURE_COLUMN] = self.data[TEMPERATURE_COLUMN].apply(lambda _: to_float(_))
        self.data[TEMPERATURE_COLUMN] = self.data[TEMPERATURE_COLUMN].interpolate()
        self.data[TEMPERATURE_COLUMN] = self.data[TEMPERATURE_COLUMN].astype('float')

        # remove unwanted columns
        self.data = self.data.drop(['Latitude', 'Longitude', 'AverageTemperatureUncertainty'], axis=1)

        self.data = self.data.set_index('date').sort_index()
        return self.data

    def train_test_split_and_add_lags(self, lags=1, key='City', test_size=0.2):
        train_size = 1.0 - test_size
        grouped_data = make_groups(self.data, key)

        train = []
        test = []
        for k in grouped_data.keys():
            y = grouped_data[k][TEMPERATURE_COLUMN]
            grouped_data[k] = make_lags(grouped_data[k], lags)
            grouped_data[k]['City'] = k
            grouped_data[k]['target'] = y
            grouped_data[k] = grouped_data[k].reset_index()
            temp = grouped_data[k]
            train.append(grouped_data[k][:int(train_size * len(temp))].copy())
            test.append(grouped_data[k][int(train_size * len(temp)):].copy())

        train, test = pd.concat(train).sort_index(), pd.concat(test).sort_index()
        y_train, y_test = train[TEMPERATURE_COLUMN], test[TEMPERATURE_COLUMN]
        x_train, x_test = train.drop(TEMPERATURE_COLUMN, axis=1), test.drop(TEMPERATURE_COLUMN,axis=1)
        return x_train, x_test, y_train, y_test
