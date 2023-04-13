import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
            break

        train, test = pd.concat(train).sort_index(), pd.concat(test).sort_index()
        print(train.columns)
        print(test.columns)
        # one_hot1 = pd.get_dummies(train['City'], prefix='City')
        # one_hot2 = pd.get_dummies(test['City'], prefix='City')
        #
        # train = pd.concat([train, one_hot1], axis=1)
        # test = pd.concat([test, one_hot2], axis=1)
        encoder = LabelEncoder()
        train['EncodedCity'] = encoder.fit_transform(train['City'])
        encoder = LabelEncoder()
        test['EncodedCity'] = encoder.fit_transform(test['City'])

        if 'City' in train.columns:
            train = train.drop('City', axis=1)
        if 'City' in test.columns:
            test = test.drop('City', axis=1)

        train['date'] = pd.to_datetime(train['date'])
        train['day'] = train['date'].dt.day
        train['month'] = train['date'].dt.month
        train['year'] = train['date'].dt.year
        train = train.drop('date', axis=1)
        train = train.dropna()

        # columns = train.columns
        # for c in columns:
        #     train[c] = train[c].interpolate()

        test['date'] = pd.to_datetime(test['date'])
        test['day'] = test['date'].dt.day
        test['month'] = test['date'].dt.month
        test['year'] = test['date'].dt.year
        test = test.drop('date', axis=1)
        test = test.dropna()

        # columns = test.columns
        # for c in columns:
        #     test[c] = test[c].interpolate()

        y_train, y_test = train['target'], test['target']
        x_train, x_test = train.drop('target', axis=1), test.drop('target', axis=1)
        return x_train, x_test, y_train, y_test
