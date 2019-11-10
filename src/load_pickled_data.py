import pickle
import os
from pathlib import Path


def load_pickled_data(data_path):

    training_file = data_path + os.sep + 'train.p'
    validation_file = data_path + os.sep + 'valid.p'
    testing_file = data_path + os.sep + 'test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    return X_train, y_train, X_valid, y_valid, X_test, y_test


if __name__ == '__main__':
    print("[test]: load_pickled_data")
    path = str(Path('.').resolve().parents[0]) + os.sep + 'traffic-signs-data'
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_pickled_data(path)
