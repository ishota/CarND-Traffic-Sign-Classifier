import pickle
import os
from pathlib import Path

class Data:

    def __init__(self, data_path, csv_path):

        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = self.load_pickled_data(self, data_path)


    def load_pickled_data(self, data_path):

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
    print('test: data_process.py')

