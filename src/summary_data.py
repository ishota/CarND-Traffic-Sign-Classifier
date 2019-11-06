import csv
import os
import load_pickled_data
from pathlib import Path


def summary_data(X_train, y_train, X_valid, y_valid, X_test, y_test, csv_path):
    all_labels = []

    with open(csv_path, 'r') as csvfile:
        readCSV = csv.reader(csvfile, delimiter='.')
        for row in readCSV:
            all_labels += [row[1]]

    n_train = len(X_train)
    n_validation = len(X_valid)
    n_test = len(X_test)
    image_shape = X_train[0].shape
    img_size = X_train.shape[1]
    n_classes = len(all_labels)


if __name__ == '__main__':
    print('test: summary_data')
    data_path = str(Path('.').resolve().parents[0]) + os.sep + 'traffic-signs-data'
    csv_path = str(Path('.').resolve().parents[0]) + 'signnames.csv'
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_pickled_data(data_path)
    summary_data(X_train, y_train, X_valid, y_valid, X_test, y_test, csv_path)
