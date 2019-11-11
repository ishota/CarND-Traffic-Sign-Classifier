# -*- coding: utf-8 -*-
import os
from pathlib import Path
from load_pickled_data import *
from summary_data import *


def main():

    # load_pickled_data
    data_path = str(Path('.').resolve().parents[0]) + os.sep + 'traffic-signs-data'
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_pickled_data(data_path)

    # summary_data
    csv_file = str(Path('.').resolve().parents[0]) + os.sep + 'signnames.csv'
    summary_data(X_train, y_train, X_valid, y_valid, X_test, y_test, csv_file)


if __name__ == '__main__':
    main()
