# -*- coding: utf-8 -*-
import os
from pathlib import Path
from load_pickled_data import *


def main():

    # directory path of pickled data
    data_path = str(Path('.').resolve().parents[0]) + os.sep + 'dataset'

    # load pickled_data
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_pickled_data(data_path)

if __name__ == '__main__':
    main()
