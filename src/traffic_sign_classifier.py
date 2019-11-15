# -*- coding: utf-8 -*-
import tensorflow
import os
from pathlib import Path
from load_pickled_data import *
from summary_data import *
from preprocess_data import *
from nn_model import *


def basic_lenet():

    # load_pickled_data
    data_path = str(Path('.').resolve().parents[0]) + os.sep + 'traffic-signs-data'
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_pickled_data(data_path)

    # summary_data
    csv_file = str(Path('.').resolve().parents[0]) + os.sep + 'signnames.csv'
    n_classes = summary_data(X_train, y_train, X_valid, y_valid, X_test, y_test, csv_file)

    # create the model
    model = lenet()

    # compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # train the model
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

    # evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('test_loss: ' + str(test_loss))
    print('test_accuracy: ' + str(test_acc))


def main():

    # load_pickled_data
    data_path = str(Path('.').resolve().parents[0]) + os.sep + 'traffic-signs-data'
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_pickled_data(data_path)

    # summary_data
    csv_file = str(Path('.').resolve().parents[0]) + os.sep + 'signnames.csv'
    n_classes = summary_data(X_train, y_train, X_valid, y_valid, X_test, y_test, csv_file)

    # preprocess_data
    s_bright_X = shift_brightness(X_train)

    processed_X_train = np.concatenate((X_train, s_bright_X), axis=0)
    processed_y_train = np.concatenate((y_train, y_train), axis=0)

    # create the model
    model = proposed()

    # compile the model
    model.compile(optimizer='RMSprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # train the model
    history = model.fit(processed_X_train, processed_y_train, epochs=10, validation_data=(X_valid, y_valid))

    # evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('test_loss: ' + str(test_loss))
    print('test_accuracy: ' + str(test_acc))


if __name__ == '__main__':
    # basic_lenet()
    main()
