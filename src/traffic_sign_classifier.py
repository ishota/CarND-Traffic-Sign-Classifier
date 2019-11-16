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
    lenet_model = lenet()

    # compile the model
    lenet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # train the model
    history = lenet_model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid), verbose=2)

    # evaluate the model
    test_loss, test_acc = lenet_model.evaluate(X_test, y_test, verbose=0)
    print('test_loss: ' + str(test_loss))
    print('test_accuracy: ' + str(test_acc))

    # save model
    lenet_model.save('basic_lenet_model.h5')


def main():

    # load_pickled_data
    data_path = str(Path('.').resolve().parents[0]) + os.sep + 'traffic-signs-data'
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_pickled_data(data_path)

    # summary_data
    csv_file = str(Path('.').resolve().parents[0]) + os.sep + 'signnames.csv'
    n_classes = summary_data(X_train, y_train, X_valid, y_valid, X_test, y_test, csv_file)

    # preprocess_data
    processed_X_train, processed_y_train = random_distortion(X_train, y_train)
    n_classes = summary_data(processed_X_train, processed_y_train, X_valid, y_valid, X_test, y_test, csv_file)

    # create the model
    nn_model = proposed()
    nn_model.summary()

    # compile the model
    nn_model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # train the model
    history = nn_model.fit(processed_X_train, processed_y_train,
                        epochs=20, batch_size=256, validation_data=(X_valid, y_valid), verbose=2,
                        use_multiprocessing=True)

    # evaluate the model
    test_loss, test_acc = nn_model.evaluate(X_test, y_test, verbose=0)
    print('test_loss: ' + str(test_loss))
    print('test_accuracy: ' + str(test_acc))

    # save model
    nn_model.save('proposed_nn_model.h5')


if __name__ == '__main__':
    # basic_lenet()
    main()
