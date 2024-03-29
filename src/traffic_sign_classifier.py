# -*- coding: utf-8 -*-
import tensorflow
import os
from pathlib import Path
from load_pickled_data import *
from summary_data import *
from preprocess_data import *
from nn_model import *
from learning_history_analyze import *
from prediction_analyze import *
from new_img_prediction import *
from mid_layer_analyze import *


def basic_lenet():

    # load_pickled_data
    data_path = str(Path('.').resolve().parents[0]) + os.sep + 'traffic-signs-data'
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_pickled_data(data_path)

    # summary_data
    csv_file = str(Path('.').resolve().parents[0]) + os.sep + 'signnames.csv'
    n_classes, all_labels = summary_data(X_train, y_train, X_valid, y_valid, X_test, y_test, csv_file)

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
    n_classes, all_labels = summary_data(X_train, y_train, X_valid, y_valid, X_test, y_test, csv_file)

    # preprocess_data: distortion
    processed_X_train, processed_y_train = random_distortion(X_train, y_train)
    _, _ = summary_data(processed_X_train, processed_y_train, X_valid, y_valid, X_test, y_test, csv_file)

    # preprocess data: standardization
    std_processed_X_train = tf.image.per_image_standardization(tf.convert_to_tensor(processed_X_train, np.float32))
    std_X_valid = tf.image.per_image_standardization(tf.convert_to_tensor(X_valid, np.float32))
    std_X_test = tf.image.per_image_standardization(tf.convert_to_tensor(X_test, np.float32))

    # create the model
    nn_model = proposed()
    nn_model.summary()

    # compile the model
    nn_model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # train the model
    # std_processed_X_train = input_normalize(processed_X_train)
    # std_X_valid = input_normalize(X_valid)
    history = nn_model.fit(std_processed_X_train, processed_y_train,
                        epochs=25, batch_size=256, validation_data=(std_X_valid, y_valid), verbose=2,
                        workers=2, use_multiprocessing=True)
    history_analyze(history)

    # evaluate the model
    test_loss, test_acc = nn_model.evaluate(std_X_test, y_test, verbose=0)
    print('test_loss: ' + str(test_loss))
    print('test_accuracy: ' + str(test_acc))

    # analyze test images
    prediction_analyze(nn_model, std_X_test, X_test, y_test, all_labels)

    # analyze new test images
    new_X_test, new_y_test = test_new_img(data_path)
    std_new_X_test = tf.image.per_image_standardization(tf.convert_to_tensor(new_X_test, np.float32))
    prediction_analyze(nn_model, std_new_X_test, new_X_test, new_y_test, all_labels, num_rows=3, num_cols=2)

    # save model
    nn_model.save('proposed_nn_model.h5')

    # check an intermediate layer
    layer_analyze(nn_model, std_new_X_test, new_X_test)


if __name__ == '__main__':
    # basic_lenet()
    main()
