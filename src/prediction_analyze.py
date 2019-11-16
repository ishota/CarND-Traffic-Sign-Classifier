# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from load_pickled_data import *
from pathlib import Path


def prediction_analyze(nn_model, test_images):

    # convert uint8 -> float32
    test_images = tf.convert_to_tensor(test_images, np.float32)

    # load prediction result
    prediction = nn_model.predict(test_images)

    pass


if __name__ == '__main__':
    print('[test]: prediction_analyze')

    # load test images
    data_path = str(Path('.').resolve().parents[0]) + os.sep + 'traffic-signs-data'
    csv_file = str(Path('.').resolve().parents[0]) + os.sep + 'signnames.csv'
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_pickled_data(data_path)

    # load saved model
    saved_model_path = 'proposed_nn_model.h5'
    nn_model = tf.keras.models.load_model(saved_model_path)

    # analyze prediction result
    prediction_analyze(nn_model, X_test)
    pass
