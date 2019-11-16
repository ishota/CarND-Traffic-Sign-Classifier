# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from summary_data import *
from load_pickled_data import *
from pathlib import Path
from copy import deepcopy


def prediction_analyze(nn_model, test_images, y_test, all_labels):

    # load prediction result
    prediction = nn_model.predict(tf.convert_to_tensor(test_images, np.float32))

    # plot random select result
    random_select_indexes = np.random.randint(0, len(y_test), 15)

    # plot result as tile
    for i in random_select_indexes:
        plt.figure(figsize=(6, 3))
        img = test_images[i]
        prediction_array = prediction[i]
        prediction_prob = np.max(prediction_array)
        predicted_label_idx = np.argmax(prediction_array)
        predicted_label = all_labels[predicted_label_idx]
        true_label_idx = y_test[i]
        true_label = all_labels[true_label_idx]

        plt.subplot(1, 2, 1)
        plot_image_with_label(img, predicted_label_idx, predicted_label, true_label_idx, true_label, prediction_prob)

        plt.subplot(1, 2, 2)
        plot_top5_value_array(prediction_array, all_labels)

        plt.show()


def plot_image_with_label(img, predicted_label_idx, predicted_label, true_label_idx, true_label, prediction_prob):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    if predicted_label_idx == true_label_idx:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% \n({})".format(predicted_label, prediction_prob*100, true_label), color=color)


def plot_value_array(prediction_array, predicted_label_idx, true_label):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(43), prediction_array, color='#777777')
    plt.ylim([0, 1])

    thisplot[predicted_label_idx].set_color('red')
    thisplot[true_label].set_color('blue')


def plot_top5_value_array(prediction_array, all_labels):

    prediction_top5_idx = (-prediction_array).argsort()[:5]
    prediction_top5_array = prediction_array[prediction_top5_idx]
    top5_label = [all_labels[i] for i in prediction_top5_idx]

    plt.grid(False)
    plt.xticks(range(5), top5_label, rotation=90)
    plt.yticks([])
    thisplot = plt.bar(range(5), prediction_top5_array, color='#777777')
    plt.ylim([0, 1])

    thisplot[0].set_color('blue')


if __name__ == '__main__':
    print('[test]: prediction_analyze')

    # load test images
    data_path = str(Path('.').resolve().parents[0]) + os.sep + 'traffic-signs-data'
    csv_file = str(Path('.').resolve().parents[0]) + os.sep + 'signnames.csv'
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_pickled_data(data_path)

    # get class label
    data_path = str(Path('.').resolve().parents[0]) + os.sep + 'traffic-signs-data'
    csv_file = str(Path('.').resolve().parents[0]) + os.sep + 'signnames.csv'
    n_classes, all_labels = summary_data(X_train, y_train, X_valid, y_valid, X_test, y_test, csv_file)

    # load saved model
    saved_model_path = 'proposed_nn_model.h5'
    nn_model = tf.keras.models.load_model(saved_model_path)

    # analyze prediction result
    prediction_analyze(nn_model, X_test, y_test, all_labels)
    pass
