# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from summary_data import *
from load_pickled_data import *
from pathlib import Path


def prediction_analyze(nn_model, input_images, test_images, test_label_id, all_labels, num_rows=5, num_cols=3):

    # load prediction result
    prediction = nn_model.predict(tf.convert_to_tensor(input_images, np.float32))

    # calculate number of images
    num_images = num_rows * num_cols

    # plot random select result
    idxes = np.arange(len(test_label_id))
    random_select_indexes = np.random.choice(idxes, num_images, replace=False)
    plt.figure(figsize=(3*2*num_cols, 4*num_rows))
    plt.subplots_adjust(hspace=2.0)
    count = 0
    for i in random_select_indexes:
        img = test_images[i]
        prediction_array = prediction[i]
        prediction_prob = np.max(prediction_array)
        predicted_label_idx = np.argmax(prediction_array)
        predicted_label = all_labels[predicted_label_idx]
        true_label_idx = test_label_id[i]
        true_label = all_labels[true_label_idx]

        plt.subplot(num_rows, 2*num_cols, 2*count+1)
        plot_image_with_label(img, predicted_label_idx, predicted_label, true_label_idx, true_label, prediction_prob)

        plt.subplot(num_rows, 2*num_cols, 2*count+2)
        plot_top5_value_array(prediction_array, all_labels)

        count += 1
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

    plt.xlabel("{}\n {:2.0f}% \n({})".format(predicted_label, prediction_prob*100, true_label), color=color)


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
    plt.xticks(range(5), top5_label, rotation=90, size=9, wrap=True, ha='left')
    plt.yticks([])
    thisplot = plt.bar(range(5), prediction_top5_array, color='#777777', width=0.3)
    for x in range(5):
        plt.text(x-0.35, -0.05, "{:2.1f}%".format(prediction_top5_array[x]*100), rotation=90)
    plt.ylim([0, 1])

    thisplot[0].set_color('cyan')


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
    std_X_test = tf.image.per_image_standardization(tf.convert_to_tensor(X_test, np.float32))
    prediction_analyze(nn_model, std_X_test, X_test, y_test, all_labels)
    pass
