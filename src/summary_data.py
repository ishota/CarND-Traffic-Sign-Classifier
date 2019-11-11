import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from load_pickled_data import *
from pathlib import Path
from itertools import groupby


def summary_data(X_train, y_train, X_valid, y_valid, X_test, y_test, csv_file):

    # import csv file to read labels
    all_labels = []
    with open(csv_file, 'r') as f:
        read_csv = csv.reader(f, delimiter=',')
        for row in read_csv:
            all_labels += [row[1]]
        all_labels.remove('SignName')

    n_train = len(X_train)
    n_validation = len(X_valid)
    n_test = len(X_test)
    image_shape = X_train[0].shape
    n_classes = len(all_labels)

    print("Number of training examples =", n_train)
    print("Number of validation examples =", n_validation)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

    # plot images with it label
    num_of_samples = []
    plt.figure(figsize=(10, 15))
    plt.tight_layout()
    for i in range(0, n_classes):
        plt.subplot(15, 3, i+1)
        x_selected = X_train[y_train == i]
        random_selected_index = random.randint(0, x_selected.shape[0])
        plt.imshow(x_selected[random_selected_index, :, :, :])
        plt.title(all_labels[i], fontsize=8)
        plt.axis('off')
        num_of_samples.append(len(x_selected))
    plt.show()

    # plot number of images per label
    plt.figure(figsize=(15, 10))
    hist, bins = np.histogram(y_train, bins=n_classes)
    left = (bins[:-1] + bins[1:]) / 2
    plt.barh(left, hist, align='center')
    plt.yticks(left, all_labels)
    plt.show()


if __name__ == '__main__':
    print('[test]: summary_data')
    data_path = str(Path('.').resolve().parents[0]) + os.sep + 'traffic-signs-data'
    csv_file = str(Path('.').resolve().parents[0]) + os.sep + 'signnames.csv'
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_pickled_data(data_path)
    summary_data(X_train, y_train, X_valid, y_valid, X_test, y_test, csv_file)
