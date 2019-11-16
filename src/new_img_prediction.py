# -*- coding: utf-8 -*-
import tensorflow as tf
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


def test_new_img(data_path):

    data_path = data_path + os.sep + 'new' + os.sep
    all_file_name = np.array([data_path + '1_Speed_limit_30.jpg',
                             data_path + '17_No_entry.jpg',
                             data_path + '34_Turn_left_ahead.jpg',
                             data_path + '35_Ahead_only.jpg',
                             data_path + '36_Go_straight_or_right.jpg',
                             data_path + '37_Go_straight_or_left.jpg'])

    new_X_test = np.zeros((6, 32, 32, 3), dtype=int)
    for i in range(6):
        img_cv2 = cv2.imread(all_file_name[i], cv2.IMREAD_COLOR)
        LinerImg = cv2.resize(img_cv2, (32, 32), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(all_file_name[i], LinerImg)
        img = np.array(Image.open(all_file_name[i]))
        new_X_test[i] = img

    new_y_test = np.array([1, 17, 34, 35, 36, 37])

    return new_X_test, new_y_test


def read_image_as_tensor(file_name):

    image_raw = tf.io.read_file(file_name)
    image_tensor = tf.image.decode_jpeg(image_raw)
    image_final = tf.image.resize(image_tensor, [32, 32])

    plt.imshow(image_final)

    return image_final


if __name__ == '__main__':
    print('[test]: test_new_img')
    data_path = str(Path('.').resolve().parents[0]) + os.sep + 'traffic-signs-data'
    new_X_test, new_y_test = test_new_img(data_path)
    exit()
