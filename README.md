# Project: Build a Traffic Sign Recognition Program

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The Project

The goals / steps of this project are the following:

* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---

Overview

A convolutional neural network to classify German traffic signs.

## Description

In this project, I create convolutional neural networks to classify German traffic signs.
Finaly, created model accuracy is over 95%.
You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
After the model is trained, you will then try out the model on images of new traffic signs that you find on ./traffic-signs-data/new/*.
You also try the model on images that you find.

## Demo

I show part of the classification result.
You can see the summary of this project in `Traffic_Sign_Classifier.ipynb` or `Traffic_sign_Classifier.html`.

[img1]: ./writeup_image/plot_test_result.png "plot_test_result"
![alt text][img1]

## Requirement

* tensorflow**2.0**
* numpy
* matplotlib
* picle
* opencv

## Quick start guide

You can get result quickly to implement `traffic_sign_classifier.py`.
You can write your original N.N. model at `nn_model.py`.
