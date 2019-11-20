# **Traffic Sign Recognition**

## Writeup

This is a project for Udacity lesson of Self-driving car engineer. The project is creating Traffic Sign Classifier that classifies images to 43 kinds.

---

**Build a Traffic Sign Recognition Project**
The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation

---

### Data Set Summary & Exploration

#### 1. After importing the dataset files I calculate summary statistics of the traffic signs dataset and show the exploratory visualization of the dataset

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes/labels in the data set is **43**

[img2]: ./writeup_image/summary_train_img.png "summary_train_img"
![alt text][img2]

Here are 43 randomly selected images and summary of dataset.

[img1]: ./writeup_image/train_img.png "train_img"
![alt text][img1]

### Design and Test a Model Architecture

#### 1. Preprosecced images

As a first step, I decided to changing saturation because I wanted to reduce color information on the image.
For example, the "No passing" image is similer to the "No entry" image.
I thought a N.N. model learned more a shape of image by decreacing the saturation.
But, I didn't convert training images to grayscalse because color is one of an important feature of images.

Here are 43 randomly selected preprocessed image.

[img3]: ./writeup_image/preprocessed_train_img.png "preprocessed_train_img"
![alt text][img3]

As a last step, I normalized the image data to prevent the so-called vanishing gradients.
I just normalize inputs.
The size of trainging set is finaly **69598** as double of training set before preprocessed.

#### 2. Model architecture

My final model consisted of the following layers:

```python
Model: "Proposed model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 30, 30, 6)         168
_________________________________________________________________
dropout (Dropout(0.2))       (None, 30, 30, 6)         0
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 28, 28, 6)         0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 24, 24, 14)        2114
_________________________________________________________________
dropout_1 (Dropout(0.3))     (None, 24, 24, 14)        0
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 12, 12, 14)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 6, 6, 22)          15114
_________________________________________________________________
dropout_2 (Dropout(0.5))     (None, 6, 6, 22)          0
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 5, 5, 22)          0
_________________________________________________________________
flatten (Flatten)            (None, 550)               0
_________________________________________________________________
dense (Dense)                (None, 150)               82650
_________________________________________________________________
dense_1 (Dense)              (None, 100)               15100  
_________________________________________________________________
dense_2 (Dense)              (None, 43)                4343
=================================================================
Total params: 119,489
Trainable params: 119,489
Non-trainable params: 0
_________________________________________________________________
```

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate

To train the model, I used a RMSprop optimizer.
The RMSprop optimizer is proposed by [Geoffrey Hinton's lecture](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).
The RMSprop is equivalent to using the gradient.
In this optimizer, a learning rate decreases gradually.
I use default initial learning rate which is 0.001.

I show other setting below.

* A number of epochs is **25**
* A number of batch_size is **256**

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93

The code for fitting my final model is located in the 11th cell of the Ipython notebook.

My final model results were:

* training set accuracy is **0.9916**
* validation set accuracy is **0.9757**
* test set accuracy is **0.9515**

Here I show the history of learning.

[img4]: ./writeup_image/plot_epoch_accuracy.png "plot_epoch_accuracy"
![alt text][img4]

Here I show the 15 results selected randomly.

[img13]: ./writeup_image/plot_test_result.png "plot_test_result"
![alt text][img13]

First, I implemented LeNet model which summary shown below.

```python
Model: "Lenet model (initial implemented model)"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 28, 28, 6)         456
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 6)         0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 10, 10, 16)        2416
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 16)          0
_________________________________________________________________
flatten (Flatten)            (None, 400)               0
_________________________________________________________________
dense (Dense)                (None, 120)               48120
_________________________________________________________________
dense_1 (Dense)              (None, 84)                10164
_________________________________________________________________
dense_2 (Dense)              (None, 43)                3655
=================================================================
Total params: 64,811
Trainable params: 64,811
Non-trainable params: 0
_________________________________________________________________
```

But test set accuracy was below 0.93.
So, I improve some points:

* Increase a number of epoch to improve under fitting.
* Increase a convolutional layer to capture smaller features in an image.
* Increase a max pooling layer to blur the features that emerge in the convolutional layer and to learn that they are the same as other similar feature.
* Add Dropout layer to suppress over fitting but the number of epochs increase.
* Add L2-regularization to suppress over fitting.

### Test a Model on New Images

#### 1. Choose six German traffic signs and discuss what quality or qualities might be difficult to classify

Here are six German traffic signs that I found on the web:

[img5]: ./writeup_image/new_test_img.png "new_test_img"
![alt text][img5]

I thought each traffic signs have a difficult point to be classified.

1. [Speed limit (30km/h)] : Not facing the front.
2. [No passing] : Similar to "No entry".
3. [Turn left ahead] : Similar to "Turn right ahead".
4. [Ahead only] : Similar to "Keep right" and "Keep left".
5. [Go straight or right] : Similar to "Go straight or left".
6. [Go straight or left] : Similar to "Go straight or right".

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric)

The model was able to correctly guess 5 traffic signs, which gives an accuracy of 83.3%.
Compared to accuracy of test set, the model sufficiently classified new 6 test images.

#### 3. Describe how certain the model is when predicting on each of the six new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Here are the results of the prediction:

[img6]: ./writeup_image/plot_new_test_result.png "plot_new_test_result"
![alt text][img6]

The code for making predictions on my final model is located in the 17th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Speed limit 30km/h (probability of 0.78), but also estimates a Speed limit 80km/h.
The reason why it estimates 80km/h is a 3 is similar to 8.

For the socond image, the model failed to classify.
A correct answer is "No entry" but the model classified "No passing".
I guess the model learned similar features because These two images have a similar shape with horizontal lines.

For the remaining images, the model classified correctly.
I think that the model was able to properly classify images with different orientations because it did not use rotation or flipping in the pre-processing part.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications

Since there was a possibility that "No entry" and "No passing" were learning the same feature, a feature map of each convolution layer was output.
First, three feature maps of "No entry" are shown in depth order.
I found that horizontal lines were extracted as features.

[img7]: ./writeup_image/no_entry_mid1.png "no_entry_mid1"
![alt text][img7]

[img8]: ./writeup_image/no_entry_mid2.png "no_entry_mid2"
![alt text][img8]

[img9]: ./writeup_image/no_entry_mid3.png "no_entry_mid3"
![alt text][img9]

Next, three feature maps of "No passing" are shown in depth order.
The model also learned the horizontal lines feature map.
This is one of the reasons why the model failed to classify.

[img10]: ./writeup_image/no_pass_mid1.png "no_pass_mid1"
![alt text][img10]

[img11]: ./writeup_image/no_pass_mid2.png "no_pass_mid2"
![alt text][img11]

[img12]: ./writeup_image/no_pass_mid3.png "no_pass_mid3"
![alt text][img12]
