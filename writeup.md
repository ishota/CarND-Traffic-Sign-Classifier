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

[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

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

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy is **0.9916**
* validation set accuracy is **0.9757**
* test set accuracy is **0.9515**

Here I show the history of learning.

[img4]: ./writeup_image/plot_epoch_accuracy.png "plot_epoch_accuracy"
![alt text][img4]

First, I implemented LeNet model which summary shown below.

```python
Model: "Lenet model"
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
So, I improve some point:

* Increase a number of epoch to improve under fitting.
* Increase a convolutional layer to capture smaller features in an image.
* Increase a max pooling layer to blur the features that emerge in the convolutional layer and to learn that they are the same as other similar feature.
* Add Dropout layer to suppress over fitting but the number of epochs increase.
* Add L2-regularization to suppress over fitting.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


