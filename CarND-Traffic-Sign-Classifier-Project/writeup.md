# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: CarND-Traffic-Sign-Classifier-Project/images/result/labels_histogram.png "Visualization"
[image2]: CarND-Traffic-Sign-Classifier-Project/images/result/grayscale.png "Grayscaling"
[image3]: ./images/random_noise.jpg "Random Noise"
[image4]: ./images/internet/test/label_02.jpeg "Traffic Sign 1"
[image5]: ./images/internet/test/label_13.jpg
[image6]: ./images/internet/test/label_31_0.jpeg "Traffic Sign 1"
[image7]: ./images/internet/test/label_31_1.jpeg
[image8]: ./images/internet/test/label_41.jpeg "Traffic Sign 1"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is 32 x 32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data 
is distributed over the labels.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because traffic signs have to be 
identifiable just by it's shapes and symbols, since even color-blind people should be able to read them.

Here is an example of a traffic sign image before and after gray-scaling.

![alt text][image2]

As a last step, I normalized the image data because it enhances the optimization process, 
hence increases the test accuracy.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32 gray-scale image						| 
| Convolution 4x4     	| 1x1 stride, valid padding, outputs 29x29x6 	|
| RELU					|												|
| Convolution 4x4	    | 1x1 stride, valid padding, outputs 26x26x16 	|
| RELU					|												|
| Flatten				| output 1x10816  								|
| Fully connected		| output 1x111         							|
| RELU					|												|
| Dropout				| Keep probability 50%							|
| Fully connected		| output 1x88         							|
| RELU					|												|
| Dropout				| Keep probability 50%							|
| Output layer			| output 1x43									|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer, since it's more sophisticated than plain 
gradient descent. I tuned the hyper parameters, by implementing two classes  
`hyper_parameter_handler.py` and `paramater.py`, with these classes I randomly varied 
the hyper parameters, since testing all possible combinations would have been too expensive.
I tested around 100 random combinations, where I restricted the range of the parameters.
If the validation accuracy increased, under the consideration of *the rule of 30*, I would 
save the hyper parameters in the `parameter/hyper.json` file.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.0%
* validation set accuracy of 93.4% 
* test set accuracy of 95.4%

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?
I first tried LeNet 5, since it was suggested as a good starting point.

* What were some problems with the initial architecture?
LeNet 5 is using pooling as a regularization method, where as drop-out is considered superior regularization method.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different 
model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function 
or changing the activation function. One common justification for adjusting an architecture would be due to 
overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over 
fitting; a low accuracy on both sets indicates under fitting.
Hence I adjusted the regularization method to drop-out, which lead directly to better validation accuracies. 
This suggests that the initial architecture was over-fitting. 

* Which parameters were tuned? How were they adjusted and why?
 1. Epochs: since the amount of epochs can increase the accuracy for each iteration
 2. Batch size: since it also influences tha accuracy, besides the computation time
 3. Learning rate: reason see batch size
 4. Drop out rate: do adjust the regularization to the network, helps to avoid under- and overfitting
 5. CNN filter size: which influences the accuracy
 6. CNN depth: accuracy and over-/underfitting
 7. Sigma: variation of the weights can also influence the accuracy
 8. Fully connected layers output size: accuracy and over-/underfitting

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
The convolutional layer helps to make the objects that have to be detected position invariant. The drop-out helped
to prevent potential overfitting and to prevent underfitting the drop out rate was parametrized.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

Some might be difficult to classify because:
1. Is not parallel to the image plain
2. Noisy background
3. Dark sign bright background
4. Not parallel to image plain
5. Not parallel to image plain
#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed 50      		| Speed 30   									| 
| Yield     			| Yield 										|
| Wild animal crossing	| Wild animal crossing											|
| Wild animal crossing	| Keep right					 				|
| End of no passing     | No vehicles      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. 
This a a very poor accuracy compared to the test accuracy.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a speed sign 30 (probability of 0.99), 
and the image does contain a speed 50. The remaining top 3 softmax probabilities are below 1%, even though,
speed 50 comes at second place.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99%         			| Speed 30   									| 
| 100%     				| Yield 										|
| 97%					| Wild animals crossing							|
| 49%	      			| Keep right					 				|
| 99%				    | No vehicles      							    |


For the second image is classified correctly with a probability of 100%. The second one as well with
a probability of 97%. The last two ones are classified wrong where the 4th probability
is very low and also the top 3 softmax classifications are wrong. Sam applied for the 5th, the only difference,
here is that the classifier is quiet certain it's a no vehicle sign.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


