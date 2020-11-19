# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to 
use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[image1]: ./examples/curve.jpg "Curved lane"
[image2]: ./examples/center.jpg "Center lane driving"
[image3]: ./examples/recover-1.jpg "Recovery Image"
[image4]: ./examples/recover-2.jpg "Recovery Image"
[image5]: ./examples/recover-3.jpg "Recovery Image"
[image8]: ./examples/loss.png "Loss function"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe 
how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by 
executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline 
I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of normalization, image cropping, convolutional layers, a dropout layer and fully connected layers.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 122). 

The model was trained and validated on different data sets (code lines 16 - 28) to ensure that the model was not 
overfitting (code lines 133-137). The model was tested by running it through the simulator and ensuring that the 
vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 129).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, 
recovering from the left and right sides of the road, curvatures and anti-clockwise laps. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to create a big data set with different scenarios and then 
fine tune it by applying transfer learning.

My first step was to use a convolution neural network model similar to the NVIDIA I thought this model might be 
appropriate because is was published by the autonomous team of NVIDIA.

In order to gauge how well the model was working, I split my image and steering angle data into a training and 
validation set. I found that my first model had a low mean squared error on the training set but a high mean squared 
error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model and included a dropout layer after the convolutional layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots 
where the vehicle fell off the track, to improve the driving behavior in these cases, I collected more data and used 
transfer learning to fine tune my existing model.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 115-127) consisted of a convolution neural network with the following 
layers and layer sizes:
 - Normalization 
 - Croppint to 70x25 images
 - Convolutions layer 24x5x5 with subsampling 2x2 and a RELU function
 - Convolutions layer 36x5x5 with subsampling 2x2 and a RELU function
 - Convolutions layer 48x5x5 with subsampling 2x2 and a RELU function
 - Convolutions layer 64x3x3 and a RELU function
 - Convolutions layer 64x3x3 and a RELU function
 - Fully connected layer 100 
 - Fully connected layer 50 
 - Fully connected layer 10 
 - Fully connected layer 1 

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example 
image of center lane driving:
![Center lane driving][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle
 would learn to drive back to the center of the road. These images show what a recovery looks like starting from the 
 edge of the road:

![Recover Image 1][image3]
![Recover Image 2][image4]
![Recover Image 3][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images (code lines 49-51) and angles thinking that this would compensate the 
left/right bias of the track for certain curves. And I also used the left and right image from the simulator.

![Curved right image][image1]

After the collection process, I had ~80k data points. I then preprocessed this data by normalizing and cropping.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under 
fitting. The ideal number of epochs was 5 as evidenced by the following image:
![Loss][image8]

I used an adam optimizer so that manually training the learning rate wasn't necessary.
