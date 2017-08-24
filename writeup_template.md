# Behavioral Cloning

I learn to drive from my brother who has the best record of good driver. 
The way he drive is smooth, stable, and carefully to what the degree that 
he never has any acident or driving ticket in his life of driving.

While I learn to drive from him, I inherit this behavoir and I am also never have 
any acident or traffic ticket involving a bad driving behavoir because I always maintain
the good driving behavior.

While working on "Behavioral Cloning Project" for self-driving car, I learned that
the self-driving car also immitate to what I have driven during my training session.
The self-driving car drives poorly in autonomous mode after I have driven poorly while I train
it to drive. On the other hand, the self-driving car will drive better by itself after I have 
driven really good in training session.

Now, let take a look on how the self-driving car learns from my training and how it maintain
to keep the safe and good driving behavior on the road in order to keep the passengers and 
the other cars on the road safe.



# Project Structure

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


---
# Files Submitted & Code Quality

### 1. File Structure

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* readme.md this file

### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---
# Model Architecture and Training Strategy

### 1. An appropriate model architecture has been employed

As suggest by lesson from udacity.com, I follow the nVidia CNN for my model architecture. Below is the picture of how the model is design:

<img src="images/cnn-architecture.png" width="460">

There are some change to the original model in order to make it work to the project. Below is the architecture of the model:

Layer | Description
--- | ---
Input | Input shape 160x320x3 RGB 
CNN 5x5 | 2x2 stride
Dropout | Dropout layer with 20% rate
CNN 5x5 | 2x2 stride CNN
Dropout | Dropout layer with 20% rate
CNN 5x5 | 2x2 stride CNN
Dropout | Dropout layer with 20% rate
CNN 3x3 | 
CNN 3x3 |
Flatten layer | 
Fully-connected | 100 
Fully-connected | 50
Fully-connected | 1

Therefore, I have change the original nVidia model to fit the need of this project. The original model does not provide the dropout layer. While impliment the original model, my self-driving car could not drive well on the road and it end up get off the road. While add three layers of dropout into the model, I see the improvement while training and also in the autonomous driving mode in the simulator.

### 2. Attempts to reduce overfitting in the model

As I have mention above, I have added to the model the dropout layers in order to reduce overfitting and improve the model validation. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and reverse direction of driving. 

# Model Architecture and Training Strategy

### 1. Solution Design Approach

The overall strategy for deriving a model architecture was adapted from nVideo CNN self-driving architecture. 

I have first try to use LeNet model architecture and then nVidia model architecture. After training and testing, I saw the accuration of the nVideo model was better than LeNet.Therefore, I have chosen a convolution neural network model similar to the nVidia model because I thought this model might be appropriate because nVidia has used this model for their self-driving car. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding three dropout layer with 20% rate. Then I see the improvement of the model both on training also in the autonomouse driving mode. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, this happen because I was drive bad during training. To improve the driving behavior in these cases, I have drive again in the simulator in training mode in the well driving behavior. After I train with new dataset, I get the better result.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers

Layer | Description
--- | ---
Input | Input shape 160x320x3 RGB 
CNN 5x5 | 2x2 stride
Dropout | Dropout layer with 20% rate
CNN 5x5 | 2x2 stride CNN
Dropout | Dropout layer with 20% rate
CNN 5x5 | 2x2 stride CNN
Dropout | Dropout layer with 20% rate
CNN 3x3 | 
CNN 3x3 |
Flatten layer | 
Fully-connected | 100 
Fully-connected | 50
Fully-connected | 1

### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is example a images center lane driving:

Camera | Image
---| ---
Left | <img src ="https://github.com/loynin/Behavioral-Cloning/blob/master/images/left_2017_08_21_17_35_09_714.jpg" width="300"> 
Center | <img src ="https://github.com/loynin/Behavioral-Cloning/blob/master/images/center_2017_08_21_17_35_09_714.jpg" width="300"> |
Right |<img src ="https://github.com/loynin/Behavioral-Cloning/blob/master/images/right_2017_08_21_17_35_09_714.jpg" width="300"> |


I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back to the road when it is got off from the road. 

To augment the data sat, I also flipped images and measurement thinking that this would make model more generalize to the road condition. For example, here is an image that has then been flipped:

<img src="https://github.com/loynin/Behavioral-Cloning/blob/master/images/augment_image.png" width="400">

**After the collection process, I had the following number of data:**
- Number of sample data: 14,352
- Number of Validation data: 2,871
- Image shape: (160,320,3)

I finally randomly shuffled the data set and put **20%** of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3. I used an adam optimizer so that manually training the learning rate wasn't necessary.
