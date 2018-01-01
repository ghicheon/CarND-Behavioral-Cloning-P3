# **Behavioral Cloning** 

## by Ghicheon Lee

---
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[center_image]: ./center_2017_12_30_21_40_10_739.jpg   "center"
[left_image]:  ./left_2017_12_30_21_40_10_739.jpg      "left"
[right_image]: ./right_2017_12_30_21_40_10_739.jpg     "right"

[normal_image]: ./left_2017_12_30_21_40_10_739.jpg      "normal"
[flipped_image]:  ./flipped_left_2017_12_30_21_40_10_739.png   "Flipped"

[history_image]:  ./train_valid_history.png   "train_valid_loss"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py : python code for creating and training the model.    
* model.h5 : trained convolution neural network      
* writeup_report.md : resport for the results     

FYI, I used original drive.py given by Udacity for driving the car in autonomous mode    

#### 2. Submission includes functional code
Using the Udacity provided simulator and the drive.py file,    
the car can be driven autonomously around the track by executing     

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file has the code for loading and preprocessing from data files.      
It creates the convolution neural network. Then train and validate the network.      
I added some comments in orer to help reviewer understand my code.   

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
My model(model.py 121 line to 139 lines) has 4 convolution Layers and 4 fullpy connected layers. I used 3x3 kernel for each convolution layer. 

#### 2. Attempts to reduce overfitting in the model

I added max pooling layers for each convolution layer and dropout layers for each fully connected layer in order to reduce overfitting.   
I used dropout layer progressively.Its drop out rate is 0.5     
This model was tested and the car could stay on the track successfully.     


#### 3. Model parameter tuning

This model used an Amam optimizer. It gave me good result. 

#### 4. Appropriate training data

I used nomal center lane driving for around 4 laps.    

I used center/left/right camera images and flipped images. In summary, I could get 6 training  data per a cvs file entry. 

Actually, I was about to add recovering driving and reverse center lane driver.if it's not proper.     
However for my model, center lane driving data is enough to get a expected result.   

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I created the model based on NVIDIA model.    
It seems to suffer from overfitting  because validation loss is much higher that training loss. Even if many epoches was done, validation loss didn't decrese.    
Because of this, I used less nodes in fully connected layers and added drop out layers.  0.5 was used for  drop out rate. (I tried 0.2 ,0.3 and so on. Finally,I found out small rate didn't make a good result)   

I divided the training data into 3 parts for training,validation, and test. 70% is for training. 20% is for validation. 10% is for test.

When I test it using the simulator, the car drives around the track autonomously without leaving the road.

#### 2. Final Model Architecture

I printed the model using summary() of keras.          

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 63, 318, 12)       336
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 31, 159, 12)       0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 29, 157, 24)       2616
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 14, 78, 24)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 12, 76, 48)        10416
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 6, 38, 48)         0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 4, 36, 96)         41568
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 2, 18, 96)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 3456)              0
_________________________________________________________________
dense_1 (Dense)              (None, 512)               1769984
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 128)               65664
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0
_________________________________________________________________
dense_3 (Dense)              (None, 32)                4128
_________________________________________________________________
dropout_3 (Dropout)          (None, 32)                0
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 33
=================================================================
Total params: 1,894,745
Trainable params: 1,894,745
Non-trainable params: 0

```


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center_image]         
![alt text][left_image]            
![alt text][right_image]         

I used flipped images for each images.      
![alt text][normal_image]          
![alt text][flipped_image]          

I also used early stopping and checkpoint of keras. I could save training time  because I don't need to calculate the number of epoches.    
But I experienced the validation loss was decreased after many epoches. Because of this, I set "patience" parameter to 5.    
At first, Preprocessing time took lots of time because my system doesn't have enough memory. Thanks to generator, I could figure out this problem.    
I draw the history graph using traing loss and validation loss.   

![alt text][history_image]
