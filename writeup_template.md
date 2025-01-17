# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Model_Visualization.png "Model Visualization"
[image2]: ./examples/Gray_Image.jpg "Grayscaling"
[image3]: ./examples/center_2018_01_03_21_04_16_060.jpg "Recovery Image"
[image4]: ./examples/right_2018_01_03_21_04_16_060.jpg "Recovery Image"
[image5]: ./examples/left_2018_01_03_21_04_16_060.jpg "Recovery Image"
[image6]: ./examples/Normal_Image.jpg "Normal Image"
[image7]: ./examples/Reversed_Image.jpg "Flipped Image"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 55-72) 

The model includes RELU layers to introduce nonlinearity, the data is normalized in the model using a Keras lambda layer (code line 53), and the images are cropped to focus on only the portion of the image that is useful for predicting a steering angle (code line 54). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 35-46). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 71).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and flipping the original images. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a simpler model and increase layers.

My first step was to use a convolution neural network model similar to the LeNet architecture. This architecture seemed to produce results that worked well on a straight road in the simulator but failed at edges. Then I changed my model to Nvidia architecture which has more convolution layers. I thought this model might be appropriate because by increasing the number of convolution layers the model can better extract information from images. Finally, I used adam optimizer to reduce mean squared error.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I reduced the number of training epochs from 5 to 3. One can also introduce dropouts in the architecture or introduce regularization to reduce overfitting of the model.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track and was stuck. To improve the driving behavior in these cases, I used left and right camera images with small correction added to the steering angle.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 52-65) consisted of a convolution neural network with the following layers and layer sizes ...

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back to the centre. These images show what a recovery looks like starting from centre, right and left cameras :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help with bias. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 48216 number of data points. I then preprocessed this data by normalizing the pixels and cropping the images to focus on useful portion of the image.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
