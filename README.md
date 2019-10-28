## MNIST digit classification task using PyTorch

Author: Awet Haileslassie Gebrehiwot(awethaileslassie21@gmail.com)

If you have any questions on the code or README, please feel free to contact me.

### Requirments
python 
Pytorch
Torchvision
Numpy
Matplotlib


### Performance

CNN: 99.58% (test accuracy) in 10 epoches

### Description

The code provides a predefined model.
The  model is CNN (convolutional neural networks).
The traing and testing dataset labels are modified according "1" if the image represents the number 3, and "0" otherwise. 
Model returns "1" if the image represents the number 3, and "0" otherwise.
The network is trained and evaluated on MNIST dataset with classification accuracy. 

#### Part 1: data preparation

The code use as input a MNIST image (http://yann.lecun.com/exdb/mnist/)  
We split training data (60000) 
Test data (10000) is already given.
T he data for CNN, is 4-dim tensor.

#### Part 3: CNN model

The CNN model contains two convolutional layers with max pooling and ReLU function.
Each convolutional layer has 16 and 32 maps, respectively.
Both uses 5x5 kernels.
Dropout of keep probability 0.2 is used for regularization.
After convolution, the activation is flattened and passes through a fully connected layer.
Fully connected layer has 256 dimension.

#### Part 4: Training and testing model with Training dataset 
The traing and testing dataset labels are modified according "1" if the image represents the number 3, and "0" otherwise. 
For every epoch, we train network by train set .
We use SGD optimizer with learning_rate = 0.05, momentum = 0.9 for parameter update.

#### Part 5: Test model with Test dataset

The CNN model is evaluated with test dataset

# Model Loss during Training for 50 epoches (50\*6000) 
![accuracy.png](https://upload-images.jianshu.io/upload_images/1231993-d76f92bd4f431100.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

