## MNIST digit classification task using PyTorch

Author: Awet Haileslassie Gebrehiwot(awethaileslassie21@gmail.com)

If you have any questions on the code or README, please feel free to contact me.

### Requirments
python 
Pytorch
Torchvision
Numpy
Matplotlib

### Running the code

Open terminal and navigate to the directory where the code exist
then simply type:

python mnist_cnn_A_1.py 

### Performance

CNN: 99.9% (test accuracy) in 50 epoches
Test set: Avg. loss:  3.9577173387715444e-05 	[ Accuracy:  9990 / 10000  (  99.9 %)

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

#### Accuracy calculation

Sigmoid function outputs a value in range [0,1] which corresponds to the probability of the given sample belonging to positive class (class 3). Everything below 0.5 is labeled with zero (i.e. class other than 3) and everything above 0.5 is labeled with one. So to find the predicted class, I have used:

output = network(data)
pred = output.data > 0.5

# Model Loss during Training for 50 epoches 
![Loss_50_epoch.png](https://github.com/awethaileslassie/awet_mnist_pytorch/blob/master/Loss_50_epoch.png)

# Model Accuracy during Training for 50 epoches
![Accuracy_50_epoch.png](https://github.com/awethaileslassie/awet_mnist_pytorch/blob/master/Accuracy_50_epoch.png)

