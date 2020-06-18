# mnist-classification-from-scratch
A neural network written using only Python's NumPy library to classify the MNIST digits dataset. External libraries are only utilized for preprocessing purposes. 

## Description
The program implements a three layer neural network with 784 input units, 40 hidden units, and 10 output units. External libraries are only utilized for purposes such as loading in data, image manipulation and one-hot encoding. 

The model and all helper functions are in the file new.py. best-params.p contains a dictionary of parameters outputted by the model after 2500 iterations of batch gradient descent. All the files with the .idx3-ubyte extension hold the MNIST data, which can be downloaded and utilized in a program using the MNIST library in Python. 

The number-imgs folder compiles numerous images that I inputted into the network to test its classification ability. The folder contains both a correctly classified and an incorrectly classified example for each number. The code at the bottom allows the user to input their own image into the network to make predictions. The MNIST data consists of centered white numbers on black backgrounds, making it such that the model tends to only correctly classify images that fulfill the aforementioned constraints.

## Details
After training, the model achieved approximately 94% training accuracy and 93% test accuracy. 

Details about the model:
1. Cross-entropy loss is used.
2. It employs batch gradient descent and L2 regularization.
3. The performance metric utilized to judge performance is classification accuracy.
4. All layers utilize a sigmoid activation function. 



