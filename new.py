import numpy as np
from mnist import MNIST
from mlxtend.preprocessing import one_hot
import matplotlib.pyplot as plt
from PIL import Image
import os
import pickle

#load data using MNIST libraries
dataset = MNIST('/Users/na1r_/OneDrive/Desktop')
img_train, labels_train = dataset.load_training()
img_test, labels_test = dataset.load_testing()

#convert to numpy arrays
img_train_np = np.asarray(img_train)
img_test_np = np.asarray(img_test)
labels_train_np = np.asarray(labels_train)
labels_test_np = np.asarray(labels_test)

#normalize the values
img_train_np = img_train_np / 255.0
img_test_np = img_test_np / 255.0

#define the training and testing sets, one hot encode the labels
X_train = img_train_np.T
X_test = img_test_np.T
Y_train = one_hot(labels_train_np).T
Y_test = one_hot(labels_test_np).T


#print shapes of training and test sets
print('X_train shape:', np.shape(X_train))
print('X_test shape:', np.shape(X_test))
print('Y_train shape: ', np.shape(Y_train))
print('Y_test shape: ', np.shape(Y_test))

#visualize training examples
index = np.random.randint(1, 60000)     # prints a random training example
first_array=X_train[:, index]           # get the randomly chosen image from thedata
reshaped_array = np.reshape(first_array, (28, 28))       #reshape image to display
plt.title('Image number '+ str(index))
plt.imshow(reshaped_array)
plt.show()



# define layer dimensions
layer_dims = [784, 40, 10]          #  define the network architecture here

#define functions
def initialize_params(num_layers):

    layer_count = len(num_layers)
    params = {}  #initialize parameters dictionary

    # use a loop to initialize parameters for each layer
    for l in range(1, layer_count):
        params['W'+str(l)] = np.random.randn(num_layers[l], num_layers[l-1]) / np.sqrt(num_layers[l-1])       # randomly initialize weights
        params['b'+str(l)] = np.zeros((num_layers[l], 1))       # initialize biases as zeros
        print('Shape of W' + str(l) + ': ', np.shape(params['W'+str(l)]))       # print the shape of all weights
        print('Shape of b' + str(l) + ':', np.shape(params['b'+str(l)]))        # print the shape of all biases
    return params

def compute_cost(A, Y, reg_const, params):
    m = A.shape[1]    # m = 60,000 (# of examples)
    L = len(params) // 2

    cost = (-1/m) * np.sum(Y * np.log(A) + (1-Y) * np.log(1- A))   # compute the cost

    sum_W = 0   # initialize the variable to hold all the sum(W^2) values added together

    # sum up all the W^2 terms for regularization
    for l in range(L):
        W = params['W' + str(l+1)]
        sum_W += np.sum(W ** 2)     # compute the sum of the weights for regularization

    cost += (reg_const / (2*m)) * sum_W         # add the regularization term
    return cost

def sigmoid(Z):
    # compute the sigmoid
    sigmoid_Z = 1 / (1 + np.exp(-Z))
    return sigmoid_Z

def dSigmoid(Z):
    # compute the derivative of the sigmoid
    sigmoid_Z = 1 / (1 + np.exp(-Z))
    dSigmoid_Z = sigmoid_Z * (1 - sigmoid_Z)
    return dSigmoid_Z

def forward_prop(X, params):
    L = len(params) // 2  # initialize L to # of layers

    Z_cache = []      # initialize Z_cache for easy backprop computation
    A_cache = []      # initialize A_cache for easy backprop computation
    A = X       # initialize A0 as X
    A_cache.append(X)

    for l in range(L):    # perform computation for all layers
        W = params['W' + str(l+1)]   #W & b for each layer are taken from params dict
        b = params['b' + str(l+1)]
        A_prev = A     # equivalent of A(L-1) in equations
        Z = np.dot(W, A_prev) + b   # compute Z
        Z_cache.append(Z)     # store Z for each layer in Z_cache
        A = sigmoid(Z)    # activation function (computes AL for each layer)
        A_cache.append(A) # store all activations in A_cache


    return A, np.asarray(Z_cache), np.asarray(A_cache)


def backprop(AL, Y, Z_cache, A_cache, params, reg_const):

    L = len(params) // 2  # number of layers

    m = AL.shape[1]     # m = 60,000 (# of training examples)

    grads = {}  # initialize grads dict to store all computed derivs
    grads['dA' + str(L)] = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))   # initialize dAL

    for l in reversed(range(L)):    # compute from last layer ---> first layer
        #print(l)
        Z_current = Z_cache[l]
        A_current = A_cache[l]
        #print('Shape of A' + str(l), np.shape(A_current))  used for debugging

        W = params['W' + str(l+1)]  # collect vals of params from params dict
        b = params['b' + str(l+1)]

        grads['dZ' + str(l+1)] = grads['dA' + str(l+1)] * dSigmoid(Z_current)
        #print('Shape of dZ' + str(l+1), np.shape(grads['dZ' + str(l+1)]))   used to debug errors
        grads['dW' + str(l+1)] = (1/m)*np.dot(grads['dZ' + str(l+1)], A_current.T) + (reg_const/m)*W
        #print('Shape of dW' + str(l+1), np.shape(grads['dW' + str(l+1)]))    used to debug errors
        grads['db' + str(l+1)] = (1/m) * np.sum(grads['dZ' + str(l+1)], axis=1, keepdims = True)
        grads['dA' + str(l)] = np.dot(W.T, grads['dZ' + str(l+1)])

    return grads


def update_params(params, grads, alpha):

    L = len(params) // 2

    #update the paramaters using the gradients from backprop w/a loop for each layer
    for l in range(L):

        params['W' + str(l + 1)] = params['W' + str(l+1)] - alpha * grads['dW' + str(l+1)]      # update each layer's parameters
        params['b' + str(l + 1)] = params['b' + str(l+1)] - alpha * grads['db' + str(l+1)]

    return params

def predict(X, params):
    # returns a matrix of 0s and 1s
    AL, z, a = forward_prop(X,params)
    #print('Probabilities: ', AL)
    preds = (AL > 0.5)     # sets all values where AL>0.5 to 1

    return preds

def accuracy(X, Y, params):

    m  = X.shape[1]
    score = 0      # keeps track of the # of correctly classified examples
    preds = predict(X, params)      # get the predictions from predict()

    for i in range(m):

        current_pred = preds[:, i]      # set the current prediction to a slice of the prediction matrix
        current_truths = Y[:, i]        # set the current truth label to a slice of the ground truth matrix

        current_pred = np.asarray(current_pred)         # convert to numpy arrays
        current_truths = np.asarray(current_truths)

        comp = current_pred == current_truths           # check if current_pred is equal to current truth
        equality = comp.all()

        if equality == True:            # if the current_pred is equal, add one to the score
            score +=1

    return ( (score/m) * 100.0)         # prints out a number between 0-100, where 100 indicates perfect classification


def model(X_train, Y_train, X_test, Y_test, layer_dims, alpha = 1.5, iterations = 2500, reg_const=0.5):

    costs = []          # initialize costs array to display the cost function
    params = initialize_params(layer_dims)      # initialize parameters

    for i in range(iterations):
        AL, lin_cache, a_cache = forward_prop(X_train, params)          # forward through netwrok

        cost = compute_cost(AL, Y_train, reg_const, params)     # compute the current cost, append to the cost array
        costs.append(cost)
        grads = backprop(AL, Y_train, lin_cache, a_cache, params, reg_const)        # backward through network

        params = update_params(params, grads, alpha)        # update parameters

        if i % 100 == 0:
            accu = accuracy(X_train, Y_train, params)       # print cost and accuracy every 100 iterations
            print('Cost after iteration ' + str(i) + ': ', str(cost) + ',', 'Accuracy: ', str(accu))


    print('Testing accuracy: ', str(accuracy(X_test, Y_test, params)))  # print the accuracy on the test set once iters are done

    # plot the cost function
    plt.plot(costs)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Costs per Iteration')
    plt.show()

    return params


#params = model(X_train, Y_train, X_test, Y_test, layer_dims)       trains the model

#pickle.dump(params, open('best_params.p', 'wb'))       stores the parameters in a pickle file after training, stops repetitive training
params = pickle.load(open('best_params.p', 'rb'))       # load the optimized parameters into the program
# predict using your own number image

img = Image.open('/Users/na1r_/OneDrive/Desktop/9-images.png').resize((28,28)).convert('L')     # load in an image, resize and  make greyscale

img_data = np.asarray(img)      # convert img to numpy array

img_flat = img_data.flatten().reshape((784,1))      # flatten the image and reshape
img_flat = img_flat / 255.0                 # normalize
plt.imshow(img_flat.reshape((28,28)))       # temporarily reshape to display the image
plt.show()

preds = predict(img_flat, params)           # predict on the inputted image

index = None
for i in range(len(preds)):             # sets index equal to the predicted number that the network outputs
    current_val = preds[i, :]

    if current_val == True:
        index = i

if type(index) == int:                  # prints the predicted number out, if no number is predicted, an error is outputted
    print('The predicted number is ' + str(index) + '.')
else:
    print('Prediction error')
