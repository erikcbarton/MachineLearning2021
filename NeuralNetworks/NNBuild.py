

# -*- coding: utf-8 -*-
"""
Neural Network Code for 3 layer neural network. Addapted from Stanford's CS231n code structure.

@author: Erik Barton
"""
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import sys, getopt
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform



##################################################################################################################################################################################################
# Data Processing
##################################################################################################################################################################################################


'''
Loads test data from the specified file - expects ',' delim. Adds in a column of 1's as the frist column of S.
'''
def loadDataSy(fileData):
    tempSandY = np.genfromtxt(fileData, dtype=float, delimiter=',')
    S = tempSandY[:,0:-1]
    y = tempSandY[:,-1]

    #oneCol = np.ones(S.shape[0])
    #oneCol = oneCol[..., np.newaxis]
    #S = all_data = np.append(S, oneCol, axis = 1)

    return S, y

'''
Convert 0.0 in y to -1.0
'''
def changeY(y):
    for i in range(y.shape[0]):
        if (y[i] < 1e-3):
            y[i] = -1.0



##################################################################################################################################################################################################
# NN Algorithms
##################################################################################################################################################################################################

'''
Three layer deep neural network. Sigmoid acts and square loss.
'''
class ThreeLayerNet(object):

  def __init__(self, input_size, hidden_size, std=1):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, hidden_size)
    self.params['b2'] = np.zeros(hidden_size)
    self.params['W3'] = std * np.random.randn(hidden_size, 1)
    self.params['b3'] = np.zeros(1)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # Forward pass        #
    #############################################################################
    hidden01 = X.dot(W1) + np.expand_dims(b1, axis=0) # =NxH
    temp01 = np.ones(hidden01.shape)
    hiddenActivated01 = np.divide(temp01, np.add(temp01, np.exp(np.negative(hidden01)))) # =NxH

    hidden12 = hiddenActivated01.dot(W2) + np.expand_dims(b2, axis=0) # =NxH
    temp12 = np.ones(hidden12.shape)
    hiddenActivated12 = np.divide(temp12, np.add(temp12, np.exp(np.negative(hidden12)))) # =NxH

    scores = hiddenActivated12.dot(W3) + np.expand_dims(b3, axis=0) # =Nx1
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # Compute the loss                                                        #
    #############################################################################
    loss = 0.0

    loss = np.sum(0.5 * np.square(np.subtract(scores, y)))



    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # Comp Grads #
    #############################################################################
    # Grad


    dscores = np.subtract(scores, y) # Nx1

    dB3 = np.sum(dscores, axis=0) # = 1
    dW3 = (hiddenActivated12.T).dot(dscores) # = Hx1

    dHidden2 = dscores.dot(W3.T) # = NxH
    temp = np.ones(hiddenActivated12.shape)
    dHidden2 =  hiddenActivated12 * (np.subtract(temp, hiddenActivated12)) * dHidden2 # = NxH

    dB2 = np.sum(dHidden2, axis=0) # = H 
    dW2 = (hiddenActivated01.T).dot(dHidden2) # = HxH

    dHidden1 = dHidden2.dot(W2.T) # = NxH
    temp = np.ones(hiddenActivated01.shape)
    dHidden1 =  hiddenActivated01 * (np.subtract(temp, hiddenActivated01)) * dHidden1 # = NxH 

    dB1 = np.sum(dHidden1, axis=0) # = H
    dW1 = (X.T).dot(dHidden1) # = DxH


    grads['W1'] = dW1
    grads['b1'] = dB1
    grads['W2'] = dW2
    grads['b2'] = dB2
    grads['W3'] = dW3
    grads['b3'] = dB3

    return loss, grads

  def train(self, X, y, X_val, y_val,
            ephocs, learnRateFunc,
            verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    train_acc_history = []
    val_acc_history = []
    ephocsCount = 0

    for e in range(ephocs):
        ephocsCount = e + 1

        idx = np.random.choice(X.shape[0], X.shape[0], replace=False) 
        XTemp = X[idx]
        yTemp = y[idx]

        if XTemp.shape != X.shape:
            print("WARNING temp array not correct shape.")

        gamma = learnRateFunc(ephocsCount)

        for i in range(XTemp.shape[0]):
            X_batch = XTemp[i, :]
            y_batch = yTemp[i]

            X_batch = X_batch[np.newaxis, :] # Make sure 1xInput
            y_batch = np.atleast_2d(y_batch)

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch)

            self.params['W1'] += -1 * gamma * grads['W1']
            self.params['b1'] += -1 * gamma * grads['b1']
            self.params['W2'] += -1 * gamma * grads['W2']
            self.params['b2'] += -1 * gamma * grads['b2']
            self.params['W3'] += -1 * gamma * grads['W3']
            self.params['b3'] += -1 * gamma * grads['b3']

        
        # Check loss train and valid
        predX = self.predict(X)
        predX_val = self.predict(X_val)
        sub = np.subtract(predX, y)
        square = np.square(sub)
        trainLoss = np.sum(0.5 * square)
        validLoss = np.sum(0.5 * np.square(np.subtract(predX_val, y_val)))
        train_acc_history.append(trainLoss)
        val_acc_history.append(validLoss)

        if verbose and e % 10 == 0:
            print(str(e) +": Training loss: " + ": " + str(trainLoss) + " | Validation loss: " + str(validLoss))

    return {
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }
   

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    scores = self.loss(X)
    y_pred = scores # = Nx1

    return y_pred



def schedule(t):
    gama0 = 0.001 #0.0000030517578125
    a = 2.0
    return (gama0) / (1 + (gama0 / a) * t)




##################################################################################################################################################################################################
# HELPERS
##################################################################################################################################################################################################

def countErrors(w, S, y):
    errors = 0
    for i in range(S.shape[0]):
        prediction = w.dot(S[i])

        if (prediction >= 0.0 and y[i] < 0.0) or (prediction < 0.0 and y[i] >= 0.0):
            errors+=1

    return errors

def countErrorsNN(nn, S, y):
    errors = 0
    predictions = nn.predict(S)
    for i in range(y.shape[0]):
        if predictions[i] >= 0.5 and y[i] < 0.5:
            errors+=1
        elif predictions[i] < 0.5 and y[i] >= 0.5:
            errors+=1
    return errors

def countErrorsNN0(nn, S, y):
    errors = 0
    predictions = nn.predict(S)
    for i in range(y.shape[0]):
        if predictions[i] >= 0.0 and y[i] < 0.0:
            errors+=1
        elif predictions[i] < 0.0 and y[i] >= 0.0:
            errors+=1
    return errors




##################################################################################################################################################################################################
# MAIN
##################################################################################################################################################################################################


def main(argv):

    #net = ThreeLayerNet(3, 3)
    #X = np.ones((1,3))
    #y = np.ones((1,1))
    #scores = net.loss(X, y)

    #tempx = np.ones((2,2))
    #temp = tempx[0]
    #print(temp.shape)


    #script_dir = os.path.dirname(__file__)
    #start = str(script_dir)
    
    S, y = loadDataSy("C:/Users/erikc/source/repos/PerceptronBuild/PerceptronBuild/bank-note/bank-note/train.csv")
    STest, yTest = loadDataSy("C:/Users/erikc/source/repos/PerceptronBuild/PerceptronBuild/bank-note/bank-note/test.csv")

    WVals = [5, 10, 25, 50, 100]

    print("Non Zero Init")
    w = WVals[0]
    width = w-1
    net = ThreeLayerNet(4, width)
    net.train(S, y, STest, yTest, 200, schedule, verbose=False)

    print("Width " + str(w))
    print("Training Error: " + str(countErrorsNN(net, S, y)/S.shape[0]))
    print("Test Error: " + str(countErrorsNN(net, STest, yTest)/STest.shape[0]))
    print("")

    #if (argv[0] == "c"):
    #    print("Zero Init")
    #    for w in WVals:
    #        width = w-1
    #        net = ThreeLayerNet(4, width, std=0)
    #        net.train(S, y, STest, yTest, 200, schedule, verbose=True)

    #        print("Width " + str(w))
    #        print("Training Error: " + str(countErrorsNN(net, S, y)/S.shape[0]))
    #        print("Test Error: " + str(countErrorsNN(net, STest, yTest)/STest.shape[0]))
    


if __name__ == '__main__':
    main(sys.argv[1:])




