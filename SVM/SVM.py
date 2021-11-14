

# -*- coding: utf-8 -*-
"""


@author: Erik Barton
"""
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import sys, getopt
from scipy.optimize import minimize



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

    oneCol = np.ones(S.shape[0])
    oneCol = oneCol[..., np.newaxis]
    S = all_data = np.append(S, oneCol, axis = 1)

    return S, y

'''
Convert 0.0 in y to -1.0
'''
def changeY(y):
    for i in range(y.shape[0]):
        if (y[i] < 1e-3):
            y[i] = -1.0



##################################################################################################################################################################################################
# SVM Algorithms
##################################################################################################################################################################################################

'''
SSGD SVM algorithm
'''
def SSGD_SVM(S, y, ephocs, learnRateFunc, C=1, doRando=False):
    w = np.zeros(S.shape[1])
    N = S.shape[0]

    ephocsCount = 0
    errors = []

    for e in range(ephocs):
        ephocsCount = e + 1

        if (not doRando):
            STemp = np.array(S)
            yTemp = np.array(y)
        else:
            idx = np.random.choice(S.shape[0], S.shape[0], replace=False)
            STemp = S[idx]
            yTemp = y[idx]

        if STemp.shape != S.shape:
            print("WARNING temp array not correct shape.")

        for i in range(STemp.shape[0]):
            # Make prediction
            prediction = w.dot(S[i]) * yTemp[i]
            #print("Prediction: " + str(prediction))

            gama = learnRateFunc(ephocsCount) # note: must change for test
            #print("Gama: " + str(gama))

            W0 = np.array(w)
            W0[-1] = 0

            # If wrong take step
            if (prediction <= 1):
                gamaW0 = W0 * gama
                #print("GamaW0: " + str(gamaW0))

                gamaCNyx = gama * C * N * yTemp[i] * S[i]
                #print("GamaCNyx: " + str(gamaCNyx))

                
                #subgrad = W0 - C * N * yTemp[i] * S[i]
                #print("Subgrad: " + str(subgrad))

                w = w - gamaW0 + gamaCNyx

            else:
                w = W0 * (1 - gama)
                #print("Subgrad: " + str(W0))

            #print("New W: " + str(w))

        errorsCurr = countErrors(w, S, y)
        errors.append(errorsCurr)

    return w, ephocsCount, errors

'''
Learning rate gama / (1 + gama/a * t)
'''
def scheduleA(numEphoc):
    gama0 = 0.0000030517578125
    a = 1.0
    return (gama0) / (1 + (gama0 / a) * numEphoc)

'''
Learning rate gama / (1 + t)
'''
def scheduleB(numEphoc):
    gama0 = 0.0000030517578125
    return (gama0) / (1 + numEphoc)

'''
For testing purposes
'''
def fixedLRTesting(t):
    options = [0.01, 0.005, 0.0025]
    return options[t]




################ Dual ####################

'''
Dual objective function. NON-linearized.
'''
def svmDualObjective(alpha, *args):
    x, y = args
    sumIJ = 0.0
    for i in range(alpha.shape[0]):
        for j in range(alpha.shape[0]):
            sumIJ += y[i] * y[j] * alpha[i] * alpha[j] * (x[i].dot(x[j]))

    sumAlpha = 0.0
    for i in range(alpha.shape[0]):
        sumAlpha += alpha[i]

    return 0.5 * sumIJ - sumAlpha




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

def countErrorsVoted(Ws, Cs, S, y):
    errors = 0
    for i in range(S.shape[0]):
        prediction = 0.0
        for j in range(len(Ws)):
            prediction += Cs[j] * Ws[j].dot(S[i])
        

        if (prediction >= 0.0 and y[i] < 0.0) or (prediction < 0.0 and y[i] >= 0.0):
            errors+=1

    return errors



##################################################################################################################################################################################################
# MAIN
##################################################################################################################################################################################################


def calVolume(x):
    L = x[0]
    W = x[1]
    H = x[2]
    Volume = L*H*W
    return Volume


# define the function of surface
def calSurface(x):
    L = x[0]
    W = x[1]
    H = x[2]
    Surface = 2*(L*H + H*W + L*W)
    return Surface

# define the objective function: maximize vol <=> minimize -vol
def objective(x):
    return -calVolume(x)


# define the constrain for optimization
def constrain(x):
    return 10-calSurface(x)





def main(argv):
    script_dir = os.path.dirname(__file__)
    start = str(script_dir)
    
    S, y = loadDataSy(start + "/bank-note/bank-note/train.csv")
    STest, yTest = loadDataSy(start + "/bank-note/bank-note/test.csv")

    changeY(y)
    changeY(yTest)

    C = [(100.0/873.0), (500.0/873.0), (700.0/873.0)]

    if(argv[0] == "1a"):
        for c in C:
            print("C value of " + str(c))
            w, numEphocs, _ = SSGD_SVM(S, y, 100, scheduleA, C=c, doRando=True)
            print("Error train set: ")
            print(countErrors(w, STest, yTest) / S.shape[0])
            print("Error test set: ")
            print(countErrors(w, STest, yTest) / STest.shape[0])
            print("Weights: ")
            print(w)

    if(argv[0] == "1b"):
        for c in C:
            print("C value of " + str(c))
            w, numEphocs, _ = SSGD_SVM(S, y, 100, scheduleB, C=c, doRando=True)
            print("Error train set: ")
            print(countErrors(w, STest, yTest) / S.shape[0])
            print("Error test set: ")
            print(countErrors(w, STest, yTest) / STest.shape[0])
            print("Weights: ")
            print(w)

    if(argv[0] == "2a"):
        c = C[1]
        args = (S, y)
        bounds = [(0.0, c)] * S.shape[0]
        cons = ({'type':'eq','fun':lambda x: x.dot(y)})
        x0 = np.zeros(S.shape[0], dtype='float')

        sol = minimize(fun=svmDualObjective, x0=x0, args=args, method='SLSQP', constraints=cons, bounds=bounds)
    
        print(sol)






if __name__ == '__main__':
    main(sys.argv[1:])



