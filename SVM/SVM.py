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
from scipy.spatial.distance import pdist, squareform



##################################################################################################################################################################################################
# Data Processing
##################################################################################################################################################################################################


'''
Loads test data from the specified file - expects ',' delim. Adds in a column of 1's as the frist column of S.
'''
def loadDataSyWithOnes(fileData):
    tempSandY = np.genfromtxt(fileData, dtype=float, delimiter=',')
    S = tempSandY[:,0:-1]
    y = tempSandY[:,-1]

    oneCol = np.ones(S.shape[0])
    oneCol = oneCol[..., np.newaxis]
    S = all_data = np.append(S, oneCol, axis = 1)

    return S, y

def loadDataSy(fileData):
    tempSandY = np.genfromtxt(fileData, dtype=float, delimiter=',')
    S = tempSandY[:,0:-1]
    y = tempSandY[:,-1]

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
            prediction = w.dot(STemp[i]) * yTemp[i]
            #print("Prediction: " + str(prediction))

            gama = learnRateFunc(ephocsCount) # note: must change for test
            #print("Gama: " + str(gama))

            W0 = np.array(w)
            W0[-1] = 0

            # If wrong take step
            if (prediction <= 1):
                gamaW0 = W0 * gama
                #print("GamaW0: " + str(gamaW0))

                gamaCNyx = gama * C * N * yTemp[i] * STemp[i]
                #print("GamaCNyx: " + str(gamaCNyx))

                
                #subgrad = W0 - C * N * yTemp[i] * S[i]
                #print("Subgrad: " + str(subgrad))

                w = w - gamaW0 + gamaCNyx

            else:
                w = W0 * (1 - gama)
                #print("Subgrad: " + str(W0))

            #print("New W: " + str(w))

        #errorsCurr = countErrors(w, S, y)
        #errors.append(errorsCurr)

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

'''
Dual objective function. Linearized.
'''
def svmDualObjectiveVector(alpha, *args):
    x, y = args
    
    yAlpha = np.multiply(np.outer(y, y), np.outer(alpha, alpha))
    yAlphaX = np.multiply(yAlpha, x.dot(x.T))
    sumIJ = np.sum(yAlphaX)

    sumAlpha = np.sum(alpha)

    return 0.5 * sumIJ - sumAlpha

'''
Dual objective function. Linearized. Using Kernel K
'''
def svmDualObjectiveKernel(alpha, *args):
    x, y, gamma = args
    Kval = K(x, gamma)
    yAlpha = np.multiply(np.outer(y, y), np.outer(alpha, alpha))
    yAlphaX = np.multiply(yAlpha, Kval)
    sumIJ = np.sum(yAlphaX)

    sumAlpha = np.sum(alpha)
    return 0.5 * sumIJ - sumAlpha

'''
Gaussian kernel for a batch.
'''
def K(X, gamma):
    p = pdist(X, 'euclidean')
    pairwise_dists = squareform(p)
    K = np.exp(-1 * np.square(pairwise_dists) / gamma)
    return K

'''
Gaussian kernel for a single example.
'''
def KSingle(xi, xj, gamma):
    normedSquared = np.linalg.norm(xi - xj) ** 2
    return np.exp(-normedSquared / gamma)


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

def getIdxNonZero(alphas, c):
    allNon0 = []
    btw = []
    for i in range(alphas.shape[0]):
        if alphas[i] > 0.0:
            allNon0.append(i)
            if alphas[i] < c:
                btw.append(i)
    if len(allNon0) == alphas.shape[0]:
        print("Same Length!!!")
    return allNon0, btw

def getOptimalW(alphaStars, supportVecs, y):
    ay = np.multiply(alphaStars, y)
    #oneCol[..., np.newaxis]
    ay = ay[..., np.newaxis]
    allMulted = np.multiply(ay, supportVecs)
    return np.sum(allMulted, axis=0)

def getOptimalB(w, bVecs, y):
    sum = 0.0
    for i in range(bVecs.shape[0]):
        sum += y[i] - w.dot(bVecs[i])
    return sum / bVecs.shape[0]

def getOptimalBK(alphas, supportVecs, ySupport, gamma, bVecs, y):
    sum = 0.0
    for i in range(bVecs.shape[0]):
        
        partialSum = 0.0
        for j in range(alphas.shape[0]):
            partialSum += alphas[j] * ySupport[j] * KSingle(supportVecs[j], bVecs[i], gamma)

        sum += y[i] - partialSum

    return sum / bVecs.shape[0]

def countErrorsK(alphas, supportVecs, ySupport, b, gamma, S, y):
    errors = 0
    for i in range(S.shape[0]):
        prediction = 0.0
        for j in range(alphas.shape[0]):
            prediction += alphas[j] * ySupport[j] * KSingle(supportVecs[j], S[i], gamma)
        prediction += b

        # print(prediction)

        if (prediction >= 0.0 and y[i] < 0.0) or (prediction < 0.0 and y[i] >= 0.0):
            errors+=1

    return errors


##################################################################################################################################################################################################
# MAIN
##################################################################################################################################################################################################



def main(argv):
    script_dir = os.path.dirname(__file__)
    start = str(script_dir)
    
    SOnes, y = loadDataSyWithOnes(start + "/bank-note/bank-note/train.csv")
    STestOnes, yTest = loadDataSyWithOnes(start + "/bank-note/bank-note/test.csv")
    S, _ = loadDataSy(start + "/bank-note/bank-note/train.csv")
    STest, _ = loadDataSy(start + "/bank-note/bank-note/test.csv")

    changeY(y)
    changeY(yTest)

    C = [(100.0/873.0), (500.0/873.0), (700.0/873.0)]
    gamma = [0.1, 0.5, 1, 5, 100]

    if(argv[0] == "2a"):
        print("2.2.a - Linear Primal")
        for c in C:
            print("C value of " + str(c))
            w, numEphocs, _ = SSGD_SVM(SOnes, y, 100, scheduleA, C=c, doRando=True)
            print("Error train set: ")
            print(countErrors(w, SOnes, y) / S.shape[0])
            print("Error test set: ")
            print(countErrors(w, STestOnes, yTest) / STestOnes.shape[0])
            print("Weights: ")
            print(w)

    if(argv[0] == "2b"):
        print("2.2.b - Linear Primal")
        for c in C:
            print("C value of " + str(c))
            w, numEphocs, _ = SSGD_SVM(SOnes, y, 100, scheduleB, C=c, doRando=True)
            print("Error train set: ")
            print(countErrors(w, SOnes, y) / S.shape[0])
            print("Error test set: ")
            print(countErrors(w, STestOnes, yTest) / STestOnes.shape[0])
            print("Weights: ")
            print(w)

    if(argv[0] == "3a"):
        print("2.3.a - Linear Dual")
        print("WARNING: Slow!")
        for c in C:
            print("C value of " + str(c))
            args = (S, y)
            bounds = [(0.0, c)] * S.shape[0]
            cons = ({'type':'eq','fun':lambda x: x.dot(y)})
            x0 = np.zeros(S.shape[0], dtype='float')

            sol = minimize(fun=svmDualObjectiveVector, x0=x0, args=args, method='SLSQP', constraints=cons, bounds=bounds)
            alphas = sol.x

            allNon0Idx, btw0andC = getIdxNonZero(alphas, c)

            supportVecs = S[allNon0Idx, :]
            supportVecY = y[allNon0Idx]
            alphaStar = alphas[allNon0Idx]
            bVecs = S[btw0andC, :]
            bVecY = y[btw0andC]

            wStar = getOptimalW(alphaStar, supportVecs, supportVecY)
            bStar = getOptimalB(wStar, bVecs, bVecY)
            w = np.append(wStar, bStar)

            print("Error train set: ")
            print(countErrors(w, SOnes, y) / S.shape[0])
            print("Error test set: ")
            print(countErrors(w, STestOnes, yTest) / STestOnes.shape[0])
            print("Weights: ")
            print(w)

    if(argv[0] == "3b"):
        print("2.3.b - Gaussain Dual")
        print("WARNING: Very Slow!")
        for c in C:
            for g in gamma:
                #c = C[1]
                #g = gamma[1]
                print("C value of: " + str(c) + " Gamma value of: " + str(g))
                args = (S, y, g)
                bounds = [(0.0, c)] * S.shape[0]
                cons = ({'type':'eq','fun':lambda x: x.dot(y)})
                x0 = np.zeros(S.shape[0], dtype='float')

                sol = minimize(fun=svmDualObjectiveKernel, x0=x0, args=args, method='SLSQP', constraints=cons, bounds=bounds)
                alphas = sol.x

                allNon0Idx, btw0andC = getIdxNonZero(alphas, c)

                supportVecs = S[allNon0Idx, :]
                supportVecY = y[allNon0Idx]
                alphaStar = alphas[allNon0Idx]
                bVecs = S[btw0andC, :]
                bVecY = y[btw0andC]

                bStar = getOptimalBK(alphaStar, supportVecs, supportVecY, g, bVecs, bVecY)

                print("Error train set: ")
                print(countErrorsK(alphaStar, supportVecs, supportVecY, bStar, g, S, y) / S.shape[0])
                print("Error test set: ")
                print(countErrorsK(alphaStar, supportVecs, supportVecY, bStar, g, STest, yTest) / STest.shape[0])

    print("")

if __name__ == '__main__':
    main(sys.argv[1:])



