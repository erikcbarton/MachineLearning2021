
# -*- coding: utf-8 -*-
"""


@author: Erik Barton
"""
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import sys, getopt






##################################################################################################################################################################################################
# Data Processing
##################################################################################################################################################################################################


'''
Loads test data from the specified file - expects ',' delim.
'''
def loadDataSy(fileData):
    tempSandY = np.genfromtxt(fileData, dtype=float, delimiter=',')
    S = tempSandY[:,0:-1]
    y = tempSandY[:,-1]

    oneCol = np.ones(S.shape[0])
    oneCol = oneCol[..., np.newaxis]
    S = all_data = np.append(oneCol, S, axis = 1)

    return S, y


##################################################################################################################################################################################################
# Random Forest
##################################################################################################################################################################################################

'''
Calculate the entropy from a dictionary mapping value (lbl) to counts.
'''
def entropy(dictionaryYValues, countTotal, logBase):
    entFinal = 0.0
    countTotal += 0.0
    #print("Calculating the entropy.")

    for key, value in dictionaryYValues.items():
        px = value / countTotal
        #print("p_x of " + str(key) + " is %f" % px)
        entFinal += (px * np.log(px))/np.log(logBase)

    entFinal *= -1
    #print("Entropy: " + str(entFinal))


    return entFinal

'''
Stocastic algorithm
'''
def BatchGradientDescent(S, y, lr, maxItters, convergenceBound = 0.0):
    costList = []
    w = np.zeros(S.shape[1])
    gradientPrev = np.zeros(w.shape)
    printItter = 0
    for i in range(maxItters):
        
        if i == printItter:
            #print(str(i) + " / " + str(maxItters))
            printItter += 5000
            costList.append(costCurr(S, y, w))

        gradient = calcGradient(S, y, w)

        gradDif = gradient - gradientPrev
        norm = np.linalg.norm(gradDif, ord=1)
        if(norm < convergenceBound):
            return w, i+1, costList
        
        w = w - lr * gradient
        gradientPrev = gradient
    return w, maxItters, costList

def StochasticGradientDescent(S, y, lr, maxItters):
    costList = []
    w = np.zeros(S.shape[1])
    gradientPrev = np.zeros(w.shape)
    printItter = 0
    for i in range(maxItters):
        index = i % y.shape[0]

        if i == printItter:
            printItter += 100
            costList.append(costCurr(S, y, w))
        
        w = w + lr * (y[index] - w.dot(S[index,:]))  * S[index,:]
    return w, maxItters, costList


def costCurr(S, y, w):
    xw = S.dot(w)
    yminus = y - xw
    yminusSquare = np.square(yminus)
    sum = np.sum(yminusSquare)
    return 0.5 * sum

def calcGradient(S, y, w):
    xw = S.dot(w)
    yminus = y - xw # 1xm
    wAlmost = yminus.dot(S)
    return -1*wAlmost

##################################################################################################################################################################################################
# HELPERS
##################################################################################################################################################################################################

def countErrors(classifier, S, y):
    errors = 0
    for i in range(S.shape[0]):
        predicted = classifier.label(S[i,:])
        if predicted != y[i]:
            errors += 1

    return errors

##################################################################################################################################################################################################
# MAIN
##################################################################################################################################################################################################

def main(argv):
    script_dir = os.path.dirname(__file__)
    start = str(script_dir)
    
    S, y = loadDataSy(start + "/concrete/concrete/train.csv")
    STest, yTest = loadDataSy(start + "/concrete/concrete/test.csv")

    #x = np.array(
    #    [
    #        [1, 1, -1, 2],
    #        [1, 1, 1, 3],
    #        [1, -1, 1, 0],
    #        [1, 1, 2, -4],
    #        [1, 3, -1, -1]
    #     ])
    #y = np.array([1, 4, -1, -2, 0])

    #w, numberItterations, costList = BatchGradientDescent(S, y, 0.00045, 2000, convergenceBound=1e-4)

    if(argv[0] == "a"):
        w, numberItterations, costList = BatchGradientDescent(S, y, 0.0000152, 200000, convergenceBound=1e-6)
        print("Batch Grad")
        print("Number of itters: " + str(numberItterations))
        print("w: " + str(w))
        print("Cost jumping by 5000's of itterations:")
        print(costList)
        print("Cost test: ")
        print(costCurr(STest, yTest, w))

    if(argv[0] == "b"):
        w, numberItterations, costList = StochasticGradientDescent(S, y, 0.0005, 200000)
        print("Stochast Grad")
        print("Number of itters: " + str(numberItterations))
        print("w: " + str(w))
        print("Cost jumping by 100's of itterations:")
        print(costList)
        print("Cost test: ")
        print(costCurr(STest, yTest, w))

    if(argv[0] == "c"):
        xxt = (S.T).dot(S)
        xy = (S.T).dot(y)
        xxinv = np.linalg.inv(xxt)
        fin = xxinv.dot(xy)
        print("Optimal")
        print("w: " + str(fin))
        print("Cost test: ")
        print(costCurr(STest, yTest, fin))




if __name__ == '__main__':
    main(sys.argv[1:])


