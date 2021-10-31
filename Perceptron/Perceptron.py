
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
Loads test data from the specified file - expects ',' delim. Adds in a column of 1's as the frist column of S.
'''
def loadDataSy(fileData):
    tempSandY = np.genfromtxt(fileData, dtype=float, delimiter=',')
    S = tempSandY[:,0:-1]
    y = tempSandY[:,-1]

    oneCol = np.ones(S.shape[0])
    oneCol = oneCol[..., np.newaxis]
    S = all_data = np.append(oneCol, S, axis = 1)

    return S, y

'''
Convert 0.0 in y to -1.0
'''
def changeY(y):
    for i in range(y.shape[0]):
        if (y[i] < 1e-3):
            y[i] = -1.0



##################################################################################################################################################################################################
# Perceptron Algorithms
##################################################################################################################################################################################################

'''
Perceptron algorithm
'''
def Perceptron(S, y, ephocs, r=1):
    w = np.zeros(S.shape[1])

    for e in range(ephocs):
        for i in range(S.shape[0]):
            # Make prediction
            prediction = w.dot(S[i])

            # If wrong take step
            if (prediction >= 0.0 and y[i] < 0.0) or (prediction < 0.0 and y[i] >= 0.0):
                step = r * (y[i] * S[i])
                w += step
    return w

'''
Voted Perceptron algorithm
'''
def VotedPerceptron(S, y, ephocs, r=1):
    w = np.zeros(S.shape[1])

    wVectors = []
    Cs = []
    C = 0
    for e in range(ephocs):
        for i in range(S.shape[0]):
            # Make prediction
            prediction = w.dot(S[i])

            # If wrong take step
            if (prediction >= 0.0 and y[i] < 0.0) or (prediction < 0.0 and y[i] >= 0.0):
                # Store the old one
                wVectors.append(np.array(w))
                Cs.append(C)

                # Take the step
                step = r * (y[i] * S[i])
                w += step

                # Reset
                C=1
            else:
                C+=1
    wVectors.append(np.array(w))
    Cs.append(C)
    return wVectors, Cs

'''
Averaged Perceptron algorithm
'''
def AvgPerceptron(S, y, ephocs, r=1):
    w = np.zeros(S.shape[1])
    a = np.zeros(S.shape[1])

    for e in range(ephocs):
        for i in range(S.shape[0]):
            # Make prediction
            prediction = w.dot(S[i])

            # If wrong take step
            if (prediction >= 0.0 and y[i] < 0.0) or (prediction < 0.0 and y[i] >= 0.0):
                step = r * (y[i] * S[i])
                w += step
            a += w
    return a


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

def printVoted(Ws, Cs):
    for i in range(len(Ws)):
        print("W_" + str(i) +": " + str(Ws[i]))
        print("C_" + str(i) +": " + str(Cs[i]))



##################################################################################################################################################################################################
# MAIN
##################################################################################################################################################################################################

def main(argv):
    script_dir = os.path.dirname(__file__)
    start = str(script_dir)
    
    S, y = loadDataSy(start + "/bank-note/bank-note/train.csv")
    STest, yTest = loadDataSy(start + "/bank-note/bank-note/test.csv")

    changeY(y)
    changeY(yTest)

    if(argv[0] == "a"):
        w = Perceptron(S, y, 10)
        print("Perceptron W:")
        print(w)
        errorsPerceptron = countErrors(w, STest, yTest)
        print("With " + str(errorsPerceptron / (0.0 + yTest.shape[0])) + " average prediction error on the test set.")

    if(argv[0] == "b"):
        wVectors, cVectors = VotedPerceptron(S, y, 10)
        print("Voting Perceptron W vectors and Cs:")
        printVoted(wVectors, cVectors)
        errorsVPerceptron = countErrorsVoted(wVectors, cVectors, STest, yTest)
        print("With " + str(errorsVPerceptron / (0.0 + yTest.shape[0])) + " average prediction error on the test set.")

    if(argv[0] == "c"):
        w = AvgPerceptron(S, y, 10)
        print("Average Perceptron W:")
        print(w)
        errorsPerceptron = countErrors(w, STest, yTest)
        print("With " + str(errorsPerceptron / (0.0 + yTest.shape[0])) + " average prediction error on the test set.")




if __name__ == '__main__':
    main(sys.argv[1:])



