
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

    wToCVectors = {}
    C = 0
    for e in range(ephocs):
        for i in range(S.shape[0]):
            # Make prediction
            prediction = w.dot(S[i])

            # If wrong take step
            if (prediction >= 0.0 and y[i] < 0.0) or (prediction < 0.0 and y[i] >= 0.0):
                # Store the old one
                stringW = str(w) 
                if stringW in wToCVectors:
                    wToCVectors[stringW] = (wToCVectors[stringW][0], wToCVectors[stringW][1] + C)
                else:
                    wToCVectors[stringW] = (np.array(w), C) # make sure is copy

                # Take the step
                step = r * (y[i] * S[i])
                w += step

                # Reset
                C=1
            else:
                C+=1
    stringW = str(w)
    if stringW in wToCVectors:
        wToCVectors[stringW] = (wToCVectors[stringW][0], wToCVectors[stringW][1] + C)
    else:
        wToCVectors[stringW] = (np.array(w), C) # make sure is copy
    return wToCVectors

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

def countErrorsVoted(wToC, S, y):
    errors = 0
    for i in range(S.shape[0]):
        prediction = 0.0
        for key in wToC:
            W, C = wToC[key]
            prediction += C * W.dot(S[i])
        

        if (prediction >= 0.0 and y[i] < 0.0) or (prediction < 0.0 and y[i] >= 0.0):
            errors+=1

    return errors

def printVoted(wToC):
    i = 0
    for key in wToC:
        W, C = wToC[key]
        print("W_" + str(i) +": [" + str(W[0]) + ", " + str(W[1]) + ", " + str(W[2]) + ", " + str(W[3]) + ", " + str(W[4]) 
            + "] " + "C_" + str(i) +": " + str(C))
        i+= 1



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
        wToCMap = VotedPerceptron(S, y, 10)
        print("Voting Perceptron W vectors and Cs:")
        printVoted(wToCMap)
        errorsVPerceptron = countErrorsVoted(wToCMap, STest, yTest)
        print("With " + str(errorsVPerceptron / (0.0 + yTest.shape[0])) + " average prediction error on the test set.")

    if(argv[0] == "c"):
        w = AvgPerceptron(S, y, 10)
        print("Average Perceptron W:")
        print(w)
        errorsPerceptron = countErrors(w, STest, yTest)
        print("With " + str(errorsPerceptron / (0.0 + yTest.shape[0])) + " average prediction error on the test set.")




if __name__ == '__main__':
    main(sys.argv[1:])



