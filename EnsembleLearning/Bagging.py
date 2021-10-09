
# -*- coding: utf-8 -*-
"""


@author: Erik Barton
"""
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
from DecisionTree import DTree



##################################################################################################################################################################################################
# Data Processing
##################################################################################################################################################################################################

'''
Finds the most common S attribute value for each attribute. 
'''
def findMostCommon(S, unknownIndicator):
    #print("Selecting the most common value for this attribute.")
    mostCommonAttribValues = []
    for j in range(S.shape[1]):
        counts = {}
        for i in range(S.shape[0]):
            if S[i,j] == unknownIndicator:
                continue
            if S[i,j] in counts:
                counts[S[i,j]] += 1
            else:
                counts[S[i,j]] = 1
        maxCount = -1
        maxAVal = None
        for key in counts:
            if counts[key] > maxCount:
                maxCount = counts[key]
                maxAVal = key 

        mostCommonAttribValues.append(maxAVal)
        counts.clear()

    #print("Most common attribute values: " + str(mostCommonAttribValues))
    return mostCommonAttribValues

'''
Replaces any values with unknown with their most common value.
'''
def replaceUnknowns(S, unknownIndicator, mostCommon):
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if S[i,j] == unknownIndicator:
                S[i,j] = mostCommon[j]


'''
Loads data from the specified file - expects ',' delim.
'''
def loadTrainData(fileNames, fileData):
    attributes = []
    attributeValues = []
    numYTypes = 0
    attributesAvaliable = set()
    haveLabelValue = False
    with open(fileNames, 'r') as f:
        for line in f:
            splitTerms = line.strip().split(',')
            if len(splitTerms) > 1:
                if ':' in splitTerms[0]:
                    tempSubSplit = splitTerms[0].split(' ')
                    tempAttributeValues = []
                    tempAttributeValues.append(tempSubSplit[-1])
                    for i in range(1, len(splitTerms)):
                        tempAttributeValues.append(splitTerms[i].replace(".", "").replace(" ", ""))
                    attributeValues.append(tempAttributeValues)
    
                else:
                    if not haveLabelValue:
                        numYTypes = len(splitTerms)
                        haveLabelValue = True
                    else:
                        attributes = splitTerms[0:-1]
    
    for i in range(len(attributes)):
        attributesAvaliable.add(i)

    tempSandY = np.genfromtxt(fileData, dtype=str, delimiter=',')
    S = tempSandY[:,0:-1]
    y = tempSandY[:,-1]

    return S, y, np.array(attributes, dtype=str), attributeValues, attributesAvaliable, numYTypes

'''
Loads test data from the specified file - expects ',' delim.
'''
def loadDataSy(fileData):
    tempSandY = np.genfromtxt(fileData, dtype=str, delimiter=',')
    S = tempSandY[:,0:-1]
    y = tempSandY[:,-1]
    return S, y

'''
Hard coded attribute information
'''
def getAttributeInformation():
    attributes = np.array([ "age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"])
    attributeValues = [
        ["1","0"],
        ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"],
        ["married","divorced","single"],
        ["unknown","secondary","primary","tertiary"],
        ["yes","no"],
        ["1", "0"],
        ["yes","no"],
        ["yes","no"],
        ["unknown","telephone","cellular"],
        ["1","0"],
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        ["1","0"],
        ["1","0"],
        ["1","0"],
        ["1","0"],
        ["unknown","other","failure","success"]
        ]
    attributesAvaliable = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
    numYTypes = 2
    indexNumerical = [0,5,9,11,12,13,14]
    return attributes, attributeValues, attributesAvaliable, numYTypes, indexNumerical


'''
finds median and splits data set if no medians provided, if medians provided then just updates the dataset.
"1" if above or equal
"0" if below
'''
def doMedian(S, indexNum, median=np.array([])):
    if median.shape[0] == 0:
        #print(S)
        stringNumericCols = S[:,indexNum]
        #print(stringNumericCols)
        floatNumericCols = stringNumericCols.astype(float)
        #print(floatNumericCols)
        medians = np.median(floatNumericCols, axis=0)
        #print(medians)
        for c in range(floatNumericCols.shape[1]):
            for r in range(S.shape[0]):
                if floatNumericCols[r, c] >= medians[c]:
                    S[r,indexNum[c]] = "1"
                else:
                    S[r,indexNum[c]] = "0"
        #print(S[:,indexNum])
        return medians
    else:
        #print(median)
        stringNumericCols = S[:,indexNum]
        #print(stringNumericCols)
        floatNumericCols = stringNumericCols.astype(float)
        for c in range(floatNumericCols.shape[1]):
            for r in range(S.shape[0]):
                if floatNumericCols[r, c] >= median[c]:
                    S[r,indexNum[c]] = "1"
                else:
                    S[r,indexNum[c]] = "0"
        #print(S[:,indexNum])
        return median







##################################################################################################################################################################################################
# AdaBoost
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
Adaboost algorithm on descision stumps
'''
class Bagging(object):
    '''
    Initalize objects
    '''
    def __init__(self):
        self.Trees = []

    '''
    Builds adaboost with T classifiers and tracks error rates on training and test for the boost and the sum of the stumps
    '''
    def buldCollectionTracking(self, STrain, yTrain, attributes, attributeValues, func, numYTypes, attributesAvaliable, STest, yTest, maxDepth, T):
        # Define running vote for test and train and error lists
        runningVotesTrain = np.zeros(yTrain.shape)
        runningVotesTest = np.zeros(yTest.shape)
        errorsTest = []
        errorsTrain = []
        # Loop over T
        for i in range(T):
            # Get a new data subset
            tempS, tempY = self.GetSubset(STrain, yTrain)

            # Build a tree and add to list
            dTree = DTree()
            dTree.buldTree(tempS, tempY, attributes, attributeValues, func, numYTypes, attributesAvaliable, maxDepth)
            self.Trees.append(dTree)

            # Find choices
            choicesTrain = self.MakeChoices(dTree, STrain, runningVotesTrain) # add to running votes and use to make choices using running
            choicesTest = self.MakeChoices(dTree, STest, runningVotesTest)

            # Find errors
            errorsTrain.append(self.CountErrors(choicesTrain, yTrain)) # compare and return count
            errorsTest.append(self.CountErrors(choicesTest, yTest))

        return errorsTrain, errorsTest

    '''
    Builds T trees of maximum depth maxDepth and stores for use in bagging evaluation.
    '''
    def buildCollection(self, STrain, yTrain, attributes, attributeValues, func, numYTypes, attributesAvaliable, maxDepth, T):
        # Loop over T
        for i in range(T):
            # Get a new data subset
            tempS, tempY = self.GetSubset(STrain, yTrain)

            # Build a tree and add to list
            dTree = DTree()
            dTree.buldTree(tempS, tempY, attributes, attributeValues, func, numYTypes, attributesAvaliable, maxDepth)
            self.Trees.append(dTree)

    '''
    Returns the evaluated label for the bagged trees.
    '''
    def label(self, row):
        running = 0.0
        for i in range(len(self.Trees)):
            lbl = self.Trees[i].label(row)
            if lbl == "yes":
                running += 1
            else:
                running -= 1

        if running >= 0.0:
            return "yes"
        else:
            return "no"

    '''
    Gets the rando subset of half of the training size.
    '''
    def GetSubset(self, S, y):
        numSelect = int(S.shape[0] / 2) # TODO: Could be a param
        # Get out the random rows and return
        random_indices = np.random.choice(S.shape[0], size=numSelect, replace=True)
        return S[random_indices, :], y[random_indices]

    '''
    Finds the choice label for the given tree and adds 1 or -1 to running then uses the running to decide if the bagged choice is yes or no.
    '''
    def MakeChoices(self, dTree, S, running):
        choices = []
        for i in range(S.shape[0]):
            lbl = dTree.label(S[i])
            if lbl == "yes":
                running[i] += 1
            else:
                running[i] -= 1

            if running[i] >= 0.0:
                choices.append("yes")
            else:
                choices.append("no")
        return choices

    '''
    Counts the number of errors between the choices made and the true y values
    '''
    def CountErrors(self, choices, y):
        errors = 0
        for i in range(y.shape[0]):
            if choices[i] != y[i]:
                errors += 1
        return errors

##################################################################################################################################################################################################
# HELPERS
##################################################################################################################################################################################################

def countErrors(bagged, S, y):
    errors = 0
    for i in range(S.shape[0]):
        predicted = bagged.label(S[i,:])
        if predicted != y[i]:
            errors += 1

    return errors

##################################################################################################################################################################################################
# MAIN
##################################################################################################################################################################################################

def main():
    script_dir = os.path.dirname(__file__)
    start = str(script_dir)
    
    S, y = loadDataSy(start + "/bank/train.csv")
    Stest, ytest = loadDataSy(start + "/bank/test.csv")
    attributes, attributeValues, attributesAvaliable, numYTypes, indexNumerical = getAttributeInformation()

    medians = doMedian(S, indexNumerical)
    doMedian(STest, indexNumerical, medians)

    bag = Bagging()
    errorsTrain, errorsTest = bag.buldCollectionTracking(S, y, attributes, attributeValues, entropy, numYTypes, attributesAvaliable, STest, yTest, 16, 500)
    print(errorsTrain)
    print(errorsTest)


if __name__ == '__main__':
    main()
