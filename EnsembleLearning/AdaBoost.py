
# -*- coding: utf-8 -*-
"""


@author: Erik Barton
"""
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
from DecisionTreePartialWeights import WeightedDTree




##################################################################################################################################################################################################
# Data Processing
##################################################################################################################################################################################################

'''
Convert yes and no into 1 and -1
'''
def convertToOneMinusOne(y):
    out = np.ones(y.shape)
    for i in range(y.shape[0]):
        if y[i] == "no":
            out[i] = -1
    return out.astype(int)

'''
Finds the most common S attribute value for each attribute. 
'''
def findMostCommon(S, unknownIndicator):
    ##print("Selecting the most common value for this attribute.")
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

    ##print("Most common attribute values: " + str(mostCommonAttribValues))
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
        ##print(S)
        stringNumericCols = S[:,indexNum]
        ##print(stringNumericCols)
        floatNumericCols = stringNumericCols.astype(float)
        ##print(floatNumericCols)
        medians = np.median(floatNumericCols, axis=0)
        ##print(medians)
        for c in range(floatNumericCols.shape[1]):
            for r in range(S.shape[0]):
                if floatNumericCols[r, c] >= medians[c]:
                    S[r,indexNum[c]] = "1"
                else:
                    S[r,indexNum[c]] = "0"
        ##print(S[:,indexNum])
        return medians
    else:
        ##print(median)
        stringNumericCols = S[:,indexNum]
        ##print(stringNumericCols)
        floatNumericCols = stringNumericCols.astype(float)
        for c in range(floatNumericCols.shape[1]):
            for r in range(S.shape[0]):
                if floatNumericCols[r, c] >= median[c]:
                    S[r,indexNum[c]] = "1"
                else:
                    S[r,indexNum[c]] = "0"
        ##print(S[:,indexNum])
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
    ##print("Calculating the entropy.")

    for key, value in dictionaryYValues.items():
        px = value / countTotal
        ##print("p_x of " + str(key) + " is %f" % px)
        entFinal += (px * np.log(px))/np.log(logBase)

    entFinal *= -1
    ##print("Entropy: " + str(entFinal))


    return entFinal

'''
Calculate the majority error from a dictionary mapping value (lbl) to counts.
'''
def ME(dictionaryYValues, countTotal, logBase):
    #print("Calculating the majority error.")
    countTotal += 0.0
    maxCount = -1
    maxLbl = None
    for lbl in dictionaryYValues:
        if dictionaryYValues[lbl] > maxCount:
            maxCount = dictionaryYValues[lbl]
            maxLbl = lbl
    #print("Max label is: " + str(maxLbl) + " with count %f" % (maxCount))
    sumCountsNotMax = 0.0
    for lbl in dictionaryYValues:
        if lbl == maxLbl:
            continue
        sumCountsNotMax += dictionaryYValues[lbl]
    #print("Number of non max label is: " + str(sumCountsNotMax))
    me = sumCountsNotMax / countTotal
    #print("Majority error is: %f" % (me))
    return me

'''
Adaboost algorithm on descision stumps
'''
class AdaBoostStumps(object):
    '''
    Initalize objects
    '''
    def __init__(self):
        self.Stumps = []
        self.Alphas = []

    '''
    Builds adaboost with T classifiers and tracks error rates on training and test for the boost and the sum of the stumps
    '''
    def buldCollectionTracking(self, STrain, yTrain, attributes, attributeValues, func, numYTypes, attributesAvaliable, STest, yTest, T):
        initWeight = 1.0 / yTrain.shape[0]
        weightsValues = np.full(yTrain.shape, initWeight)

        runningWeightedSumTrain = np.zeros(yTrain.shape)
        runningWeightedSumTest = np.zeros(yTest.shape)

        ErrorsAdaBoostTrain = []
        ErrorsAdaBoostTest = []
        ErrorsStumpsTrain = []
        ErrorsStumpsTest = []

        for t in range(T):
            #print("Weight Values: " + str(weightsValues))
            dStump = WeightedDTree()
            dStump.buldTree(STrain, yTrain, weightsValues, attributes, attributeValues, func, numYTypes, attributesAvaliable, 1)

            numberOfErrorsThisStumpTrain, stumpPredictionsTrain, epsT = self.countErrorsStump(dStump, STrain, yTrain, weightsValues) # y must stay as strings for now
            numberOfErrorsThisStumpTest, stumpPredictionsTest, _ = self.countErrorsStump(dStump, STest, yTest, weightsValues)

            #print("Number of errors stump train = " + str(numberOfErrorsThisStumpTrain))
            #print("Stump predictions train = " + str(stumpPredictionsTrain))
            #print("Eps train = " + str(epsT))

            ErrorsStumpsTrain.append(numberOfErrorsThisStumpTrain)
            ErrorsStumpsTest.append(numberOfErrorsThisStumpTest)

            alphat = 0.0
            if(epsT > 1e-8):
                alphat = 0.5 * (np.log( (1 - epsT)/(epsT) ))
            else:
                alphat = 1.0

            self.Stumps.append(dStump)
            self.Alphas.append(alphat)

            stumpPredictionsTrain = stumpPredictionsTrain.astype(float)
            stumpPredictionsTest = stumpPredictionsTest.astype(float)

            weightedVotesLevelTTrain = alphat * stumpPredictionsTrain
            weightedVotesLevelTTest = alphat * stumpPredictionsTest

            #print("Weighted votes train = " + str(weightedVotesLevelTTrain))

            runningWeightedSumTrain += weightedVotesLevelTTrain
            runningWeightedSumTest += weightedVotesLevelTTest

            #print("Running weights train = " + str(runningWeightedSumTrain))
            #print("Training y = " + str(yTrain))

            errorsBoostTrain, errorIdxTrain = self.findErrorsRunningWeightedAvg(runningWeightedSumTrain, yTrain)
            errorsBoostTest, _ = self.findErrorsRunningWeightedAvg(runningWeightedSumTest, yTest)

            #print("Errors in boosted train = " + str(errorsBoostTrain))
            #print("Idx errors = " + str(errorIdxTrain))

            ErrorsAdaBoostTrain.append(errorsBoostTrain)
            ErrorsAdaBoostTest.append(errorsBoostTest)

            yFloat = yTrain.astype(float)
            preNorm = self.DtPlus(stumpPredictionsTrain, yFloat, alphat, weightsValues)
            sumValue = np.sum(preNorm)
            weightsValues = preNorm / sumValue
            #print("New weights = " + str(weightsValues))
            #print("")

        return ErrorsAdaBoostTrain, ErrorsAdaBoostTest, ErrorsStumpsTrain, ErrorsStumpsTest

    '''
    Builds adaboost with T classifiers and tracks error rates on training and test for the boost and the sum of the stumps
    '''
    def buldCollection(self, STrain, yTrain, attributes, attributeValues, func, numYTypes, attributesAvaliable, T):
        initWeight = 1.0 / yTrain.shape[0]
        weightsValues = np.full(yTrain.shape, initWeight)

        for t in range(T):
            dStump = WeightedDTree()
            dStump.buldTree(STrain, yTrain, weightsValues, attributes, attributeValues, func, numYTypes, attributesAvaliable, 1)

            numberOfErrorsThisStumpTrain, stumpPredictionsTrain, epsT = self.countErrorsStump(dStump, STrain, yTrain, weightsValues)

            if(epsT > 1e-8):
                alphat = 0.5 * (np.log( (1 - epsT)/(epsT) ))
            else:
                alphat = 1

            self.Stumps.append(dStump)
            self.Alphas.append(alphat)

            stumpPredictionsTrain = stumpPredictionsTrain.astype(float)

            yFloat = yTrain.astype(float)
            preNorm = self.DtPlus(stumpPredictionsTrain, yFloat, alphat, weightsValues)

            sumValue = np.sum(preNorm)
            weightsValues = preNorm / sumValue

    '''
    Choose a label (int)
    '''
    def label(self, y):
        runningSum = 0.0
        for i in range(len(self.Stumps)):
            lbl = self.Stumps[i].label(y)
            runningSum += self.Alphas[i] * lbl
        if runningSum >= 0.0:
            return 1
        else:
            return -1

    '''
    Checks the value for the stump for S and returns the number of errors, the decisions (as str), and the epsilon values of the summed weights of errors
    '''
    def countErrorsStump(self, dStump, S, y, weights):
        # Todo find the choices of the stump for all values in np.array and then check and count the errors as well and eps
        stumpChoice = []
        for i in range(S.shape[0]):
            stumpChoice.append(dStump.label(S[i]))
        stumpChoices = np.array(stumpChoice)

        errorCount = 0
        sumWeightedMistakes = 0.0
        for i in range(y.shape[0]):
            if (stumpChoices[i] >= 0 and y[i] < 0) or (stumpChoices[i] < 0 and y[i] >= 0):
                sumWeightedMistakes += weights[i]
                errorCount += 1

        return errorCount, stumpChoices, sumWeightedMistakes

    '''
    Finds the errors in running weighted averages of the decision stumps
    '''
    def findErrorsRunningWeightedAvg(self, runningWeightedSum, y):
        errorCount = 0
        errorIdx = []
        for i in range(y.shape[0]):
            if (y[i] >= 0.0 and runningWeightedSum[i] < 0.0) or (y[i] < 0.0 and runningWeightedSum[i] >= 0.0):
                errorCount += 1
                errorIdx.append(i)
        return errorCount, errorIdx

    '''
    Builds the pre-normalized new weights
    '''
    def DtPlus(self, runningWeightedSum, yFloat, alphat, weightsValues):
        result = np.zeros(weightsValues.shape)
        for i in range(yFloat.shape[0]):
            if (yFloat[i] >= 0.0 and runningWeightedSum[i] < 0.0) or (yFloat[i] < 0.0 and runningWeightedSum[i] >= 0.0):
                result[i] = weightsValues[i] * np.exp(1.0 * alphat)
            else:
                result[i] = weightsValues[i] * np.exp(-1.0 * alphat)
        return result

##################################################################################################################################################################################################
# HELPERS
##################################################################################################################################################################################################

def countErrors(ada, S, y):
    errors = 0
    for i in range(S.shape[0]):
        predicted = ada.label(S[i,:])
        if predicted != y[i]:
            errors += 1

    return errors


##################################################################################################################################################################################################
# MAIN
##################################################################################################################################################################################################

def main():

    # ---
    print("Adaboost")
    script_dir = os.path.dirname(__file__)
    start = str(script_dir)
    S, y = loadDataSy(start + "/bank/train.csv")
    STest, yTest = loadDataSy(start + "/bank/test.csv")
    attributes, attributeValues, attributesAvaliable, numYTypes, indexNumerical = getAttributeInformation()
    y = convertToOneMinusOne(y)
    yTest = convertToOneMinusOne(yTest)
    medians = doMedian(S, indexNumerical)
    doMedian(STest, indexNumerical, medians)

    adaBuild = AdaBoostStumps()
    ErrorsAdaBoostTrain, ErrorsAdaBoostTest, ErrorsStumpsTrain, ErrorsStumpsTest = adaBuild.buldCollectionTracking(S, y, attributes, attributeValues, ME, numYTypes, attributesAvaliable, STest, yTest, 500)
    print("Stumps train, test:")
    print(ErrorsStumpsTrain)
    print(ErrorsStumpsTest)
    print("Boost train, test:")
    print(ErrorsAdaBoostTrain)
    print(ErrorsAdaBoostTest)

    # ---



if __name__ == '__main__':
    main()