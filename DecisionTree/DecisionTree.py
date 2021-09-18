# -*- coding: utf-8 -*-
"""
Decision Tree based on the ID3 algorithm. 

@author: erikcbarton
"""
import matplotlib.pyplot as plt
import numpy as np
import copy
import os

###############################################################################################################
# Decision Tree Class
###############################################################################################################
class DTree(object):
    '''
    ID3 Based DTree
    '''

    '''
    Initalize objects
    '''
    def __init__(self):
        self.RootNode = None

    '''
    Builds the tree using the specified input values
    '''
    def buldTree(self, S, y, attributes, attributeValues, func, numYTypes, attributesAvaliable, depthLimit):
        self.RootNode = self.ID3(S, y, attributes, attributeValues, func, numYTypes, attributesAvaliable, depthLimit)

    '''
    '''
    def label(self, attribs):
        return self.RootNode.getLabel(attribs)

    '''
    ID3 algo
    '''
    def ID3(self, S, y, attributes, attributeValues, func, numYTypes, attributesAvaliable, depthRemaining):
        node = Node()
        # TODO: Check empty y?
        if self.allSameLbl(y):
            #print("All same y")
            lbl = y[0]
            node.setupNode(lbl, None, None, None)
            return node

        if len(attributes) == 0:
            #print("No More Attributes")
            lbl = self.mostCommonLbl(y)
            node.setupNode(lbl, None, None, None)
            return node

        if depthRemaining <= 0:
            #print("Depth limit reached")
            lbl = self.mostCommonLbl(y)
            node.setupNode(lbl, None, None, None)
            return node

        attributeToSplit = self.findBestAttrib(S, y, attributes, attributeValues, func, numYTypes, attributesAvaliable)
        attributesAvaliable.remove(attributeToSplit) # remove for recursion add back before completion
        attribValueToSets, attribValueToY = self.splitOnAttrib(attributeToSplit, S, y, attributeValues)

        attribValueToNodes = {}

        for key in attribValueToSets:
            if attribValueToY[key].size == 0:
                #print("Empty set")
                lbl = self.mostCommonLbl(y)
                tempNode = Node()
                tempNode.setupNode(lbl, None, None, None)
                attribValueToNodes[key] = tempNode
            else:
                attribValueToNodes[key] = self.ID3(attribValueToSets[key], attribValueToY[key], attributes, attributeValues, func, numYTypes, attributesAvaliable, (depthRemaining - 1))

        node.setupNode(None, attributeToSplit, attributeValues[attributeToSplit], attribValueToNodes)
        attributesAvaliable.add(attributeToSplit) # added back 
        return node

    '''
    Finds the best attribute using the avaliable attributes with the maximum information gain.
    '''
    def findBestAttrib(self, S, y, attributes, attributeValues, func, numYTypes, attributesAvaliable):
        #print("Calculating information gain of set:")
        #print(S)
        #print(y)

        bestAttribute = None
        bestGain = -1

        numS = y.shape[0]
        numS += 0.0
        #print("Number of elements (|S|): %d" % (numS))
        entropyThisLevel = self.findEntCurr(y, func, numYTypes)

        for i in attributesAvaliable:
            #print("")
            #print("Gain for Attribute: " + str(attributes[i]))

            entSplit = self.findEntSubset(i, attributes, attributeValues, S, y, func, numS, numYTypes)

            gain = entropyThisLevel - entSplit
            #print("GAIN: %f" % (gain))
        
            if gain > bestGain:
                bestGain = gain
                bestAttribute = i

        #print("")
        #print("Best gain %f for attribute: " % (bestGain) + str(attributes[bestAttribute]) )

        return bestAttribute

    '''
    Finds the entropy of subsets of the given S.
    '''
    def findEntSubset(self, attributeNumber, attributes, attributeValues, S, y, func, Scount, numYTypes):
        valuesForThisAttrib = attributeValues[attributeNumber] 

        summedEnt = 0.0

        for attribValue in valuesForThisAttrib: # all attrib valus like: hot warm cold
            splitCount = 0.0
            dictYValuesToCounts = {}
            entAttrib = 0.0

            #print("Attribute Value: " + str(attribValue))

            for j in range(S.shape[0]):
                if attribValue == S[j, attributeNumber]:
                    # add y val to dict
                    yVal = y[j]
                    if yVal in dictYValuesToCounts:
                        dictYValuesToCounts[yVal] = dictYValuesToCounts[yVal] + 1
                    else:
                        dictYValuesToCounts[yVal] = 1
                    # add 1 to the total counter
                    splitCount += 1.0
            
            # call the purity function
            if splitCount > 1e-8:
                entAttrib = func(dictYValuesToCounts, splitCount, numYTypes)
            else:
                entAttrib = 0.0

            #print("H value is: " + str(entAttrib))

            # use ent value to sum to final value
            summedEnt += (splitCount / (0.0 + Scount)) * entAttrib
            # reset count to 0.0
            splitCount = 0.0
            # reset dict
            dictYValuesToCounts.clear()

        #print("Final expected purity: %f" % (summedEnt))
        return summedEnt

    '''
    Finds the entropy of the current set (no subsets).
    '''
    def findEntCurr(self, y, func, numYTypes):
        #print("Purity of the current level")
        total = y.shape[0]
        dictYValuesToCounts = {}

        for yVal in y:
            if yVal in dictYValuesToCounts:
                dictYValuesToCounts[yVal] = dictYValuesToCounts[yVal] + 1
            else:
                dictYValuesToCounts[yVal] = 1

        return func(dictYValuesToCounts, total, numYTypes)

    '''
    Split the sets S and lbls based on the given attribute. Dictionaries are returned
    mapping the attribute's values to the corresponding subsets for S and lbls.
    '''
    def splitOnAttrib(self, attrib, S, y, attributeValues):
        #print("")
        #print("")
        #print("New sets.")
        valuesForThisAttrib = attributeValues[attrib]

        attribValueToSets = {}
        attribValueToY = {}

        attribValueToSetsNP = {}
        attribValueToYNP = {}

        for attribValue in valuesForThisAttrib:
            attribValueToSets[attribValue] = []
            attribValueToY[attribValue] = []

        for i in range(S.shape[0]):
            value = S[i, attrib]
            attribValueToSets[value].append(S[i])
            attribValueToY[value].append(y[i])
    
        for key in attribValueToSets:
            arrayNPAttributes = np.array(attribValueToSets[key])
            arrayNPY = np.array(attribValueToY[key])
            #print(arrayNPAttributes)
            #print(arrayNPY)
            attribValueToSetsNP[key] = arrayNPAttributes
            attribValueToYNP[key] = arrayNPY

        #print("")
        #print("")

        return attribValueToSetsNP, attribValueToYNP

    '''
    Check if all values are the same in y
    '''
    def allSameLbl(self, y):
        # Assume not empty
        val = y[0] # = y[0,0]
        for i in range(1, y.shape[0]):
            if val != y[i]: # != y[i,0]:
                return False
        return True

    '''
    Find the most common Y value (ties are decided arbitrarily)
    '''
    def mostCommonLbl(self, y):
        counts = {}
        for i in range(y.shape[0]):
            if y[i] in counts:
                counts[y[i]] += 1
            else:
                counts[y[i]] = 1

        maxCount = -1
        maxLbl = None
        for key in counts:
            if counts[key] > maxCount:
                maxCount = counts[key]
                maxLbl = key

        #print("Most common label is " + str(maxLbl))
        return maxLbl

###############################################################################################################
# Node Class for Nodes in a DTree
###############################################################################################################
class Node(object):
    '''
    CLASS representing a node in the tree (leaf or inner). Must be initalized.
    '''

    def __init__(self):
        self.Attribute = None
        self.AttributeValueList = None
        self.ChildrenAtrbValueToNode = None
        self.LblValue = None

    def setupNode(self, returnValue, attribute, attributeValueList, childrenAtrbValueToNode):
        if returnValue != None:
            self.LblValue = returnValue
            return

        self.Attribute = attribute
        self.AttributeValueList = attributeValueList
        self.ChildrenAtrbValueToNode = childrenAtrbValueToNode
        return

    def getLabel(self, value):
        if self.LblValue != None:
            return self.LblValue
        else:
            return self.ChildrenAtrbValueToNode[value[self.Attribute]].getLabel(value)

###############################################################################################################
# Support Methods
###############################################################################################################

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
Calculate the gini index from a dictionary mapping value (lbl) to counts.
'''
def GI(dictionaryYValues, countTotal, logBase):
    giFinal = 0.0
    countTotal += 0.0
    #print("Calculating the gini index.")

    for key, value in dictionaryYValues.items():
        px = value / countTotal
        #print("p_x of " + str(key) + " is %f" % px)
        giFinal += px**2

    giFinal = 1.0 - giFinal
    #print("Gini index is: %f" % (giFinal))

    return giFinal

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
Hard coded attribute information for part 2a
'''
def getAttributeInformationPart2a():
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
Hard coded attribute information for part 2b
'''
def getAttributeInformationPart2b():
    attributes = np.array([ "age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"])
    attributeValues = [
        ["1","0"],
        ["admin.","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"],
        ["married","divorced","single"],
        ["secondary","primary","tertiary"],
        ["yes","no"],
        ["1", "0"],
        ["yes","no"],
        ["yes","no"],
        ["telephone","cellular"],
        ["1","0"],
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        ["1","0"],
        ["1","0"],
        ["1","0"],
        ["1","0"],
        ["other","failure","success"]
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

'''
Trains DTrees with all three heuristics and from min depth to max depth specified (inclusive).
Prints the test errors on the specified test set.
'''
def testTrainMultiLevel(S, y, Stest, ytest, attributes, attributeValues, attributesAvaliable, numYTypes, minDepth, maxDepth):
    # Try multi level with multiple functions
    print("Depth|    Ent     |     ME     |     GI   ")
    for d in range(minDepth, maxDepth+1):
        # Do for each depth
        dTreeEnt = DTree()
        dTreeME = DTree()
        dTreeGI = DTree()
        tempAttrib = copy.deepcopy(attributesAvaliable)
        dTreeEnt.buldTree(S, y, attributes, attributeValues, entropy, numYTypes, tempAttrib, d)
        tempAttrib = copy.deepcopy(attributesAvaliable)
        dTreeME.buldTree(S, y, attributes, attributeValues, ME, numYTypes, tempAttrib, d)
        tempAttrib = copy.deepcopy(attributesAvaliable)
        dTreeGI.buldTree(S, y, attributes, attributeValues, GI, numYTypes, tempAttrib, d)

        errorsEnt = 0
        errorsME = 0
        errorsGI = 0
        total = 0.0 + ytest.shape[0]

        for i in range(Stest.shape[0]):
            lblEnt = dTreeEnt.label(Stest[i])
            lblME = dTreeME.label(Stest[i])
            lblGI = dTreeGI.label(Stest[i])
            if lblEnt != ytest[i]:
                errorsEnt += 1
            if lblME != ytest[i]:
                errorsME += 1
            if lblGI != ytest[i]:
                errorsGI += 1

        print("  %d  |  %f  |  %f  |  %f  " % (d, errorsEnt / total, errorsME / total, errorsGI / total))
        



###############################################################################################################
# MAIN Method
###############################################################################################################

'''
Main method to run the ID3 program hard coded
'''
def main(): 
    script_dir = os.path.dirname(__file__)
    start = str(script_dir)
    # 1 a, b
    print("Part 1")
    S, y, attributes, attributeValues, attributesAvaliable, numYTypes = loadTrainData(start + "/car/data-desc.txt", start + "/car/train.csv")
    Stest, ytest = loadDataSy(start + "/car/test.csv")
    
    print("Train Set")
    #testTrainMultiLevel(S, y, S, y, attributes, attributeValues, attributesAvaliable, numYTypes, 1, 6)
    print("Test Set")
    testTrainMultiLevel(S, y, Stest, ytest, attributes, attributeValues, attributesAvaliable, numYTypes, 1, 6)

    # 2 a
    print("Part 2a")
    S, y = loadDataSy(start + "/bank/train.csv")
    Stest, ytest = loadDataSy(start + "/bank/test.csv")
    attributes, attributeValues, attributesAvaliable, numYTypes, indexNumerical = getAttributeInformationPart2a()

    medians = doMedian(S, indexNumerical)
    doMedian(Stest, indexNumerical, medians)

    testTrainMultiLevel(S, y, Stest, ytest, attributes, attributeValues, attributesAvaliable, numYTypes, 1, 16)

    # 2 b
    print("Part 2b")
    attributes, attributeValues, attributesAvaliable, numYTypes, indexNumerical = getAttributeInformationPart2a()
    mostCommon = findMostCommon(S, "unknown")
    replaceUnknowns(S,"unknown", mostCommon)
    replaceUnknowns(Stest,"unknown", mostCommon)

    testTrainMultiLevel(S, y, Stest, ytest, attributes, attributeValues, attributesAvaliable, numYTypes, 1, 16)


if __name__ == '__main__':
    main()

