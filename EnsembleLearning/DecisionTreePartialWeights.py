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
class WeightedDTree(object):
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
    def buldTree(self, S, y, rowPartials, attributes, attributeValues, func, numYTypes, attributesAvaliable, depthLimit):
        self.RootNode = self.ID3(S, y, rowPartials, attributes, attributeValues, func, numYTypes, attributesAvaliable, depthLimit)

    '''
    '''
    def label(self, attribs):
        return self.RootNode.getLabel(attribs)

    '''
    ID3 algo
    '''
    def ID3(self, S, y, rowPartials, attributes, attributeValues, func, numYTypes, attributesAvaliable, depthRemaining):
        node = Node()
        # TODO: Check empty y?
        if self.allSameLbl(y):
            #print("All same y")
            lbl = y[0]
            node.setupNode(lbl, None, None, None)
            return node

        if len(attributesAvaliable) == 0: # TODO: Fix
            #print("No More Attributes")
            lbl = self.mostCommonLbl(y, rowPartials)
            node.setupNode(lbl, None, None, None)
            return node

        if depthRemaining <= 0:
            #print("Depth limit reached")
            lbl = self.mostCommonLbl(y, rowPartials)
            node.setupNode(lbl, None, None, None)
            return node

        attributeToSplit = self.findBestAttrib(S, y, rowPartials, attributes, attributeValues, func, numYTypes, attributesAvaliable)
        attributesAvaliable.remove(attributeToSplit) # remove for recursion add back before completion
        attribValueToSets, attribValueToY, partials = self.splitOnAttrib(attributeToSplit, S, y, rowPartials, attributeValues)

        attribValueToNodes = {}

        for key in attribValueToSets:
            if attribValueToY[key].size == 0:
                #print("Empty set")
                lbl = self.mostCommonLbl(y, rowPartials)
                tempNode = Node()
                tempNode.setupNode(lbl, None, None, None)
                attribValueToNodes[key] = tempNode
            else:
                attribValueToNodes[key] = self.ID3(attribValueToSets[key], attribValueToY[key], partials[key], attributes, attributeValues, func, numYTypes, attributesAvaliable, (depthRemaining - 1))

        node.setupNode(None, attributeToSplit, attributeValues[attributeToSplit], attribValueToNodes)
        attributesAvaliable.add(attributeToSplit) # added back 
        return node

    '''
    Finds the best attribute using the avaliable attributes with the maximum information gain.
    '''
    def findBestAttrib(self, S, y, rowPartials, attributes, attributeValues, func, numYTypes, attributesAvaliable):
        #print("Calculating information gain of set:")
        #print(S)
        #print(y)
        #print(rowPartials)

        bestAttribute = None
        bestGain = -1

        numS = self.sumPartials(rowPartials)
        #print("The summed y values are %f\\\\" % (numS))
        entropyThisLevel = self.findEntCurr(y, rowPartials, func, numYTypes)

        for i in attributesAvaliable:
            #print("")
            #print("Gain for Attribute: " + str(attributes[i]))

            entSplit = self.findEntSubset(i, attributes, attributeValues, S, y, rowPartials, func, numS, numYTypes)

            gain = entropyThisLevel - entSplit
            #print("GAIN: %f" % (gain))
        
            if gain > bestGain:
                bestGain = gain
                bestAttribute = i

        #print("")
        #print("Best gain %f for attribute: " % (bestGain) + str(attributes[bestAttribute]) )

        return bestAttribute

    '''
    Count up the total weights
    '''
    def sumPartials(self, parts):
        sum = 0.0
        for i in range(parts.shape[0]):
            sum += parts[i]
        return sum

    '''
    Finds the entropy of subsets of the given S.
    '''
    def findEntSubset(self, attributeNumber, attributes, attributeValues, S, y, rowPartials, func, Scount, numYTypes):
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
                        dictYValuesToCounts[yVal] += rowPartials[j]
                    else:
                        dictYValuesToCounts[yVal] = rowPartials[j]
                    # add 1 to the total counter
                    splitCount += rowPartials[j]
            
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
    def findEntCurr(self, y, rowPartials, func, numYTypes):
        #print("Purity of the current level")
        total = self.sumPartials(rowPartials)
        dictYValuesToCounts = {}

        for i in range(y.shape[0]):
            yVal = y[i]
            if yVal in dictYValuesToCounts:
                dictYValuesToCounts[yVal] += rowPartials[i]
            else:
                dictYValuesToCounts[yVal] = rowPartials[i]

        return func(dictYValuesToCounts, total, numYTypes)

    '''
    Split the sets S and lbls based on the given attribute. Dictionaries are returned
    mapping the attribute's values to the corresponding subsets for S and lbls.
    '''
    def splitOnAttrib(self, attrib, S, y, rowPartials, attributeValues):
        #print("")
        #print("")
        #print("New sets.")
        valuesForThisAttrib = attributeValues[attrib]

        attribValueToSets = {}
        attribValueToY = {}
        partialsValueToPartial = {}

        attribValueToSetsNP = {}
        attribValueToYNP = {}
        attribValueToPartialNP = {}

        for attribValue in valuesForThisAttrib:
            attribValueToSets[attribValue] = []
            attribValueToY[attribValue] = []
            partialsValueToPartial[attribValue] = []

        for i in range(S.shape[0]):
            value = S[i, attrib]
            attribValueToSets[value].append(S[i])
            attribValueToY[value].append(y[i])
            partialsValueToPartial[value].append(rowPartials[i])
    
        for key in attribValueToSets:
            arrayNPAttributes = np.array(attribValueToSets[key])
            arrayNPY = np.array(attribValueToY[key])
            arrayNPPartial = np.array(partialsValueToPartial[key])
            #print(arrayNPAttributes)
            #print(arrayNPY)
            #print(arrayNPPartial)
            attribValueToSetsNP[key] = arrayNPAttributes
            attribValueToYNP[key] = arrayNPY
            attribValueToPartialNP[key] = arrayNPPartial

        #print("")
        #print("")

        return attribValueToSetsNP, attribValueToYNP, attribValueToPartialNP

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
    def mostCommonLbl(self, y, weights):
        counts = {}
        for i in range(y.shape[0]):
            if y[i] in counts:
                counts[y[i]] += weights[i]
            else:
                counts[y[i]] = weights[i]

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






