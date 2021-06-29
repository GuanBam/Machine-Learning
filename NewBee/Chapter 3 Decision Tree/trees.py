'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@Python  Version: 2
@author: Peter Harrington

Updated and Commented on Jun28,2021
@Python Version:3.7.4
@author: GuanBam
'''

from math import log
import operator

"""
create example dataset
Output: fixed dataSet and labels
"""
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

"""
calculate Shannon Entropy
Input: dataset matrix (features with label)
Output: entropy
"""
def calcShannonEnt(dataSet):
    # get number of instances
    numEntries = len(dataSet)
    labelCounts = {}
    #calcuilate the number of unique elements and their occurance
    for featVec in dataSet: 
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    # calculate the entropy
    for key in labelCounts:
        # calculate the probability of certain class
        prob = float(labelCounts[key])/numEntries
        # update the entropy
        shannonEnt -= prob * log(prob,2) 
    return shannonEnt

"""
Split the given dataset based on given feature index and value
Input: dataset, feature index, vaule of target feature
Output: dataset only with given value on given feature
"""
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    # transverse all instances
    for featVec in dataSet:
        # if the feature value meet the requirement, take it off
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis] 
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

"""
Find best feature for splitting
Input: dataset
Output:index of best feature
"""
def chooseBestFeatureToSplit(dataSet):
    # count the number of features, the last column is used for the labels
    numFeatures = len(dataSet[0]) - 1
    # calculate the original entory
    baseEntropy = calcShannonEnt(dataSet)
    # initialize the Information gain
    bestInfoGain = 0.0
    bestFeature = -1
    # tranverse all features
    for i in range(numFeatures):
        # tranverse all instance to get value for the feature at index i
        featList = [example[i] for example in dataSet]
        # using set() to get a set (whose elements will only occur once)
        uniqueVals = set(featList) 
        newEntropy = 0.0
        # tranverse all values for current feature
        for value in uniqueVals:
            # split the dataset based on the value of the feature
            subDataSet = splitDataSet(dataSet, i, value)
            # calculate the probability of split dataset
            prob = len(subDataSet)/float(len(dataSet))
            # update the entrop
            newEntropy += prob * calcShannonEnt(subDataSet)
        # calculate the info gain with current feature
        infoGain = baseEntropy - newEntropy
        # update the info gain and feature index if split with current feature has larger infomation gain
        if (infoGain > bestInfoGain):     
            bestInfoGain = infoGain         
            bestFeature = i
    return bestFeature

"""
Count the class which occurred most
Input: class list
Output: class occurred most
"""
def majorityCnt(classList):
    classCount={}
    # count class with dict
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    # sort the dict in decreasing order
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

"""
Generate Decision Tree
Input: dataset, and labels for feature
Ouput: Decision Tree (which will be a nested dict)
"""
def createTree(dataSet,labels):
    # get classList from dataSet
    classList = [example[-1] for example in dataSet]
    # baseline 1: only one class remaining in the dataset, no need for further splitting
    if classList.count(classList[0]) == len(classList): 
        return classList[0]
    # baseline 2: only one feature remaining in the dataset, not able to further splitting
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # find best feature for splitting
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # get the label of the best feature
    bestFeatLabel = labels[bestFeat]
    # initialize the tree as nested dict
    myTree = {bestFeatLabel:{}}
    # delate the label for the best feature, since it will not occurred in next steps
    del(labels[bestFeat])
    # get unique values for current feature
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    # tranverse all valuese and update to the tree
    for value in uniqueVals:
        # copy the labels
        subLabels = labels[:]
        # recursively for next layer decision
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    # return the nested dict, which is the tree
    return myTree                            

"""
Classification based on given tree
Input: Decision Tree, Feature Labels, input X
Output: class of input X
"""
def classify(inputTree,featLabels,testVec):
    # get the first feature label in inputTree
    firstStr = list(inputTree.keys())[0]
    # get the nested tree with key of first feature label
    secondDict = inputTree[firstStr]
    # get feature index
    featIndex = featLabels.index(firstStr)
    # key as value of testVec at splitting feature index
    key = testVec[featIndex]
    # value as key of the nested tree
    valueOfFeat = secondDict[key]
    # if the type of value is dict, means didn't come to the leaf of the decsion tree
    if isinstance(valueOfFeat, dict):
        # recursively for next layer
        classLabel = classify(valueOfFeat, featLabels, testVec)
    # if not, then the class if matched
    else:
        classLabel = valueOfFeat
    # return the result class
    return classLabel

"""
Store the tree with a file
Input: Decision Tree, and expected filename
"""
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

"""
Reload the stored tree
Input: filename
Output:Decision Tree
"""
def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)
    
