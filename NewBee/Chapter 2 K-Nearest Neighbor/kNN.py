'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors
@Python Version: 2
@author: pbharrin

Updated and Commented on Jun 24,2021
@Python Version: 3.7.4
@author: GuanBam
'''
from numpy import *
import operator
from os import listdir

"""
Function used to classification the input according to the dataset.
Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
Output:     the most popular class label
"""
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]                      # Get dataset length, number of training data
                                                        # distance equation d = ((an-a)^2+(bn-b)^2+...+(zn-z)^2)^0.5
                                                        # d: distance
                                                        # a,b,c...,z: features(number depends on the training dataset)
                                                        # n:features index(maximum is the amount of dataset)
    diffMat = tile(inX, (dataSetSize,1)) - dataSet      # Calculate the difference between input feature matrixs and all training features
    sqDiffMat = diffMat**2                              # diffMat^2
    sqDistances = sqDiffMat.sum(axis=1)                 # Sum the diffMat^2 according to feature groups
    distances = sqDistances**0.5                        # Caculate the distance
    sortedDistIndicies = distances.argsort()            # Sort the distance array in increasing order(The elements with lower index, means it's more similar to the input)
    classCount={}          
    for i in range(k):                                  # Tranverse the distance array, and calculate the matched label for the index
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)     # Sort the frequency dictionary in decreasing order 
    return sortedClassCount[0][0]                       # The first label will be the best match for input

"""
An Function used to create example dataset
Output: feature matrix and label array
"""
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

"""
turn data file into feature matirxs and label array (For different file format, the way to deal with data will be different)
Input: path of dataset
Output: Feature Matrix, and Label Array
"""
def file2matrix(filename):
    love_dictionary={'largeDoses':3, 'smallDoses':2, 'didntLike':1}   # There's other method to map different labels into Number
    fr = open(filename)                         # Open the dataset
    arrayOLines = fr.readlines()                # Read the dataset
    numberOfLines = len(arrayOLines)            # Get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        # Prepare matrix to return, size will be numberOfLines * 3
    classLabelVector = []                       # Prepare labels return   
    index = 0

    for line in arrayOLines:                    # Deal with the data line by line
        line = line.strip()                     # Remove space at begining and end of the line
        listFromLine = line.split('\t')         # Seperate the line with tab
        returnMat[index,:] = listFromLine[0:3]  # Store features into feature matrix

        if(listFromLine[-1].isdigit()):         # Store labels into label array
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

"""
Normalize Dataset with scale 0~1
Input: feature matrixs
Output: normalized feature matrixs, ranges of orignial features, minimum values of original features.
"""
def autoNorm(dataSet):
    # Only the first column need to be normalized for example dating perference
    minVals = dataSet.min(0)                    # Get minimum value 
    maxVals = dataSet.max(0)                    # Get maximum value
    ranges = maxVals - minVals                  # Get range
    normDataSet = zeros(shape(dataSet))         # Initialize new feature matrixs
    m = dataSet.shape[0]                        # m = total row number of feature matrixs
    normDataSet = dataSet - tile(minVals, (m,1))            # minus the minium value to set new minum as 0         
    normDataSet = normDataSet/tile(ranges, (m,1))           # Divide the ranges(ranges will be the maximum after minus) to get value between 0~1
    return normDataSet, ranges, minVals
   
"""
Testing Function for Dating Perference, will print the result for Testing dataset
"""
def datingClassTest():
    hoRatio = 0.50                                                      # Rate for testing dataset
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')      # Load data set from file
    normMat, ranges, minVals = autoNorm(datingDataMat)                  # Normalize data
    m = normMat.shape[0]                                                # Get entire data
    numTestVecs = int(m*hoRatio)                                        # Define number for test
    errorCount = 0.0
    for i in range(numTestVecs):                                        # Pass data to test and check if the result is correct
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)

"""
Take features from user side and classification the user input with testing dataset
"""
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input(\
                                  "percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream, ])
    classifierResult = classify0((inArr - \
                                  minVals)/ranges, normMat, datingLabels, 3)
    print("You will probably like this person: %s" % resultList[classifierResult - 1])
    
"""
Turn img files (txt file) into feature matrix
Input: img files (.txt)
Output: feature matrix
"""
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

"""
Testing function for handwriting recognize
"""
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')            # Load the training dataset
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):                                      # Process training data
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')                    # Load the testing dataset
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):                                  # Process testing data and check the result for each case
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))
