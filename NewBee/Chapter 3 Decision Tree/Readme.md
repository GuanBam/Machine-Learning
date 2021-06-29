# 3 Decision Tree
Decision tree do not required too much knowledge of machine learning.
Basic concept of Decision Tree is judge the data one step by one step.

Pros: Not complex, easy to understand the result, do not senstive to missing values, able to deal with non-relevant features.
Cons: May overfitting

# 3.1 Algorithm Logic
. user need figure out which feature is used to split the data.
. split the data into several subsets which will traverse down the branches of the first decision node.
. repeat first and second step, until all the data in the subset belongs to same class.
Below is the pseudo-code: 
```Python
def createBranch():
  if match class:
    return class label
  else:
    find the best feature to split data
    split data according to the feature
    create branch node
    for each subset:
      createBranch(each)
    return brach node
```
# 3.1.1 Information Gain (Entropy)
The key for this algorithm is how to split the dataset to make the unorganized data more organized.
In "Machine Learning In Action", the author introduced the method of measuring the information.
The change in information before and after the split is known as the Information Gain. Information Gain is based on quantifiable information, user may consider about which feature will have more weight in determing the class of the data and that will be consider as has higher information gain.

## How to calculate the Information Gain
If clssifying something that can take on multiple values, the information for symbol xi is defined as:

![1](http://latex.codecogs.com/svg.latex?l(x_i)=-log_2P(x_i))

![2](http://latex.codecogs.com/svg.latex?P(x_i)) is the probability of choosing this class

To calculate the Information Gain, we need the expected value of all the information of all possible values of class:

![3](http://latex.codecogs.com/svg.latex?H=-\sum_{i=1}^nP(x_i)log_2P(x_i))

where n is the number of class.

Below is the code in Python to calculate the Information Gain
```Python
from math import log

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    # record the total time of each class occurred
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 1
        else:
            labelCounts[currentLabel] += 1
    shannoEnt = 0.0
    # calculate the Entropy
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannoEnt
```
### What the Information Gain Represent
With more class, the probility for each class will be lower and the log result will be much more close to -inf, which means the Entropy will be larger.

If we can find a splitting way that the subset return largest Entropy, then we just find a way to split data that contains most classes, in another word, we picked out data which may belong to a certain kind of class.

To find the best split way, we will need try it with all the features and all possible values.

## 3.1.2 Splitting the dataset
Below is the method to split dataset with choosen feature:
```Python
# split dataset according to feature(axis)=value
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        # if the value of certain axis meets the decision tree if-statement, take the remaing features value
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    # return the splited dataset
    return retDataSet
```
Below is the method to find the best feature for splitting:
```Python
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannoEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    # tranverse each features for splitting
    for i in range(numFeatures):
        # get all occurred vale for each feature
        featList = [examples[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        # tranverse each values for current feature
        for value in uniqueVals:
            # split dataset
            subDataSet = splitDataSet(dataSet, i , value)
            # calculate the entropy
            prob = len(subDataSet)/float((len/dataSet))
            newEntropy += prob * calcShannoEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain>bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
```
## 3.1.3 Building the Decision Tree
Recursion will be used to splitting data until run out of features for splitting or all data in a branch are the same class.
```Python
import operator

# count the number of class occurred
def majorityCnt(classList):
    classCount = {}
    for vote in classCount.keys():
        if vote not in classCount:
            classCount[vote]=0
        else:
            classCount[vote]+=1
    # sort the dict in decreasing order
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),reverse=True)
    return sortedClassCount

# make the decision tree according to input dataset and labels
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    # Base Line
    # if there's only one class left
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # if there's only one feature left
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
        
    # find the best feature for splitting
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    # get all possible values for best feature
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    # recursively building the tree
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree
```
## 3.2 Plot to show the tree
"Matplotlib" library can be used to help show of the tree. Codes is given but won't show here, since it's just used to help understand the process but not necessary for this algorithm.

## 3.3 Testing and storing the classifier
### 3.3.1 Test: using the tree for classification
```Python
# The inputTree could be a nested tree
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    # find the feature index according to the given decision tree
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            # The inputTree could be a nested dict
            # if the type of the value of the key is "dict", then we need go to next level of tree 
            if type(secondDict[key]).__name__=="dict":
                classLabel = classify(secondDict[key],featLabels,testVec)
            # else we just finished the classification
            else:
                classLabel = secondDict[key]
    return classLabel
```
### 3.3.2 Use: persisting the decision tree
Store the tree will help decrease the time for classification, sine you don't have to find the decision tree each time according to the same training dataset.
With "pickle" library, we can read and store the data set("json" should able to do the same thing).
```Python
def storeTree(inputTree,filename):
    import pickle
    f = open(filename,'wb')
    pickle.dump(inputTree,f)
    f.close()
    
def grabTree(filename):
    import pickle
    f = open(filename,'rb')
    return pickle.load(f)
```
## 3.4 Example: predict contact lens type
Follow the step as the code given before
. Collect and prepare: Text file provided.
. Train: with createTree() function.
. Test: with a given instance.
. Use: persist the tree data structure for next using.
