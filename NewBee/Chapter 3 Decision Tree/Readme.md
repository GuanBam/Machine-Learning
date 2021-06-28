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

