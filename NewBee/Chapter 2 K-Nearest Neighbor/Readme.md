# 2. KNN (K-Nearest Neighbor Algorithm)
Pros: High accurate, not sensitive to outliers, no assumpted data input.
Cons: High computational complexity, high space complexity.
Range: Numerical and Nominal.

KNN do classification by measuring the distance between different feature values.

# 2.1 Algorithm Logic
Suppose we already have a bunch of labled data, consider as training dataset.
Now, we got a set of unlabled data, the module will compare the set of data with training dataset and tell what should be the lable for this dataset.

1. Caculate the distance between the point in the known category dataset and the current point.
2. Sort in increasing order of distance.
3. Select the first K points (the closest K points).
4. Caculate the frequency of the Lable of the frist K points.
5. Return the label with highest occurred frequency.
```Python
def classify0(inX, dataSet, labels, k):
    # inX is the input data which need to be classified
    # dataSet is the training dataset
    # labels is the label of dataSet
    # k is the range of choosen points
    
    dataSetSize = dataSet.shape[0]
    
    ## Caculate the distance
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis - 1)
    distances = sqDistances**0.5
    
    ## Sort the distance
    sortedDistIndicies = distances.argsort()
    classCount = {}
    
    ## Choose the closest K points and caculate their label frequency 
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlable] = classCount.get(voteIlabel, 0) + 1
    
    ## Sort the counted class in decreasing order and return the first one (Label with highest frequency)
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]
```
The python file "kNN.py" attached in this folder including the function specific for the example dataset.
# 2.2 Example: KNN on Dating Perference
With Function:
```
"""
Dump data from dataset into matrix.
Input: path of dataset file
Output: feature matrix, and class label array
"""
file2matrix(filename)

"""
Normalize the value in the feature matrix.
Input: feature matrix
Output: normalized feature matrix, ranges (max value - min value), min value
"""
autoNorm(dataSet)

"""
Take 50% data from Testing dataset and test how accurate the classify0() function can be.
"""
datingClassTest()

"""
Required inX from user, and match the input with Testing dataset.
"""
classifyPerson()
```
## 2.2.1 Normalize
The reason why normalization needed is because when the value for different feature has large difference, it will affect the distance calculation.
It will be better to normalize all features to the same scale, so the weight of them will be the same.

# 2.3 Example: KNN on Handwriting Recognize
```Python
"""
Turn Imgs into feature matrix, features will be the value of each pixels.(Here the img given examples are in txt already.)
Input: Img folder
Output: Feature matix
"""
img2vector(filename)

"""
This Function will take pre-assigned data as traning dataset and check the accuracy with testing dataset.
"""
handwritingClassTest()
```
