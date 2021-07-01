# 4 Naïve Bayes
Another classifying algorithm with probability theory.
With KNN and decision tree, they will directly tell you what is the class of given data, but the result could be wrong.
So, Naïve Bayes, is used to solve the problem, it will tell you the probability of which class the given data could be.

Pros: Works for small data sample, can process multiple classes.
Cons: Sensitive to how the input data is prepared
Works with: Nominal values
Naïve Bayes is a subset of Bayesian decision theory.
## 4.1 Bayesian Decision Theory
suppose we have a data set with two features x and y, and two class labels 1, and 2.
Consider the probability calculation function is p(x,y)
Then the probability for the input to be certain class will be P1(x,y) and P2(x,y)
Just compare which probability is larger, we can say the data will be much more like of the one.

## 4.2 Conditional Probability
In a real case, the way to calculate probability could be much more complex and we may need using conditional probability to solve problems.

[Wiki: Conditional Probability](https://en.wikipedia.org/wiki/Conditional_probability)

![1](http://latex.codecogs.com/svg.latex?P(c|x)=\frac{P(x|c)P(c)}{P(x)})


## 4.3 Classifying with Conditional Probability
As the example in 4.1, the equation could be extend to:

![2](http://latex.codecogs.com/svg.latex?P(c_i|x,y)=\frac{P(x,y|c_i)P(c_i)}{P(x,y)})

If P(c1|x,y) > P(c2|x,y), then the class is c1.
If P(c1|x,y) < P(c2|x,y), then the class is c2.

# 4.4 Document classification with Naïve Bayes
One example is document classification, using Naïve Bayes to classify any news, message board discussion, or other type text.
The concept will be looking at the documents by the words used and treat the presence or absence of each word as a feature.
Gereral Approach:
. Collect Data: (RSS)
. Prepare: True data into Numeric or Boolean values
. Analyze: Histograms could help
. Train
. Test
. Use

# 4.5 Classifying text
*Token:* Combination of characters.
In this example, two class are given: abusive and not. Use 1 represent abusive and 0 for not.

## 4.5.1 Prepare: Making work vectores from text
```Python
"""
Initialize example dataset and label
Output: Data array, label array
"""
def loadDataSet():
    # data list
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # label for the sentence
    classVec = [0,1,0,1,0,1]
    return postingList,classVec
 
"""
Process dataset and get all unique words occurred in dataset
Input: dataset
Output: unique word array
"""
def createVocabList(dataSet):
    #create empty set
    vocabSet = set([]) 
    # tranverse each sentence
    for document in dataSet:
        #union of the two sets to get all possible unique words
        vocabSet = vocabSet | set(document) 
    return list(vocabSet)

"""
Check the if any words of unique word array occurred in input
Input: Unique word array, Test Data
Output: Word occurred situation
"""
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        # check if the word in the list
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec
```

## 4.5.2 Train: calculating probabilities from word vectors
Now try to calculate the probabilities according to the example and the given equation:

![3](http://latex.codecogs.com/svg.latex?P(c_i|w)=\frac{P(w|c_i)P(c_i)}{P(w)})

P(ci) would be the times of the class occurred dividing by the total number of data.

w is the vector of words. To calculate P(w|ci), we need calculate P(w0|ci)P(w1|ci)....P(wn|ci), the pesudocode would look like this:
```
for every training document:
    for each class:
        if a token appears in the document:
            increment the count for tokens
    for each class:
        for each token:
            divide the token count by the total token count to get conditional probabilities
return conditional probabilities for each class
```
Below is the python code:
```Python
from numpy import *

def trainNB0(trainMatrix, trainCategory):
    # get number of dataset
    numTrainDocs = len(trainMatrix)
    # get length of longest sentence
    numWords = len(trainMatrix[0])
    # get probability of class abusive
    pAbusive = sum(trainCategory/float(numTrainDocs))
    # count for token occurred times
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    # count for token number
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # count probability of each token
    p1Vect = p1Num/p1Denom
    p0Vect = p0Num/p0Denom
    return p0Vect,p1Vect,pAbusive
```
## 4.5.3 Problems will occurr
