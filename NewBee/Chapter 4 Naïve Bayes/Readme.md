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
Output: Words occurred situation
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
## 4.5.3 Test: For real-world conditions
### Problem 1: 0 in multiplier
when initialize p0Num and p1Num, zeros() function is used, in this case, if there's any words occured 0 times, the final multiple result will be 0.
So it will be better to use ones() function, and set p0Denom, p1Denom start with 2.
```Python
# reinitialize variable
p0Num = ones(numWords)
p1Num = ones(numWords)
p0Denom = 2.0
p1Denom = 2.0
```
### Problem 2: Underflow
When the data amount is large, the probability for each word could be extremely low that will be show as 0. Then it will back to the problem 1.
To solve this problem, natural logarithm could be used. Consider ln(a*b) = ln(a) + ln(b), we can turn the product of probability for each words into the sum of log for probability of each words.
```Python
p1Vect = log(p1Num/p1Denom)
p0Vect = log(p0Num/P0Denom)
```
### Ready for test
Now we need to calculate P(c1|w) and P(c0|w).
As mentioned before, the equation is ![3](http://latex.codecogs.com/svg.latex?P(c_i|w)=\frac{P(w|c_i)P(c_i)}{P(w)}).
since P(w) will be the same, we just need to compare the result P(w|ci)P(ci).
The probability has go through natural logarithm, here the product will be replaced by plus. 
```Python
"""
classify the given words
Input: Word array, p0Vec = P(wi|c0) array, p1Vec = P(wi|C1) array, probability of class 1
Output: class
"""
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    #Calculate P1 and P0
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
        
"""
Testing the given words
"""
def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
```
# 4.5.4 Prepare: the bag-of-words document model
In previous codes, only record if a word occurred. Now turn it into record how many times a word occurred, to make the result more accurate.
```Python
"""
Similar with function setOfWords2Vec(vocabList, inputSet)
"""
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
```
# 4.6 Example: classifying spam email with Naïve Bayes
# 4.6.1 Prepare: tokens, data process
You can check it in book if you interested, basiclly, it's about how to split string and remove empty tokens.
# 4.6.2 Test:
regulartion expression is used to process string.[Python Regulartion Expression Document](https://docs.python.org/3/library/re.html).
```Python
def textParse(bigString):
    import re
    # split data based on all none alphabet, none nums, none uder line characters.
    listOfTokens = re.split(r'\W',bigString)
    # return tokens longer than 2 in lower cases
    return [tok.lower() for tok in listOfTokens if len(tok) > 2 ]

def spamTest():
    docList=[]
    classList = []
    fullText =[]
    # Here the range is the name of file, both of spam and ham are from 1-25
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    #create vocabulary
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet=[]           #create test set
    # Take ten instance as testing dataset, here just generate the random index
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]
    trainClasses = []
    #train the classifier (get probs) trainNB0
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    #classify the testing set
    for docIndex in testSet:        
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error",docList[docIndex])
    print('the error rate is: ',float(errorCount)/len(testSet))
    #return vocabList,fullText
```
# 4.7 Example: using Naïve Bayes to reveal local attitudes from personal ads
## 4.7.1 Data Collection with RSS source
The link author given seems not working anymore, you may have a test with it. The code is in the given "bayes.py" file.
Below is the rss link given by author and code to call the function.
```Python
import feedparser
ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
vocabList,psF,pNY = localWords(ny,sf)
```
## 4.7.2 RSS feed classifier and frequent word removal functions
```python
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    # calculate times the word occurred
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    # sort the dict
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    # return thirty most frequent word
    return sortedFreq[:30]   
    
def localWords(feed1,feed0):
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    #create vocabulary
    vocabList = createVocabList(docList)
    # remove top 30 words
    top30Words = calcMostFreq(vocabList,fullText)   
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    #create test set
    trainingSet = list(range(2*minLen)); testSet=[]           
    print(len(trainingSet))
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    #train the classifier (get probs) trainNB0
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    #classify the remaining items
    for docIndex in testSet:        
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V
```
## 4.7.3 Display Locally Used Words
```python
def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])
```
