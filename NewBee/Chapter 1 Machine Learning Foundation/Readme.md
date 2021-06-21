# Machine Learning ---- using Machine to show the meaning of data

# 1.1 What is Machine Learning
Get key information from unordered data and lead to the meaning of these data.

# 1.2 Terms

w[1]\*x[1] + w[2]\*x[2] ..... + w[n]\*x[n] + b = y

### 1. Features
  x[1],x[2],.....,x[n]
  The data used are considered as Features. For different situations, the feature will be different.
  For example, for figures, the features will be value of pixels. 
### 2. Weight
  w[1],w[2],.....,w[n]
  For each feature, there will be a weight for it, represent the importance of this feature in final meaning.
### 3. Bias
### 4. Class
  y
  The final meaning of the data.
  For example, to determin if a figure is dog or not, the class will be "yes" and "no"; to determin which number is the figure represent, the class will be 0,1,2,3,....,9. 
### 5. Training Dataset
  Dataset been used for training the module.
### 6. Testing Dataset
  Dataset been used for testing the module.

# 1.3 What is the main job for Machine Learning
  Classification and Regression
  According to the given data, tell what kind of data it is.
### Supervised Learning
  Label the training data (Known Features and Classes)
  |Supervised Algorithm|Regression Algorithm|
  |--|--|
  |K-Nearest Neighbor Algorithm |Linear Regression|
  |Naive Bayes Algorithm |Locally weighted regression|
  |Support Vector Machine |Ridger Regression|
  |Decision Tree|Lasso Minimum Regression Coefficient Estimation|
### Unspervised Learning
  Unlabled data, while training, module will cluster similar data.
  |||
  |--|--|
  |K-Mean|Maximum Expectation Algorithm|
  |DBSCAN|Parzen Window Design|
  
# 1.4 Select Suitable Algorithm
1. If you want to predict the value of the target variable, you can use a supervised learning algorithm. If the target variable is a discrete value, use a classification algorithm. If the target variable is a continuous value, use a regression algorithm.

2. If it is not to predict the value of the target variable, use unsupervised machine learning. If you need to divide the data set into discrete groups, use a clustering algorithm. If you need to estimate the similarity between the data and each group, you need to use density Estimate algorithm.

3. The second thing to consider is the data problem: whether the characteristic value is a discrete variable or a continuous type, whether there are missing values, outliers, etc.
4. 
# 1.5 Steps for Machine Learning Development
1. Collect data:
2. Prepare to input data: clean data
3. Analyze the input data
4. Training algorithm
5 test algorithm
6. Use algorithm 
