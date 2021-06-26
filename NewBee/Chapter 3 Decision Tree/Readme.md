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
