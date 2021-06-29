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


## 4.3  Classifying with Conditional Probability
As the example in 4.1, the equation could be extend to:

![2](http://latex.codecogs.com/svg.latex?P(c_i|x,y)=\frac{P(x,y|c_i)P(c_i)}{P(x,y)})

If P(c1|x,y) > P(c2|x,y), then the class is c1.
If P(c1|x,y) < P(c2|x,y), then the class is c2.
