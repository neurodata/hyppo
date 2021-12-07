r"""
.. _friedmanRafsky:

Friedman Rafsky Test for Randomness
*********************

This notebook will introduce the usage of the Friedman Rafsky test, a multivariate
extension of the Wald-Wolfowitz runs test to test for randomness between two multivariate
samples. More specifically, the function tests whether two multivariate samples were
independently drawn from the same distribution.

The question proposed in 'Multivariate Generalizations of the Wald-Wolfowitz and Smirnov Two-Sample Tests' (1) is that of how to extend the univariate Wald-Wolfowitz runs test to a multivariate setting.

The univariate Wald-Wolfowitz runs test is a non-parametric statistical test that checks a randomness hypothesis for a two-valued data sequence. More specifically, it can be used to test the hypothesis that the elements of a sequence are mutually independent. For a data sequence with identifiers of two groups, say: $$ X \in \mathbb{R}^n , Y \in \mathbb{R}^m $$ we begin by sorting the combined data set :math: `W \in \mathbb{R}^{m+n}` in numerical ascending order. The number of runs is then defined by the number of maximal, non-empty segments of the sequence consisting of adjacent and equal elements. So if we designate every :math: `X = +,  Y = -` an example assortment of the 15 element long sequence of both sets could be given as :math: `+++----++---+++` which contains 5 runs, 3 of which positive and 2 of which negative. By randomly permuting the labels of our data a large number of times, we can compare our true number of runs to that of the random permutations to determine the test statistic and p-value.

For extension to the multivariate case, we determine a neighbor not by proximity in numerical order, but instead by euclidean distance between each point. The data is then 'sorted' via a calculation of the minimum spanning tree in which each point is connected to each other point with edge weight equal to that of the euclidean distance between the two points. The number of runs is then determined by severing each edge of the MST for which the points connected do not belong to the same family. Similarly to the univariate case, the labels associated with each node are then permuted randomly a large number of times, we can compare our number of true runs to the random distribution of runs to determine the multivariate test statistic and p-value. To further understand, let us consider an exmaple in the 2-dimensional case. 

########################################################################################
#
from IPython.display import Image
Image(filename='Full Tree.PNG')
#
# Here the MST has been calculated with the euclidean distance between points used to determine edge weights. If we remove all edges such that the neighboring nodes are not of the same class, we find our new # collection of subgraphs to be:
#
Image(filename='Runs Tree.PNG')
#
# As such, we see that this set of data contains 3 such runs, and again by randomizing the labels of each data point we can determine the test statistic and p-value for the hypothesis that to determine the # randomness of our combined sample.
#
# To test, we first generate two independent samples from the same distribution, in this case a multivariate normal of mean 0, variance diagonal matrix of ones

import numpy as np
import numpy.random as random
from friedmanRafsky import friedmanRafsky

mean = [0, 0, 0, 0, 0]

cov = [[1, 0, 0, 0, 0],
       [0, 1, 0, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 0, 1, 0],
       [0, 0, 0, 0, 1]]

x = random.multivariate_normal(mean, cov, 100)
y = random.multivariate_normal(mean, cov, 100)

x = np.transpose(x)
y = np.transpose(y)

#Concatenate the data sets into one, contiguous array such that each column is an observation

data = np.concatenate((x, y), axis = 1)

#Assign labels for X and Y classes

labels = np.append(np.zeros(len(x[0])), np.ones(len(y[0])))

#Initialize the test and print results

fR = friedmanRafksy()

# Inputs given as --
# .. note::
#
#    data: Array of x and y values such that each column is an observation
#    labels: Numeric labels corresponding to respective x and y observations
#    perm: Number of permutations with which to calculate permutation statistic
#    aglorithm: Algorithm with which to calculate minimum spanning tree (only Kruskal's and Prim's available at this time)


print(fR.test(data, labels, perm = 1000, algorithm = 'Kruskal'))

# The test class returns --
# .. note::
#
#    statistic: The test statistic determined for the true runs count
#    pvalue: The p-value determined via the permutation testing
#
# As expected, a low p-value suggests that we fail to reject the null hypothesis that these samples are drawn
# independently from the same distribution. 

#######################################################################################
# Now, we examine a case in which our samples are not drawn from the same distribution.
# In this case, X will be drawn from the same multivariate normal while Y will be drawn from
# a multivariate normal of varying mean and covariance.

meanX = [0, 0, 0, 0, 0]

covX = [[1, 0, 0, 0, 0],
       [0, 1, 0, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 0, 1, 0],
       [0, 0, 0, 0, 1]]
       
meanY = [1, 2, -1, 0, 1]

covY = [[1, 0, 0, 0, 0],
       [0, 10, 0, 0, 0],
       [0, 0, 4, 0, 0],
       [0, 0, 0, 10, 0],
       [0, 0, 0, 0, 5]]

x = random.multivariate_normal(meanX, covX, 100)
y = random.multivariate_normal(meanY, covY, 100)

x = np.transpose(x)
y = np.transpose(y)

data = np.concatenate((x, y), axis = 1)

labels = np.append(np.zeros(len(x[0])), np.ones(len(y[0])))

fR = friedmanRafksy()

print(fR.test(data, labels, perm = 1000, algorithm = 'Kruskal'))

# As we see in this case, a large p-value suggests we can reject the null hypothesis that these
# samples are independently drawn from the same distribution.

 Important note
# ---------------------------------------------
#
# A few points worth mentioning
#
# It is of note that the Friedman Rafsky test is not a test of independence: rather, it is a test of randomness
# to determine if two samples were independently drawn from the same distribution. If independence testing is 
# desired, a seperate test from this module should be utilized. Additionally, it need not be necessary that
# the samples X and Y have the same number of samples, just that they posess the same number of multivariate
# features. Lastly, labels for X and Y need


