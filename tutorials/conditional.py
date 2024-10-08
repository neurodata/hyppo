r"""
.. _conditional:

Conditional Independence Testing
********************************

Conditional independence testing is similar to independence testing but introduces
the presence of a third conditioning variable. Consider random variables :math:`X`,
:math:`Y`, and :math:`Z` with distributions :math:`F_X`, :math:`F_Y`, and :math:`F_Z`.
When performing conditional independence testing, we are evaluating whether
:math:`F_{X, Y|Z} = F_{X|Z}F_{Y|Z}`. Specifically, we are testing

.. math::

    H_0 &: X \perp \!\!\! \perp Y \mid Z \\
    H_A &: X \not\!\perp\!\!\!\perp Y \mid Z

Like all the other tests within hyppo, each method has a :func:`statistic` and
:func:`test` method. The :func:`test` method is the one that returns the test statistic
and p-values, among other outputs, and is the one that is used most often in the
examples, tutorials, etc.

Specifics about how the test statistics are calculated for each in
:class:`hyppo.conditional` can be found the docstring of the respective test. Here,
we overview subsets of the types of conditional tests we offer in hyppo, and special
parameters unique to those tests.

Now, let's look at unique properties of some of the tests in :mod:`hyppo.conditional`:
"""

########################################################################################
# Fast Conditional Independence Test (FCIT)
# ---------------------------------------------
#
# The **Fast Conditional Independence Test (FCIT)** is a non-parametric conditional
# independence test. The test is based on a weak assumption that if the conditional
# independence alternative hypothesis is true, then prediction of the independent
# variable with only the conditioning variable should be just as accurate as
# prediction of the independent variable using the dependent variable conditioned on
# the conditioning variable.
# More details can be found in :class:`hyppo.conditional.FCIT`.
#
# .. note::
#
#    This algorithm is currently under review at a preprint on arXiv.
#
# .. note::
#
#    :Pros: - Very fast due on high-dimensional data due to parallel processes
#    :Cons: - Heuristic method; above assumption, though weak, is not always true
#
# The test uses a regression model to construct predictors for the indendent variable.
# By default, the regressor used is the decision tree regressor but the user can also
# specify other forms of regressors to use along with a set of hyperparameters to be
# tuned using cross-validation. Below is an example where the null hypothesis is true:

import numpy as np
from hyppo.conditional import FCIT
from sklearn.tree import DecisionTreeRegressor
np.random.seed(1234)
dim = 2
n = 100000
z1 = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim), size=(n))
A1 = np.random.normal(loc=0, scale=1, size=dim * dim).reshape(dim, dim)
B1 = np.random.normal(loc=0, scale=1, size=dim * dim).reshape(dim, dim)
x1 = (A1 @ z1.T + np.random.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim), size=(n)).T)
y1 = (B1 @ z1.T + np.random.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim), size=(n)).T)
model = DecisionTreeRegressor()
cv_grid = {"min_samples_split": [2, 8, 64, 512, 1e-2, 0.2, 0.4]}
stat, pvalue = FCIT(model=model, cv_grid=cv_grid).test(x1.T, y1.T, z1)
print("Statistic: ", stat)
print("p-value: ", pvalue)

########################################################################################
# Kernel Conditional Independence Test (KCI)
# ---------------------------------------------
#
# The Kernel Conditional Independence Test (KCI) is a conditional independence test
# that works based on calculating the RBF kernels of distinct samples of data.
# The respective kernels are then normalized and multiplied together to determine
# the test statistic via the trace of the matrix product. The test then employs
# a gamma approximation based on the mean and variance of the independent
# sample kernel values to determine the p-value of the test.
# More details can be found in :class:`hyppo.conditional.KCI`.
#
# .. note::
#
#    :Pros: - Very fast on high-dimensional data due to simplicity and approximation
#    :Cons: - Dispute in literature as to ideal theta value, loss of accuracy on very large datasets
#
# Below is a linear example where we reject the null hypothesis:

import numpy as np
from hyppo.conditional import KCI
from hyppo.tools import linear
np.random.seed(123456789)
x, y = linear(100, 1)
stat, pvalue = KCI().test(x, y)
print("Statistic: ", stat)
print("p-value: ", pvalue)

########################################################################################
# Partial Correlation (PCorr) and Partial Distance Correlation (PDcorr)
# ----------------------------------------------------------------------
#
# Partial Correlation (PCorr) and Partial Distance Correlation (PDcorr)
# are conditional independence tests that are extensions of Pearson's Correlation
# and Distance Correlation, respectively.
# Partial distance correlation introduces a new Hilbert space where the squared
# distance covariance is the inner product.
# More details can be found in :class:`hyppo.conditional.PartialCorr`
# and :class:`hyppo.conditional.PartialDcorr`.
#
# .. note::
#
#    :Pros: - Simplest extension of Pearson's Correlation and Distance Correlation
#    :Cons: - Literature may suggest that this is not actually a dependence measure
#           - Partial correlation makes strong linearity assumptions about the data
#
# Below is a linear example where we reject the null hypothesis:

import numpy as np
from hyppo.conditional import PartialDcorr
from hyppo.tools import correlated_normal
np.random.seed(123456789)
x, y, z = correlated_normal(100, 1)
stat, pvalue = PartialDcorr().test(x, y, z)
print("Statistic: ", stat)
print("p-value: ", pvalue)

########################################################################################
# Conditional Distance Correlation (CDcorr)
# ---------------------------------------------
#
# Conditional Dcorr (CDcorr) is a nonparametric measure of conditional dependence
# for multivariate random variables. The sample version takes the same statistical
# form of Dcorr but is conditioned on a third variable. It has also has strong
# guarantees regarding convergence and asymptotic normality.
# More details can be found in :class:`hyppo.conditional.CDcorr`.
#
# .. note::
#
#    :Pros: - Has stronger theoretical guarantees than PCorr and PDcorr
#    :Cons: - Computationally expensive on very large datasets
#
# Below is a linear example where we reject the null hypothesis:

import numpy as np
from hyppo.conditional import ConditionalDcorr
from hyppo.tools import correlated_normal
np.random.seed(123456789)
x, y, z = correlated_normal(100, 1)
stat, pvalue = ConditionalDcorr().test(x, y, z)
print("Statistic: ", stat)
print("p-value: ", pvalue)
