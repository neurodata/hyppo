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
# The **Fast Conditional Independence Test (CCA)** is a non-parametric conditional
# independence test. The test is based on a weak assumption that if the conditional
# independence alternative hypothesis is true, then prediction of the independent
# variable with only the conditioning variable should be just as accurate as
# prediction of the independent variable using the dependent variable conditioned on
# the conditioning variable.
# More details can be found in :class:`hyppo.conditional.FCIT`
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
print("Statistc: ", stat)
print("p-value: ", pvalue)