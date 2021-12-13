r"""
.. _d_var:

D-Variate Independence Testing
*******************************

Here, we consider joint independence testing of :math:`d` random variables. This
is a more difficult task than pairwise independence testing, but this can be very
useful when we are asking the question of whether three or more groups are affecting
one another. Joint independence can be tested by combining pairwise independence
tests, but using a :math:`d`-variate independence test is generally faster.

The :math:`d`-variate independence test can be found in :mod:`hyppo.d_variate`, and
will be explained in detail below. Like all the other tests within hyppo, each
method has a :func:`statistic` and :func:`test` method. The :func:`test` method is
the one that returns the test statistic and p-values, among other outputs, and is
the one that is used most often in the examples, tutorials, etc. The p-value returned
is calculated using a permutation test using :meth:`hyppo.tools.multi_perm_test`.

Specifics about how the statistic is calculated in :class:`hyppo.d_variate` can be
found in the docstring of the test. Here, we give an overview of the :math:`d`-variate
independence test we offer in hyppo and some of its properties compared to those
in :mod:`hyppo.independence`.
"""

########################################################################################
# D-variable Hilbert Schmidt Independence Criterion (dHsic)
# ---------------------------------------------------------
#
# **dHsic** is an extension of :class:`hyppo.independence.Hsic`, and it uses the
# reproducing kernel Hilbert space to test for the joint independence of :math:`d`
# random variables. More details can be found in :class:`hyppo.d_variate.dHsic`.
# Note that unlike :class:`hyppo.independence.Hsic`, there is no fast version of
# the test. It always uses the permutation method to compute its p-value.
#
# .. note::
#
#    :Pros: - Highly accurate independence test for d random variables
#           - Much faster than constructing a joint independence test from multiple
#             pairwise independence tests
#    :Cons: - Is not always more powerful than pairwise Hsic, depends on simulation
#             and the dependence structure of the variables
#
# dHsic is often computationally less expensive than using pairwise Hsic, and if
# dimension :math:`d` is too large, a pairwise Hsic approach may fail to reject
# the null hypothesis.
#
# The following is a general use case of dHsic using data points that simulate a
# 1D linear relationship between random variables :math:`X`, :math:`Y`, :math:`U`,
# and :math:`V`. Note that here we use the default gaussian kernel with a gamma
# value of 0.5. For a full list of parameters, see :class:`hyppo.d_variate.dHsic`.

from hyppo.d_variate import dHsic
from hyppo.tools import linear

x, y = linear(100, 1)
u, v = linear(100, 1)
stat, pvalue = dHsic(gamma=0.5).test(x, y, u, v)
print(stat, pvalue)

########################################################################################

