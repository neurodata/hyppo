r"""
.. _indep:

Independence Testing
*********************

Finding trends in data is pretty hard. Independence testing is a fundamental tool that
helps us find trends in data and make further decisions based on these results. This
testing is desirable to answer many question such as: does brain connectivity effect
creativity? Does gene expression effect cancer? Does something effect something else?

If you are interested in questions of this mold, this module of the package is for you!
All our tests can be found in :mod:`hyppo.independence`, and will be elaborated in
detail below. But before that, let's look at the mathematical formulations:

Consider random variables :math:`X` and :math:`Y` with distributions :math:`F_X` and
:math:`F_Y` respectively, and joint distribution is :math:`F_{XY} = F_{Y|X} F_X`.
When performing independence testing, we are seeing whether or not
:math:`F_{Y|X} = F_Y`. That is, we are testing

.. math::

    H_0 &: F_{XY} = F_X F_Y \\
    H_A &: F_{XY} \neq F_X F_Y

Like all the other tests within hyppo, each method has a :func:`statistic` and
:func:`test` method. The :func:`test` method is the one that returns the test statistic
and p-values, among other outputs, and is the one that is used most often in the
examples, tutorials, etc.
The p-value returned is calculated using a permutation test using
:meth:`hyppo.tools.perm_test` unless otherwise specified.

Specifics about how the test statistics are calculated for each in
:class:`hyppo.independence` can be found the docstring of the respective test. Here,
we overview subsets of the types of independence tests we offer in hyppo, and special
parameters unique to those tests. But first, we present the general testing workflow
present for all our tests using one of the tests, :class:`hyppo.independence.MGC`, as
an example.

.. _general indep:

Independence Testing Workflow
---------------------------------------

The first thing that we must do is import the test that we desire. We also import a
simulation from the :mod:`hyppo.tools` module to generate some data to test against;
in this case, :meth:`hyppo.tools.w_shaped`.
"""

from hyppo.independence import MGC
from hyppo.tools import w_shaped

# 100 samples, 1D x and 1D y, noise
x, y = w_shaped(n=100, p=1, noise=True)

########################################################################################
# The data are points simulating a 2D w-shaped relationship between random variables
# :math:`X` and :math:`Y` and returns realizations as :class:`numpy.ndarray`.

import matplotlib.pyplot as plt
import seaborn as sns

# make plots look pretty
sns.set(color_codes=True, style="white", context="talk", font_scale=1)

# look at the simulation
plt.figure(figsize=(5, 5))
plt.scatter(x, y)
plt.xticks([])
plt.yticks([])
sns.despine(left=True, bottom=True, right=True)
plt.show()

########################################################################################
# Let's ask the question: are ``x`` and ``y`` independent? From the description given
# above, the answer to that is obviously yes.
# From the simulation visualization, it's hard to tell.
# We can verify whether or not we can see a trend within the data by
# running our test.
#
# First, we initalize the class. Most tests have a ``compute_distance`` parameter that
# can use accept any metric from :func:`sklearn.metric.pairwise_distances`
# (or :func:`sklearn.metrics.pairwise.pairwise_kernels` for kernel-based methods)
# and additional keyword arguments for the method.
# The parameter can also accept a custom function, or ``None`` in the case where the
# inputs are already distance matrices.
#
# Each test also has a :func:`test` method that has a
# ``reps`` parameter that controls the replications of
# :meth:`hyppo.tools.perm_test` and the ``workers`` parameter controls the number of
# threads when running the parallelized code (``-1`` uses all available cores). We
# highly recommend using a number >= 1 in general since speed increases are noticeable.

stat, pvalue, _ = MGC().test(x, y, reps=1000, workers=-1)
print(stat, pvalue)

########################################################################################
# Note: MGC, like some tests, have 3 outputs. In general, tests in
# :mod:`hyppo.independence` have 2 outputs.
#
# We see that we are right! Since the p-value is less than the alpha level of 0.05, we
# can conclude that random variables :math:`X` and :math:`Y` are independent.
#
# Now, let's look at unique properties of some of the tests in
# :mod:`hyppo.independence`:

########################################################################################
# Pearson's Correlation Multivariate Variants
# ---------------------------------------------
#
# **Cannonical correlation (CCA)** and **Rank value (RV)** are multivariate analogues
# of Pearson's correlation.
# More details can be found in :class:`hyppo.independence.CCA` and
# :class:`hyppo.independence.RV`.
# The following applies to both:
#
# .. note::
#
#    :Pros: - Very fast
#           - Similar to tests found in scientific literature
#    :Cons: - Not accurate when compared to other tests in most situations
#           - Makes dangerous variance assumptions about the data, among others
#             (similar assumptions to Pearson's correlation)
#
# Neither of these test are distance based, and so do not have a ``compute_distance``
# parameter.
# These tests runs like :ref:`any other test<general indep>`.

########################################################################################
# Distance (and Kernel) Based Tests
# -----------------------------------
#
# A number of tests within :mod:`hyppo.independence` use the concept of inter-sample
# distance, or kernel similarity, to generate powerful indpendence tests.
#
# **Heller Heller Gorfine (HHG)** is a powerful multivariate independence test that
# compares intersample distance, and computes a Pearson statistic.
# More details can be found in :class:`hyppo.independence.HHG`.
#
# .. note::
#
#    :Pros: - Very accurate in certain settings
#    :Cons: - Very slow (computationally complex)
#           - Test statistic not very interpretable, not between (-1, 1)
#
# This test runs like :ref:`any other test<general indep>`.
#
# ------------
#
# **Distance Correlation (Dcorr)** is a powerful multivariate independence test based on
# energy distance.
# **Hilbert Schmidt Independence Criterion (Hsic)** is a kernel-based analogue to Dcorr
# that uses the a gaussian median kernel by default.
# More details can be found in :class:`hyppo.independence.Dcorr` and
# :class:`hyppo.independence.Hsic`.
# The following applies to both:
#
# .. note::
#
#    :Pros: - Accurate, powerful independence test for multivariate and nonlinear
#             data
#           - Has enticing empiral properties (foundation of some of the other tests in
#             the package)
#           - Has fast implementations (fastest test in the package)
#    :Cons: - Slightly less accurate as the above tests
#
# For Hsic, kernels are used instead of distances with the ``compute_kernel`` parameter.
# Otherwise, this test runs like :ref:`any other test<general indep>`.
# Any addition, if the bias variant of the test statistic is required, then the ``bias``
# parameter can be set to ``True``. In general, we do not recommend doing this.
# Otherwise, these tests runs like :ref:`any other test<general indep>`.
# Since Dcorr and Hsic are implemented similarly, let's look at Dcorr.

import timeit

import numpy as np

setup_code = """
from hyppo.independence import Dcorr
from hyppo.tools import w_shaped
x, y = w_shaped(n=100, p=1, noise=True)
"""

t_perm = timeit.Timer(stmt="Dcorr().test(x, y, auto=False)", setup=setup_code)
t_fast = timeit.Timer(stmt="Dcorr().test(x, y, auto=True)", setup=setup_code)

perm_time = np.array(t_perm.timeit(number=1))  # permutation Dcorr
fast_time = np.array(t_fast.timeit(number=1000)) / 1000  # fast Dcorr

print(u"Permutation time: {0:.3g}s".format(perm_time))
print(u"Fast time: {0:.3g}s".format(fast_time))

########################################################################################
# Look at the time increases when using the fast test!
# The fast test approximates the null distribution as a chi-squared random variable,
# and so is far faster than the permutation method.
# To call it, simply set ``auto`` to ``True``, which is the default, and if your data
# has a sample size greater than 20, then the test will run.
#
# ------------

########################################################################################
# **Multiscale graph correlation (MGC)** is a powerful independence test the uses the
# power of Dcorr
# and `k`-nearest neighbors to create an efficient and powerful independence test.
# More details can be found in :class:`hyppo.independence.MGC`.
#
# .. note::
#
#    We recently added the majority of the source code of this algorithm to
#    :func:`scipy.stats.multiscale_graphcorr`.
#    This class serves as a wrapper for that implementation.
#
# .. note::
#
#    :Pros: - Highly accurate, powerful independence test for multivariate and nonlinear
#             data
#           - Gives information about geometric nature of the dependency
#    :Cons: - Slightly slower than similar tests in this section
#
# MGC has some specific differences outlined below, but creating the instance of the
# class was demonstrated in the :ref:`general indep` section.

from hyppo.independence import MGC

# get the MGC map and optimal scale (only need the statistic)
_, _, mgc_dict = MGC().test(x, y, reps=0)

########################################################################################
# A unique property of MGC is a MGC map, which is a ``(n, n)`` map (or ``(n, k)`` where
# `k` is less than `n` if there are repeated values). This shows you the test statistics
# at all these different "scales". The optimal scale is an ordered pair that indicates
# nearest neighbors where the test statistic is maximized and is marked by a red "X".
# This optimal scale is the location of the test statistic.
# The heat map gives insights into the nature of the dependency between `x` and `y`.
# Otherwise, this test runs like :ref:`any other test<general indep>`.
# We can plot it below:

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# make plots look pretty
sns.set(color_codes=True, style="white", context="talk", font_scale=1)

mgc_map = mgc_dict["mgc_map"]
opt_scale = mgc_dict["opt_scale"]  # i.e. maximum smoothed test statistic
print("Optimal Scale:", opt_scale)

# create figure
fig, (ax, cax) = plt.subplots(
    ncols=2, figsize=(9, 8.5), gridspec_kw={"width_ratios": [1, 0.05]}
)

# draw heatmap and colorbar
ax = sns.heatmap(mgc_map, cmap="YlGnBu", ax=ax, cbar=False)
fig.colorbar(ax.get_children()[0], cax=cax, orientation="vertical")
ax.invert_yaxis()

# optimal scale
ax.scatter(opt_scale[1], opt_scale[0], marker="X", s=200, color="red")

# make plots look nice
ax.set_title("MGC Map")
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xlabel("Neighbors for x")
ax.set_ylabel("Neighbors for y")
ax.set_xticks([0, 50, 100])
ax.set_yticks([0, 50, 100])
ax.xaxis.set_tick_params()
ax.yaxis.set_tick_params()
cax.xaxis.set_tick_params()
cax.yaxis.set_tick_params()
plt.show()

########################################################################################
# For linear data, we would expect the optimal scale to be at the maximum nearest
# neighbor pair. Since we consider nonlinear data, this is not the case.

########################################################################################
# Random Forest Based Tests
# --------------------------------------------
#
# Random-forest based tests exploit the theoretical properties of decision tree based
# classifiers to create highly accurate tests (especially in cases of high dimensional
# data sets).
#
# **Kernel mean embedding random forest (KMERF)** is one such test, which uses
# similarity matrices as a result of random forest to generate a test statistic and
# p-value. More details can be found in :class:`hyppo.independence.KMERF`.
#
# .. note::
#
#    :Pros: - Highly accurate, powerful independence test for multivariate and nonlinear
#             data
#           - Gives information about releative dimension (or feature) importance
#    :Cons: - Very slow (requires training a random forest for each permutation)
#
# Let's go over the test initialization process. Unlike other tests, there is no
# ``compute_distance``
# parameter. Instead, the number of trees can be set explicityly, and the type of
# classifier can be set ("classifier" in the case where the return value :math:`y` is
# categorical, and "regressor" when that is not the case). Check out
# :class:`sklearn.ensemble.RandomForestClassifier` and
# :class:`sklearn.ensemble.RandomForestRegressor` for additional parameters to change.
# Otherwise, this test runs like :ref:`any other test<general indep>`.
# Using :meth:`hyppo.tools.cubic`:

from hyppo.independence import KMERF
from hyppo.tools import cubic

# 100 samples, 5D sim
x2, y2 = cubic(n=100, p=5)

# get the feature importances (only need the statistic)
_, _, kmerf_dict = KMERF(ntrees=5000).test(x2, y2, reps=0)

########################################################################################
# Because this test is random-forest based, we can get feature importances. This gives
# us relative importances value of dimension (i.e. feature). KMERF returns a normalized
# version of this parameter, and we can plot these importances:

importances = kmerf_dict["feat_importance"]
dims = range(1, 6)  # range of dimensions of simulation


import matplotlib.pyplot as plt
import seaborn as sns

# make plots look pretty
sns.set(color_codes=True, style="white", context="talk", font_scale=1)

# plot the feature importances
plt.figure(figsize=(9, 6.5))
plt.plot(dims, importances)
plt.xlim([1, 5])
plt.xticks([1, 2, 3, 4, 5])
plt.xlabel("Dimensions")
plt.ylim([0, 1])
plt.ylabel("Normalized Feature Importance")
plt.title("Feature Importances")
plt.show()

########################################################################################
# We see that feature importances decreases as dimension increases. This is true for
# most of the simulations in :mod:`hyppo.tools`.
