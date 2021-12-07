r"""
.. _kgof:

Kernel Goodness-of-Fit Testing
*********************

This notebook will introduce usage of kgof, a kernel goodness-of-fit hyppo package 
implementing a linear time kernel-based GoF test.

Given a known probability density :math: `p` (model) and a sample 
:math: `\{ \mathbf{x}_i \}_{i=1}^n \sim q` where :math:`q` is an unknown 
density, a goodness-of-fit test proposes null and alternative hypotheses:

.. math::

    H_0 &: p = q \\
    H_1 &: p \neq q

In other words, the GoF test tests whether or not the sample :math: `\{ \mathbf{x}_i \}_{i=1}^n` 
is distributed according to a known :math:`p`.

The implemented test relies on a new test statistic called The Finite-Set Stein Discrepancy (FSSD) 
which is a discrepancy measure between a density and a sample. Unique features of the new goodness-of-fit test are:

It makes only a few mild assumptions on the distributions :math: `p` and :math: `q`. The model :math: `p` 
can take almost any form. The normalizer of :math: `p` is not assumed known. The test only assesses the goodness of 
:math: `p` through :math: `\nabla_{\mathbf{x}} \log p(\mathbf{x})` i.e., the first derivative of the log density.

The runtime complexity of the full test (both parameter tuning and the actual test) is 
:math: `\mathcal{O}(n)` i.e., linear in the sample size.

It returns a set of points (features) which indicate where :math: `p` fails to fit the data.

For demonstration purposes, let us consider a simple two-dimensional problem where :math: `p` 
is the standard Gaussian.
"""

########################################################################################
# .. _gauss fssd:
#
# A simple Gaussian model
# ---------------------------------------------
#
# Assume that :math: `p(\mathbf{x}) = \mathcal{N}(\mathbf{0}, \mathbf{I})`
# in :math: `\mathbb{R}^2` (two-dimensional space).
# The data :math: `\{ \mathbf{x}_i \}_{i=1}^n \sim q = \mathcal{N}([m, 0], \mathbf{I})`
# where :math: `m` specifies the mean of the first coordinate of :math: `q`.
# From this setting, if :math: `m\neq 0`, then :math: `H_1` is true and the test should
# reject :math: `H_0`.
# First construct the log density function for the model

import data 
import density
import kernel 
import _utils
import matplotlib
import matplotlib.pyplot as plt
import autograd.numpy as np
import scipy.stats as stats

def isogauss_log_den(X):
  mean = np.zeros(2)
  variance = 1
  unden = -np.sum((X - mean)**2, 1) / (2.0 * variance)
  return unden

########################################################################################
# This function computes the log of an unnormalized density. This works fine as this test
# only requires a :math: `nabla_{\mathbf{x}} \log p(\mathbf{x})` which does not depend on 
# the normalizer. The gradient :math: `\nabla_{\mathbf{x}} \log p(\mathbf{x})` will be 
# automatically computed by autograd. In this kgof package, a model :math: `p` can be 
# specified by implementing the class :class: `hyppo.kgof.density.UnnormalizedDensity`. Implementing
# this directly is a bit tedious, however. An easier way is to use the function 
# :function: `hyppo.kgof.density.from_log_den(d, f)` which takes 2 arguments as inputs ``d``, 
# the dimension of the input space, and ``f``, a function
# taking in a 2D numpy array of size :math: `n` x :math: `d` and producing a one-dimensional array
# of size :math: `n` for the :math: `n` values of the log unnormalized density.
# Construct an :class: `UnnormalizedDensity` which will represent a Gaussian model. All the implemented
# goodness-of-fit tests take this object as input.

p = density.from_log_den(2, isogauss_log_den) # UnnormalizedDensity object

# Next, draw a sample from q.
# Drawing n points from q
m = 1 # If m = 0, p = q and H_0 is true

seed = 4
np.random.seed(seed)
n = 400
X = np.random.randn(n, 2) + np.array([m, 0])

# Plot the data from q
plt.plot(X[:, 0], X[:, 1], 'ko', label='Data from $q$')
plt.legend()

########################################################################################
# All the implemented tests take the data in the form of a :class: `hyppo.kgof.data.Data` object. 
# This is just an encapsulation of the sample :math: `X`. To construct :class: `data.Data` do the following:

# dat will be fed to the test.
dat = data.Data(X) # Creates a fssdgof Data object here

########################################################################################
# Now that the data has been generated, randomly split it into two disjoint halves: ``train`` and ``test``. 
# The training set ``train`` will be used for parameter optimization. The testing set ``test`` will be used 
# for the actual goodness-of-fit ``test``. ``train``` and ``test`` are again of type :class: `data.Data`.

train, test = dat.split_tr_te(tr_proportion=0.2, seed=2)

# Optimize the parameters of the test on train. The optimization relies on autograd to compute the gradient. 
# A Gaussian kernel is being used for the test.

# J is the # of test locs (features), not larger than 10
J = 1

opts = {
    'reg': 1e-2, # regularization parameter in the optimization objective
    'max_iter': 50, # maximum number of gradient ascent iterations
    'tol_fun':1e-7, # termination tolerance of the objective
}

# make sure to give train (NOT test).
# do the optimization with the options in opts.
# V_opt, gw_opt, opt_info = GaussFSSD.optimize_auto_init(p, train, J, **opts)

########################################################################################
# The optimization procedure returns --
# .. note::
#
#    V_opt: optimized test locations (features). A :math:`J \times d` ``numpy`` array.
#    gw_opt: optimized Gaussian width (for the Gaussian kernel). A floating point number.
#    opt_info: a dictionary containing information gathered during the optimization.

# opt_info

########################################################################################
# Use these optimized parameters to construct the FSSD test. The test using a Gaussian 
# kernel is implemented in :class: `hyppo.kgof.goftest.GaussFSSD`.

alpha = 0.01 # significance level of the test (99% confidence)
## fssd_opt = GaussFSSD(p, gw_opt, V_opt, alpha)

# Perform the goodness-of-fit test on the testing data test.
# return a dictionary of testing results
## test_result = fssd_opt.test(test)
## test_result

########################################################################################
# It can be seen that the test correctly rejects :math:`H_0` with a very small p-value.

########################################################################################
#
# Important note
# ---------------------------------------------
#
# A few points worth mentioning
#
# The FSSD test requires that the derivative of :math: `\log p` exists.
# The test requires a technical condition called the "vanishing boundary" condition for it to be consistent. 
# The condition is :math: `\lim_{\|\mathbf{x} \|\to \infty} p(\mathbf{x}) \mathbf{g}(\mathbf{x}) = \mathbf{0}` where 
# :math: `\mathbf{g}` is the so called the Stein witness function which depends on the kernel and 
# :math: `\nabla_{\mathbf{x}} \log p(\mathbf{x})`. For a density :math: `p` which has support everywhere e.g., 
# Gaussian, there is no problem at all. 
# However, for a density defined on a domain with a boundary, one has to be careful. For example, if :math: `p` is a 
# Gamma density defined on the positive orthant of :math: `\mathbb{R}`, the density itself can actually be evaluated on negative points. 
# Looking at the way the Gamma density is written, there is nothing that tells the test that it cannot be evaluated on negative orthant. 
# Therefore, if :math: `p` is Gamma, and the observed sample also follows :math: `p` 
# (i.e., :math:`H_0` is true), the test will still reject :math:`H_0`! 
# The reason is that the data do not match the left tail (in the negative region!) of the Gamma. 
# It is necessary to include the fact that negative region has 0 density into the density itself.

