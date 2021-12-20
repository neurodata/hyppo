r"""
.. _kgof:

Kernel Goodness-of-Fit Testing
******************************

Figuring out the origin of a particular set of data is difficult. However, insights gained from 
the origin of a set of samples (i.e., its target density function) can be instrumental in understanding
its characteristics. This sort of testing is desirable in practice to answer questions such as: does this
sample correctly represent this population? Can a certain statistical phenomena be attributed to a given 
distribution?

For anyone interested in questions like the above, this module of the package is perfect! 
The goal of goodness-of-fit testing is to determine, given a set of samples, 
how likely it is that these were generated from a target density function. The tests can be found in
:mod:`hyppo.kgof`, and will be elaborated upon in detail below. But let's take a look 
at some of the mathematical theory behind the test formation:

Consider a known probability density :math:`p` and a sample :math:`\{ \mathbf{x}_i \}_{i=1}^n \sim q`
where :math:`q` is an unknown density, a goodness-of-fit test proposes null and alternative hypotheses: 

.. math::

    H_0 &: p = q \\
    H_1 &: p \neq q

In other words, the GoF test tests whether or not the sample :math:`\{ \mathbf{x}_i \}_{i=1}^n` 
is distributed according to a known :math:`p`.

The implemented test relies on a new test statistic called The Finite-Set Stein Discrepancy (FSSD) 
which is a discrepancy measure between a density and a sample. Unique features of the new goodness-of-fit test are:

It makes only a few mild assumptions on the distributions :math:`p` and :math:`q`. The model :math:`p` 
can take almost any form. The normalizer of :math:`p` is not assumed known. The test only assesses the goodness of 
:math:`p` through :math:`\nabla_{\mathbf{x}} \log p(\mathbf{x})` i.e., the first derivative of the log density.

It returns a set of points (features) which indicate where :math:`p` fails to fit the data.

"""

########################################################################################
# .. _gauss fssd:
#
# A simple Gaussian model
# ---------------------------------------------
#
# Assume that :math:`p(\mathbf{x}) = \mathcal{N}(\mathbf{0}, \mathbf{I})`
# in :math:`\mathbb{R}^2` (two-dimensional space).
# The data :math:`\{ \mathbf{x}_i \}_{i=1}^n \sim q = \mathcal{N}([m, 0], \mathbf{I})`
# where :math:`m` specifies the mean of the first coordinate of :math:`q`.
# From this setting, if :math:`m\neq 0`, then :math:`H_1` is true and the test should
# reject :math:`H_0`.
# First construct the log density function for the model

from hyppo.kgof import Data
from hyppo.kgof import KGauss
from hyppo.kgof import IsotropicNormal
from hyppo.kgof import fit_gaussian_draw, meddistance
from hyppo.kgof import FSSD, FSSDH0SimCovObs
import matplotlib.pyplot as plt
import autograd.numpy as np
from numpy.random import default_rng

def isogauss_log_den(X):
  mean = np.zeros(2)
  variance = 1
  unden = -np.sum((X - mean)**2, 1) / (2.0 * variance)
  return unden

########################################################################################
# This function computes the log of an unnormalized density. This works fine as this test
# only requires a :math:`nabla_{\mathbf{x}} \log p(\mathbf{x})` which does not depend on 
# the normalizer. The gradient :math:`\nabla_{\mathbf{x}} \log p(\mathbf{x})` will be 
# automatically computed by autograd. In this kgof package, a model :math:`p` can be 
# specified by implementing the class :class:`hyppo.kgof.density.UnnormalizedDensity`. Implementing
# this directly is a bit tedious, however. Construct an :class:`UnnormalizedDensity` which will 
# represent a Gaussian model. All the implemented goodness-of-fit tests take this object as input.

# Next, draw a sample from q.
# Drawing n points from q
m = 1 # If m = 0, p = q and H_0 is true

seed = 4
rng = default_rng(seed)
n = 400
X = rng.standard_normal(size=(n, 2)) + np.array([m, 0])

# Plot the data from q
plt.plot(X[0], X[1], 'ko', label='Data from $q$')
plt.legend()

########################################################################################
# All the implemented tests take the data in the form of a :class:`hyppo.kgof.data.Data` object. 
# This is just an encapsulation of the sample :math:`X`. To construct :class:`data.Data` do the following:

dat = Data(X)

########################################################################################
# Conduct an FSSD kernel goodness-of-fit test for an isotropic normal density with mean 
# 0 and variance 1. 

mean = 0
variance = 1

J = 1
sig2 = meddistance(X, subsample=1000) ** 2
k = KGauss(sig2)
isonorm = IsotropicNormal(mean, variance)

# random test locations
V = fit_gaussian_draw(X, J, seed=seed + 1)
null_sim = FSSDH0SimCovObs(n_simulate=200, seed=3)
fssd = FSSD(isonorm, k, V, null_sim=null_sim, alpha=0.01)

tresult = fssd.test(dat, return_simulated_stats=True)
tresult

