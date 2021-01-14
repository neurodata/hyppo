"""
.. _overview:

Overview
========

``hyppo`` is a multivariate hypothesis testing package, dealing with problems such as
independence testing, *k*-sample testing, time series independence testing, etc. It
includes algorithms developed by the `neurodata lab <https://neurodata.io/mgc/>`_ as
well as relevant important tests within the field.

The primary motivation for creating ``hyppo`` was simply the limitations of tools that
data scientists are afforded in Python, which then leads to complex workflows using
other languages such as R or MATLAB. This is especially true for hypothesis testing,
which is a very important part of data science.

Conventions
-----------

Before we get started, here are a few of the conventions we use within ``hyppo``:

* All tests are releagted to a single class, and all classes have a ``.test`` method.
  This method returns a test statistic and p-value, as well as other informative
  outputs depending on the test.
* All functions and classes accept numpy arrays as inputs. Optional inputs vary between
  tests within the package.

The Library
-----------

Most classes and functions are available through the :mod:`hyppo` top level package,
though our workflow generally involves importing specific classes or methods from our
modules.

Our goal is to create a comprehensive hypothesis testing package in a simple and easy
to use interface. Currently, we include the following modules:
:mod:`hyppo.independence`, :mod:`hyppo.ksample`, :mod:`hyppo.time_series`, and
:mod:`hyppo.tools`. The last of which does not contain any tests, but functions to
generate simulated data, that we used to evalue our methods, as well as functions to
calculate p-values or other important functions commmonly used between modules.

Brief Example
--------------

As an example, let's generate some simulated data using :class:`hyppo.tools.linear`:
"""

from hyppo.tools import spiral

x, y = spiral(100, 1, noise=True)


########################################################################################
# The data are points simulating a noisey spiral relationship between random variables
# :math:`X` and :math:`Y` and returns realizations as :class:`numpy.ndarray`:

import matplotlib.pyplot as plt

plt.figure()
plt.scatter(x, y)
plt.show()

########################################################################################
# Let's ask the question: are ``x`` and ``y`` independent? From the description given
# above, the answer to that is obviously yes. We can verify that this is in fact true by
# running an independence test. Let's using the test multiscale graph correlation (MGC)
# which, as an aside, was the test that started the creation of the package. First, we
# have to import it, and then run the test:

from hyppo.independence import MGC

stat, pvalue, mgc_dict = MGC().test(x, y)
# Printing a gridder shows the class and all of it's configuration options.
print(stat, pvalue)

########################################################################################
# We see that we are right! Since the p-value is less than the alpha level of 0.05, we
# can conclude that random variables :math:`X` and :math:`Y` are independent. A cool
# thing about MGC (and something we will expand upon in it's documentation) is that we
# get a map of test statistics that inform the nature of the relationships, i.e. linear,
# nonlinear, etc.

import matplotlib.ticker as ticker
import seaborn as sns

mgc_map = mgc_dict["mgc_map"]
opt_scale = mgc_dict["opt_scale"]  # i.e. maximum smoothed test statistic

print("Optimal Scale:", opt_scale)
fig, (ax, cax) = plt.subplots(
    ncols=2, figsize=(9.45, 7.5), gridspec_kw={"width_ratios": [1, 0.05]}
)

# draw heatmap and colorbar
ax = sns.heatmap(mgc_map, cmap="YlGnBu", ax=ax, cbar=False)
fig.colorbar(ax.get_children()[0], cax=cax, orientation="vertical")
ax.invert_yaxis()

# optimal scale
ax.scatter(opt_scale[0], opt_scale[1], marker="X", s=200, color="red")

# make plots look nice
fig.suptitle("MGC Map", fontsize=17)
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
# And that's it! Aside from the last section (which is unique to MGC), this is how you
# run a test in ``hyppo``.


########################################################################################
# Wrap Up
# -------
#
# This covers the basics of using most tests in ``hyppo``. Most use cases and examples
# in the documentation will involve some variation of the following workflow:
#
# 1. Load your data and convert to :class:`numpy.ndarray`
# 2. Import the desired test
# 3. Run the test on your data
# 4. Obtain a test statistic and p-value (among other outputs)
