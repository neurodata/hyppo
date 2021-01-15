..  -*- coding: utf-8 -*-

Changelog
=========

*Note: People with a "+" by their names contributed a patch for the first time.*

v0.1.3
------
| **Release date:** 24 July 2020
| **Supports:** Python 3.6+.

**Bug Fixes:**

* Prevent division by zero when calculating using default gaussian median kernel

**Maintenance:**

* Used ``chi2.sf`` instead of ``1 - chi2.cdf`` for ``chi2_approx``

**Authors:**

* Benjamin Pedigo +
* Anton Alayakin +


v0.1.2
------
| **Release date:** 5 May 2020
| **Supports:** Python 3.6+.

**Bug Fixes:**

* Fixed MMD/k-sample Hsic not running

**Authors:**

+ Sambit Panda

v0.1.1
------
| **Release date:** 28 April 2020
| **Supports:** Python 3.6+.

**Documentation:**

* arXiv badge added to docs.
* OS/Software requirements and license changes updated in README
* Reference docs and tutorials added to Time Series module

**Maintenance:**

* Pearson, Spearman, and Kendall are no longer tests within the package.
* Python 3.5 no longer supported.
* ``pairwise_distances`` from ``sklearn`` used instead of ``cdist`` from ``scipy``.
* Null distribution added as a class atribute
* Calculate kernel once before calculating p-value
* Upper and lower-case inputs are available for ``indep_test``

**Authors:**

+ Ronak Mehta +
+ Sambit Panda
+ Bijan Varjavand +


v0.1.0
------
| **Release date:** 25 February 2020
| **Supports:** Python 3.5+.

*Note: as compared to `mgcpy`_*

.. _mgcpy: https://github.com/neurodata/mgcpy-old

**New features:**

* Parallelization added to all tests
* ``hyppo.independence.Hsic`` is now a stand alone class
* Simulations are given module, with new k-sample and time series modules
* Discrimnability ported from r-mgc
* Benchmarks folder added with relevant notebooks comparing implementations

**Maintenance:**

* Modified scikit-learn API adopted (classes given unique files, organized in
  independence, *k*-sample, and time series modules.

**Authors:**

+ Jayanta Dey +
+ Sambit Panda +

