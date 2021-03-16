..  -*- coding: utf-8 -*-

Changelog
=========

*Note: People with a "+" by their names contributed a patch for the first time.*

v0.2.1
------
| **Release date:** 25 February 2021
| **Supports:** Python 3.6+.

**Bug Fixes:**

* :class:`hyppo.independence.Dcorr` and :class:`hyppo.independence.Hsic` when ``auto=True`` had low finite power, fixed.
* Fast exact statistic for :class:`hyppo.independence.Dcorr` was incorrect, fixed

**Documentation:**

* Fix PyPi description rendering issues
* Add more descriptive contirubting guidelines
* Add ``ROLES.md`` for specification about maintainers
* Added power computation for independence increasing sample size and dimension
* Add benchmark section and move relevant examples there
* Add base classes

**Maintenance:**

* Fix Gaussian kernel to prevent division by 0
* Add checks for Type I error

**Authors:**

+ Sambit Panda

v0.2.0
------
| **Release date:** 08 February 2021
| **Supports:** Python 3.6+.

**New features:**

* Added restricted permutation functionality for Dcorr
* Kernel functions now wrap scikit-learn and support keyword arguments
* :class:`hyppo.ksample.Energy`
* :class:`hyppo.ksample.DISCO`
* :class:`hyppo.ksample.MMD`
* Fast 1D exact Dcorr :math:`\mathcal{O}(n \log n)`
* :class:`hyppo.ksample.Hotelling`
* :class:`hyppo.ksample.MANOVA`
* :class:`hyppo.independence.MaxMargin`

**Bug Fixes:**

* Fixed error check for k-sample tests to be between samples instead of within
* Time series doesn't clip negative values
* Fix docs not building on netlify
* Fix p-value calculations for permutation tests to be more in line with literature
* Fix :class:`hyppo.independence.Dcorr` and :class:`hyppo.independence.Hsic` incorrect stats

**Documentation:**

* Update badges and README to FIRM guidelines
* Incorrect equation in :meth:`hyppo.tools.circle` docstring
* Update README to be in line with scikit-learn
* Remove literature reference section in docstrings, add links to papers
* Add docstrings for :mod:`hyppo.tools` functions
* Add ``overview.py`` file for an overview of the package
* Add tutorials folder, rewrite so it is more user-friendly (port independence, k-sample, time series)
* Add examples folder with more information about unique features
* Move to ``sphinx-gallery`` instead of nbconvert
* Use ``automodule`` instead of ``autoclass``
* Make clear about the package requirements and docs requirements
* Make ``changelog`` into a single file
* Add external links to neurodata and code of conduct
* Add citing page to cite the package papers
* Make index page a clone of README
* Update MakeFile for more options
* Add intersphinx mapping with links externally (``numpy``, ``scipy``, etc.) and internally
* Add docs for statistic function
* Add discriminability tutorial

**Maintenance:**

* Fix typos in warning commits
* Updated tests to precalculate distance matrix
* Moved from Travis CI to Circle CI
* Raise base ``requirements.txt`` to fix failing tests on CircleCI
* Add code coverage config files
* Add documentation folders and files to ``.gitignore``
* Remove ``reps`` warning test
* Cache numba after first call to speed up runs
* Fix netlify config to new doc build structure

**Authors:**

+ Sambit Panda
+ Vivek Gopalakrishnan +
+ Ronak Mehta
+ Ronan Perry +

v0.1.3
------
| **Release date:** 24 July 2020
| **Supports:** Python 3.6+.

**Bug Fixes:**

* Prevent division by zero when calculating using default Gaussian median kernel

**Maintenance:**

* Used ``sf`` from :meth:`scipy.stats.chi2` instead of ``1 - cdf`` for :meth:`hyppo.tools.chi2_approx`

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
* :meth:`sklearn.pairwise.pairwise_distances` used instead of :meth:`scipy.spatial.distance.cdist`.
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
* :class:`hyppo.independence.Hsic` is now a stand alone class
* Simulations are given module, with new k-sample and time series modules
* Discrimnability ported from r-mgc
* Benchmarks folder added with relevant notebooks comparing implementations

**Maintenance:**

* Modified scikit-learn API adopted (classes given unique files, organized in
  independence, *k*-sample, and time series modules.

**Authors:**

+ Jayanta Dey +
+ Sambit Panda +

