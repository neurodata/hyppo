# hyppo
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[all-contrib]: https://img.shields.io/badge/all_contributors-36-orange.svg?style=flat 'All Contributors'
<!-- ALL-CONTRIBUTORS-BADGE:END -->

[![CircleCI](https://img.shields.io/circleci/build/github/neurodata/hyppo/main?style=flat)](https://app.circleci.com/pipelines/github/neurodata/hyppo?branch=main)
[![Codecov](https://img.shields.io/codecov/c/github/neurodata/hyppo?style=flat)](https://codecov.io/gh/neurodata/hyppo)
[![Netlify](https://img.shields.io/netlify/e5242ebd-631e-4330-b43e-85e428dac66a?style=flat)](https://app.netlify.com/sites/hyppo/deploys)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hyppo?style=flat)](https://pypi.org/project/hyppo/)
[![PyPI](https://img.shields.io/pypi/v/hyppo?style=flat)](https://pypi.org/project/hyppo/)
[![arXivshield](https://img.shields.io/badge/arXiv-1907.02088-red.svg?style=flat)](https://arxiv.org/abs/1907.02088)

hyppo (**HYP**othesis Testing in **P**yth**O**n, pronounced "Hippo") is an open-source software package for multivariate hypothesis testing. We decided to develop hyppo for the following reasons:

* With the increase in the amount of data in many fields, hypothesis testing for high-dimensional and nonlinear data is important.
* Libraries in R exist, but their interfaces are inconsistent, and most are not available in Python.

hyppo intends to be a comprehensive multivariate hypothesis testing package that runs on all major versions of operating systems. It also includes novel tests not found in other packages. It is quick to install and free of charge. If you need to use multivariate hypothesis testing, be sure to give hyppo a try!

Website: [https://hyppo.neurodata.io/](https://hyppo.neurodata.io/)

## Installation

### Dependencies

hyppo requires the following:

* [python](https://www.python.org/) (>= 3.8)
* [numba](https://numba.pydata.org/) (>= 0.46)
* [numpy](https://numpy.org/)  (>= 1.17)
* [scipy](https://docs.scipy.org/doc/scipy/reference/) (>= 1.4.0)
* [scikit-learn](https://scikit-learn.org/stable/) (>= 0.22)
* [joblib](https://joblib.readthedocs.io/en/latest/) (>= 0.17.0)
* [statsmodels](https://www.statsmodels.org/) (>= 0.14.4)
* [patsy](https://patsy.readthedocs.io/en/latest/) (>= 0.5.1)
* [future](https://pypi.org/project/future/) (>=1.0.0)

### User installation

The easiest way to install hyppo is using `pip`.

```sh
pip install hyppo
```

To upgrade to a newer release, use the `--upgrade` flag.

```sh
pip install --upgrade hyppo
```

The documentation includes more detailed [installation instructions](https://hyppo.neurodata.io/get_start/install.html).
hyppo is free software; you can redistribute it and/or modify it under the
terms of the [license](https://hyppo.neurodata.io/development/license.html).

## Release Notes

See the [release notes](https://hyppo.neurodata.io/changelog/index.html)
for a history of notable changes to hyppo.

## Development

We welcome new contributors of all experience levels. The hyppo
community's goals are to be helpful, welcoming, and effective. The
[contributor guide](https://hyppo.neurodata.io/development/contributing.html)
has detailed information about contributing code, documentation, and tests.

* Official source code: [https://github.com/neurodata/hyppo/tree/main/hyppo](https://github.com/neurodata/hyppo/tree/main/hyppo)
* Download releases: [https://pypi.org/project/hyppo/](https://pypi.org/project/hyppo/)
* Issue tracker: [https://github.com/neurodata/hyppo/issues](https://github.com/neurodata/hyppo/issues)

**Note: We have recently moved our benchmarks (with relevant figure replication code for our papers) folder to a new [repo](https://github.com/neurodata/hyppo-papers)!** We aim to add test development code and paper figure replication codes there, and we will add relevant tests (with tutorials) to hyppo.

## Contributors

Thanks goes to these wonderful people:

<a href="https://github.com/neurodata/hyppo/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=neurodata/hyppo" />
</a>

Made with [contrib.rocks](https://contrib.rocks).

## Project History

hyppo is a rebranding of mgcpy, which was founded in November 2018.
mgcpy was designed and written by [@tpsatish95](https://github.com/tpsatish95), [@sampan501](https://github.com/sampan501), [@junhaobearxiong](https://github.com/junhaobearxiong), [@sundaysundya](https://github.com/sundaysundya), [@ananyas713](https://github.com/ananyas713), and [@ronakdm](https://github.com/ronakdm). hyppo
was designed and written by [@sampan501](https://github.com/sampan501) and first released in February 2020.
