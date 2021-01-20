# hyppo

[![Build Status](https://circleci.com/gh/neurodata/hyppo/tree/master.svg?style=shield&circle-token=:circle-token)](https://circleci.com/gh/neurodata/hyppo)
[![Codecov](https://codecov.io/gh/neurodata/hyppo/branch/master/graph/badge.svg?token=a2TXyRVW0a)](https://codecov.io/gh/neurodata/hyppo)
[![Netlify](https://img.shields.io/netlify/e5242ebd-631e-4330-b43e-85e428dac66a)](https://app.netlify.com/sites/hyppo/deploys)
[![PythonVersion](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)
[![PyPi](https://badge.fury.io/py/hyppo.svg)](https://pypi.org/project/hyppo/)
[![arXiv shield](https://img.shields.io/badge/arXiv-1907.02088-red.svg?style=flat)](https://arxiv.org/abs/1907.02088)

`hyppo` (**HYP**othesis Testing in **P**yth**O**n, pronounced "Hippo") is an open-source software package for multivariate hypothesis testing.

- **Documentation:** [https://hyppo.neurodata.io/](https://hyppo.neurodata.io/)
- **Tutorials:** [https://hyppo.neurodata.io/tutorials.html](https://hyppo.neurodata.io/tutorials.html)
- **Source Code:** [https://github.com/neurodata/hyppo/tree/master/hyppo](https://github.com/neurodata/hyppo/tree/master/hyppo)
- **Issues:** [https://github.com/neurodata/hyppo/issues](https://github.com/neurodata/hyppo/issues)
- **Contribution Guide:** [https://hyppo.neurodata.io/contributing.html](https://hyppo.neurodata.io/contributing.html)

**Note: We have recently moved our `benchmarks` (with relevant figure replication code for our papers) folder to a new [repo](https://github.com/neurodata/hyppo-papers)!** We aim to add test development code and paper figure replication code there, and will add relevant tests (with tutorials) to ``hyppo``.

`hyppo` intends to be a comprehensive multivariate hypothesis testing package and runs on all major versions of operating systems. It also includes novel tests not found in other packages. It is quick to install and free of charge. If you need to use multivariate hypothesis testing, be sure to give `hyppo` a try!

## Install Guide

The installation guide cange found found here: [https://hyppo.neurodata.io/install.html](https://hyppo.neurodata.io/install.html). Relevant sections have been added below:

Below we assume you have the default Python environment already configured on your computer and you intend to install `hyppo` inside of it.  If you want to create and work with Python virtual environments, please follow instructions on [venv](https://docs.python.org/3/library/venv.html) and [virtual environments](http://docs.python-guide.org/en/latest/dev/virtualenvs/). We also highly recommend conda. For instructions to install this, please look at [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

First, make sure you have the latest version of `pip` (the Python package manager) installed. If you do not, refer to the `pip` [documentation](https://pip.pypa.io/en/stable/installing/) and install `pip` first.

### Install from PyPi

Install the current release of `hyppo` from the Terminal with `pip`:

```sh
pip install hyppo
```

To upgrade to a newer release use the `--upgrade` flag:

```sh
pip install --upgrade hyppo
```

If you do not have permission to install software systemwide, you can install into your user directory using the `--user` flag:

```sh
pip install --user hyppo
```

### Install from Github

You can manually download `hyppo` by cloning the git repo master version and running the `setup.py` file. That is, unzip the compressed package folder and run the following from the top-level source directory using the Terminal:

```sh
git clone https://github.com/neurodata/hyppo
cd hyppo
python3 setup.py install
```

Or, alternatively, you can use `pip`:

```sh
git clone https://github.com/neurodata/hyppo
cd hyppo
pip install .
```

### Other Important Information

`hyppo` requires the following packages as dependencies:

```python
numba
numpy
scipy
scikit-learn
joblib
```

`hyppo` package requires only a standard computer with enough RAM to support the in-memory operations. This package is supported for all major operating systems. The following versions of operating systems was tested on Travis CI:

- **Linux:** Ubuntu Xenial 16.04
- **Windows:** Windows Server, version 1803
