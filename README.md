# hyppo

[![Build Status](https://circleci.com/gh/neurodata/hyppo/tree/master.svg?style=shield&circle-token=:circle-token)](https://circleci.com/gh/neurodata/hyppo)
[![Codecov](https://codecov.io/gh/neurodata/hyppo/branch/master/graph/badge.svg?token=a2TXyRVW0a)](https://codecov.io/gh/neurodata/hyppo)
[![Netlify](https://img.shields.io/netlify/e5242ebd-631e-4330-b43e-85e428dac66a)](https://app.netlify.com/sites/hyppo/deploys)
[![PythonVersion](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)
[![PyPi](https://badge.fury.io/py/hyppo.svg)](https://pypi.org/project/hyppo/)
[![arXiv shield](https://img.shields.io/badge/arXiv-1907.02088-red.svg?style=flat)](https://arxiv.org/abs/1907.02088)

hyppo (**HYP**othesis Testing in **P**yth**O**n, pronounced "Hippo") is an open-source software package for multivariate hypothesis testing.

hyppo intends to be a comprehensive multivariate hypothesis testing package and runs on all major versions of operating systems. It also includes novel tests not found in other packages. It is quick to install and free of charge. If you need to use multivariate hypothesis testing, be sure to give hyppo a try!

Website: https://hyppo.neurodata.io/

## Installation

### Dependencies

hyppo requires:

- Python (>= 3.6)
- NumPy (>= 1.17)
- Numba (>= 0.46)
- SciPy (>= 1.4.0)
- Joblib (>= 0.17.0)
- scikit-learn (>= 0.22)

### User installation

The easiest way to install hyppo is using `pip`

```sh
pip install hyppo
```

To upgrade to a newer release use the `--upgrade` flag:

```sh
pip install --upgrade hyppo
```

The documentation includes more detailed [installation instructions](https://hyppo.neurodata.io/install.html).
`hyppo` is free software; you can redistribute it and/or modify it under the
terms of the [license](https://hyppo.neurodata.io/license.html).

## Changelog

See the [changelog](https://hyppo.neurodata.io/news.html)
for a history of notable changes to hyppo.

## Development

We welcome new contributors of all experience levels. The hyppo
community goals are to be helpful, welcoming, and effective. The
[Contributer Guide](https://hyppo.neurodata.io/contributing.html)
has detailed information about contributing code, documentation and tests.

- Official source code: [https://github.com/neurodata/hyppo/tree/master/hyppo](https://github.com/neurodata/hyppo/tree/master/hyppo)
- Download releases: [https://pypi.org/project/hyppo/](https://pypi.org/project/hyppo/)
- Issue tracker: [https://github.com/neurodata/hyppo/issues](https://github.com/neurodata/hyppo/issues)

**Note: We have recently moved our `benchmarks` (with relevant figure replication code for our papers) folder to a new [repo](https://github.com/neurodata/hyppo-papers)!** We aim to add test development code and paper figure replication code there, and will add relevant tests (with tutorials) to `hyppo`.

## Project History


`hyppo` is a rebranding of `mgcpy`, which was founded in Novemember 2018.
The original version was designed and written by Satish Palaniappan, Sambit
Panda, Junhao Xiong, Sandhya Ramachandran, and Ronak Mehtra. This new version
was designed and written by Sambit Panda and first released February 2020.

**Note**: `hyppo` was previously referred to as `mgcpy`.
