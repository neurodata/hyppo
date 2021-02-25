.. -*- mode: rst -*-

hyppo
******

|CircleCI|_ |Codecov|_ |Netlify|_ |PythonVersion|_ |PyPi|_ |arXivshield|_

.. |CircleCI| image:: https://circleci.com/gh/neurodata/hyppo/tree/main.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/neurodata/hyppo

.. |Codecov| image:: https://codecov.io/gh/neurodata/hyppo/branch/main/graph/badge.svg?token=a2TXyRVW0a
.. _Codecov: https://codecov.io/gh/neurodata/hyppo

.. |Netlify| image:: https://img.shields.io/netlify/e5242ebd-631e-4330-b43e-85e428dac66a
.. _`Netlify`: https://app.netlify.com/sites/hyppo/deploys

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue
.. _PythonVersion: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue

.. |PyPi| image:: https://badge.fury.io/py/hyppo.svg
.. _PyPi: https://pypi.org/project/hyppo/

.. |arXivshield| image:: https://img.shields.io/badge/arXiv-1907.02088-red.svg?style=flat
.. _arXivshield: https://arxiv.org/abs/1907.02088

.. placeholder-for-doc-index

hyppo (\ **HYP**\ othesis Testing in **P**\ yth\ **O**\ n, pronounced "Hippo") is an open-source software package for multivariate hypothesis testing. We decided to develop hyppo for the following reasons:

* With the increase in the amount of data in many fields, hypothesis testing for high dimensional and nonlinear data is important
* Libraries in R exist, but their interfaces are inconsistent and most are not available in Python

hyppo intends to be a comprehensive multivariate hypothesis testing package and runs on all major versions of operating systems. It also includes novel tests not found in other packages. It is quick to install and free of charge. If you need to use multivariate hypothesis testing, be sure to give hyppo a try!

Website: https://hyppo.neurodata.io/

Installation
------------

Dependencies
=============

hyppo requires the following:

- `python <https://www.python.org/>`_ (>= 3.6)
- `numba <https://numba.pydata.org/>`_ (>= 0.46)
- `numpy <https://numpy.org/>`_  (>= 1.17)
- `scipy <https://docs.scipy.org/doc/scipy/reference/>`_ (>= 1.4.0)
- `scikit-learn <https://scikit-learn.org/stable/>`_ (>= 0.22)
- `joblib <https://joblib.readthedocs.io/en/latest/>`_ (>= 0.17.0)

User installation
==================

The easiest way to install hyppo is using ``pip``::

    pip install hyppo

To upgrade to a newer release use the ``--upgrade`` flag::

    pip install --upgrade hyppo

The documentation includes more detailed `installation instructions <https://hyppo.neurodata.io/install.html>`_.
hyppo is free software; you can redistribute it and/or modify it under the
terms of the `license <https://hyppo.neurodata.io/license.html>`_.

Changelog
----------

See the `changelog <https://hyppo.neurodata.io/news.html>`_
for a history of notable changes to hyppo.

Development
------------

We welcome new contributors of all experience levels. The hyppo
community goals are to be helpful, welcoming, and effective. The
`contributor guide <https://hyppo.neurodata.io/contributing.html>`_
has detailed information about contributing code, documentation and tests.

- Official source code: https://github.com/neurodata/hyppo/tree/master/hyppo
- Download releases: https://pypi.org/project/hyppo/
- Issue tracker: https://github.com/neurodata/hyppo/issues

**Note: We have recently moved our** ``benchmarks`` **(with relevant figure replication code for our papers) folder to a new** `repo <https://github.com/neurodata/hyppo-papers>`_\ **!** We aim to add test development code and paper figure replication code there, and will add relevant tests (with tutorials) to hyppo.

Project History
----------------

hyppo is a rebranding of mgcpy, which was founded in Novemember 2018.
mgcpy was designed and written by Satish Palaniappan, Sambit
Panda, Junhao Xiong, Sandhya Ramachandran, and Ronak Mehtra. hyppo
was designed and written by Sambit Panda and first released February 2020.
