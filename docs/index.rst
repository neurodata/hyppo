..  -*- coding: utf-8 -*-

.. _contents:

Overview of mgc_
===================

.. _mgc: https://mgc.neurodata.io/

``mgc`` (pronounced "Magic") is an open-source software package for
independence and k-sample testing.

Motivation
----------

With the increase in the amount of data in many fields, a method to
consistently and efficiently decipher relationships within high dimensional
data sets is important. Because many modern datasets are multivariate,
univariate independence tests are not applicable. While many multivariate
independence tests have R packages available, the interfaces are inconsistent
and most are not available in Python. ``mgc`` is an extensive Python library
that includes many state of the art multivariate independence testing
procedures using a common interface. The package is easy-to-use and is
flexible enough to enable future extensions.

Python
------

Python is a powerful programming language that allows concise expressions of
network algorithms.  Python has a vibrant and growing ecosystem of packages
that mgc uses to provide more features such as numerical linear algebra and
plotting.  In order to make the most out of ``mgc`` you will want to know how
to write basic programs in Python.  Among the many guides to Python, we
recommend the `Python documentation <https://docs.python.org/3/>`_.

Free software
-------------

``mgc`` is free software; you can redistribute it and/or modify it under the
terms of the :doc:`MIT </license>`.  We welcome contributions. Join us on
`GitHub <https://github.com/neurodata/mgc>`_.

History
-------

``mgc`` is a rebranding of ``mgcpy``, which was founded in September 2018. The
original version was designed and written by Satish Palaniappan, Sambit Panda
Junhao Xiong, Sandhya Ramachandran, and Ronak Mehtra. This new version was
written by Sambit Panda.

Documentation
=============

.. toctree::
   :maxdepth: 1

   install
   reference/index
   contributing
   license

.. toctree::
   :maxdepth: 1
   :caption: Useful Links

   mgc @ GitHub <https://github.com/neurodata/mgc>
   mgc @ PyPi <https://pypi.org/project/mgc/>
   Issue Tracker <https://github.com/neurodata/mgc/issues>


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
