Install
=======

Which Python?
-------------

You’ll need **Python 3.6 or greater**.

If you want to
create and work with Python virtual environments, please follow instructions
on `venv <https://docs.python.org/3/library/venv.html>`_ and `virtual
environments <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_. We
also highly recommend conda. For instructions to install this, please look
at
`conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_.

Install from PyPi
-----------------

First, make sure you have the latest version of ``pip`` (the Python package
manager) installed. If you do not, refer to the `Pip documentation
<https://pip.pypa.io/en/stable/installing/>`_ and install ``pip`` first.

Install the current release of ``hyppo`` from the Terminal with ``pip``::

    $ pip install hyppo

To upgrade to a newer release use the ``--upgrade`` flag::

    $ pip install --upgrade hyppo

If you do not have permission to install software systemwide, you can install
into your user directory using the ``--user`` flag::

    $ pip install --user hyppo

Install from Github
-------------------
You can manually download ``hyppo`` by cloning the git repo master version and
running the ``setup.py`` file. That is, unzip the compressed package folder
and run the following from the top-level source directory using the Terminal::

    $ git clone https://github.com/neurodata/hyppo
    $ cd hyppo
    $ python3 setup.py install

Or, alternatively, you can use ``pip``::

    $ git clone https://github.com/neurodata/hyppo
    $ cd hyppo
    $ pip install .

Python package dependencies
---------------------------
``hyppo`` requires the following packages:

- `numba <https://numba.pydata.org/>`_
- `numpy <https://numpy.org/>`_
- `scipy <https://docs.scipy.org/doc/scipy/reference/>`_
- `scikit-learn <https://scikit-learn.org/stable/>`_

Hardware requirements
---------------------
``hyppo`` package requires only a standard computer with enough RAM to support
the in-memory operations.

OS Requirements
---------------
This package is supported for all major operating systems. The following
versions of operating systems was tested on Travis CI:

- **Linux:** Ubuntu Xenial 16.04
- **Windows:** Windows Server, version 1803

Testing
-------
``hyppo`` uses the Python ``pytest`` testing package.  If you don't already have
that package installed, follow the directions on the `pytest homepage
<https://docs.pytest.org/en/latest/>`_.
