Install
=======

Below we assume you have the default Python environment already configured on
your computer and you intend to install ``mgc`` inside of it.  If you want to
create and work with Python virtual environments, please follow instructions
on `venv <https://docs.python.org/3/library/venv.html>`_ and `virtual
environments <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_. We
also highly recommend conda. For instructions to install this, please look
at
`conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_.

First, make sure you have the latest version of ``pip`` (the Python package
manager) installed. If you do not, refer to the `Pip documentation
<https://pip.pypa.io/en/stable/installing/>`_ and install ``pip`` first.

Install from Github
-------------------
You can manually download ``mgc`` by cloning the git repo master version and
running the ``setup.py`` file. That is, unzip the compressed package folder
and run the following from the top-level source directory using the Terminal::

    $ git clone https://github.com/sampan501/mgc
    $ cd mgc
    $ python3 setup.py install

Python package dependencies
---------------------------
mgc requires the following packages:

- numba
- numpy
- scipy

Hardware requirements
---------------------
``mgc`` package requires only a standard computer with enough RAM to support
the in-memory operations.

OS Requirements
---------------
This package is supported for all major operating systems. The following
versions of operating systems was tested on Travis CI:

- Linux: Ubuntu Xenial 16.04
- Windows: Windows Server, version 1803

Testing
-------
``mgc`` uses the Python ``pytest`` testing package.  If you don't already have
that package installed, follow the directions on the `pytest homepage
<https://docs.pytest.org/en/latest/>`_.
