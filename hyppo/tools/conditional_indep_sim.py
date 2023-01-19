import numpy as np

from .indep_sim import _CheckInputs


def independent_normal(n):
    r"""
    Conditionally independent normal distributions

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).

    Returns
    -------
    x,y,z : ndarray of float
        Simulated data matrices. ``x``, ``y``, and ``z`` have shapes
        ``(n, 3)`` where `n` is the number of samples and `3` is the
        number of dimensions.
    """
    check_in = _CheckInputs(n, p=3)
    check_in()

    mean = np.array([0, 0, 0])
    cov = np.array([[1, 0.36, 0.6], [0.36, 1, 0.6], [0.6, 0.6, 1]])

    x = np.random.multivariate_normal(mean, cov, size=n)
    y = np.random.multivariate_normal(mean, cov, size=n)
    z = np.random.multivariate_normal(mean, cov, size=n)

    return x, y, z


def independent_binomial(n, p):
    r"""
    Conditionally independent binomial distributions

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).

    Returns
    -------
    x,y,z : ndarray of float
        Simulated data matrices. ``x`` and ``y``, and ``z`` have shapes
        ``(n, 1)`` where `n` is the number of samples and `1` is the
        number of dimensions.
    """
    check_in = _CheckInputs(n, p=p)
    check_in()

    size = (n, 1)

    z = np.random.binomial(10, 0.5, size=(n, p))
    x = np.random.binomial(10, 0.5, size=size) + z.sum(axis=1)
    y = np.random.binomial(10, 0.5, size=size) + z.sum(axis=1)

    return x, y, z


def independent_normal_nonlinear(n):
    r"""
    Conditionally independent normal distributions

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).

    Returns
    -------
    x,y,z : ndarray of float
        Simulated data matrices. ``x``, ``y``, and ``z`` have shapes
        ``(n, 3)`` where `n` is the number of samples and `3` is the
        number of dimensions.
    """
    check_in = _CheckInputs(n, p=1)
    check_in()

    size = (n, 1)
    z = np.random.normal(size=size)
    z1 = 0.5 * (np.power(z, 3) / 7 + z / 2)
    z2 = (np.power(z, 3) / 2 + z) / 3

    x1 = np.random.normal(size=size)
    x2 = z1 + np.tanh(x1)
    x = x2 + np.power(x2, 3) / 3

    y1 = np.random.normal(size=size)
    y2 = z2 + y1
    y = y2 + np.tanh(y2 / 3)

    return x, y, z
