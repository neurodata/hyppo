import numpy as np


class _CheckInputs:
    """ Check if additional arguments are correct """

    def __init__(self, n):
        self.n = n

    def __call__(self, *args):
        if type(self.n) is not int:
            raise ValueError("n must be int")

        if self.n < 5:
            raise ValueError("n must be greater than or equal to 5")

        for arg in args:
            if arg[1] is float and type(arg[0]) is int:
                continue
            if type(arg[0]) is not arg[1]:
                raise ValueError("Incorrect input variable type")


def indep_ar(n, lag=1, phi=0.5, sigma=1):
    r"""
    Simulates two independent, stationary, autoregressive time series.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    lag : float, optional (default: 1)
        The maximum time lag considered between `x` and `y`.
    phi : float, optional  (default: 0.5)
        The AR coefficient.
    sigma : float, optional  (default: 1)
        The variance of the noise.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n,)` and `(n,)`
        where `n` is the number of samples.

    Notes
    -----
    :math:`X_t` and :math:`Y_t` are univarite AR(``1ag``) with
    :math:`\phi = 0.5` for both series. Noise follows
    :math:`\mathcal{N}(0, \sigma)`. With lag (1), this is

    .. math::

        \begin{bmatrix} X_t \\ Y_t \end{bmatrix} =
        \begin{bmatrix} \phi & 0 \\ 0 & \phi \end{bmatrix}
        \begin{bmatrix} X_{t - 1} \\ Y_{t - 1} \end{bmatrix} +
        \begin{bmatrix} \epsilon_t \\ \eta_t \end{bmatrix}

    Examples
    --------
    >>> from hyppo.tools import indep_ar
    >>> x, y = indep_ar(100)
    >>> print(x.shape, y.shape)
    (100,) (100,)
    """
    extra_args = [
        (lag, float),
        (phi, float),
        (sigma, float),
    ]
    check_in = _CheckInputs(n)
    check_in(*extra_args)

    epsilons = np.random.normal(0, sigma, n)
    etas = np.random.normal(0, sigma, n)

    x = epsilons
    y = etas

    # AR process
    for t in range(lag, n):
        x[t] = phi * x[t - lag] + epsilons[t]
        y[t] = phi * y[t - lag] + etas[t]

    return x, y


def cross_corr_ar(n, lag=1, phi=0.5, sigma=1):
    r"""
    Simulates two linearly dependent time series.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    lag : float, optional (default: 1)
        The maximum time lag considered between `x` and `y`.
    phi : float, optional  (default: 0.5)
        The AR coefficient.
    sigma : float, optional  (default: 1)
        The variance of the noise.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n,)` and `(n,)`
        where `n` is the number of samples.

    Notes
    -----
    :math:`X_t` and :math:`Y_t` are together a bivariate univarite AR(``1ag``) with
    :math:`\phi = \begin{bmatrix} 0 & 0.5 \\ 0.5 & 0 \end{bmatrix}` for both series.
    Noise follows :math:`\mathcal{N}(0, \sigma)`. With lag (1), this is

    .. math::

        \begin{bmatrix} X_t \\ Y_t \end{bmatrix} =
        \begin{bmatrix} 0 & \phi \\ \phi & 0 \end{bmatrix}
        \begin{bmatrix} X_{t - 1} \\ Y_{t - 1} \end{bmatrix} +
        \begin{bmatrix} \epsilon_t \\ \eta_t \end{bmatrix}

    Examples
    --------
    >>> from hyppo.tools import cross_corr_ar
    >>> x, y = cross_corr_ar(100)
    >>> print(x.shape, y.shape)
    (100,) (100,)
    """
    extra_args = [
        (lag, float),
        (phi, float),
        (sigma, float),
    ]
    check_in = _CheckInputs(n)
    check_in(*extra_args)

    epsilons = np.random.normal(0, sigma, n)
    etas = np.random.normal(0, sigma, n)

    x = epsilons
    y = etas

    for t in range(lag, n):
        x[t] = phi * y[t - lag] + epsilons[t]
        y[t] = phi * x[t - lag] + etas[t]

    return x, y


def nonlinear_process(n, lag=1, phi=1, sigma=1):
    r"""
    Simulates two nonlinearly dependent time series.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    lag : float, optional (default: 1)
        The maximum time lag considered between `x` and `y`.
    phi : float, optional  (default: 1)
        The AR coefficient.
    sigma : float, optional  (default: 1)
        The variance of the noise.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n,)` and `(n,)`
        where `n` is the number of samples.

    Notes
    -----
    :math:`X_t` and :math:`Y_t` are together a bivariate nonlinear process.
    Noise follows :math:`\mathcal{N}(0, \sigma)`. With lag (1), this is

    .. math::

        \begin{bmatrix} X_t \\ Y_t \end{bmatrix} =
        \begin{bmatrix} \phi \epsilon_t Y_{t - 1} \\ \eta_t \end{bmatrix}

    Examples
    --------
    >>> from hyppo.tools import cross_corr_ar
    >>> x, y = cross_corr_ar(100)
    >>> print(x.shape, y.shape)
    (100,) (100,)
    """
    extra_args = [
        (lag, float),
        (phi, float),
        (sigma, float),
    ]
    check_in = _CheckInputs(n)
    check_in(*extra_args)

    epsilons = np.random.normal(0, sigma, n)
    etas = np.random.normal(0, sigma, n)

    x = np.zeros(n)
    y = etas

    for t in range(lag, n):
        x[t] = phi * epsilons[t] * y[t - lag]

    return x, y
