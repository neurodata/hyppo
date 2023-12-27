import numpy as np


class _CheckInputs:
    """Check if additional arguments are correct"""

    def __init__(self, n):
        self.n = n

    def __call__(self, **kwargs):
        if type(self.n) is not int:
            raise ValueError("Expected n of type int, got {}".format(type(self.n)))

        if self.n < 5:
            raise ValueError("n must be >= 5, got {}".format(self.n))

        for key, value in kwargs.items():
            if value[1] is float and type(value[0]) is int:
                continue
            if type(value[0]) is not value[1]:
                raise ValueError(
                    "Expected {} type {} got {}".format(key, value[1], type(value[0]))
                )


def indep_ar(n, lag=1, phi=0.5, sigma=1):
    r"""
    2 independent, stationary, autoregressive time series simulation.

    :math:`X_t` and :math:`Y_t` are univarite AR(``1ag``) with
    :math:`\phi = 0.5` for both series. Noise follows
    :math:`\mathcal{N}(0, \sigma)`. With lag (1), this is

    .. math::

        \begin{bmatrix} X_t \\ Y_t \end{bmatrix} =
        \begin{bmatrix} \phi & 0 \\ 0 & \phi \end{bmatrix}
        \begin{bmatrix} X_{t - 1} \\ Y_{t - 1} \end{bmatrix} +
        \begin{bmatrix} \epsilon_t \\ \eta_t \end{bmatrix}

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 3).
    lag : float, default: 1
        The maximum time lag considered between ``x`` and ``y``.
    phi : float, default: 0.5
        The AR coefficient.
    sigma : float, default: 1
        The variance of the noise.

    Returns
    -------
    x,y : ndarray of float
        Simulated data matrices. ``x`` and ``y`` have shape ``(n,)``
        where `n` is the number of samples.
    """
    extra_args = {
        "lag": (lag, float),
        "phi": (phi, float),
        "sigma": (sigma, float),
    }
    check_in = _CheckInputs(n)
    check_in(**extra_args)

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
    2 linearly dependent time series simulation.

    :math:`X_t` and :math:`Y_t` are together a bivariate univarite AR(``1ag``) with
    :math:`\phi = \begin{bmatrix} 0 & 0.5 \\ 0.5 & 0 \end{bmatrix}` for both series.
    Noise follows :math:`\mathcal{N}(0, \sigma)`. With lag (1), this is

    .. math::

        \begin{bmatrix} X_t \\ Y_t \end{bmatrix} =
        \begin{bmatrix} 0 & \phi \\ \phi & 0 \end{bmatrix}
        \begin{bmatrix} X_{t - 1} \\ Y_{t - 1} \end{bmatrix} +
        \begin{bmatrix} \epsilon_t \\ \eta_t \end{bmatrix}

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 3).
    lag : float, default: 1
        The maximum time lag considered between ``x`` and ``y``.
    phi : float, default: 0.5
        The AR coefficient.
    sigma : float, default: 1
        The variance of the noise.

    Returns
    -------
    x,y : ndarray of float
        Simulated data matrices. ``x`` and ``y`` have shape ``(n,)``
        where `n` is the number of samples.
    """
    extra_args = {
        "lag": (lag, float),
        "phi": (phi, float),
        "sigma": (sigma, float),
    }
    check_in = _CheckInputs(n)
    check_in(**extra_args)

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
    2 nonlinearly dependent time series simulation.

    :math:`X_t` and :math:`Y_t` are together a bivariate nonlinear process.
    Noise follows :math:`\mathcal{N}(0, \sigma)`. With lag (1), this is

    .. math::

        \begin{bmatrix} X_t \\ Y_t \end{bmatrix} =
        \begin{bmatrix} \phi \epsilon_t Y_{t - 1} \\ \eta_t \end{bmatrix}

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 3).
    lag : float, default: 1
        The maximum time lag considered between `x` and `y`.
    phi : float, default: 1
        The AR coefficient.
    sigma : float, default: 1
        The variance of the noise.

    Returns
    -------
    x,y : ndarray of float
        Simulated data matrices. ``x`` and ``y`` have shape ``(n,)``
        where `n` is the number of samples.
    """
    extra_args = {
        "lag": (lag, float),
        "phi": (phi, float),
        "sigma": (sigma, float),
    }
    check_in = _CheckInputs(n)
    check_in(**extra_args)

    epsilons = np.random.normal(0, sigma, n)
    etas = np.random.normal(0, sigma, n)

    x = np.zeros(n)
    y = etas

    for t in range(lag, n):
        x[t] = phi * epsilons[t] * y[t - lag]

    return x, y


def extinct_gaussian_process(n, phi=0.2, extinction_rate=0.5, radius=1):
    r"""
    2 nonlinearly dependent time series simulation.

    :math:`X_t` and :math:`Y_t` are together a bivariate nonlinear process.
    Noise :math:`\epsilon_t, \eta_t` are :math:`\mathcal{N}(0, \sigma)`, but chosen
    such that given radius, :math:`r`, uniform variable :math:`d \sim \mathcal{U}(0, 1)`,
    and extinction rate, :math:`p`:

    .. math::
        \begin{align*}
        \epsilon_t^2 + \eta_t^2 &> r^2\\
        d &> p
        \end{align*}

    With :math:`\phi = 0.2`,  :math:`:math:`X_t` and :math:`Y_t` follows:

    .. math::
        \begin{bmatrix} X_t \\ Y_t \end{bmatrix} =
        \begin{bmatrix} \phi & 0 \\ 0 & \phi \end{bmatrix}
        \begin{bmatrix} X_{t - 1} \\ Y_{t - 1} \end{bmatrix} +
        \begin{bmatrix} \epsilon_t \\ \eta_t \end{bmatrix}

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 3).
    phi : float, default: 0.2
        The variance of the noise.
    extinction_rate : float, default: 0.5
        The rate at which the Gaussian process is extinct.
    radius : float, default: 1
        The radius of the Gaussian process.
    """
    extra_args = {
        "extinction_rate": (extinction_rate, float),
        "radius": (radius, float),
        "phi": (phi, float),
    }
    check_in = _CheckInputs(n)
    check_in(**extra_args)

    if (extinction_rate < 0) or (extinction_rate > 1):
        msg = "extinction_rate must be between 0 and 1, inclusive."
        raise ValueError(msg)

    x = np.zeros(n)
    y = np.zeros(n)

    for t in range(0, n):
        while True:
            eta = np.random.normal(0, 1)
            epsilon = np.random.normal(0, 1)
            d = np.random.uniform()

            if (eta**2 + epsilon**2 > radius**2) or (d > extinction_rate):
                break

        if t == 0:
            x[t] = eta
            y[t] = epsilon
        else:
            x[t] = phi * x[t - 1] + epsilon
            y[t] = phi * y[t - 1] + eta

    return x, y


TS_SIMS = {
    "indep_ar": indep_ar,
    "cross_corr_ar": cross_corr_ar,
    "nonlinear_process": nonlinear_process,
    "extinct_gaussian_process": extinct_gaussian_process,
}


def ts_sim(sim, n, **kwargs):
    r"""
    Time-series simulation generator.

    Takes a simulation and the required parameters, and outputs the simulated
    data matrices.

    Parameters
    ----------
    sim : str
        The name of the simulation (from the :mod:`hyppo.tools` module) that is to be
        rotated.
    n : int
        The number of samples desired by the simulation (>= 3).
    **kwargs
        Additional keyword arguements for the desired simulation.

    Returns
    -------
    x,y : ndarray of float
        Simulated data matrices.
    """
    if sim not in TS_SIMS.keys():
        raise ValueError(
            "sim_name must be one of the following: {}".format(list(TS_SIMS.keys()))
        )
    else:
        sim = TS_SIMS[sim]

    x, y = sim(n, **kwargs)

    if x.ndim == 1:
        x = x[:, np.newaxis]
    if y.ndim == 1:
        y = y[:, np.newaxis]

    return x, y
