import numpy as np


class _CheckInputs:
    """ Check if additional arguments are correct """
    def __init__(self, n, p):
        self.n = n
        self.p = p

    def __call__(self, *args):
        if type(self.n) is not int or type(self.p) is not int:
            raise ValueError("n and p must be ints")

        if self.n < 5 or self.p < 1:
            raise ValueError("n must be greater than or equal to 5 and p "
                             "must be greater than or equal to than 1")

        for arg in args:
            if arg[1] is float and type(arg[0]) is int:
                continue
            if type(arg[0]) is not arg[1]:
                raise ValueError("Incorrect input variable type")


def _gen_coeffs(p):
    """Calculates coefficients polynomials"""
    return np.array([1/(i+1) for i in range(p)]).reshape(-1, 1)


def _random_uniform(n, p, low=-1, high=1):
    """Generate random uniform data"""
    return np.array(np.random.uniform(low, high, size=(n, p)))


def _calc_eps(n):
    """Calculate noise"""
    return np.random.normal(0, 1, size=(n, 1))


def linear(n, p, noise=False, low=-1, high=1):
    r"""
    Simulates univariate or multivariate linear data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : float, (default: 1)
        The noise amplitude of the simulation.
    low : float, (default: -1)
        The lower limit of the uniform distribution simulated from.
    high : float, (default: -1)
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, 1)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    Linear :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}`:

    .. math::

        X &\sim \mathcal{U}(-1, 1)^p \\
        Y &= w^T X + \kappa \epsilon

    Examples
    --------
    >>> from mgc.sims import linear
    >>> x, y = linear(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 1)
    """
    extra_args = [
        (noise, bool),
        (low, float),
        (high, float)
    ]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    x = _random_uniform(n, p, low, high)
    coeffs = _gen_coeffs(p)
    eps = _calc_eps(n)
    y = x @ coeffs + 1*noise*eps

    return x, y


def exponential(n, p, noise=False, low=0, high=3):
    r"""
    Simulates univariate or multivariate exponential data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : float, (default: 10)
        The noise amplitude of the simulation.
    low : float, (default: 0)
        The lower limit of the uniform distribution simulated from.
    high : float, (default: 3)
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, 1)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    Exponential :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}`:

    .. math::

        X &\sim \mathcal{U}(0, 3)^p \\
        Y &= \exp (w^T X) + 10 \kappa \epsilon

    Examples
    --------
    >>> from mgc.sims import exponential
    >>> x, y = exponential(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 1)
    """
    extra_args = [
        (noise, bool),
        (low, float),
        (high, float)
    ]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    x = _random_uniform(n, p, low, high)
    coeffs = _gen_coeffs(p)
    eps = _calc_eps(n)
    y = np.exp(x @ coeffs) + 10*noise*eps

    return x, y


def cubic(n, p, noise=False, low=-1, high=1, cubs=[-12, 48, 128], scale=1/3):
    r"""
    Simulates univariate or multivariate cubic data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : float, (default: 80)
        The noise amplitude of the simulation.
    low : float, (default: -1)
        The lower limit of the uniform distribution simulated from.
    high : float, (default: -1)
        The upper limit of the uniform distribution simulated from.
    cubs : list of ints (default: [-12, 48, 128])
        Coefficients of the cubic function where each value corresponds to the
        order of the cubic polynomial.
    scale : float (default: 1/3)
        Scaling center of the cubic.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, 1)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    Cubic :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}`:

    .. math::

        X &\sim \mathcal{U}(-1, 1)^p \\
        Y &= 128 \left( w^T X - \frac{1}{3} \right)^3
             + 48 \left( w^T X - \frac{1}{3} \right)^2
             - 12 \left( w^T X - \frac{1}{3} \right)
             + 80 \kappa \epsilon

    Examples
    --------
    >>> from mgc.sims import cubic
    >>> x, y = cubic(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 1)
    """
    extra_args = [
        (noise, bool),
        (low, float),
        (high, float),
        (cubs, list),
        (scale, float)
    ]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    x = _random_uniform(n, p, low, high)
    coeffs = _gen_coeffs(p)
    eps = _calc_eps(n)

    x_coeffs = x @ coeffs - scale
    y = (cubs[2] * x_coeffs**3
         + cubs[1] * x_coeffs**2
         + cubs[0] * x_coeffs**3
         + 80*noise*eps)

    return x, y

def joint_normal(n, p, noise=False):
    r"""
    Simulates univariate or multivariate spiral data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : int, (default: 0.4)
        The noise amplitude of the simulation.
    low : float, (default: 0)
        The lower limit of the uniform distribution simulated from.
    high : float, (default: 5)
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, 1)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    Spiral :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}`: For
    :math:`U \sim \mathcal{U}(0, 5)`, :math:`\epsilon \sim \mathcal{N}(0, 1)`

    .. math::

        X_{|d|} &= U \sin(\pi U) \cos^d(\pi U)\ \mathrm{for}\ d = 1,...,p-1 \\
        X_{|p|} &= U \cos^p(\pi U) \\
        Y &= U \sin(\pi U) + 0.4 p \epsilon

    Examples
    --------
    >>> from mgc.sims import spiral
    >>> x, y = spiral(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 1)
    """
    if p > 10:
        raise ValueError("Covariance matrix for p>10 is not positive"
                         "semi-definite")

    extra_args = [
        (noise, bool)
    ]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    rho = 1 / (2*p)
    cov1 = np.concatenate((np.identity(p), rho*np.ones((p, p))), axis=1)
    cov2 = np.concatenate((rho*np.ones((p, p)), np.identity(p)), axis=1)
    covT = np.concatenate((cov1.T, cov2.T), axis=1)

    eps = _calc_eps(n)
    x = np.random.multivariate_normal(np.zeros(2*p), covT, n)
    y = x[:, p:2*p] + 0.5*noise*eps
    x = x[:, :p]

    return x, y


def step(n, p, noise=False, low=-1, high=1):
    r"""
    [summary]

    Parameters
    ----------
    n : [type]
        [description]
    p : [type]
        [description]
    noise : float, optional
        [description], by default 0.1
    low : int, optional
        [description], by default -1
    high : int, optional
        [description], by default 1
    """
    extra_args = [
        (noise, bool),
        (low, float),
        (high, float)
    ]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    if p > 1:
        noise = True
    x = _random_uniform(n, p, low, high)
    coeffs = _gen_coeffs(p)
    eps = _calc_eps(n)

    x_coeff = ((x @ coeffs) > 0) * 1
    y = x_coeff + noise*eps

    return x, y


def quadratic(n, p, noise=False, low=-1, high=1):

    extra_args = [
        (noise, bool),
        (low, float),
        (high, float)
    ]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    x = _random_uniform(n, p, low, high)
    coeffs = _gen_coeffs(p)
    eps = _calc_eps(n)

    x_coeffs = x @ coeffs
    y = x_coeffs ** 2 + 0.5*noise*eps

    return x, y


def w_shaped(n, p, noise=False, low=-1, high=1):

    extra_args = [
        (noise, bool),
        (low, float),
        (high, float)
    ]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    x = _random_uniform(n, p, low, high)
    u = _random_uniform(n, p, 0, 1)
    coeffs = _gen_coeffs(p)
    eps = _calc_eps(n)

    x_coeffs = x @ coeffs
    u_coeffs = u @ coeffs
    y = 4 * ((x_coeffs**2 - 0.5) ** 2 + u_coeffs/500) + 0.5*noise*eps

    return x, y


def spiral(n, p, noise=False, low=0, high=5):
    r"""
    Simulates univariate or multivariate spiral data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : int, (default: 0.4)
        The noise amplitude of the simulation.
    low : float, (default: 0)
        The lower limit of the uniform distribution simulated from.
    high : float, (default: 5)
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, 1)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    Spiral :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}`: For
    :math:`U \sim \mathcal{U}(0, 5)`, :math:`\epsilon \sim \mathcal{N}(0, 1)`

    .. math::

        X_{|d|} &= U \sin(\pi U) \cos^d(\pi U)\ \mathrm{for}\ d = 1,...,p-1 \\
        X_{|p|} &= U \cos^p(\pi U) \\
        Y &= U \sin(\pi U) + 0.4 p \epsilon

    Examples
    --------
    >>> from mgc.sims import spiral
    >>> x, y = spiral(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 1)
    """
    extra_args = [
        (noise, bool),
        (low, float),
        (high, float)
    ]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    if p > 1:
        noise = True
    rx = _random_uniform(n, p=1, low=low, high=high)
    ry = rx
    rx = np.repeat(rx, p, axis=1)
    z = rx
    x = np.zeros((n, p))
    x[:, 0] = np.cos(z[:, 0] * np.pi)
    for i in range(p - 1):
        x[:, i+1] = np.multiply(x[:, i], np.cos(z[:, i+1] * np.pi))
        x[:, i] = np.multiply(x[:, i], np.sin(z[:, i+1] * np.pi))
    x = np.multiply(rx, x)
    y = np.multiply(ry, np.sin(z[:, 0].reshape(-1, 1) * np.pi))

    eps = _calc_eps(n)
    y = y + 0.4*p*noise*eps

    return x, y


def uncorrelated_bernoulli(n, p, noise=False, prob=0.5):
    """
    [summary]

    Parameters
    ----------
    n : [type]
        [description]
    p : [type]
        [description]
    noise : float, optional
        [description], by default 0.05
    prob : float, optional
        [description], by default 0.5
    """
    extra_args = [
        (noise, bool),
        (prob, float)
    ]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    binom = np.random.binomial(1, prob, size=(n, 1))
    sig = np.identity(p)
    gauss_noise = np.random.multivariate_normal(np.zeros(p), sig, size=n)

    x = np.random.binomial(1, prob, size=(n, p)) + 0.5*noise*gauss_noise
    coeffs = _gen_coeffs(p)

    eps = _calc_eps(n)
    x_coeffs = x @ coeffs
    y = binom*2 - 1
    y = np.multiply(x_coeffs, y) + 0.5*noise*eps

    return x, y


def logarithmic(n, p, noise=False):

    extra_args = [
        (noise, bool)
    ]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    sig = np.identity(p)
    x = np.random.multivariate_normal(np.zeros(p), sig, size=n)
    eps = _calc_eps(n)

    y = np.log(x ** 2) + 3*noise*eps

    return x, y


def fourth_root(n, p, noise=False, low=-1, high=1):

    extra_args = [
        (noise, bool),
        (low, float),
        (high, float)
    ]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    x = _random_uniform(n, p, low, high)
    eps = _calc_eps(n)
    coeffs = _gen_coeffs(p)

    x_coeffs = x @ coeffs
    y = np.abs(x_coeffs) ** 0.25 + 0.25*noise*eps

    return x, y


def _sin(n, p, noise=False, low=-1, high=1, period=4*np.pi):

    extra_args = [
        (noise, bool),
        (low, float),
        (high, float),
        (period, float)
    ]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    x = _random_uniform(n, p, low, high)
    if p > 1 or noise:
        sig = np.identity(p)
        v = np.random.multivariate_normal(np.zeros(p), sig, size=n)
        x = x + 0.02*p*v
    eps = _calc_eps(n)

    if period == 4*np.pi:
        cc = 1
    else:
        cc = 0.5

    y = np.sin(x * period) + cc*noise*eps

    return x, y


def sin_four_pi(n, p, noise=False, low=-1, high=1):

    return _sin(n, p, noise=noise, low=low, high=high, period=4*np.pi)


def sin_sixteen_pi(n, p, noise=False, low=-1, high=1):

    return _sin(n, p, noise=noise, low=low, high=high, period=16*np.pi)


def _square_diamond(n, p, noise=False, low=-1, high=1, period=-np.pi/2):

    extra_args = [
        (noise, bool),
        (low, float),
        (high, float),
        (period, float)
    ]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    u = _random_uniform(n, p, low, high)
    v = _random_uniform(n, p, low, high)
    sig = np.identity(p)
    gauss_noise = np.random.multivariate_normal(np.zeros(p), sig, size=n)

    x = u*np.cos(period) + v*np.sin(period) + 0.05*p*gauss_noise
    y = -u*np.sin(period) + v*np.cos(period)

    return x, y


def square(n, p, noise=False, low=-1, high=1):

    return _square_diamond(n, p, noise=noise, low=low, high=high, period=-np.pi/8)


def two_parabolas(n, p, noise=False, low=-1, high=1, prob=0.5):

    extra_args = [
        (noise, bool),
        (low, float),
        (high, float),
        (prob, float)
    ]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    x = _random_uniform(n, p, low, high)
    coeffs = _gen_coeffs(p)
    u = np.random.binomial(1, prob, size=(n, 1))
    rand_noise = _random_uniform(n, p, low=0, high=1)

    x_coeffs = x @ coeffs
    y = (x_coeffs**2 + 2*noise*rand_noise) * (u-0.5)

    return x, y


def _circle_ellipse(n, p, noise=False, low=-1, high=1, radius=1):

    extra_args = [
        (noise, bool),
        (low, float),
        (high, float),
        (radius, float)
    ]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    if p > 1:
        noise = True
    x = _random_uniform(n, p, low, high)
    rx = radius * np.ones((n, p))
    unif = _random_uniform(n, p, low, high)
    sig = np.identity(p)
    gauss_noise = np.random.multivariate_normal(np.zeros(p), sig, size=n)

    ry = np.ones((n, p))
    x[:, 0] = np.cos(unif[:, 0] * np.pi)
    for i in range(p - 1):
        x[:, i+1] = x[:, i] * np.cos(unif[:, i+1] * np.pi)
        x[:, i] = x[:, i] * np.sin(unif[:, i+1] * np.pi)

    x = rx * x + 0.4*noise*rx*gauss_noise
    y = ry * np.sin(unif[:, 0].reshape(n, 1) * np.pi)

    return x, y


def circle(n, p, noise=False, low=-1, high=1):

    return _circle_ellipse(n, p, noise=noise, low=low, high=high, radius=1)


def ellipse(n, p, noise=False, low=-1, high=1):

    return _circle_ellipse(n, p, noise=noise, low=low, high=high, radius=5)


def diamond(n, p, noise=False, low=-1, high=1):

    return _square_diamond(n, p, noise=noise, low=low, high=high, period=-np.pi/4)


def multiplicative_noise(n, p):

    extra_args = []
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    sig = np.identity(p)
    x = np.random.multivariate_normal(np.zeros(p), sig, size=n)
    y = np.random.multivariate_normal(np.zeros(p), sig, size=n)
    y = np.multiply(x, y)

    return x, y


def multimodal_independence(n, p, prob=0.5, sep1=3, sep2=2):

    extra_args = [
        (prob, float),
        (sep1, float),
        (sep2, float)
    ]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    sig = np.identity(p)
    u = np.random.multivariate_normal(np.zeros(p), sig, size=n)
    v = np.random.multivariate_normal(np.zeros(p), sig, size=n)
    u_2 = np.random.binomial(1, prob, size=(n, p))
    v_2 = np.random.binomial(1, prob, size=(n, p))

    x = u/sep1 + sep2*u_2 - 1
    y = v/sep1 + sep2*v_2 - 1

    return x, y
