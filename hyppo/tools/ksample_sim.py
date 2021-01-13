import numpy as np

from .indep_sim import (
    circle,
    cubic,
    diamond,
    ellipse,
    exponential,
    fourth_root,
    joint_normal,
    linear,
    logarithmic,
    multimodal_independence,
    multiplicative_noise,
    quadratic,
    sin_four_pi,
    sin_sixteen_pi,
    spiral,
    square,
    step,
    two_parabolas,
    uncorrelated_bernoulli,
    w_shaped,
)

_SIMS = [
    linear,
    spiral,
    exponential,
    cubic,
    joint_normal,
    step,
    quadratic,
    w_shaped,
    uncorrelated_bernoulli,
    logarithmic,
    fourth_root,
    sin_four_pi,
    sin_sixteen_pi,
    two_parabolas,
    circle,
    ellipse,
    diamond,
    multiplicative_noise,
    square,
    multimodal_independence,
]


def _normalize(x, y):
    """Normalize input data matricies."""
    x[:, 0] = x[:, 0] / np.max(np.abs(x[:, 0]))
    y[:, 0] = y[:, 0] / np.max(np.abs(y[:, 0]))
    return x, y


def _2samp_rotate(sim, x, y, p, degree=90, pow_type="samp"):
    angle = np.radians(degree)
    data = np.hstack([x, y])
    same_shape = [
        "joint_normal",
        "logarithmic",
        "sin_four_pi",
        "sin_sixteen_pi",
        "two_parabolas",
        "square",
        "diamond",
        "circle",
        "ellipse",
        "multiplicative_noise",
        "multimodal_independence",
    ]
    if sim.__name__ in same_shape:
        rot_shape = 2 * p
    else:
        rot_shape = p + 1
    rot_mat = np.identity(rot_shape)
    if pow_type == "dim":
        if sim.__name__ not in [
            "exponential",
            "cubic",
            "spiral",
            "uncorrelated_bernoulli",
            "fourth_root",
            "circle",
        ]:
            for i in range(rot_shape):
                mat = np.random.normal(size=(rot_shape, 1))
                mat = mat / np.sqrt(np.sum(mat ** 2))
                if i == 0:
                    rot = mat
                else:
                    rot = np.hstack([rot, mat])
                rot_mat, _ = np.linalg.qr(rot)
                if (p % 2) == 1:
                    rot_mat[0] *= -1
        else:
            rot_mat[np.ix_((0, -1), (0, -1))] = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
    elif pow_type == "samp":
        rot_mat[np.ix_((0, 1), (0, 1))] = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
    else:
        raise ValueError("pow_type not a valid flag ('dim', 'samp')")
    rot_data = (rot_mat @ data.T).T

    if sim.__name__ in same_shape:
        x_rot, y_rot = np.hsplit(rot_data, 2)
    else:
        x_rot, y_rot = np.hsplit(rot_data, [-p])

    return x_rot, y_rot


def rot_2samp(sim, n, p, noise=True, degree=90):
    r"""
    Rotates input simulations to produce a 2-sample simulation.

    Parameters
    ----------
    sim : callable()
        The simulation (from the ``hyppo.tools`` module) that is to be rotated.
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : bool, (default: True)
        Whether or not to include noise in the simulation.
    degree : float, (default: 90)
        The number of degrees to rotate the input simulation by (in first dimension).

    Returns
    -------
    samp1, samp2 : ndarray
        Rotated data matrices. `samp1` and `samp2` have shapes `(n, p+1)` and `(n, p+1)`
        or `(n, 2p)` and `(n, 2p)` depending on the independence simulation. Here, `n`
        is the number of samples and `p` is the number of dimensions.

    Examples
    --------
    >>> from hyppo.tools import rot_2samp, linear
    >>> x, y = rot_2samp(linear, 100, 1)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 2)
    """
    if sim not in _SIMS:
        raise ValueError("Not valid simulation")

    if sim.__name__ == "multimodal_independence":
        x, y = sim(n, p)
        x_rot, y_rot = sim(n, p)
    else:
        if sim.__name__ == "multiplicative_noise":
            x, y = sim(n, p)
        else:
            x, y = sim(n, p, noise=noise)
        x_rot, y_rot = _2samp_rotate(sim, x, y, p, degree=degree, pow_type="samp")
    samp1 = np.hstack([x, y])
    samp2 = np.hstack([x_rot, y_rot])

    return samp1, samp2


def trans_2samp(sim, n, p, noise=True, degree=90, trans=0.3):
    r"""
    Translates and rotates input simulations to produce a 2-sample
    simulation.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : bool, (default: False)
        Whether or not to include noise in the simulation.
    degree : float, (default: 90)
        The number of degrees to rotate the input simulation by (in first dimension).
    trans : float, (default: 0.3)
        The amount to translate the second simulation by (in first dimension).

    Returns
    -------
    samp1, samp2 : ndarray
        Translated/rotated data matrices. `samp1` and `samp2` have shapes `(n, p+1)` and
        `(n, p+1)` or `(n, 2p)` and `(n, 2p)` depending on the independence simulation.
        Here, `n` is the number of samples and `p` is the number of dimensions.

    Examples
    --------
    >>> from hyppo.tools import trans_2samp, linear
    >>> x, y = trans_2samp(linear, 100, 1)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 2)
    """
    if sim not in _SIMS:
        raise ValueError("Not valid simulation")

    if sim.__name__ == "multimodal_independence":
        x, y = sim(n, p)
        x_trans, y_trans = sim(n, p)
    else:
        if sim.__name__ == "multiplicative_noise":
            x, y = sim(n, p)
        else:
            x, y = sim(n, p, noise=noise)
        x, y = _normalize(x, y)
        x_trans, y_trans = _2samp_rotate(sim, x, y, p, degree=degree, pow_type="dim")
        x_trans[:, 0] += trans
        y_trans[:, 0] = y_trans[:, -1]

    samp1 = np.hstack([x, y])
    samp2 = np.hstack([x_trans, y_trans])

    return samp1, samp2


def gaussian_3samp(n, epsilon=1, weight=0, case=1):
    r"""
    Generates 3 sample of gaussians corresponding to 5 cases.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    epsilon : float, (default: 1)
        The amount to translate simulation by (amount  depends on case).
    weight : float, (default: False)
        Number between 0 and 1 corresponding to weight of the second Gaussian
        (used in case 4 and 5 to produce a mixture of Gaussians)
    case : {1, 2, 3, 4, 5}, (default: 1)
        The case in which to evaluate statistical power for each test.

    Returns
    -------
    sims : list of ndarray
        List of 3 2-dimensional multivariate Gaussian each
        corresponding to the desired case.

    Examples
    --------
    >>> from hyppo.tools import gaussian_3samp
    >>> sims = gaussian_3samp(100)
    >>> print(sims[0].shape, sims[1].shape, sims[2].shape)
    (100, 2) (100, 2) (100, 2)
    """
    old_case = case
    if old_case == 4:
        case = 2
    elif old_case == 5:
        case = 3
    sigma = np.identity(2)
    mu1 = [0] * 3
    mu2 = [0] * 3

    if case == 1:
        pass
    elif case == 2:
        mu2 = [0, 0, epsilon]
    elif case == 3:
        mu1 = [0, -epsilon / 2, epsilon / 2]
        mu2 = [
            (np.sqrt(3) / 3) * epsilon,
            -(np.sqrt(3) / 6) * epsilon,
            -(np.sqrt(3) / 6) * epsilon,
        ]
    else:
        raise ValueError("Not valid case, must be 1, 2, or 3")

    means = list(zip(mu1, mu2))
    sims = [np.random.multivariate_normal(mean, sigma, n) for mean in means]
    if old_case == 4:
        sims[-1] = (1 - weight) * sims[-1] + weight * np.random.multivariate_normal(
            means[-1], sigma * 1.5, n
        )
    elif old_case == 5:
        sims = [
            (1 - weight) * sims[i]
            + weight * np.random.multivariate_normal(means[i], sigma * 1.5, n)
            for i in range(len(sims))
        ]

    return sims
