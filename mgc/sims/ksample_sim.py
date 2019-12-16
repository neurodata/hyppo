import numpy as np

from .indep_sim import *


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


def _2samp_rotate(sim, x, y, p, degree=90):
    angle = np.radians(degree)
    if sim.__name__ in [
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
    ]:
        rot_mat = np.identity(2 * p)
    else:
        rot_mat = np.identity(p + 1)
    rot_mat[np.ix_((0, 1), (0, 1))] = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    data = np.hstack([x, y])
    rot_data = (rot_mat @ data.T).T
    x_rot, y_rot = np.hsplit(rot_data, 2)

    return x_rot, y_rot


def rot_2samp(sim, n, p, noise=True, degree=90):
    """Rotated 2 sample test"""
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
        x_rot, y_rot = _2samp_rotate(sim, x, y, p, degree=degree)
    samp1 = np.hstack([x, y])
    samp2 = np.hstack([x_rot, y_rot])

    return samp1, samp2


def trans_2samp(sim, n, p, noise=False, trans=0.3):
    """Translated 2 sample test"""
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
        degree = np.random.randint(90)
        x_trans += trans
        x_trans, y_trans = _2samp_rotate(sim, x_trans, y, p, degree=degree)
    samp1 = np.hstack([x, y])
    samp2 = np.hstack([x_trans, y_trans])

    return samp1, samp2


def gaussian_3samp(n, epsilon, case=1):
    cov = np.identity(2)
    means = [0] * 3
    epsilons = [epsilon] * 3

    if case == 1:
        pass
    elif case == 2:
        epsilons = [0, 0, epsilon]
    elif case == 3:
        means = [0, -epsilon / 2, epsilon / 2]
        epsilons = [
            (np.sqrt(3) / 3) * epsilon,
            -(np.sqrt(3) / 6) * epsilon,
            -(np.sqrt(3) / 6) * epsilon,
        ]
    else:
        raise ValueError("Not valid case, must be 1, 2, or 3")

    total_means = list(zip(means, epsilons))
    sims = [np.random.multivariate_normal(mean, cov, n) for mean in total_means]

    return sims
