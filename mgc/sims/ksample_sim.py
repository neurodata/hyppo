import numpy as np

from .indep_sim import *


def _k_sample_rotate2d(x, y, degree=90):
    angle = np.radians(degree)
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
    data = np.hstack([x, y])
    rot_data = (rot_mat @ data.T).T
    x_rot, y_rot = np.hsplit(rot_data, 2)

    return x_rot, y_rot


def rot_2samp(sim, n, p, noise=1, low=-1, high=1):
    """Rotated 2 sample test"""
    sims = [linear, spiral, exponential, cubic]
    if sim not in sims:
        raise ValueError("Not valid simulation")

    x, y = sim(n, 1, noise=noise, low=low, high=high)
    x_rot, y_rot = _k_sample_rotate2d(x, y)
    samp1 = np.hstack([x, y])
    samp2 = np.hstack([x_rot, y_rot])

    return samp1, samp2
