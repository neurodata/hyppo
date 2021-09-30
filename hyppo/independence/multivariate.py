
import numpy as np


def _perm_stat(calc_stat, *data_matrices):
    """Permute every entry and calculate test statistic"""
    # permute within each row
    # construct new (dxn) matrix
    # calculate test stat (dhsic) using permuted matrix


def multi_perm_test(calc_stat, *data_matrices, reps=1000, workers=1):
    """
    Permutation test for the p-value of a nonparametric test with multiple variables.
    
    Parameters
    ----------
    calc_stat: callable
        The method used to calculate the test statistic (must use hyppo API).
    *data_matrices: Tuple[np.ndarray]
        Input data matrices.
    reps: int, default=1000
    workers: int, default=1

    Returns
    -------
    stat : float
        The computed test statistic.
    pvalue : float
        The computed p-value.
    null_dist : list of float
        The approximated null distribution of shape ``(reps,)``.
    """
    # calculate observed test statistic
    # calculate null distribution ''reps'' number of times using _perm_stat
    # calculate p-value


def compute_kern(*data_matrices, metric="gaussian", workers=1, **kwargs):
    """
    Kernel similarity matrices for the inputs.

    Parameters
    ----------
    *data_matrices: Tuple[np.ndarray]
    metric: str, callable, or None, default="gaussian"
    workers: int, default=1
    **kwargs
        Arbitrary keyword arguments provided to
        :func:`sklearn.metrics.pairwise.pairwise_kernels`
        or a custom kernel function.

    Returns
    -------
    sim_matrices: Tuple[np.ndaaray]
        Similarity matrices based on the metric provided by the user.
        Must be same shape as ''data_matrices''.
    """