from __future__ import print_function, division, unicode_literals, absolute_import

from builtins import int

import autograd.numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from numpy.random import default_rng


def constrain(val, min_val, max_val):
    return min(max_val, max(min_val, val))


def meddistance(X, subsample=None, mean_on_fail=True):
    r"""
    Compute the median of pairwise distances (not distance squared) of points
    in the matrix.  Useful as a heuristic for setting Gaussian kernel's width.

    Parameters
    ----------
    X : n x d numpy array
    mean_on_fail: True/False. If True, use the mean when the median distance is 0.
        This can happen especially, when the data are discrete e.g., 0/1, and
        there are more slightly more 0 than 1. In this case, the m

    Returns
    -------
    median distance

    From: https://github.com/wittawatj/fsic-test
    """
    if subsample is None:
        D = euclidean_distances(X, X)
        Itri = np.tril_indices(D.shape[0], -1)
        Tri = D[Itri]
        med = np.median(Tri)
        if med <= 0:
            # use the mean
            return np.mean(Tri)
        return med

    else:
        assert subsample > 0
        rand_state = np.random.get_state()
        rng = default_rng(9827)
        n = X.shape[0]
        ind = rng.choice(n, min(subsample, n), replace=False)
        np.random.set_state(rand_state)
        # recursion just one
        return meddistance(X[ind, :], None, mean_on_fail)


def fit_gaussian_draw(X, J, seed=28, reg=1e-7, eig_pow=1.0):
    r"""
    Fit a multivariate normal to the data X (n x d) and draw J points
    from the fit.

    Parameters
    ----------
    reg : regularizer to use with the covariance matrix
    eig_pow : raise eigenvalues of the covariance matrix to this power to construct
        a new covariance matrix before drawing samples. Useful to shrink the spread
        of the variance.

    From: https://github.com/wittawatj/fsic-test
    """
    rng = default_rng(seed)
    d = X.shape[1]
    mean_x = np.mean(X, 0)
    cov_x = np.cov(X.T)
    if d == 1:
        cov_x = np.array([[cov_x]])
    [evals, evecs] = np.linalg.eig(cov_x)
    evals = np.maximum(0, np.real(evals))
    assert np.all(np.isfinite(evals))
    evecs = np.real(evecs)
    shrunk_cov = evecs.dot(np.diag(evals**eig_pow)).dot(evecs.T) + reg * np.eye(d)
    V = rng.multivariate_normal(mean_x, shrunk_cov, J)
    return V


def outer_rows(X, Y):
    r"""
    Compute the outer product of each row in X, and Y.

    Parameters
    ----------
    X : n x dx numpy array
    Y : n x dy numpy array

    Returns
    -------
    Return an n x dx x dy numpy array.
    """
    return np.einsum("ij,ik->ijk", X, Y)
