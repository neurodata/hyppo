import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy.stats.distributions import chi2
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state


# Explicitly copying private function from scipy 1.7.3
# Modified to only use nan_policy 'raise'
# REF: https://github.com/scipy/scipy/blob/59e6539cf80dc04b16b0f0ab52343381f0a7a2fa/scipy/stats/stats.py#L79
def contains_nan(a):
    nan_policy = "raise"
    try:
        # Calling np.sum to avoid creating a huge array into memory
        # e.g. np.isnan(a).any()
        with np.errstate(invalid="ignore"):
            contains_nan_var = np.isnan(np.sum(a))
    except TypeError:
        # This can happen when attempting to sum things which are not
        # numbers (e.g. as in the function `mode`). Try an alternative method:
        try:
            contains_nan_var = np.nan in set(a.ravel())
        except TypeError:
            # Don't know what to do. Fall back to omitting nan values and
            # issue a warning.
            contains_nan_var = False
            nan_policy = "omit"
            warnings.warn(
                "The input array could not be properly "
                "checked for nan values. nan values "
                "will be ignored.",
                RuntimeWarning,
            )

    if contains_nan_var:
        raise ValueError("The input contains nan values")

    return contains_nan_var, nan_policy


def check_ndarray_xy(x, y):
    """Check if x or y is an ndarray of float"""
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("x and y must be ndarrays")


def convert_xy_float64(x, y):
    """Convert x or y to np.float64 (if not already done)"""
    # convert x and y to floats
    x = np.asarray(x).astype(np.float64)
    y = np.asarray(y).astype(np.float64)

    return x, y


def check_reps(reps):
    """Check if reps is valid"""
    # check if reps is an integer > than 0
    if not isinstance(reps, int) or reps < 0:
        raise ValueError("Number of reps must be an integer greater than 0.")

    # check if reps is under 1000 (recommended)
    elif reps < 1000:
        msg = (
            "The number of replications is low (under 1000), and p-value "
            "calculations may be unreliable. Use the p-value result, with "
            "caution!"
        )
        warnings.warn(msg, RuntimeWarning)


def _check_distmat(x, y):
    """Check if x and y are distance matrices."""
    if (
        not np.allclose(x, x.T)
        or not np.allclose(y, y.T)
        or not np.all((x.diagonal() == 0))
        or not np.all((y.diagonal() == 0))
    ):
        raise ValueError(
            "x and y must be distance matrices, {is_sym} symmetric and "
            "{zero_diag} zeros along the diagonal".format(
                is_sym="x is not"
                if not np.array_equal(x, x.T)
                else "y is not"
                if not np.array_equal(y, y.T)
                else "both are",
                zero_diag="x doesn't have"
                if not np.all((x.diagonal() == 0))
                else "y doesn't have"
                if not np.all((y.diagonal() == 0))
                else "both have",
            )
        )


def _check_kernmat(x, y):
    """Check if x and y are similarity matrices."""
    if (
        not np.allclose(x, x.T)
        or not np.allclose(y, y.T)
        or not np.all((x.diagonal() == 1))
        or not np.all((y.diagonal() == 1))
    ):
        raise ValueError(
            "x and y must be kernel similarity matrices, "
            "{is_sym} symmetric and {one_diag} "
            "ones along the diagonal".format(
                is_sym="x is not"
                if not np.array_equal(x, x.T)
                else "y is not"
                if not np.array_equal(y, y.T)
                else "both are",
                one_diag="x doesn't have"
                if not np.all((x.diagonal() == 1))
                else "y doesn't have"
                if not np.all((y.diagonal() == 1))
                else "both have",
            )
        )


def compute_kern(x, y, metric="gaussian", workers=1, **kwargs):
    """
    Kernel similarity matrices for the inputs.

    Parameters
    ----------
    x,y : ndarray of float
        Input data matrices. ``x`` and ``y`` must have the same number of
        samples. That is, the shapes must be ``(n, p)`` and ``(n, q)`` where
        `n` is the number of samples and `p` and `q` are the number of
        dimensions. Alternatively, ``x`` and ``y`` can be kernel similarity matrices,
        where the shapes must both be ``(n, n)``.
    metric : str, callable, or None, default: "gaussian"
        A function that computes the kernel similarity among the samples within each
        data matrix.
        Valid strings for ``metric`` are, as defined in
        :func:`sklearn.metrics.pairwise.pairwise_kernels`,

            [``"additive_chi2"``, ``"chi2"``, ``"linear"``, ``"poly"``,
            ``"polynomial"``, ``"rbf"``,
            ``"laplacian"``, ``"sigmoid"``, ``"cosine"``]

        Note ``"rbf"`` and ``"gaussian"`` are the same metric.
        Set to ``None`` or ``"precomputed"`` if ``x`` and ``y`` are already similarity
        matrices. To call a custom function, either create the similarity matrix
        before-hand or create a function of the form :func:`metric(x, **kwargs)`
        where ``x`` is the data matrix for which pairwise kernel similarity matrices are
        calculated and kwargs are extra arguements to send to your custom function.
    workers : int, default: 1
        The number of cores to parallelize the p-value computation over.
        Supply ``-1`` to use all cores available to the Process.
    **kwargs
        Arbitrary keyword arguments provided to
        :func:`sklearn.metrics.pairwise.pairwise_kernels`
        or a custom kernel function.

    Returns
    -------
    simx, simy : ndarray of float
        Similarity matrices based on the metric provided by the user.
    """
    if not metric:
        metric = "precomputed"
    if metric in ["gaussian", "rbf"]:
        if "gamma" not in kwargs:
            l2 = pairwise_distances(x, metric="l2", n_jobs=workers)
            n = l2.shape[0]
            # compute median of off diagonal elements
            med = np.median(
                np.lib.stride_tricks.as_strided(
                    l2, (n - 1, n + 1), (l2.itemsize * (n + 1), l2.itemsize)
                )[:, 1:]
            )
            # prevents division by zero when used on label vectors
            med = med if med else 1
            kwargs["gamma"] = 1.0 / (2 * (med**2))
        metric = "rbf"
    if callable(metric):
        simx = metric(x, **kwargs)
        simy = metric(y, **kwargs)
        _check_kernmat(
            simx, simy
        )  # verify whether matrix is correct, built into sklearn func
    else:
        simx = pairwise_kernels(x, metric=metric, n_jobs=workers, **kwargs)
        simy = pairwise_kernels(y, metric=metric, n_jobs=workers, **kwargs)
    return simx, simy


def _multi_check_kernmat(*args):
    """Check if every input is a similarity matrix."""
    for x in args:
        if not np.allclose(x, x.T) or not np.all((x.diagonal() == 1)):
            raise ValueError(
                "x must be a kernel similarity matrix, "
                "{is_sym} symmetric and {one_diag} "
                "ones along the diagonal".format(
                    is_sym="is not" if not np.array_equal(x, x.T) else "is",
                    one_diag="doesn't have"
                    if not np.all((x.diagonal() == 1))
                    else "has",
                )
            )


def multi_compute_kern(*args, metric="gaussian", workers=1, **kwargs):
    """
    Kernel similarity matrices for the input matrices.

    Parameters
    ----------
    *args: ndarray of float
        Variable length input data matrices. All inputs must have the same
        number of samples. That is, the shapes must be ``(n, p)``, ``(n, q)``,
        etc., where `n` is the number of samples and `p` and `q` are the
        number of dimensions.
    metric: str, callable, or None, default="gaussian"
        A function that computes the kernel similarity among the samples within each
        data matrix.
        Valid strings for ``metric`` are, as defined in
        :func:`sklearn.metrics.pairwise.pairwise_kernels`,

            [``"additive_chi2"``, ``"chi2"``, ``"linear"``, ``"poly"``,
            ``"polynomial"``, ``"rbf"``,
            ``"laplacian"``, ``"sigmoid"``, ``"cosine"``]

        Note ``"rbf"`` and ``"gaussian"`` are the same metric.
        Set to ``None`` or ``"precomputed"`` if ``x`` and ``y`` are already similarity
        matrices. To call a custom function, either create the similarity matrix
        before-hand or create a function of the form :func:`metric(x, **kwargs)`
        where ``x`` is the data matrix for which pairwise kernel similarity matrices are
        calculated and kwargs are extra arguements to send to your custom function.
    workers: int, default=1
        The number of cores to parallelize the p-value computation over.
        Supply ``-1`` to use all cores available to the Process.
    **kwargs
        Arbitrary keyword arguments provided to
        :func:`sklearn.metrics.pairwise.pairwise_kernels`
        or a custom kernel function.

    Returns
    -------
    sim_matrices: ndarray of float
        Similarity matrices based on the metric provided by the user.
        Must be same shape as ''args''.
    """
    if not metric:
        metric = "precomputed"
    if metric in ["gaussian", "rbf"]:
        if "gamma" not in kwargs:
            l2 = pairwise_distances(args[0], metric="l2", n_jobs=workers)
            n = l2.shape[0]
            # compute median of off diagonal elements
            med = np.median(
                np.lib.stride_tricks.as_strided(
                    l2, (n - 1, n + 1), (l2.itemsize * (n + 1), l2.itemsize)
                )[:, 1:]
            )
            # prevents division by zero when used on label vectors
            med = med if med else 1
            kwargs["gamma"] = 1.0 / (2 * (med**2))
        metric = "rbf"
    if callable(metric):
        sim_mats = []
        for mat in args:
            sim_mat = metric(mat, **kwargs)
            sim_mats.append(sim_mat)
        sim_matrices = tuple(sim_mats)
        _multi_check_kernmat(*sim_matrices)
    else:
        sim_mats = []
        for mat in args:
            sim_mat = pairwise_kernels(mat, metric=metric, n_jobs=workers, **kwargs)
            sim_mats.append(sim_mat)
        sim_matrices = tuple(sim_mats)
    return sim_matrices


def compute_dist(x, y, metric="euclidean", workers=1, **kwargs):
    """
    Distance matrices for the inputs.

    Parameters
    ----------
    x,y : ndarray of float
        Input data matrices. ``x`` and ``y`` must have the same number of
        samples. That is, the shapes must be ``(n, p)`` and ``(n, q)`` where
        `n` is the number of samples and `p` and `q` are the number of
        dimensions. Alternatively, ``x`` and ``y`` can be distance matrices,
        where the shapes must both be ``(n, n)``.
    metric : str, callable, or None, default: "euclidean"
        A function that computes the distance among the samples within each
        data matrix.
        Valid strings for ``metric`` are, as defined in
        :func:`sklearn.metrics.pairwise_distances`,

            - From scikit-learn: [``"euclidean"``, ``"cityblock"``, ``"cosine"``,
              ``"l1"``, ``"l2"``, ``"manhattan"``] See the documentation for
              :mod:`scipy.spatial.distance` for details
              on these metrics.
            - From scipy.spatial.distance: [``"braycurtis"``, ``"canberra"``,
              ``"chebyshev"``, ``"correlation"``, ``"dice"``, ``"hamming"``,
              ``"jaccard"``, ``"kulsinski"``, ``"mahalanobis"``, ``"minkowski"``,
              ``"rogerstanimoto"``, ``"russellrao"``, ``"seuclidean"``,
              ``"sokalmichener"``, ``"sokalsneath"``, ``"sqeuclidean"``,
              ``"yule"``] See the documentation for :mod:`scipy.spatial.distance` for
              details on these metrics.

        Set to ``None`` or ``"precomputed"`` if ``x`` and ``y`` are already distance
        matrices. To call a custom function, either create the distance matrix
        before-hand or create a function of the form ``metric(x, **kwargs)``
        where ``x`` is the data matrix for which pairwise distances are
        calculated and ``**kwargs`` are extra arguements to send to your custom
        function.
    workers : int, default: 1
        The number of cores to parallelize the p-value computation over.
        Supply ``-1`` to use all cores available to the Process.
    **kwargs
        Arbitrary keyword arguments provided to
        :func:`sklearn.metrics.pairwise_distances` or a
        custom distance function.

    Returns
    -------
    distx, disty : ndarray of float
        Distance matrices based on the metric provided by the user.
    """
    if not metric:
        metric = "precomputed"
    if callable(metric):
        distx = metric(x, **kwargs)
        disty = metric(y, **kwargs)
        _check_distmat(
            distx, disty
        )  # verify whether matrix is correct, built into sklearn func
    else:
        distx = pairwise_distances(x, metric=metric, n_jobs=workers, **kwargs)
        disty = pairwise_distances(y, metric=metric, n_jobs=workers, **kwargs)
    return distx, disty


def check_perm_blocks(perm_blocks):
    # Checks generic properties of perm_blocks
    if perm_blocks is None:
        return None
    elif isinstance(perm_blocks, list):
        perm_blocks = np.asarray(perm_blocks)
    elif not isinstance(perm_blocks, np.ndarray):
        raise TypeError("perm_blocks must be an ndarray or list")
    if perm_blocks.ndim == 1:
        perm_blocks = perm_blocks[:, np.newaxis]
    elif perm_blocks.ndim > 2:
        raise ValueError("perm_blocks must be of at most dimension 2")

    return perm_blocks


def check_perm_blocks_dim(perm_blocks, y):
    if not perm_blocks.shape[0] == y.shape[0]:
        raise ValueError("perm_bocks first dimension must be same length as y")


def check_perm_block(perm_block):
    # checks a hierarchy level of perm_blocks for proper exchangeability
    if not isinstance(perm_block[0], int):
        unique, perm_blocks, counts = np.unique(
            perm_block, return_counts=True, return_inverse=True
        )
        pos_counts = counts
    else:
        unique, counts = np.unique(perm_block, return_counts=True)
        pos_counts = [c for c, u in zip(counts, unique) if u >= 0]
    if len(set(pos_counts)) > 1:
        raise ValueError(
            f"Exchangeable hiearchy has groups with {min(pos_counts)} to \
                {max(pos_counts)} elements"
        )

    return perm_block


class _PermNode(object):
    """Helper class for nodes in _PermTree."""

    def __init__(self, parent, label=None, index=None):
        self.children = []
        self.parent = parent
        self.label = label
        self.index = index

    def get_leaf_indices(self):
        if len(self.children) == 0:
            return [self.index]
        else:
            indices = []
            for child in self.children:
                indices += child.get_leaf_indices()
            return indices

    def add_child(self, child):
        self.children.append(child)

    def get_children(self):
        return self.children


class _PermTree(object):
    """Tree representation of dependencies for restricted permutations"""

    def __init__(self, perm_blocks):
        perm_blocks = check_perm_blocks(perm_blocks)
        self.root = _PermNode(None)
        self._add_levels(self.root, perm_blocks, np.arange(perm_blocks.shape[0]))
        indices = self.root.get_leaf_indices()
        self._index_order = np.argsort(indices)

    def _add_levels(self, root: _PermNode, perm_blocks, indices):
        # Add new child node for each unique label, then recurse or end
        if perm_blocks.shape[1] == 0:
            for idx in indices:
                child_node = _PermNode(parent=root, label=1, index=idx)
                root.add_child(child_node)
        else:
            perm_block = check_perm_block(perm_blocks[:, 0])
            for label in np.unique(perm_block):
                idxs = np.where(perm_block == label)[0]
                child_node = _PermNode(parent=root, label=label)
                root.add_child(child_node)
                self._add_levels(child_node, perm_blocks[idxs, 1:], indices[idxs])

    def _permute_level(self, node, rng=None):
        if rng is None:
            rng = np.random
        if len(node.get_children()) == 0:
            return [node.index]
        else:
            indices, labels = zip(
                *[
                    (self._permute_level(child), child.label)
                    for child in node.get_children()
                ]
            )
            shuffle_children = [i for i, label in enumerate(labels) if label >= 0]
            indices = np.asarray(indices)
            if len(shuffle_children) > 1:
                indices[shuffle_children] = indices[rng.permutation(shuffle_children)]
            return np.concatenate(indices)

    def permute_indices(self, rng=None):
        return self._permute_level(self.root, rng)[self._index_order]

    def original_indices(self):
        return np.arange(len(self._index_order))


# permutation group shuffling class
class _PermGroups(object):
    """Helper function to calculate parallel p-value."""

    def __init__(self, y, perm_blocks=None):
        self.n = y.shape[0]
        if perm_blocks is None:
            self.perm_tree = None
        else:
            self.perm_tree = _PermTree(perm_blocks)

    def __call__(self, rng=None):
        if rng is None:
            rng = np.random
        if self.perm_tree is None:
            order = rng.permutation(self.n)
        else:
            order = self.perm_tree.permute_indices(rng)

        return order


# p-value computation
def _perm_stat(calc_stat, x, y, is_distsim=True, permuter=None, random_state=None):
    """Permute the test statistic"""
    rng = check_random_state(random_state)
    if permuter is None:
        order = rng.permutation(y.shape[0])
    else:
        order = permuter(rng)

    if is_distsim:
        permy = y[order][:, order]
    else:
        permy = y[order]

    perm_stat = calc_stat(x, permy)

    return perm_stat


def perm_test(
    calc_stat,
    x,
    y,
    reps=1000,
    workers=1,
    is_distsim=True,
    perm_blocks=None,
    random_state=None,
):
    """
    Permutation test for the p-value of a nonparametric test.

    This process is completed by first randomly permuting :math:`y` to estimate the null
    distribution and then calculating the probability of observing a test
    statistic, under the null, at least as extreme as the observed test
    statistic.

    Parameters
    ----------
    calc_stat : callable
        The method used to calculate the test statistic (must use hyppo API).
    x,y : ndarray of float
        Input data matrices. ``x`` and ``y`` must have the same number of
        samples. That is, the shapes must be ``(n, p)`` and ``(n, q)`` where
        `n` is the number of samples and `p` and `q` are the number of
        dimensions. Alternatively, ``x`` and ``y`` can be distance or similarity
        matrices,
        where the shapes must both be ``(n, n)``.
    reps : int, default: 1000
        The number of replications used to estimate the null distribution
        when using the permutation test used to calculate the p-value.
    workers : int, default: 1
        The number of cores to parallelize the p-value computation over.
        Supply ``-1`` to use all cores available to the Process.
    is_distsim : bool, default: True
        Whether or not ``x`` and ``y`` are distance or similarity matrices.
    perm_blocks : ndarray, default: None
        Defines blocks of exchangeable samples during the permutation test.
        If ``None``, all samples can be permuted with one another. Requires `n`
        rows. Constructs a tree graph with all samples initially at
        the root node. Each column partitions samples from the same leaf with
        shared column label into a child of that leaf. During the permutation
        test, samples within the same final leaf node are exchangeable
        and blocks of samples with a common parent node are exchangeable. If a
        column value is negative, the resulting block is unexchangeable.
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
    stat = calc_stat(x, y)

    # make RandomState seeded array
    if random_state is not None:
        rng = check_random_state(random_state)
        random_state = rng.randint(np.iinfo(np.int32).max, size=reps)

    # make random array
    else:
        random_state = np.random.randint(np.iinfo(np.int32).max, size=reps)

    # calculate null distribution
    permuter = _PermGroups(y, perm_blocks)

    null_dist = np.array(
        Parallel(n_jobs=workers)(
            [
                delayed(_perm_stat)(calc_stat, x, y, is_distsim, permuter, rng)
                for rng in random_state
            ]
        )
    )
    pvalue = (1 + (null_dist >= stat).sum()) / (1 + reps)

    return stat, pvalue, null_dist


def _multi_perm_stat(calc_stat, *args):
    """Permute every entry and calculate test statistic"""
    # permute each row
    comb_matrix = np.concatenate(args, axis=1)
    perm_matrix = np.zeros(np.shape(comb_matrix))
    for j in range(comb_matrix.shape[1]):
        order = np.random.permutation(comb_matrix.shape[0])
        perm_matrix[:, j] = comb_matrix[order, j]
    perm_args = tuple(np.split(perm_matrix, len(args), axis=1))

    # calculate test statistic using permuted matrices
    perm_stat = calc_stat(*perm_args)

    return perm_stat


def multi_perm_test(calc_stat, *args, reps=1000, workers=1):
    """
    Permutation test for the p-value of a nonparametric test with multiple variables.

    Parameters
    ----------
    calc_stat : callable
        The method used to calculate the test statistic (must use hyppo API).
    *args : ndarray
        Variable length input data matrices. All inputs must have the same
        number of samples. That is, the shapes must be ``(n, p)``, ``(n, q)``,
        etc., where `n` is the number of samples and `p` and `q` are the
        number of dimensions.
    reps : int, default: 1000
        The number of replications used to estimate the null distribution
        when using the permutation test used to calculate the p-value.
    workers : int, default: 1
        The number of cores to parallelize the p-value computation over.
        Supply ``-1`` to use all cores available to the Process.

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
    stat = calc_stat(*args)

    # calculate null distribution
    null_dist = np.array(
        Parallel(n_jobs=workers)(
            [delayed(_multi_perm_stat)(calc_stat, *args) for _ in range(reps)]
        )
    )
    pvalue = (1 + (null_dist >= stat).sum()) / (1 + reps)

    return stat, pvalue, null_dist


def chi2_approx(calc_stat, x, y):
    """
    Fast chi-squared approximation for the p-value.

    In the case of distance and kernel methods, Dcorr (and by extension Hsic
    :footcite:p:`shenExactEquivalenceDistance2020`)
    can be approximated via a chi-squared distribution
    :footcite:p:`shenChiSquareTestDistance2021`.
    This approximation is also applicable for the nonparametric MANOVA via
    independence testing method in our package
    :footcite:p:`pandaNonparMANOVAIndependence2021`.

    Parameters
    ----------
    calc_stat : callable
        The method used to calculate the test statistic (must use hyppo API).
    x,y : ndarray of float
        Input data matrices. ``x`` and ``y`` must have the same number of
        samples. That is, the shapes must be ``(n, p)`` and ``(n, q)`` where
        `n` is the number of samples and `p` and `q` are the number of
        dimensions. Alternatively, ``x`` and ``y`` can be distance or similarity
        matrices,
        where the shapes must both be ``(n, n)``.

    Returns
    -------
    stat : float
        The computed test statistic.
    pvalue : float
        The computed p-value.

    References
    ----------
    .. footbibliography::
    """
    n = x.shape[0]
    stat = calc_stat(x, y)
    pvalue = chi2.sf(stat * n + 1, 1)

    return stat, pvalue
