from ..independence import CCA, HHG, KMERF, MGC, RV, Dcorr, Hsic, MaxMargin
from ._utils import _CheckInputs, k_sample_transform
from .base import KSampleTest


class KSample(KSampleTest):
    r"""
    Nonparametric `K`-Sample Testing test statistic and p-value.

    A *k*-sample test tests equality in distribution among groups. Groups
    can be of different sizes, but generally have the same dimensionality.
    There are not many non-parametric *k*-sample tests, but this version
    cleverly leverages the power of some of the implemented independence
    tests to test this equality of distribution.
    The formulation for this implementation is as follows `[1]`_:

    The *k*-sample testing problem can be thought of as a generalization of
    the two sample testing problem. Define
    :math:`\{ u_i \stackrel{iid}{\sim} F_U,\ i = 1, ..., n \}` and
    :math:`\{ v_j \stackrel{iid}{\sim} F_V,\ j = 1, ..., m \}` as two groups
    of samples deriving from different distributions with the same
    dimensionality. Then, problem that we are testing is thus,

    .. math::

        H_0: F_U &= F_V \\
        H_A: F_U &\neq F_V

    The closely related independence testing problem can be generalized
    similarly: Given a set of paired data
    :math:`\{\left(x_i, y_i \right) \stackrel{iid}{\sim} F_{XY}, \ i = 1, ..., N\}`,
    the problem that we are testing is,

    .. math::

        H_0: F_{XY} &= F_X F_Y \\
        H_A: F_{XY} &\neq F_X F_Y

    By manipulating the inputs of the *k*-sample test, we can create
    concatenated versions of the inputs and another label matrix which are
    necessarily paired. Then, any nonparametric test can be performed on
    this data.

    Letting :math:`n = \sum_{i=1}^k n_i`, define new data matrices
    :math:`\mathbf{x}` and :math:`\mathbf{y}` such that,

    .. math::

        \begin{align*}
        \mathbf{x} &=
        \begin{bmatrix}
            \mathbf{u}_1 \\
            \vdots \\
            \mathbf{u}_k
        \end{bmatrix} \in \mathbb{R}^{n \times p} \\
        \mathbf{y} &=
        \begin{bmatrix}
            \mathbf{1}_{n_1 \times 1} & \mathbf{0}_{n_1 \times 1}
            & \ldots & \mathbf{0}_{n_1 \times 1} \\
            \mathbf{0}_{n_2 \times 1} & \mathbf{1}_{n_2 \times 1}
            & \ldots & \mathbf{0}_{n_2 \times 1} \\
            \vdots & \vdots & \ddots & \vdots \\
            \mathbf{0}_{n_k \times 1} & \mathbf{0}_{n_k \times 1}
            & \ldots & \mathbf{1}_{n_k \times 1} \\
        \end{bmatrix} \in \mathbb{R}^{n \times k}
        \end{align*}

    Additionally, in the two-sample case,

    .. math::

        \begin{align*}
        \mathbf{x} &=
        \begin{bmatrix}
            \mathbf{u}_1 \\
            \mathbf{u}_2
        \end{bmatrix} \in \mathbb{R}^{n \times p} \\
        \mathbf{y} &=
        \begin{bmatrix}
            \mathbf{0}_{n_1 \times 1} \\
            \mathbf{1}_{n_2 \times 1}
        \end{bmatrix} \in \mathbb{R}^n
        \end{align*}

    Given :math:`\mathbf{u}` and :math:`\mathbf{v}`$` as defined above,
    to perform a :math:`w`-way test where :math:`w < k`,

    .. math::

        \mathbf{y} =
        \begin{bmatrix}
            \mathbf{1}_{n_1 \times 1} & \mathbf{0}_{n_1 \times 1}
            & \ldots & \mathbf{1}_{n_1 \times 1} \\
            \mathbf{1}_{n_2 \times 1} & \mathbf{1}_{n_2 \times 1}
            & \ldots & \mathbf{0}_{n_2 \times 1} \\
            \vdots & \vdots & \ddots & \vdots \\
            \mathbf{0}_{n_k \times 1} & \mathbf{1}_{n_k \times 1}
            & \ldots & \mathbf{1}_{n_k \times 1} \\
        \end{bmatrix} \in \mathbb{R}^{n \times k}.

    where each row of :math:`\mathbf{y}` contains :math:`w`
    :math:`\mathbf{1}_{n_i}` elements. This leads to label matrix distances
    proportional to how many labels (ways) samples differ by, a hierarchy of distances
    between samples thought to be true if the null hypothesis is rejected.

    Performing a multilevel test involves constructing :math:`x` and :math:`y` using
    either of the methods above and then performing a block permutation `[2]`_.
    Essentially, the permutation is striated, where permutation is limited to be within
    a block of samples or between blocks of samples, but not both. This is done because
    the data is not freely exchangeable, so it is necessary to block the permutation to
    preserve the joint distribution `[2]`_.

    The p-value returned is calculated using a permutation test uses
    :meth:`hyppo.tools.perm_test`.
    The fast version of the test uses :meth:`hyppo.tools.chi2_approx`.

    .. _[1]: https://arxiv.org/abs/1910.08883
    .. _[2]: https://www.sciencedirect.com/science/article/pii/S105381191500508X

    Parameters
    ----------
    indep_test : "CCA", "Dcorr", "HHG", "RV", "Hsic", "MGC", "KMERF"
        A string corresponding to the desired independence test from
        :mod:`hyppo.independence`. This is not case sensitive.
    compute_distkern : str, callable, or None, default: "euclidean" or "gaussian"
        A function that computes the distance among the samples within each
        data matrix.
        Valid strings for ``compute_distance`` are, as defined in
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
        or similarity
        matrices. To call a custom function, either create the distance or similarity
        matrix
        before-hand or create a function of the form ``metric(x, **kwargs)``
        where ``x`` is the data matrix for which pairwise distances or similarities are
        calculated and ``**kwargs`` are extra arguements to send to your custom
        function.

        Alternatively, this function computes the kernel similarity among the
        samples within each data matrix.
        Valid strings for ``compute_kernel`` are, as defined in
        :func:`sklearn.metrics.pairwise.pairwise_kernels`,

            [``"additive_chi2"``, ``"chi2"``, ``"linear"``, ``"poly"``,
            ``"polynomial"``, ``"rbf"``,
            ``"laplacian"``, ``"sigmoid"``, ``"cosine"``]

        Note ``"rbf"`` and ``"gaussian"`` are the same metric.
    bias : bool, default: False
        Whether or not to use the biased or unbiased test statistics (for
        ``indep_test="Dcorr"`` and ``indep_test="Hsic"``).
    **kwargs
        Arbitrary keyword arguments for ``compute_distkern``.
    """

    def __init__(self, indep_test, compute_distkern="euclidean", bias=False, **kwargs):
        indep_test = indep_test.lower()
        test_names = {
            "rv": RV,
            "cca": CCA,
            "hhg": HHG,
            "hsic": Hsic,
            "dcorr": Dcorr,
            "mgc": MGC,
            "kmerf": KMERF,
            "maxmargin": MaxMargin,
        }
        if indep_test not in test_names.keys():
            raise ValueError("Test is not a valid independence test")
        if indep_test == "hsic" and compute_distkern == "euclidean":
            compute_distkern = "gaussian"
        self.indep_test_name = indep_test
        indep_test = test_names[indep_test]

        if self.indep_test_name in ["dcorr", "hhg", "hsic", "mgc"]:
            if self.indep_test_name == "hsic":
                self.indep_test = indep_test(
                    compute_kernel=compute_distkern, bias=bias, **kwargs
                )
            elif self.indep_test_name == "dcorr":
                self.indep_test = indep_test(
                    compute_distance=compute_distkern, bias=bias, **kwargs
                )
            else:
                self.indep_test = indep_test(
                    compute_distance=compute_distkern, **kwargs
                )
        elif self.indep_test_name == "kmerf":
            self.indep_test = indep_test(forest_type="classifier", **kwargs)
        elif self.indep_test_name == "maxmargin":
            self.indep_test = indep_test(
                indep_test=self.indep_test_name,
                compute_distkern=compute_distkern,
                bias=bias,
                **kwargs
            )
        else:
            self.indep_test = indep_test()

        # set is_distance to true if compute_distance is None
        self.is_distance = False
        if not compute_distkern:
            self.is_distance = True

        KSampleTest.__init__(
            self, compute_distance=compute_distkern, bias=bias, **kwargs
        )

    def statistic(self, *args):
        r"""
        Calulates the *k*-sample test statistic.

        Parameters
        ----------
        *args : ndarray
            Variable length input data matrices. All inputs must have the same
            number of dimensions. That is, the shapes must be `(n, p)` and
            `(m, p)`, ... where `n`, `m`, ... are the number of samples and `p` is
            the number of dimensions.

        Returns
        -------
        stat : float
            The computed *k*-Sample statistic.
        """
        inputs = list(args)
        if self.indep_test_name == "kmerf":
            u, v = k_sample_transform(inputs, test_type="rf")
        else:
            u, v = k_sample_transform(inputs)

        return self.indep_test.statistic(u, v)

    def test(self, *args, reps=1000, workers=1, auto=True):
        r"""
        Calculates the *k*-sample test statistic and p-value.

        Parameters
        ----------
        *args : ndarray
            Variable length input data matrices. All inputs must have the same
            number of dimensions. That is, the shapes must be `(n, p)` and
            `(m, p)`, ... where `n`, `m`, ... are the number of samples and `p` is
            the number of dimensions.
        reps : int, default: 1000
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, default: 1
            The number of cores to parallelize the p-value computation over.
            Supply ``-1`` to use all cores available to the Process.
        auto : bool, default: True
            Only applies to ``"Dcorr"`` and ``"Hsic"``.
            Automatically uses fast approximation when `n` and size of array
            is greater than 20. If ``True``, and sample size is greater than 20, then
            :class:`hyppo.tools.chi2_approx` will be run. Parameters ``reps`` and
            ``workers`` are
            irrelevant in this case. Otherwise, :class:`hyppo.tools.perm_test` will be
            run.

        Returns
        -------
        stat : float
            The computed *k*-Sample statistic.
        pvalue : float
            The computed *k*-Sample p-value.
        dict
            A dictionary containing optional parameters for tests that return them.
            See the relevant test in :mod:`hyppo.independence`.

        Examples
        --------
        >>> import numpy as np
        >>> from hyppo.ksample import KSample
        >>> x = np.arange(7)
        >>> y = x
        >>> z = np.arange(10)
        >>> stat, pvalue = KSample("Dcorr").test(x, y)
        >>> '%.3f, %.1f' % (stat, pvalue)
        '0.000, 1.0'
        """
        inputs = list(args)
        check_input = _CheckInputs(
            inputs=inputs,
            indep_test=self.indep_test_name,
        )
        inputs = check_input()
        if self.indep_test_name == "kmerf":
            u, v = k_sample_transform(inputs, test_type="rf")
        else:
            u, v = k_sample_transform(inputs)

        kwargs = {}
        if self.indep_test_name in ["dcorr", "hsic"]:
            kwargs = {"auto": auto}

        return self.indep_test.test(u, v, reps, workers, **kwargs)
