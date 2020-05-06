from .._utils import euclidean, gaussian
from .base import KSampleTest
from ..independence import CCA, Dcorr, HHG, RV, Hsic, MGC
from ._utils import _CheckInputs, k_sample_transform


class KSample(KSampleTest):
    r"""
    Class for calculating the *k*-sample test statistic and p-value.

    A *k*-sample test tests equality in distribution among groups. Groups
    can be of different sizes, but generally have the same dimensionality.
    There are not many non-parametric *k*-sample tests, but this version
    cleverly leverages the power of some of the implemented independence
    tests to test this equality of distribution.

    Parameters
    ----------
    indep_test : {"CCA", "Dcorr", "HHG", "RV", "Hsic", "MGC"}
        A string corresponding to the desired independence test from
        ``mgc.independence``. This is not case sensitive.
    compute_distance : callable(), optional (default: euclidean)
        A function that computes the distance among the samples within each
        data matrix. Set to `None` if `x` and `y` are already distance
        matrices. To call a custom function, either create the distance matrix
        before-hand or create a function of the form ``compute_distance(x)``
        where `x` is the data matrix for which pairwise distances are
        calculated.
    bias : bool (default: False)
        Whether or not to use the biased or unbiased test statistics. Only
        applies to ``Dcorr`` and ``Hsic``.

    Notes
    -----
    The ideas behind this can be found in an upcoming paper:

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
    :math:`\{\left(x_i, y_i \right) \stackrel{iid}{\sim} F_{XY},
    \ i = 1, ..., N\}`, the problem that we are testing is,

    .. math::

        H_0: F_{XY} &= F_X F_Y \\
        H_A: F_{XY} &\neq F_X F_Y

    By manipulating the inputs of the *k*-sample test, we can create
    concatenated versions of the inputs and another label matrix which are
    necessarily paired. Then, any nonparametric test can be performed on
    this data.
    """

    def __init__(self, indep_test, compute_distance=euclidean, bias=False):
        indep_test = indep_test.lower()
        test_names = {
            "rv": RV,
            "cca": CCA,
            "hhg": HHG,
            "hsic": Hsic,
            "dcorr": Dcorr,
            "mgc": MGC,
        }
        if indep_test not in test_names.keys():
            raise ValueError("Test is not a valid independence test")
        if indep_test == "hsic" and compute_distance == euclidean:
            compute_distance = gaussian
        self.indep_test_name = indep_test
        indep_test = test_names[indep_test]

        if self.indep_test_name in ["dcorr", "hhg", "hsic", "mgc"]:
            if self.indep_test_name == "hsic":
                self.indep_test = indep_test(compute_kernel=compute_distance, bias=bias)
            elif self.indep_test_name == "dcorr":
                self.indep_test = indep_test(
                    compute_distance=compute_distance, bias=bias
                )
            else:
                self.indep_test = indep_test(compute_distance=compute_distance)
        else:
            self.indep_test = indep_test()

        KSampleTest.__init__(self, compute_distance=compute_distance)

    def _statistic(self, *args):
        r"""
        Calulates the *k*-sample test statistic.

        Parameters
        ----------
        *args : ndarrays
            Variable length input data matrices. All inputs must have the same
            number of samples. That is, the shapes must be `(n, p)` and
            `(m, p)` where `n` and `m` are the number of samples and `p` are
            the number of dimensions. Alternatively, inputs can be distance
            matrices, where the shapes must all be `(n, n)`.
        """
        inputs = list(args)
        u, v = k_sample_transform(inputs)

        return self.indep_test._statistic(u, v)

    def test(self, *args, reps=1000, workers=1, auto=True):
        r"""
        Calculates the *k*-sample test statistic and p-value.

        Parameters
        ----------
        *args : ndarrays
            Variable length input data matrices. All inputs must have the same
            number of samples. That is, the shapes must be `(n, p)` and
            `(m, p)` where `n` and `m` are the number of samples and `p` are
            the number of dimensions. Alternatively, inputs can be distance
            matrices, where the shapes must all be `(n, n)`.
        reps : int, optional (default: 1000)
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, optional (default: 1)
            The number of cores to parallelize the p-value computation over.
            Supply -1 to use all cores available to the Process.
        auto : bool (default: True)
            Automatically uses fast approximation when sample size and size of array
            is greater than 20. If True, and sample size is greater than 20, a fast
            chi2 approximation will be run. Parameters ``reps`` and ``workers`` are
            irrelevant in this case. Only applies to ``Dcorr`` and ``Hsic``.

        Returns
        -------
        stat : float
            The computed *k*-Sample statistic.
        pvalue : float
            The computed *k*-Sample p-value.

        Examples
        --------
        >>> import numpy as np
        >>> from hyppo.ksample import KSample
        >>> x = np.arange(7)
        >>> y = x
        >>> z = np.arange(10)
        >>> stat, pvalue = KSample("Dcorr").test(x, y)
        >>> '%.3f, %.1f' % (stat, pvalue)
        '-0.136, 1.0'
        """
        inputs = list(args)
        check_input = _CheckInputs(
            inputs=inputs,
            indep_test=self.indep_test,
            compute_distance=self.compute_distance,
        )
        inputs = check_input()
        u, v = k_sample_transform(inputs)

        if self.indep_test_name in ["dcorr", "hsic"]:
            return self.indep_test.test(u, v, reps, workers, auto=auto)
        else:
            return self.indep_test.test(u, v, reps, workers)
