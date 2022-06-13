import numpy as np
from numba import jit
from scipy.stats import f

from ._utils import _CheckInputs
from .base import KSampleTest


class MANOVA(KSampleTest):
    r"""
    Multivariate analysis of variance (MANOVA) test statistic and p-value.

    MANOVA is the current standard for multivariate `k`-sample testing.

    Notes
    -----
    The test statistic is formulated as below
    :footcite:p:`pandaNonparMANOVAIndependence2021`:

    In MANOVA, we are testing if the mean vectors of each of the `k`-samples are the
    same. Define
    :math:`\{ {x_1}_i \stackrel{iid}{\sim} F_{X_1},\ i = 1, ..., n_1 \}`,
    :math:`\{ {x_2}_j \stackrel{iid}{\sim} F_{X_2},\ j = 1, ..., n_2 \}`, ... as `k`
    groups
    of samples deriving from different a multivariate Gaussian distribution with the
    same dimensionality and same covariance matrix.
    That is, the null and alternate hypotheses are,

    .. math::

       H_0 &: \mu_1 = \mu_2 = \cdots = \mu_k, \\
       H_A &: \exists \ j \neq j' \text{ s.t. } \mu_j \neq \mu_{j'}

    Let :math:`\bar{x}_{i \cdot}` refer to the columnwise means of :math:`x_i`; that is,
    :math:`\bar{x}_{i \cdot} = (1/n_i) \sum_{j=1}^{n_i} x_{ij}`. The pooled sample
    covariance of each group, :math:`W`, is

    .. math::

       W = \sum_{i=1}^k \sum_{j=1}^{n_i} (x_{ij} - \bar{x}_{i\cdot}
       (x_{ij} - \bar{x}_{i\cdot})^T

    Next, define :math:`B` as the  sample covariance matrix of the means. If
    :math:`n = \sum_{i=1}^k n_i` and the grand mean is
    :math:`\bar{x}_{\cdot \cdot} = (1/n) \sum_{i=1}^k \sum_{j=1}^{n} x_{ij}`,

    .. math::

       B = \sum_{i=1}^k n_i (\bar{x}_{i \cdot} - \bar{x}_{\cdot \cdot})
       (\bar{x}_{i \cdot} - \bar{x}_{\cdot \cdot})^T

    Some of the most common statistics used when performing MANOVA include the Wilks'
    Lambda, the Lawley-Hotelling trace, Roy's greatest root, and
    Pillai-Bartlett trace (PBT)
    :footcite:p:`bartlettNoteTestsSignificance1939`
    :footcite:p:`raoTestsSignificanceMultivariate1948`
    (PBT was chosen to be the best of these
    as it is the most conservative
    :footcite:p:`warnePrimerMultivariateAnalysis2019`) and
    :footcite:p:`everittMonteCarloInvestigation1979`
    has shown that there are
    minimal differences in statistical power among these statistics.
    More information about the specific statistics and how the p-values are calculated
    are described at this reference :footcite:p:`SASHelpCenter`.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self):
        self.MANOVA_STATS = {
            "Wilks' lambda": self._wilks_lambda,
            "Pillai's trace": self._pillai_bartlett,
            "Hotelling-Lawley trace": self._hotelling_lawley,
            "Roy's greatest root": self._roy_max_root,
        }
        KSampleTest.__init__(self)

    def _wilks_lambda(self, only_stat=False, **kwargs):
        r"""
        Calulates the Wilks' lambda test and F statistic.

        Parameters
        ----------
        only_stat : bool
            Whether to only compute the test statistic.
        **kwargs
            Arbitrary keyword arguments for F statistic computation.

        Returns
        -------
        stat : float
            The computed Wilks' lambda statistic.
        F : float, optional
            The computed F statistic.
        df : tuple of floats, optional
            The degree of freedom for the test.
        """
        p, q, v, s = kwargs["p"], kwargs["q"], kwargs["v"], kwargs["s"]
        stat = np.linalg.det(self.E) / np.linalg.det(self.T)

        results = stat
        if not only_stat:
            r = v - (p - q + 1) / 2
            u = (p * q - 2) / 4
            if (p**2 + q**2 - 5) > 0:
                t = np.sqrt((p**2 * q**2 - 4) / (p**2 + q**2 - 5))
            else:
                t = 1
            num = r * t - 2 * u
            denom = p * q
            F = (num / denom) * (s - stat ** (1 / t)) / (stat ** (1 / t))
            df = (s * denom, s * num)
            results = (stat, F, df)

        return results

    def _hotelling_lawley(self, only_stat=False, **kwargs):
        r"""
        Calulates the Hotelling-Lawley Trace test statistic.

        Parameters
        ----------
        only_stat : bool, optional
            Whether to only compute the test statistic.
        **kwargs
            Arbitrary keyword arguments for F statistic computation.

        Returns
        -------
        stat : float
            The computed Hotelling-Lawley Trace statistic.
        F : float, optional
            The computed F statistic.
        df : tuple of floats, optional
            The degree of freedom for the test.
        """
        p, q, n, m, s = kwargs["p"], kwargs["q"], kwargs["n"], kwargs["m"], kwargs["s"]
        stat = np.trace(self.H @ np.linalg.inv(self.E))

        results = stat
        if not only_stat:
            if n > 0:
                b = (p + 2 * n) * (q + 2 * n) / (2 * (2 * n + 1) * (n - 1))
                c = (2 + (p * q + 2) / (b - 1)) / (2 * n)
                num = 4 + (p * q + 2) / (b - 1)
                denom = p * q
                F = (num / denom) * (stat / c)
                df = (denom, num)
            else:
                num = 2 * (s * n + 1)
                denom = s * (2 * m + s + 1)
                F = (num / denom) * (stat / s)
                df = (denom, num)
            results = (stat, F, df)

        return results

    def _pillai_bartlett(self, only_stat=False, **kwargs):
        r"""
        Calulates the Pillai-Bartlett Trace test and F statistic.

        Parameters
        ----------
        only_stat : bool, optional
            Whether to only compute the test statistic.
        **kwargs
            Arbitrary keyword arguments for F statistic computation.

        Returns
        -------
        stat : float
            The computed Pillai-Bartlett Trace statistic.
        F : float, optional
            The computed F statistic.
        df : tuple of floats, optional
            The degree of freedom for the test.
        """
        n, m, s = kwargs["n"], kwargs["m"], kwargs["s"]
        stat = np.trace(self.H @ np.linalg.inv(self.T))

        results = stat
        if not only_stat:
            num = 2 * n + s + 1
            denom = 2 * m + s + 1
            F = (num / denom) * stat / (s - stat)
            df = (s * denom, s * num)
            results = (stat, F, df)

        return results

    def _roy_max_root(self, only_stat=False, **kwargs):
        r"""
        Calulates the Roy's Maximum Root test statistic.

        Parameters
        ----------
        only_stat : bool, optional
            Whether to only compute the test statistic.
        **kwargs
            Arbitrary keyword arguments for F statistic computation.

        Returns
        -------
        stat : float
            The computed Roy's Maximum Root statistic.
        F : float, optional
            The computed F statistic.
        df : tuple of floats, optional
            The degree of freedom for the test.
        """
        p, q, v = kwargs["p"], kwargs["q"], kwargs["v"]
        stat = np.linalg.eigvals(self.H @ np.linalg.inv(self.E)).real.max()
        print(stat)

        results = stat
        if not only_stat:
            r = np.max([p, q])
            num = v - r + q
            denom = r
            F = (num / denom) * stat
            df = (denom, num)
            results = (stat, F, df)

        return results

    def _compute_scss(self, *args):
        """Compute sum of squares hypothesis, error, and total."""
        p = args[0].shape[1]
        cmean = tuple(i.mean(axis=0).reshape(-1, 1) for i in args)
        gmean = np.vstack(args).mean(axis=0).reshape(-1, 1)
        self.E = _compute_e(args, p, cmean)
        self.H = _compute_h(args, p, cmean, gmean)
        self.T = self.H + self.E

    def statistic(self, *args):
        r"""
        Calulates the MANOVA test statistic.

        Parameters
        ----------
        *args : ndarray of float
            Variable length input data matrices. All inputs must have the same
            number of dimensions. That is, the shapes must be `(n, p)` and
            `(m, p)`, ... where `n`, `m`, ... are the number of samples and `p` is
            the number of dimensions.

        Returns
        -------
        stat : dict of float
            The computed MANOVA statistic. Contains the following keys:

                - Wilks' lambda: float
                    The Wilks' lambda test statistic
                - Pillai's trace: float
                    The Pillai-Bartlett trace test statistic
                - Hotelling-Lawley trace: float
                    The Hotelling-Lawley trace test statistic
                - Roy's greatest root: float
                    The Roy's maximum root test statistic
        """
        self._compute_scss(*args)

        stat = {}
        for key, value in self.MANOVA_STATS.items():
            stat[key] = value(only_stat=True)
        self.stat = stat

        return stat

    def test(self, *args, summary=False):
        r"""
        Calculates the MANOVA test statistic and p-value.

        Parameters
        ----------
        *args : ndarray of float
            Variable length input data matrices. All inputs must have the same
            number of dimensions. That is, the shapes must be `(n, p)` and
            `(m, p)`, ... where `n`, `m`, ... are the number of samples and `p` is
            the number of dimensions.
        summary : bool, optional
            Whether to view the results in a summary table.

        Returns
        -------
        results : dict of dict
            The computed MANOVA statistic. Contains the following keys:

                - Wilks' lambda : dict of float
                    The Wilks' lambda test results.
                - Pillai's trace : dict of float
                    The Pillai-Bartlett test results.
                - Hotelling-Lawley trace : dict of float
                    The Hotelling-Lawley trace test results.
                - Roy's greatest root : dict of float
                    The Roy's maximum root test test results.

            Each of these tests have the following keys:

                - statistic : float
                    The test statistic.
                - num df : float
                    The degrees of freedom for the numerator.
                - denom df : float
                    The degrees of freedom for the denominator.
                - f statistic : float
                    The F statistic for the test.
                - p-value : float
                    The p-value approximated via the F-distribution.
        summary
            A summary table for the test statistics, p-values, and degrees of freedom.

        Examples
        --------
        >>> import numpy as np
        >>> from hyppo.ksample import MANOVA
        >>> x = np.arange(7)
        >>> y = x
        >>> stat, pvalue = MANOVA().test(x, y)
        >>> '%.3f, %.1f' % (stat, pvalue)
        '0.000, 1.0'
        """
        inputs = list(args)
        check_input = _CheckInputs(
            inputs=inputs,
        )
        inputs = check_input()

        self._compute_scss(*args)
        N = np.sum([i.shape[0] for i in inputs])
        p = inputs[0].shape[1]
        v = N - len(inputs)

        if v < p:
            raise ValueError("Test cannot be run, degree of freedoms is off")

        q = len(inputs) - 1
        kwargs = {
            "p": p,
            "q": q,
            "v": v,
            "s": np.min([p, q]),
            "m": (np.abs(p - q) - 1) / 2,
            "n": (v - p - 1) / 2,
        }
        results = {}
        for key, value in self.MANOVA_STATS.items():
            results[key] = list(value(only_stat=False, **kwargs))
            pvalue = f.sf(results[key][1], results[key][2][0], results[key][2][1])
            results[key].append(pvalue)

        test_results = results.copy()
        for key, value in results.items():
            test_results[key] = {
                "statistic": value[0],
                "num df": value[2][0],
                "denom df": value[2][1],
                "f statistic": value[1],
                "p-value": value[3],
            }
        self.test_results = test_results

        if summary:
            header = "{:<25} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                "Criterion", "Statistic", "DF Num", "DF Denom", "F", "P-Value"
            )
            print(header)
            print("-" * len(header))

            for key, value in test_results.items():
                print(
                    "{:<25} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                        key, *value.values()
                    )
                )

        # return dictionary, not tuple
        return test_results


@jit(nopython=True, cache=True)
def _compute_e(inputs, p, cmean):  # pragma: no cover
    """Calculate the W matrix"""
    E = np.zeros((p, p))

    for i in range(len(inputs)):
        n_i = inputs[i].shape[0]
        for j in range(n_i):
            E += (inputs[i][j, :].reshape(-1, 1) - cmean[i]) @ (
                inputs[i][j, :].reshape(-1, 1) - cmean[i]
            ).T

    return E


@jit(nopython=True, cache=True)
def _compute_h(inputs, p, cmean, gmean):  # pragma: no cover
    """Calculate the B matrix"""
    H = np.zeros((p, p))

    for i in range(len(inputs)):
        n_i = inputs[i].shape[0]
        H += n_i * (cmean[i] - gmean) @ (cmean[i] - gmean).T

    return H
