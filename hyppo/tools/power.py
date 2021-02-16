from math import ceil

import numpy as np

from ..independence import INDEP_TESTS
from ..ksample import KSAMP_TESTS, k_sample_transform
from .indep_sim import indep_sim
from .ksample_sim import gaussian_3samp, rot_ksamp

_ALL_SIMS = {
    "indep": indep_sim,
    "ksamp": rot_ksamp,
    "gauss": gaussian_3samp,
}

_NONPERM_TESTS = {
    "Dcorr": "fast",
    "Hsic": "fast",
    "Energy": "fast",
    "MMD": "fast",
    "DISCO": "fast",
    "MANOVA": "analytical",
    "Hotelling": "analytical",
}


def _sim_gen(pow_type, **kwargs):
    """
    Generate ``sims`` for the desired simulations.
    """
    if pow_type in ["indep", "ksamp"]:
        if kwargs["sim"] in ["multiplicative_noise", "multimodal_independence"]:
            kwargs = kwargs.copy().pop("noise")

    sims = _ALL_SIMS[pow_type](**kwargs)

    return sims


def _indep_perm_stat(test, pow_type, **kwargs):
    """
    Generates null and alternate distributions for the independence test.
    """
    x, y = _sim_gen(pow_type=pow_type, **kwargs)
    obs_stat = test.statistic(x, y)
    permy = np.random.permutation(y)
    perm_stat = test.statistic(x, permy)

    return obs_stat, perm_stat


def _ksamp_perm_stat(test, pow_type, **kwargs):
    """
    Generates null and alternate distributions for the k-sample test.
    """
    sims = _sim_gen(pow_type=pow_type, **kwargs)
    u, v = k_sample_transform(sims)
    obs_stat = test.statistic(u, v)
    permv = np.random.permutation(v)
    perm_stat = test.statistic(u, permv)

    return obs_stat, perm_stat


_PERM_STATS = {
    "indep": _indep_perm_stat,
    "ksamp": _ksamp_perm_stat,
}


def _nonperm_pval(test, **kwargs):
    """
    Generates fast  permutation pvalues
    """
    x, y = _sim_gen(pow_type="indep", **kwargs)
    pvalue = test.test(x, y, auto=True)[1]

    return pvalue


def power(test, pow_type, sim=None, n=100, alpha=0.05, reps=1000, auto=False, **kwargs):
    """
    Computes empircal power for k-sample tests

    Parameters
    ----------
    test : str or list
        The name of the independence test (from the :mod:`hyppo.independence` module)
        that is to be tested. If MaxMargin, accepts list with first entry "MaxMargin"
        and second entry the name of another independence test.
    pow_type : "indep", "ksamp", "gauss"
        Type of power method to calculate. Depends on the type of ``sim``.
    sim : str, default: None
        The name of the independence simulation (from the :mod:`hyppo.tools` module).
        that is to be used. Set to ``None`` if using gaussian simulation curve.
    n : int, default: None
        The number of samples desired by the simulation (>= 5).
    alpha : float, default: 0.05
        The alpha level of the test.
    reps : int, default: 1000
        The number of replications used to estimate the null distribution
        when using the permutation test used to calculate the p-value.
    auto : bool, default: False
        Automatically uses fast approximation when `n` and size of array
        is greater than 20 or test has a non-permutation based p-value.
        If ``True``, and sample size is greater than 20, then
        :class:`hyppo.tools.chi2_approx` will be run. ``reps`` is
        irrelevant in this case.
        See documentation for ``test`` if this parameter applies.
    **kwargs
        Additional keyword arguements for ``sim``.

    Returns
    -------
    empirical_power : ndarray
        Estimated empirical power for the test.
    """
    if pow_type not in _ALL_SIMS.keys():
        raise ValueError(
            "pow_type {}, must be in {}".format(pow_type, list(_ALL_SIMS.keys()))
        )
    if type(test) is list:
        test = [t.lower() for t in test]
        if test[0] != "maxmargin" or test[1] not in INDEP_TESTS.keys():
            raise ValueError(
                "Test 1 must be Maximal Margin, currently {}; Test 2 must be an "
                "independence test, currently {}".format(*test)
            )
        test = INDEP_TESTS[test[0]](indep_test=test[1])
    else:
        test = test.lower()
        if test in INDEP_TESTS.keys():
            test = INDEP_TESTS[test]()
        elif test in KSAMP_TESTS.keys():
            test = KSAMP_TESTS[test]()
        else:
            raise ValueError(
                "Test {} not in {}".format(
                    test, list(INDEP_TESTS.keys()) + list(KSAMP_TESTS.keys())
                )
            )

    kwargs["n"] = n
    perm_type = "indep"
    if pow_type in ["ksamp", "gauss"]:
        perm_type = "ksamp"
    if pow_type != "gauss":
        kwargs["sim"] = sim

    if test in _NONPERM_TESTS.keys() and auto:
        if n < 20 and _NONPERM_TESTS[test] == "fast":
            empirical_power = np.nan
        else:
            pvals = np.array([_nonperm_pval(test=test, **kwargs) for _ in range(reps)])
            empirical_power = (pvals <= alpha).sum() / reps
    else:
        alt_dist, null_dist = map(
            np.float64,
            zip(
                *[
                    _PERM_STATS[perm_type](test=test, pow_type=pow_type, **kwargs)
                    for _ in range(reps)
                ]
            ),
        )
        cutoff = np.sort(null_dist)[ceil(reps * (1 - alpha))]
        empirical_power = (alt_dist >= cutoff).sum() / reps

    if empirical_power == 0:
        empirical_power = 1 / reps

    return empirical_power
