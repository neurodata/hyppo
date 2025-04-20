r"""
.. :

Causal Treatment Effect Testing
*******************************

Consider an outcome variable :math:`Y`, a treatment or grouping variable :math:`T`, and a set of covariates :math:`X`, where :math:`T` is a :math:`K`-valued grouping variable. For each group :math:`t`, imagine we have a potential outcome :math:`Y(t)`, corresponding to the outcome we would have observed had the treatment/grouping been :math:`t`. Further, assume that the outcome :math:`Y` is the outcome corresponding to the observed treatment/grouping variable; e.g., :math:`Y = Y(T)`. is a realization (causal consistency).

A standard question might be whether, for an indicated level of the covariates, the treatment variable causes changes in the potential outcomes. Specifically, we may wish to test whether, for any :math:`t, t' \in [K]` and for any :math:`x`:

..math::

    H_0 &: Y(t) = Y(t') \mid x \\
    H_A &: Y(t) \neq Y(t') \mid x

Under the alternative, in the case where :math:`Y(t)` and :math:`Y(t')` given covariates :math:`x` differ only in expected value, this is equivalent to the conditional average treatment effect (CATE). 

Under standard causal assumptions:

1. Positivity (equivalently, covariate overlap): for each covariate :math:`x` and for all treatment groups :math:`t`, :math:`P(T = t \mid x) > 0` (each item has a non-zero probability of being assigned to every group),
2. No unmeasured confounding (NUC, also known as conditional ignorability): the measured covariates capture the differences between how individuals ended up being measured in one group as opposed to another; e.g., :math:`\left(Y(1), ..., Y(K)\right) \perp T \mid X`,
3. No interference: the potential outcomes across items do not depend on the treatments or groups of other items.

The aforementioned hypothesis can be tested via standard conditional independence tests. 

Note that neither the positivity nor the NUC assumptions can be directly verified from observed data, as they are statements about hypothetical assignment mechanisms to different groups (e.g., :math:`P(T = t \mid x)`) and about the distributions of hypothetical potential outcomes. For this reason, in order to derive causal conclusions from observed data, domain expertise is often required to reasonably justify the NUC assumption. However, the positivity assumption can often be reasoned from the empricial covariate distributions for each group. This is typically done via propensity score modeling techniques. 

Let's take a look at one such method implemented in :mod:`hyppo.causal` below:
"""

########################################################################################
# Generalised Propensity Score Modeling
# ---------------------------------------------
# 
# The generalised propensity score is a quantity :math:`r(t, x) = P(T = t \mid x)`
# which represents the probability of an individual/item with covariates :math:`x`
# being assigned to treatment/group :math:`t`. While the propensity score cannot be
# known at the time of analysis, it can be estimated from the observed data. This 
# is typically done via logistic regression. More details can be found in 
# :class:`hyppo.causal.GeneralisedPropensityModel`. Below is an example of fitting
# a generalised propensity score model to a set of data where the relationship between
# the covariates and the outcomes for a given treatment/grouping level is non-linear,
# and the treatment effect takes a different value for different levels of the covariates.
import numpy as np
from hyppo.causal import GeneralisedPropensityModel
from hyppo.tools import sigmoidal_sim
d = 2; n = 200
sim = sigmoidal_sim(n, d, balance=0.5, random_state=1234, eff_sz=1)
gps = GeneralisedPropensityModel()
gps.fit(sim["Ts"], sim["Xs"])

########################################################################################
# Propensity Methods
# ------------------
# 
# Once we estimate propensity scores :math:`\hat r(t, x)` via a chosen estimation technique,
# we can incorporate these scores downstream for subsequent causal analytical methods which
# leverage the generalised propensity scores in various ways.
# 
# Vector matching is an approach to pre-process observational data, by trimming samples which
# are poorly or overly represented in certain treatments/groups of a particular . 
# In so doing, this intuitively "filters" observations whose covariates may suggest violations
# of the positivity assumption. More details can be found in 
# :function:`hyppo.causal.GeneralisedPropensityModel.fit`.
balanced_ids = gps.vector_match()

########################################################################################
# Causal Conditional Distance Correlation
# ---------------------------------------
# 
# The Causal Conditional Distance Correlation (Causal CDcorr) is a flexible test for
# whether an assigned group or treatment variable yields changes downstream in potential 
# outcomes across grouping or treatment levels. The testing procedure is closely related 
# to the Conditional Distance Correlation (CDcorr), from :class:`hyppo.conditional.CDcorr`. 
# More details can be found in :class:`hyppo.causal.CausalCDcorr`.
# 
# Below we re-use the non-linear example from above, and we reject the null hypothesis:
from hyppo.causal import CausalCDcorr
stat, pvalue = CausalCDcorr(use_cov=False).test(sim["Ys"], sim["Ts"],
                                                sim["Xs"], random_state=1234)
print("Statistic: ", stat)
print("p-value: ", pvalue)