Tutorials
*********

.. _indep_tutorials:

Independence Tests
------------------
The independence testing problem is generalized as follows: consider random
variables :math:`X` and :math:`Y` that have joint density
:math:`F_{XY} = F_{X|Y} F_Y`. We are testing:

.. math::

    H_0: F_{XY} &= F_X F_Y \\
    H_A: F_{XY} &\neq F_X F_Y

These tutorials overview how to use these tests as well as benchmarks comparing
the algorithms included against each other.

.. toctree::
    :maxdepth: 1

    tutorials/independence/independence
    tutorials/independence/indep_power
    tutorials/independence/indep_alg_speed


.. _ksamp_tutorials:

*K*-sample Tests
------------------
The *k*-sample testing problem is generalized as follows: consider random variables
:math:`X_1, X_2, \ldots, X_k` that have densities
:math:`F_1, F_2, \ldots, F_k`. Then, we are testing

.. math::

    H_0:\ &F_1 = F_2 = \ldots F_k \\
    H_A:\ &\exists \ j \neq j' \text{ s.t. } F_j \neq F_{j'}

This tutorial overview how to use *k*-sample tests in ``hyppo``.

.. toctree::
    :maxdepth: 1

    tutorials/ksample/ksample

.. _ts_tutorials:

Time-Series Tests
-----------------
Time-series tests of independence consider the following problem: consider
random variables :math:`X` and :math:`Y` with joint density :math:`F_{XY}` and
marginal densities :math:`F_X` and :math:`F_Y`. Let :math:`F_{X_t}`,
:math:`F_{Y_s}`, and :math:`F_{X_t Y_s}` represent the marginal and joint
distributions of time-indexed random varlables :math:`X_t` and :math:`Y_s` at
timesteps :math:`t` and :math:`s`. Let
:math:`\{ (X_t, Y_t) \}_{t = -\infty}^\infty` be a full jointly-sampled
strictly stationary time series with the observed sample
:math:`\{ (X_1, Y_1), \ldots (X_n, Y_n) \}`. Choose some nonnegative integer
:math:`M` as the maximium lag hyperparamater. Then we are testing,

.. math::

    H_0: F_{X_t Y_{t - j}} &= F_{X_t} F_{Y_{t - j}} \text{ for each } j \in \{ 0, 1, \ldots, M \} \\
    H_A: F_{X_t Y_{t - j}} &\neq F_{X_t} F_{Y_{t - j}} \text{ for some } j \in \{ 0, 1, \ldots, M \}

This tutorial overview how to use time_series based tests in ``hyppo``.

.. toctree::
    :maxdepth: 1

    tutorials/time_series/time_series


.. _sims_tutorials:

Sims
----
To evaluate existing implmentations and benchmark against other packages,
we have developed a suite of 20 dependency structures. The simulation settings
include polynomial (linear, quadratic, cubic), trigonometric (sinusoidal,
circular, ellipsoidal, spiral), geometric (square, diamond, w-shaped), and
other functions. We also include 3 sample Gaussian simulations as well,
which are sampled from multivariate normal distribusions.

.. toctree::
    :maxdepth: 1

    tutorials/sims/indep_simulations
