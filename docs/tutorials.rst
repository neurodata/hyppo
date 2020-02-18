Tutorials
*********

.. _tut_package:

Independence Tests
------------------
The independence testing problem is generalized as follows: Consider random
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
    tutorials/independence/indep_simulations


*K*\ -sample Tests
------------------
The simulations module is used to benchmark all the tests included within the
package against one another.

.. toctree::
    :maxdepth: 1

    tutorials/
