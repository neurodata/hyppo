r"""
.. _discrim:

Discriminability Testing
***************************

If you have repeated measures from the same subject, and want to see if these are
different than those from other subjects. Let's look at the mathematical formulations:

With :math:`D_x` as the sample discriminability of :math:`x`, one sample test performs
the following test:

.. math::

   H_0 &: D_x = D_0 \\
   H_A &: D_x > D_0

where :math:`D_0` is the discriminability that would be observed by random chance.

This can also be formulated as a two-sample test. Let :math:`\hat{D}_x` denote the
sample discriminability of one approach, and :math:`\hat{D}_y` denote the sample
discriminability of another approach. Then,

.. math::

   H_0 &: D_x = D_y \\
   H_A &: D_x > D_y

Alternative tests can be done for :math:`D_x < D_y` and :math:`D_x \neq D_y`.

Like all the other tests within hyppo, each method has a :func:`statistic` and
:func:`test` method. The :func:`test` method is the one that returns the test statistic
and p-values, among other outputs, and is the one that is used most often in the
examples, tutorials, etc.
The p-value returned is calculated using a permutation test.
"""

########################################################################################
# **Discrimnability one-sample** and
# **Discrimnability two-sample** are time series tests of independence.
# More details can be found in :class:`hyppo.discrim.DiscrimOneSample` and
# :class:`hyppo.discrim.DiscrimTwoSample`.
#
# Each class has a ``is_dist`` parameter that indicates whether or not inputs are
# distance matrices. These distances must be euclidean distance.
# Also, ``remove_isolates`` indicates whether or not to remove measurements with a single
# instance.
# Otherwise, these tests runs like :ref:`any other test<general indep>`.
