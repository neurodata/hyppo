.. _api:

API Reference
=============

.. automodule:: hyppo.independence

.. currentmodule:: hyppo.independence

Independence
-------------

.. autosummary::
   :toctree: generated/

   MaxMargin
   KMERF
   MGC
   Dcorr
   Hsic
   HHG
   CCA
   RV
   FriedmanRafsky



.. automodule:: hyppo.d_variate

.. currentmodule:: hyppo.d_variate

*D*-Variate
-------------

.. autosummary::
    :toctree: generated/

    dHsic



.. automodule:: hyppo.ksample

.. currentmodule:: hyppo.ksample

*K*-Sample
-------------

.. autosummary::
   :toctree: generated/

   KSample
   Energy
   MMD
   DISCO
   MANOVA
   Hotelling
   SmoothCFTest
   MeanEmbeddingTest


.. automodule:: hyppo.time_series

.. currentmodule:: hyppo.time_series

Time-Series
-------------

.. autosummary::
   :toctree: generated/

   MGCX
   DcorrX



.. automodule:: hyppo.discrim

.. currentmodule:: hyppo.discrim

Discriminability
-----------------

.. autosummary::
   :toctree: generated/

   DiscrimOneSample
   DiscrimTwoSample



.. automodule:: hyppo.kgof

.. currentmodule:: hyppo.kgof

Kernel Goodness-of-Fit
-----------------------

.. autosummary::
   :toctree: generated/

   FSSD



.. automodule:: hyppo.tools

.. currentmodule:: hyppo.tools

Simulations
-------------

Independence Simulations
""""""""""""""""""""""""

.. autosummary::
   :toctree: generated/

   linear
   exponential
   cubic
   joint_normal
   step
   quadratic
   w_shaped
   spiral
   uncorrelated_bernoulli
   logarithmic
   fourth_root
   sin_four_pi
   sin_sixteen_pi
   square
   two_parabolas
   circle
   ellipse
   diamond
   multiplicative_noise
   multimodal_independence
   indep_sim

*K*-Sample Simulations
""""""""""""""""""""""""

.. autosummary::
   :toctree: generated/

   rot_ksamp
   gaussian_3samp

Time-Series Simulations
""""""""""""""""""""""""

.. autosummary::
   :toctree: generated/

   indep_ar
   cross_corr_ar
   nonlinear_process
   ts_sim


.. automodule:: hyppo

.. currentmodule:: hyppo

Miscellaneous
-----------------

.. autosummary::
   :toctree: generated/

   independence.sim_matrix
   ksample.k_sample_transform
   tools.compute_kern
   tools.multi_compute_kern
   tools.compute_dist
   tools.perm_test
   tools.multi_perm_test
   tools.chi2_approx
   tools.power
   ksample.smoothCF.smooth_cf_distance
   ksample.mean_embedding.mean_embed_distance


Base Classes
-------------

.. autosummary::
   :toctree: generated/

   independence.base.IndependenceTest
   d_variate.base.DVariateTest
   ksample.base.KSampleTest
   time_series.base.TimeSeriesTest
   discrim.base.DiscriminabilityTest
   kgof.base.GofTest
