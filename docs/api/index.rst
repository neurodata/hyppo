.. api:

API Reference
=============

.. automodule:: hyppo.independence

.. currentmodule:: hyppo.independence

Independence
-------------

.. autosummary::
   :toctree: generated/

   KMERF
   MGC
   Dcorr
   Hsic
   HHG
   CCA
   RV



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



.. automodule:: hyppo

.. currentmodule:: hyppo

Miscellaneous
-----------------

.. autosummary::
   :toctree: generated/

   independence.sim_matrix
   ksample.k_sample_transform
   tools.compute_kern
   tools.compute_dist
   tools.perm_test
   tools.chi2_approx
