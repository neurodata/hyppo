.. _api:

API Reference
=============

.. automodule:: hyppo.d_variate
    :no-members:
    :no-inherited-members:

:mod:`hyppo.d_variate`: *D*-Variate Independence tests
-----------------------------------------------------
.. currentmodule:: hyppo

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   d_variate.dHsic

.. automodule:: hyppo.discrim
    :no-members:
    :no-inherited-members:

:mod:`hyppo.discrim`: Discriminability tests
---------------------------------------------------------
.. currentmodule:: hyppo

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   discrim.DiscrimOneSample
   discrim.DiscrimTwoSample

.. automodule:: hyppo.independence
    :no-members:
    :no-inherited-members:

:mod:`hyppo.independence`: Independence tests
----------------------------------------------

Classes
^^^^^^^^^^^^^^^
.. currentmodule:: hyppo

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   independence.CCA
   independence.Dcorr
   independence.FriedmanRafsky
   independence.HHG
   independence.Hsic
   independence.KMERF
   independence.MaxMargin
   independence.MGC
   independence.RV

Functions
^^^^^^^^^^^^^^^
.. currentmodule:: hyppo

.. autosummary::
   :toctree: generated/
   :template: function.rst

   independence.sim_matrix

.. automodule:: hyppo.kgof
    :no-members:
    :no-inherited-members:

:mod:`hyppo.kgof`: Kernel Goodness-of-Fit tests
---------------------------------------------------------
.. currentmodule:: hyppo

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   kgof.FSSD

.. automodule:: hyppo.ksample
    :no-members:
    :no-inherited-members:

:mod:`hyppo.ksample`: *K*-Sample tests
----------------------------------------------

Classes
^^^^^^^^^^^^^^^
.. currentmodule:: hyppo

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   ksample.DISCO
   ksample.Energy
   ksample.Hotelling
   ksample.KSample
   ksample.MANOVA
   ksample.MeanEmbeddingTest
   ksample.MMD
   ksample.SmoothCFTest

Functions
^^^^^^^^^^^^^^^
.. currentmodule:: hyppo

.. autosummary::
   :toctree: generated/
   :template: function.rst

   ksample.k_sample_transform
   ksample.mean_embedding.mean_embed_distance
   ksample.smoothCF.smooth_cf_distance

.. automodule:: hyppo.time_series
    :no-members:
    :no-inherited-members:

:mod:`hyppo.time_series`: Time Series Independence tests
---------------------------------------------------------
.. currentmodule:: hyppo

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   time_series.DcorrX
   time_series.MGCX

.. automodule:: hyppo.tools
    :no-members:
    :no-inherited-members:

:mod:`hyppo.tools`: Base Classes and Utilities
----------------------------------------------

Base Classes
^^^^^^^^^^^^^^^
.. currentmodule:: hyppo

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   d_variate.base.DVariateTest
   discrim.base.DiscriminabilityTest
   independence.base.IndependenceTest
   kgof.base.GofTest
   ksample.base.KSampleTest
   time_series.base.TimeSeriesTest

Functions
^^^^^^^^^^^^^^^
.. currentmodule:: hyppo

.. autosummary::
   :toctree: generated/
   :template: function.rst

   tools.chi2_approx
   tools.compute_dist
   tools.compute_kern
   tools.multi_compute_kern
   tools.multi_perm_test
   tools.perm_test
   tools.power

Simulations
^^^^^^^^^^^^^^^

Independence Simulations
""""""""""""""""""""""""
.. currentmodule:: hyppo

.. autosummary::
   :toctree: generated/
   :template: function.rst

   tools.circle
   tools.cubic
   tools.diamond
   tools.ellipse
   tools.exponential
   tools.fourth_root
   tools.indep_sim
   tools.joint_normal
   tools.linear
   tools.logarithmic
   tools.multimodal_independence
   tools.multiplicative_noise
   tools.quadratic
   tools.sin_four_pi
   tools.sin_sixteen_pi
   tools.spiral
   tools.square
   tools.step
   tools.two_parabolas
   tools.uncorrelated_bernoulli
   tools.w_shaped

*K*-Sample Simulations
""""""""""""""""""""""""""""""""""""
.. currentmodule:: hyppo

.. autosummary::
   :toctree: generated/
   :template: function.rst

   tools.ksamp_sim
   tools.rot_ksamp
   tools.gaussian_3samp

Time-Series Simulations
""""""""""""""""""""""""""""""""""""
.. currentmodule:: hyppo

.. autosummary::
   :toctree: generated/
   :template: function.rst

   tools.indep_ar
   tools.cross_corr_ar
   tools.nonlinear_process
   tools.ts_sim
