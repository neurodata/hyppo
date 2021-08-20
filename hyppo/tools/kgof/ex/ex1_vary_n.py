"""Simulation to test the test power vs increasing sample size"""

__author__ = 'wittawat'

import kgof
import kgof.data as data
import kgof.glo as glo
import kgof.density as density
import kgof.goftest as gof
import kgof.intertst as tgof
import kgof.mmd as mgof
import kgof.util as util 
import kgof.kernel as kernel 

# need independent_jobs package 
# https://github.com/karlnapf/independent-jobs
# The independent_jobs and kgof have to be in the global search path (.bashrc)
import independent_jobs as inj
from independent_jobs.jobs.IndependentJob import IndependentJob
from independent_jobs.results.SingleResult import SingleResult
from independent_jobs.aggregators.SingleResultAggregator import SingleResultAggregator
from independent_jobs.engines.BatchClusterParameters import BatchClusterParameters
from independent_jobs.engines.SerialComputationEngine import SerialComputationEngine
from independent_jobs.engines.SlurmComputationEngine import SlurmComputationEngine
from independent_jobs.tools.Log import logger
import logging
import math
#import numpy as np
import autograd.numpy as np
import os
import sys 
import time

"""
All the job functions return a dictionary with the following keys:
    - goftest: test object. (may or may not return)
    - test_result: the result from calling perform_test(te).
    - time_secs: run time in seconds 
"""

def job_fssdJ1q_med(p, data_source, tr, te, r, J=1, null_sim=None):
    """
    FSSD test with a Gaussian kernel, where the test locations are randomized,
    and the Gaussian width is set with the median heuristic. Use full sample.
    No training/testing splits.

    p: an UnnormalizedDensity
    data_source: a DataSource
    tr, te: Data
    r: trial number (positive integer)
    """
    if null_sim is None:
        null_sim = gof.FSSDH0SimCovObs(n_simulate=2000, seed=r)

    # full data
    data = tr + te
    X = data.data()
    with util.ContextTimer() as t:
        # median heuristic 
        med = util.meddistance(X, subsample=1000)
        k = kernel.KGauss(med**2)
        V = util.fit_gaussian_draw(X, J, seed=r+1)

        fssd_med = gof.FSSD(p, k, V, null_sim=null_sim, alpha=alpha)
        fssd_med_result = fssd_med.perform_test(data)
    return { 'test_result': fssd_med_result, 'time_secs': t.secs}

def job_fssdJ5q_med(p, data_source, tr, te, r):
    """
    FSSD. J=5
    """
    return job_fssdJ1q_med(p, data_source, tr, te, r, J=5)

def job_fssdJ1q_opt(p, data_source, tr, te, r, J=1, null_sim=None):
    """
    FSSD with optimization on tr. Test on te. Use a Gaussian kernel.
    """
    if null_sim is None:
        null_sim = gof.FSSDH0SimCovObs(n_simulate=2000, seed=r)

    Xtr = tr.data()
    with util.ContextTimer() as t:
        # Use grid search to initialize the gwidth
        n_gwidth_cand = 5
        gwidth_factors = 2.0**np.linspace(-3, 3, n_gwidth_cand) 
        med2 = util.meddistance(Xtr, 1000)**2

        k = kernel.KGauss(med2)
        # fit a Gaussian to the data and draw to initialize V0
        V0 = util.fit_gaussian_draw(Xtr, J, seed=r+1, reg=1e-6)
        list_gwidth = np.hstack( ( (med2)*gwidth_factors ) )
        besti, objs = gof.GaussFSSD.grid_search_gwidth(p, tr, V0, list_gwidth)
        gwidth = list_gwidth[besti]
        assert util.is_real_num(gwidth), 'gwidth not real. Was %s'%str(gwidth)
        assert gwidth > 0, 'gwidth not positive. Was %.3g'%gwidth
        logging.info('After grid search, gwidth=%.3g'%gwidth)
        
        ops = {
            'reg': 1e-2,
            'max_iter': 30,
            'tol_fun': 1e-5,
            'disp': True,
            'locs_bounds_frac':30.0,
            'gwidth_lb': 1e-1,
            'gwidth_ub': 1e4,
            }

        V_opt, gwidth_opt, info = gof.GaussFSSD.optimize_locs_widths(p, tr,
                gwidth, V0, **ops) 
        # Use the optimized parameters to construct a test
        k_opt = kernel.KGauss(gwidth_opt)
        fssd_opt = gof.FSSD(p, k_opt, V_opt, null_sim=null_sim, alpha=alpha)
        fssd_opt_result = fssd_opt.perform_test(te)
    return {'test_result': fssd_opt_result, 'time_secs': t.secs, 
            'goftest': fssd_opt, 'opt_info': info,
            }

def job_fssdJ5q_opt(p, data_source, tr, te, r):
    return job_fssdJ1q_opt(p, data_source, tr, te, r, J=5)

def job_fssdJ10q_opt(p, data_source, tr, te, r):
    return job_fssdJ1q_opt(p, data_source, tr, te, r, J=10)

def job_fssdJ1q_imq_optv(p, data_source, tr, te, r, J=1, null_sim=None):
    """
    FSSD with optimization on tr. Test on te. Use an inverse multiquadric
    kernel (IMQ). Optimize only the test locations (V). Fix the kernel
    parameters to b = -0.5, c=1. These are the recommended values from

        Measuring Sample Quality with Kernels
        Jackson Gorham, Lester Mackey
    """
    if null_sim is None:
        null_sim = gof.FSSDH0SimCovObs(n_simulate=2000, seed=r)

    Xtr = tr.data()
    with util.ContextTimer() as t:
        # IMQ kernel parameters
        b = -0.5
        c = 1.0

        # fit a Gaussian to the data and draw to initialize V0
        V0 = util.fit_gaussian_draw(Xtr, J, seed=r+1, reg=1e-6)

        ops = {
            'reg': 1e-2,
            'max_iter': 40,
            'tol_fun': 1e-4,
            'disp': True,
            'locs_bounds_frac':10.0,
            }

        V_opt, info = gof.IMQFSSD.optimize_locs(p, tr, b, c, V0, **ops) 

        k_imq = kernel.KIMQ(b=b, c=c)

        # Use the optimized parameters to construct a test
        fssd_imq = gof.FSSD(p, k_imq, V_opt, null_sim=null_sim, alpha=alpha)
        fssd_imq_result = fssd_imq.perform_test(te)

    return {'test_result': fssd_imq_result, 'time_secs': t.secs, 
            'goftest': fssd_imq, 'opt_info': info,
            }

def job_fssdJ5q_imq_optv(p, data_source, tr, te, r):
    return job_fssdJ1q_imq_optv(p, data_source, tr, te, r, J=5)

def job_me_opt(p, data_source, tr, te, r, J=5):
    """
    ME test of Jitkrittum et al., 2016 used as a goodness-of-fit test.
    Gaussian kernel. Optimize test locations and Gaussian width.
    """
    data = tr + te
    X = data.data()
    with util.ContextTimer() as t:
        # median heuristic 
        #pds = p.get_datasource()
        #datY = pds.sample(data.sample_size(), seed=r+294)
        #Y = datY.data()
        #XY = np.vstack((X, Y))
        #med = util.meddistance(XY, subsample=1000)
        op = {'n_test_locs': J, 'seed': r+5, 'max_iter': 40, 
             'batch_proportion': 1.0, 'locs_step_size': 1.0, 
              'gwidth_step_size': 0.1, 'tol_fun': 1e-4, 
              'reg': 1e-4}
        # optimize on the training set
        me_opt = tgof.GaussMETestOpt(p, n_locs=J, tr_proportion=tr_proportion,
                alpha=alpha, seed=r+111)

        me_result = me_opt.perform_test(data, op)
    return { 'test_result': me_result, 'time_secs': t.secs}

def job_kstein_med(p, data_source, tr, te, r):
    """
    Kernel Stein discrepancy test of Liu et al., 2016 and Chwialkowski et al.,
    2016. Use full sample. Use Gaussian kernel.
    """
    # full data
    data = tr + te
    X = data.data()
    with util.ContextTimer() as t:
        # median heuristic 
        med = util.meddistance(X, subsample=1000)
        k = kernel.KGauss(med**2)

        kstein = gof.KernelSteinTest(p, k, alpha=alpha, n_simulate=1000, seed=r)
        kstein_result = kstein.perform_test(data)
    return { 'test_result': kstein_result, 'time_secs': t.secs}

def job_kstein_imq(p, data_source, tr, te, r):
    """
    Kernel Stein discrepancy test of Liu et al., 2016 and Chwialkowski et al.,
    2016. Use full sample. Use the inverse multiquadric kernel (IMQ) studied 
    in 

    Measuring Sample Quality with Kernels
    Gorham and Mackey 2017. 

    Parameters are fixed to the recommented values: beta = b = -0.5, c = 1. 
    """
    # full data
    data = tr + te
    X = data.data()
    with util.ContextTimer() as t:
        k = kernel.KIMQ(b=-0.5, c=1.0)

        kstein = gof.KernelSteinTest(p, k, alpha=alpha, n_simulate=1000, seed=r)
        kstein_result = kstein.perform_test(data)
    return { 'test_result': kstein_result, 'time_secs': t.secs}


def job_lin_kstein_med(p, data_source, tr, te, r):
    """
    Linear-time version of the kernel Stein discrepancy test of Liu et al.,
    2016 and Chwialkowski et al., 2016. Use full sample.
    """
    # full data
    data = tr + te
    X = data.data()
    with util.ContextTimer() as t:
        # median heuristic 
        med = util.meddistance(X, subsample=1000)
        k = kernel.KGauss(med**2)

        lin_kstein = gof.LinearKernelSteinTest(p, k, alpha=alpha, seed=r)
        lin_kstein_result = lin_kstein.perform_test(data)
    return { 'test_result': lin_kstein_result, 'time_secs': t.secs}

def job_mmd_med(p, data_source, tr, te, r):
    """
    MMD test of Gretton et al., 2012 used as a goodness-of-fit test.
    Require the ability to sample from p i.e., the UnnormalizedDensity p has 
    to be able to return a non-None from get_datasource()
    """
    # full data
    data = tr + te
    X = data.data()
    with util.ContextTimer() as t:
        # median heuristic 
        pds = p.get_datasource()
        datY = pds.sample(data.sample_size(), seed=r+294)
        Y = datY.data()
        XY = np.vstack((X, Y))

        # If p, q differ very little, the median may be very small, rejecting H0
        # when it should not?
        # If p, q differ very little, the median may be very small, rejecting H0
        # when it should not?
        medx = util.meddistance(X, subsample=1000)
        medy = util.meddistance(Y, subsample=1000)
        medxy = util.meddistance(XY, subsample=1000)
        med_avg = (medx+medy+medxy)/3.0
        k = kernel.KGauss(med_avg**2)

        mmd_test = mgof.QuadMMDGof(p, k, n_permute=400, alpha=alpha, seed=r)
        mmd_result = mmd_test.perform_test(data)
    return { 'test_result': mmd_result, 'time_secs': t.secs}

def job_mmd_opt(p, data_source, tr, te, r):
    """
    MMD test of Gretton et al., 2012 used as a goodness-of-fit test.
    Require the ability to sample from p i.e., the UnnormalizedDensity p has 
    to be able to return a non-None from get_datasource()

    With optimization. Gaussian kernel.
    """
    data = tr + te
    X = data.data()
    with util.ContextTimer() as t:
        # median heuristic 
        pds = p.get_datasource()
        datY = pds.sample(data.sample_size(), seed=r+294)
        Y = datY.data()
        XY = np.vstack((X, Y))

        med = util.meddistance(XY, subsample=1000)

        # Construct a list of kernels to try based on multiples of the median
        # heuristic
        #list_gwidth = np.hstack( (np.linspace(20, 40, 10), (med**2)
        #    *(2.0**np.linspace(-2, 2, 20) ) ) )
        list_gwidth = (med**2)*(2.0**np.linspace(-4, 4, 30) ) 
        list_gwidth.sort()
        candidate_kernels = [kernel.KGauss(gw2) for gw2 in list_gwidth]

        mmd_opt = mgof.QuadMMDGofOpt(p, n_permute=300, alpha=alpha, seed=r)
        mmd_result = mmd_opt.perform_test(data,
                candidate_kernels=candidate_kernels,
                tr_proportion=tr_proportion, reg=1e-3)
    return { 'test_result': mmd_result, 'time_secs': t.secs}

def job_mmd_dgauss_opt(p, data_source, tr, te, r):
    """
    MMD test of Gretton et al., 2012 used as a goodness-of-fit test.
    Require the ability to sample from p i.e., the UnnormalizedDensity p has 
    to be able to return a non-None from get_datasource()

    With optimization. Diagonal Gaussian kernel where there is one Gaussian width
    for each dimension.
    """
    data = tr + te
    X = data.data()
    d = X.shape[1]
    with util.ContextTimer() as t:
        # median heuristic 
        pds = p.get_datasource()
        datY = pds.sample(data.sample_size(), seed=r+294)
        Y = datY.data()
        XY = np.vstack((X, Y))

        # Get the median heuristic for each dimension
        meds = np.zeros(d)
        for i in range(d):
            medi = util.meddistance(XY[:, [i]], subsample=1000)
            meds[i] = medi

        # Construct a list of kernels to try based on multiples of the median
        # heuristic
        med_factors = 2.0**np.linspace(-4, 4, 20)  
        candidate_kernels = []
        for i in range(len(med_factors)):
            ki = kernel.KDiagGauss( (meds**2)*med_factors[i] )
            candidate_kernels.append(ki)

        mmd_opt = mgof.QuadMMDGofOpt(p, n_permute=300, alpha=alpha, seed=r+56)
        mmd_result = mmd_opt.perform_test(data,
                candidate_kernels=candidate_kernels,
                tr_proportion=tr_proportion, reg=1e-3)
    return { 'test_result': mmd_result, 'time_secs': t.secs}

# Define our custom Job, which inherits from base class IndependentJob
class Ex1Job(IndependentJob):
   
    def __init__(self, aggregator, p, data_source, prob_label, rep, job_func, n):
        #walltime = 60*59*24 
        walltime = 60*59
        memory = int(tr_proportion*n*1e-2) + 50

        IndependentJob.__init__(self, aggregator, walltime=walltime,
                               memory=memory)
        # p: an UnnormalizedDensity
        self.p = p
        self.data_source = data_source
        self.prob_label = prob_label
        self.rep = rep
        self.job_func = job_func
        self.n = n

    # we need to define the abstract compute method. It has to return an instance
    # of JobResult base class
    def compute(self):

        p = self.p
        data_source = self.data_source 
        r = self.rep
        n = self.n
        job_func = self.job_func
        data = data_source.sample(n, seed=r)
        with util.ContextTimer() as t:
            tr, te = data.split_tr_te(tr_proportion=tr_proportion, seed=r+21 )
            prob_label = self.prob_label
            logger.info("computing. %s. prob=%s, r=%d,\
                    n=%d"%(job_func.__name__, prob_label, r, n))

            job_result = job_func(p, data_source, tr, te, r)

            # create ScalarResult instance
            result = SingleResult(job_result)
            # submit the result to my own aggregator
            self.aggregator.submit_result(result)
            func_name = job_func.__name__
        logger.info("done. ex2: %s, prob=%s, r=%d, n=%d. Took: %.3g s "%(func_name,
            prob_label, r, n, t.secs))

        # save result
        fname = '%s-%s-n%d_r%d_a%.3f_trp%.2f.p' \
                %(prob_label, func_name, n, r, alpha, tr_proportion)
        glo.ex_save_result(ex, job_result, prob_label, fname)

# This import is needed so that pickle knows about the class Ex1Job.
# pickle is used when collecting the results from the submitted jobs.
from kgof.ex.ex1_vary_n import Ex1Job
from kgof.ex.ex1_vary_n import job_fssdJ1q_med
from kgof.ex.ex1_vary_n import job_fssdJ5q_med
from kgof.ex.ex1_vary_n import job_fssdJ1q_opt
from kgof.ex.ex1_vary_n import job_fssdJ5q_opt
from kgof.ex.ex1_vary_n import job_fssdJ10q_opt
from kgof.ex.ex1_vary_n import job_fssdJ1q_imq_optv
from kgof.ex.ex1_vary_n import job_fssdJ5q_imq_optv
from kgof.ex.ex1_vary_n import job_me_opt
from kgof.ex.ex1_vary_n import job_kstein_med
from kgof.ex.ex1_vary_n import job_kstein_imq
from kgof.ex.ex1_vary_n import job_lin_kstein_med
from kgof.ex.ex1_vary_n import job_mmd_med
from kgof.ex.ex1_vary_n import job_mmd_opt
from kgof.ex.ex1_vary_n import job_mmd_dgauss_opt


#--- experimental setting -----
ex = 1

# significance level of the test
alpha = 0.05

# Proportion of training sample relative to the full sample size n
tr_proportion = 0.2

# repetitions for each sample size 
reps = 200

# tests to try
method_job_funcs = [ 
    job_fssdJ5q_opt, 
    #job_fssdJ5q_med, 
    #job_me_opt,
    #job_kstein_med,
    #job_lin_kstein_med,
    job_mmd_opt,

    #job_fssdJ5q_imq_optv,
    #job_fssdJ10q_opt,
    #job_kstein_imq,
    #job_mmd_dgauss_opt,

    #job_mmd_med,
   ]

# If is_rerun==False, do not rerun the experiment if a result file for the current
# setting of (ni, r) already exists.
is_rerun = False
#---------------------------


def gbrbm_perturb(var_perturb_B, dx=50, dh=10):
    """
    Get a Gaussian-Bernoulli RBM problem where the first entry of the B matrix
    (the matrix linking the latent and the observation) is perturbed.

    - var_perturb_B: Gaussian noise variance for perturbing B.
    - dx: observed dimension
    - dh: latent dimension

    Return p (density), data source
    """
    with util.NumpySeedContext(seed=10):
        B = np.random.randint(0, 2, (dx, dh))*2 - 1.0
        b = np.random.randn(dx)
        c = np.random.randn(dh)
        p = density.GaussBernRBM(B, b, c)

        B_perturb = np.copy(B)
        if var_perturb_B > 1e-7:
            B_perturb[0, 0] = B_perturb[0, 0] + \
                np.random.randn(1)*np.sqrt(var_perturb_B)
        ds = data.DSGaussBernRBM(B_perturb, b, c, burnin=2000)

    return p, ds

def get_ns_pqsource(prob_label):
    """
    Return (ns, p, ds), a tuple of
    where 
    - ns: a list of sample sizes
    - p: a Density representing the distribution p
    - ds: a DataSource, each corresponding to one parameter setting.
        The DataSource generates sample from q.
    """
    gmd_p01_d10_ns = [1000, 3000, 5000]

    #gb_rbm_dx50_dh10_vars = [0, 1e-3, 2e-3, 3e-3]
    prob2tuples = { 

            # vary d. P = N(0, I), Q = N( (c,..0), I)
            'gmd_p03_d10_ns': (gmd_p01_d10_ns,
                density.IsotropicNormal(np.zeros(10), 1),
                data.DSIsotropicNormal(np.hstack((0.03, np.zeros(10-1))), 1) 
                ),

            # Gaussian Bernoulli RBM. dx=50, dh=10 
            # Perturbation variance to B[0, 0] is 0.1
            'gbrbm_dx50_dh10_vp1': 
                ([i*1000 for i in range(1, 4+1)], ) + 
                #([1000, 5000], ) + 
                gbrbm_perturb(var_perturb_B=0.1, dx=50, dh=10), 

            # Gaussian Bernoulli RBM. dx=50, dh=40 
            # Perturbation variance to B[0, 0] is 0.1
            'gbrbm_dx50_dh40_vp1': 
                ([i*1000 for i in range(1, 4+1)], ) + 
                #([1000, 5000], ) + 
                gbrbm_perturb(var_perturb_B=0.1, dx=50, dh=40), 

            # Gaussian Bernoulli RBM. dx=50, dh=10 
            # No perturbation
            'gbrbm_dx50_dh10_h0': 
                ([i*1000 for i in range(1, 4+1)], ) + 
                #([1000, 5000], ) + 
                gbrbm_perturb(var_perturb_B=0, dx=50, dh=10), 

            # Gaussian Bernoulli RBM. dx=50, dh=40 
            # No perturbation
            'gbrbm_dx50_dh40_h0': 
                ([i*1000 for i in range(1, 4+1)], ) + 
                #([1000, 5000], ) + 
                gbrbm_perturb(var_perturb_B=0, dx=50, dh=40), 

            # Gaussian Bernoulli RBM. dx=20, dh=10 
            # Perturbation variance to B[0, 0] is 0.1
            'gbrbm_dx20_dh10_vp1': 
                ([i*1000 for i in range(2, 5+1)], ) + 
                gbrbm_perturb(var_perturb_B=0.1, dx=20, dh=10), 

            # Gaussian Bernoulli RBM. dx=20, dh=10 
            # No perturbation
            'gbrbm_dx20_dh10_h0': 
                ([i*1000 for i in range(2, 5+1)], ) + 
                gbrbm_perturb(var_perturb_B=0, dx=20, dh=10), 


            }
    if prob_label not in prob2tuples:
        raise ValueError('Unknown problem label. Need to be one of %s'%str(prob2tuples.keys()) )
    return prob2tuples[prob_label]

def run_problem(prob_label):
    """Run the experiment"""
    ns, p, ds = get_ns_pqsource(prob_label)
    # ///////  submit jobs //////////
    # create folder name string
    #result_folder = glo.result_folder()
    from kgof.config import expr_configs
    tmp_dir = expr_configs['scratch_path']
    foldername = os.path.join(tmp_dir, 'kgof_slurm', 'e%d'%ex)
    logger.info("Setting engine folder to %s" % foldername)

    # create parameter instance that is needed for any batch computation engine
    logger.info("Creating batch parameter instance")
    batch_parameters = BatchClusterParameters(
        foldername=foldername, job_name_base="e%d_"%ex, parameter_prefix="")

    # Use the following line if Slurm queue is not used.
    #engine = SerialComputationEngine()
    #engine = SlurmComputationEngine(batch_parameters, partition='wrkstn,compute')
    engine = SlurmComputationEngine(batch_parameters)
    n_methods = len(method_job_funcs)
    # repetitions x len(ns) x #methods
    aggregators = np.empty((reps, len(ns), n_methods ), dtype=object)
    for r in range(reps):
        for ni, n in enumerate(ns):
            for mi, f in enumerate(method_job_funcs):
                # name used to save the result
                func_name = f.__name__
                fname = '%s-%s-n%d_r%d_a%.3f_trp%.2f.p' \
                        %(prob_label, func_name, n, r, alpha, tr_proportion)
                if not is_rerun and glo.ex_file_exists(ex, prob_label, fname):
                    logger.info('%s exists. Load and return.'%fname)
                    job_result = glo.ex_load_result(ex, prob_label, fname)

                    sra = SingleResultAggregator()
                    sra.submit_result(SingleResult(job_result))
                    aggregators[r, ni, mi] = sra
                else:
                    # result not exists or rerun

                    # p: an UnnormalizedDensity object
                    job = Ex1Job(SingleResultAggregator(), p, ds, prob_label,
                            r, f, n)
                    agg = engine.submit_job(job)
                    aggregators[r, ni, mi] = agg

    # let the engine finish its business
    logger.info("Wait for all call in engine")
    engine.wait_for_all()

    # ////// collect the results ///////////
    logger.info("Collecting results")
    job_results = np.empty((reps, len(ns), n_methods), dtype=object)
    for r in range(reps):
        for ni, n in enumerate(ns):
            for mi, f in enumerate(method_job_funcs):
                logger.info("Collecting result (%s, r=%d, n=%rd)" %
                        (f.__name__, r, n))
                # let the aggregator finalize things
                aggregators[r, ni, mi].finalize()

                # aggregators[i].get_final_result() returns a SingleResult instance,
                # which we need to extract the actual result
                job_result = aggregators[r, ni, mi].get_final_result().result
                job_results[r, ni, mi] = job_result

    #func_names = [f.__name__ for f in method_job_funcs]
    #func2labels = exglobal.get_func2label_map()
    #method_labels = [func2labels[f] for f in func_names if f in func2labels]

    # save results 
    results = {'job_results': job_results, 'data_source': ds, 
            'alpha': alpha, 'repeats': reps, 'ns': ns,
            'p': p,
            'tr_proportion': tr_proportion,
            'method_job_funcs': method_job_funcs, 'prob_label': prob_label,
            }
    
    # class name 
    fname = 'ex%d-%s-me%d_rs%d_nmi%d_nma%d_a%.3f_trp%.2f.p' \
        %(ex, prob_label, n_methods, reps, min(ns), max(ns), alpha,
                tr_proportion)

    glo.ex_save_result(ex, results, fname)
    logger.info('Saved aggregated results to %s'%fname)


def main():
    if len(sys.argv) != 2:
        print('Usage: %s problem_label'%sys.argv[0])
        sys.exit(1)
    prob_label = sys.argv[1]
    run_problem(prob_label)

if __name__ == '__main__':
    main()

