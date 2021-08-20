"""
Simulation to examine the P(reject) as the number of test locations
increases.  
"""
__author__ = 'wittawat'

import kgof
import kgof.data as data
import kgof.glo as glo
import kgof.density as density
import kgof.goftest as gof
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

def job_fssdq_med(p, data_source, tr, te, r, J, null_sim=None):
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



def job_fssdq_opt(p, data_source, tr, te, r, J, null_sim=None):
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

        k = kernel.KGauss(med2*2)
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
            'max_iter': 50,
            'tol_fun': 1e-4,
            'disp': True,
            'locs_bounds_frac': 10.0,
            'gwidth_lb': 1e-1,
            'gwidth_ub': 1e3,
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

def job_fssdp_opt(p, data_source, tr, te, r, J):
    """
    The suffix p means that p is sampled to get a sample for computing the
    covariance matrix under H0.
    """
    null_sim = gof.FSSDH0SimCovDraw(n_draw=2000, n_simulate=2000, seed=r)
    return job_fssdq_opt(p, data_source, tr, te, r, J, null_sim=null_sim)


# Define our custom Job, which inherits from base class IndependentJob
class Ex3Job(IndependentJob):
   
    def __init__(self, aggregator, p, data_source,
            prob_label, rep, job_func, n_locs):
        #walltime = 60*59*24 
        walltime = 60*59
        memory = int(tr_proportion*sample_size*1e-2) + 50

        IndependentJob.__init__(self, aggregator, walltime=walltime,
                               memory=memory)
        # p: an UnnormalizedDensity
        self.p = p
        self.data_source = data_source
        self.prob_label = prob_label
        self.rep = rep
        self.job_func = job_func
        self.n_locs = n_locs

    # we need to define the abstract compute method. It has to return an instance
    # of JobResult base class
    def compute(self):

        p = self.p
        data_source = self.data_source 
        r = self.rep
        n_locs = self.n_locs
        job_func = self.job_func
        # sample_size is a global variable
        data = data_source.sample(sample_size, seed=r)
        with util.ContextTimer() as t:
            tr, te = data.split_tr_te(tr_proportion=tr_proportion, seed=r+21 )
            prob_label = self.prob_label
            logger.info("computing. %s. prob=%s, r=%d,\
                    J=%d"%(job_func.__name__, prob_label, r, n_locs))

            job_result = job_func(p, data_source, tr, te, r, n_locs)

            # create ScalarResult instance
            result = SingleResult(job_result)
            # submit the result to my own aggregator
            self.aggregator.submit_result(result)
            func_name = job_func.__name__
        logger.info("done. ex2: %s, prob=%s, r=%d, J=%d. Took: %.3g s "%(func_name,
            prob_label, r, n_locs, t.secs))

        # save result
        fname = '%s-%s-n%d_r%d_J%d_a%.3f_trp%.2f.p' \
                %(prob_label, func_name, sample_size, r, n_locs, alpha,
                        tr_proportion)
        glo.ex_save_result(ex, job_result, prob_label, fname)


# This import is needed so that pickle knows about the class Ex3Job.
# pickle is used when collecting the results from the submitted jobs.
from kgof.ex.ex3_vary_nlocs import Ex3Job
from kgof.ex.ex3_vary_nlocs import job_fssdq_med
from kgof.ex.ex3_vary_nlocs import job_fssdq_opt
from kgof.ex.ex3_vary_nlocs import job_fssdp_opt

#--- experimental setting -----
ex = 3

# sample size = n (the training and test sizes are n/2)
sample_size = 500

# number of test locations / test frequencies J
alpha = 0.05
tr_proportion = 0.5
# repetitions for each parameter setting
reps = 300

# list of number of test locations/frequencies
#Js = [5, 10, 15, 20, 25]
#Js = range(2, 6+1)
#Js = [2**x for x in range(5)]
Js = [2, 8, 32, 96, 384 ]
#Js = [2, 8, 32]

method_job_funcs = [ job_fssdq_med, job_fssdq_opt, 
        #job_fssdp_opt, 
        ]

# If is_rerun==False, do not rerun the experiment if a result file for the current
# setting already exists.
is_rerun = False
#---------------------------

def gaussbern_rbm_tuple(var, dx=50, dh=10, n=sample_size):
    """
    Get a tuple of Gaussian-Bernoulli RBM problems.
    We follow the parameter settings as described in section 6 of Liu et al.,
    2016.

    - var: Gaussian noise variance for perturbing B.
    - dx: observed dimension
    - dh: latent dimension

    Return p, a DataSource
    """
    with util.NumpySeedContext(seed=1000):
        B = np.random.randint(0, 2, (dx, dh))*2 - 1.0
        b = np.random.randn(dx)
        c = np.random.randn(dh)
        p = density.GaussBernRBM(B, b, c)

        B_perturb = B + np.random.randn(dx, dh)*np.sqrt(var)
        gb_rbm = data.DSGaussBernRBM(B_perturb, b, c, burnin=50)

    return p, gb_rbm

def get_pqsource(prob_label):
    """
    Return (p, ds), a tuple of
    - p: a Density representing the distribution p
    - ds: a DataSource, each corresponding to one parameter setting.
        The DataSource generates sample from q.
    """
    prob2tuples = { 
            # H0 is true. vary d. P = Q = N(0, I)
            'sg5': (density.IsotropicNormal(np.zeros(5), 1),
                data.DSIsotropicNormal(np.zeros(5), 1) ),

            # P = N(0, I), Q = N( (0.2,..0), I)
            'gmd5': (density.IsotropicNormal(np.zeros(5), 1),
                data.DSIsotropicNormal(np.hstack((0.2, np.zeros(4))), 1) ),

            'gmd1': (density.IsotropicNormal(np.zeros(1), 1),
                data.DSIsotropicNormal(np.ones(1)*0.2, 1) ),

            # P = N(0, I), Q = N( (1,..0), I)
            'gmd100': (density.IsotropicNormal(np.zeros(100), 1),
                data.DSIsotropicNormal(np.hstack((1, np.zeros(99))), 1) ),

            # Gaussian variance difference problem. Only the variance 
            # of the first dimenion differs. d varies.
            'gvd5': (density.Normal(np.zeros(5), np.eye(5) ), 
                data.DSNormal(np.zeros(5), np.diag(np.hstack((2, np.ones(4)))) )),

            'gvd10': (density.Normal(np.zeros(10), np.eye(10) ), 
                data.DSNormal(np.zeros(10), np.diag(np.hstack((2, np.ones(9)))) )),

            # Gaussian Bernoulli RBM. dx=50, dh=10. H0 is true
            'gbrbm_dx50_dh10_v0': gaussbern_rbm_tuple(0,
                dx=50, dh=10, n=sample_size),

            # Gaussian Bernoulli RBM. dx=5, dh=3. H0 is true
            'gbrbm_dx5_dh3_v0': gaussbern_rbm_tuple(0,
                dx=5, dh=3, n=sample_size),

            # Gaussian Bernoulli RBM. dx=50, dh=10. 
            'gbrbm_dx50_dh10_v1em3': gaussbern_rbm_tuple(1e-3,
                dx=50, dh=10, n=sample_size),

            # Gaussian Bernoulli RBM. dx=5, dh=3. Perturb with noise = 1e-2.
            'gbrbm_dx5_dh3_v5em3': gaussbern_rbm_tuple(5e-3,
                dx=5, dh=3, n=sample_size),

            # Gaussian mixture of two components. Uniform mixture weights.
            # p = 0.5*N(0, 1) + 0.5*N(3, 0.01)
            # q = 0.5*N(-3, 0.01) + 0.5*N(0, 1)
            'gmm_d1': (
                density.IsoGaussianMixture(np.array([[0], [3.0]]), np.array([1, 0.01]) ),
                data.DSIsoGaussianMixture(np.array([[-3.0], [0]]), np.array([0.01, 1]) )
                ),

            # p = N(0, 1) 
            # q = 0.1*N([-10, 0,..0], 0.001) + 0.9*N([0,0,..0], 1)
            'g_vs_gmm_d5': (
                    density.IsotropicNormal(np.zeros(5), 1), 
                    data.DSIsoGaussianMixture( 
                        np.vstack(( np.hstack((0.0, np.zeros(4))), np.zeros(5) )),
                        np.array([0.0001, 1]), pmix=[0.1, 0.9] )
                    ),

            'g_vs_gmm_d2': (
                    density.IsotropicNormal(np.zeros(2), 1), 
                    data.DSIsoGaussianMixture( 
                        np.vstack(( np.hstack((0.0, np.zeros(1))), np.zeros(2) )),
                        np.array([0.01, 1]), pmix=[0.1, 0.9] )
                    ),
            'g_vs_gmm_d1': (
                    density.IsotropicNormal(np.zeros(1), 1), 
                    data.DSIsoGaussianMixture(np.array([[0.0], [0]]),
                        np.array([0.01, 1]), pmix=[0.1, 0.9] )
                    ),
            }
    if prob_label not in prob2tuples:
        raise ValueError('Unknown problem label. Need to be one of %s'%str(prob2tuples.keys()) )
    return prob2tuples[prob_label]


def run_problem(prob_label):
    """Run the experiment"""
    p, ds = get_pqsource(prob_label)

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
    engine = SlurmComputationEngine(batch_parameters)
    n_methods = len(method_job_funcs)
    # repetitions x len(Js) x #methods
    aggregators = np.empty((reps, len(Js), n_methods ), dtype=object)
    for r in range(reps):
        for ji, J in enumerate(Js):
            for mi, f in enumerate(method_job_funcs):
                # name used to save the result
                func_name = f.__name__
                fname = '%s-%s-n%d_r%d_J%d_a%.3f_trp%.2f.p' \
                        %(prob_label, func_name, sample_size, r, J, alpha,
                                tr_proportion)
                if not is_rerun and glo.ex_file_exists(ex, prob_label, fname):
                    logger.info('%s exists. Load and return.'%fname)
                    job_result = glo.ex_load_result(ex, prob_label, fname)

                    sra = SingleResultAggregator()
                    sra.submit_result(SingleResult(job_result))
                    aggregators[r, ji, mi] = sra
                else:
                    # result not exists or rerun

                    # p: an UnnormalizedDensity object
                    job = Ex3Job(SingleResultAggregator(), p, ds, prob_label,
                            r, f, J)
                    agg = engine.submit_job(job)
                    aggregators[r, ji, mi] = agg

    # let the engine finish its business
    logger.info("Wait for all call in engine")
    engine.wait_for_all()

    # ////// collect the results ///////////
    logger.info("Collecting results")
    job_results = np.empty((reps, len(Js), n_methods), dtype=object)
    for r in range(reps):
        for ji, J in enumerate(Js):
            for mi, f in enumerate(method_job_funcs):
                logger.info("Collecting result (%s, r=%d, J=%rd)" %
                        (f.__name__, r, J))
                # let the aggregator finalize things
                aggregators[r, ji, mi].finalize()

                # aggregators[i].get_final_result() returns a SingleResult instance,
                # which we need to extract the actual result
                job_result = aggregators[r, ji, mi].get_final_result().result
                job_results[r, ji, mi] = job_result

    #func_names = [f.__name__ for f in method_job_funcs]
    #func2labels = exglobal.get_func2label_map()
    #method_labels = [func2labels[f] for f in func_names if f in func2labels]

    # save results 
    results = {'job_results': job_results, 'data_source': ds, 
            'alpha': alpha, 'repeats': reps, 'Js': Js,
            'p': p,
            'tr_proportion': tr_proportion,
            'method_job_funcs': method_job_funcs, 'prob_label': prob_label,
            'sample_size': sample_size, 
            }
    
    # class name 
    fname = 'ex%d-%s-me%d_n%d_rs%d_Jmi%d_Jma%d_a%.3f_trp%.2f.p' \
        %(ex, prob_label, n_methods, sample_size, reps, min(Js),
                max(Js), alpha, tr_proportion)

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

