from abc import abstractmethod
from distutils.version import StrictVersion
import itertools
import os
import sys
import time

from independent_jobs.aggregators.ScalarResultAggregator import ScalarResultAggregator
from independent_jobs.jobs.IndependentJob import IndependentJob
from independent_jobs.tools.Log import logger
import numpy as np
import pandas as pd

pd_version_at_least="0.19.2"
if StrictVersion(pd.__version__) < StrictVersion(pd_version_at_least):
    print "Fire and forget functionality might be incompatible with the "\
        "pandas version you are using (%s). Upgrade to at least %s to get "\
        "rid of this message." % (pd.__version__, pd_version_at_least)

def store_results(fname, **kwargs):
    # create result dir if wanted
    if os.sep in fname:
        try:
            directory = os.sep.join(fname.split(os.sep)[:-1])
            os.makedirs(directory)
        except OSError:
            pass
    
    # use current time as index for the dataframe
    current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    columns = list(kwargs.keys())
    df = pd.DataFrame([[kwargs[k] for k in columns]], index=[current_time], columns=columns)
    
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///{}'.format(fname))
    df.to_sql("FireAndForgetJob", engine, if_exists="append")
    

def extract_array(fname, param_names, result_name="result",
                  non_existing=np.nan, redux_funs=[np.nanmean], return_param_values=True,
                  conditionals={}, db_is_sqlite=False):
    """
    Given a database file (as e.g. product by FireAndForgetJob, extraxts an
    array where each dimension corresponds to a provided parameter, and
    each element is a redux (e.g. mean) of all results (of given same)
    for the parameter combinations.
    An optional set of additional conditions can be specified.
    
    Database file can be csv or sqlite.
    
    Empty parameter names lead to just aggregating (sliced by conditionals) the results.
    
    A default value can be specified.
    """
    if db_is_sqlite:
        from sqlalchemy import create_engine
        engine = create_engine('sqlite:///{}'.format(fname))
        df = pd.read_sql_table("FireAndForgetJob", engine)
    else:
        with open(fname) as f:
            df = pd.read_csv(f, error_bad_lines=False, warn_bad_lines=False)
    
    for k, v in conditionals.items():
        df = df.loc[df[k] == v]
        if k in param_names:
            param_names.remove(k)
    
    # no parameter names means just return the aggregated values for the (sliced) result
    if len(param_names) == 0:
        return np.array([redux(df[result_name]) for redux in redux_funs])
    
    param_values = {param_name: np.sort(df[param_name].dropna().unique()) for param_name in param_names}
        
    sizes = [len(param_values[param_name]) for param_name in param_names]
    results = [np.zeros(tuple(sizes)) + non_existing for _ in redux_funs]
    
    # compute aggregate for each unique appearance of all parameters
    redux = df.groupby(param_names, as_index=False)[result_name].agg(redux_funs)
    
    # since not all parameter combinations might be computed, iterate and pull out computed ones
    all_combs = itertools.product(*[param_values[param_name] for param_name in param_names])
    
    for index, comb in enumerate(all_combs):
        # one element tuples should be the value itself
        if len(comb)==1:
            comb=comb[0]
            
        result_ind = np.unravel_index(index, tuple(sizes))

        # parameter combination was computed
        if comb in redux.index:
            # extract results and put them in the right place
            for i, redux_fun in enumerate(redux_funs):
                results[i][result_ind] = redux.loc[comb][redux_fun.__name__]

    if not return_param_values:
        return results
    else:
        return results, param_values

def best_parameters(db_fname, param_names, result_name, selector=np.nanmin,
                    redux_fun=np.nanmean, conditionals={},
                    db_is_sqlite=False):
    """
    Extracts the best choice of parameters using @see extract_array
    """
    results, param_values = extract_array(db_fname,
                            result_name=result_name,
                            param_names=param_names,
                            redux_funs=[redux_fun],
                            conditionals=conditionals,
                            db_is_sqlite=db_is_sqlite)
    
    results = results[0]

    best_ind = np.unravel_index(np.nanargmin(results), results.shape)

    best_params = {}
    for i, param_name in enumerate(param_names):
        best_params[param_name] = param_values[param_name][best_ind[i]]
    
    return best_params, selector(results)

class FireAndForgetJob(IndependentJob):
    def __init__(self, db_fname, result_name="result", seed=None, **param_dict):
        IndependentJob.__init__(self, ScalarResultAggregator())
        
        self.db_fname = db_fname
        self.param_dict = param_dict
        self.result_name = result_name
        
        if seed is None:
            # if no seed is set unsigned 32bit int, and store
            seed = np.random.randint(2 ** 32 - 1)
        self.seed = seed
    
    @abstractmethod
    def compute_result(self):
        raise NotImplementedError()
    
    def compute(self):
        param_string = ", ".join(["%s=%s" % (str(k), str(v)) for k, v in self.param_dict.items()])

        logger.info("Setting numpy random seed to %d" % self.seed)
        np.random.seed(self.seed)

        logger.info("Computing result for %s" % param_string)
        start_time = time.time()
        result = self.compute_result()
        end_time = time.time()
        runtime = end_time - start_time
        
        self.store_results(result, runtime)
        self.aggregator.submit_result(result)
        
        # the engine will not call this, as it "forgets"
        self.aggregator.clean_up()
    
    def store_results(self, result, runtime):
        logger.info("Storing results in %s" % self.db_fname)
        submit_dict = {}
        for k, v in self.param_dict.items():
            submit_dict[k] = v
        submit_dict[self.result_name] = result
        submit_dict["_runtime"] = runtime
        submit_dict["_seed"] = self.seed
        store_results(self.db_fname, **submit_dict)
