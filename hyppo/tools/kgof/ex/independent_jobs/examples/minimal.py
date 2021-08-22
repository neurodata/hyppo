import os
from os.path import expanduser

from independent_jobs.aggregators.ScalarResultAggregator import ScalarResultAggregator
from independent_jobs.engines.BatchClusterParameters import BatchClusterParameters
from independent_jobs.engines.SerialComputationEngine import SerialComputationEngine
from independent_jobs.engines.SlurmComputationEngine import SlurmComputationEngine
from independent_jobs.examples.MyJob import MyJob
from independent_jobs.tools.Log import Log
from independent_jobs.tools.Log import logger
import numpy as np


if __name__ == '__main__':
    """
    Simple example that shows a minimal job submission, where at runtime of
    this script, we can collect results from the cluster and potentially
    submit more jobs.
    """
    Log.set_loglevel(10)
    
    # oflder for all job files
    home = expanduser("~")
    foldername = os.sep.join([home, "minimal_example"])
    
    # parameters for the cluster (folder, name, etcI
    batch_parameters = BatchClusterParameters(foldername=foldername)
    
    # engine is the objects that jobs are submitted to
    # there are implementations for different batch cluster systems
    # the serial one runs everything locally
    engine = SerialComputationEngine()
#     engine = SGEComputationEngine(batch_parameters)
#     engine = SlurmComputationEngine(batch_parameters)

    # On submission, the engine returns aggregators that can be
    # used to retreive results after potentially doing postprocessing
    returned_aggregators = []
    
    for i in range(3):
        job = MyJob(ScalarResultAggregator())
        agg = engine.submit_job(job)
        returned_aggregators.append(agg)
        
    # This call blocks until all jobs are finished (magic happens here)
    logger.info("Waiting for all jobs to be completed.")
    engine.wait_for_all()
    
    # now that everything is done, we can collect the results
    # and or do postprocessing
    logger.info("Collecting results")
    results = np.zeros(len(returned_aggregators))
    for i, agg in enumerate(returned_aggregators):
        # the aggregator might implement postprocessing
        agg.finalize()
        
        # aggregators[i].get_final_result() here returns a ScalarResult instance,
        # which we need to extract the number from
        results[i] = agg.get_final_result().result
    
    print "Results", results
