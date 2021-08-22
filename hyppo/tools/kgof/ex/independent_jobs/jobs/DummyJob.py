import numpy as np
from time import sleep

from independent_jobs.jobs.IndependentJob import IndependentJob
from independent_jobs.results.ScalarResult import ScalarResult
from independent_jobs.tools.Log import logger


class DummyJob(IndependentJob):
    def __init__(self, aggregator, sleep_time, walltime=1, memory=1, nodes=1):
        IndependentJob.__init__(self, aggregator, walltime, memory, nodes)
        self.sleep_time = sleep_time
    
    def compute(self):
        result = ScalarResult(self.sleep_time)
        
        if self.sleep_time >= 0:
            sleep_time = self.sleep_time
        else:
            sleep_time = np.random.randint(10)
            
        logger.info("Sleeping for %d" % sleep_time)
        sleep(sleep_time)
            
        self.aggregator.submit_result(result)
