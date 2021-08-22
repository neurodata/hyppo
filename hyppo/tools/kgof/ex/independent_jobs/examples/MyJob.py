from abc import abstractmethod
from time import sleep

import numpy as np

from independent_jobs.jobs.IndependentJob import IndependentJob
from independent_jobs.results.ScalarResult import ScalarResult
from independent_jobs.tools.Log import logger


class MyJob(IndependentJob):
    """
    Simple minimal example job
    """
    def __init__(self, aggregator):
        IndependentJob.__init__(self, aggregator)
    
    @abstractmethod
    def compute(self):
        logger.info("computing")
        
        sleep_time = np.random.randint(10)
        logger.info("sleeping for %d seconds" % sleep_time)
        sleep(sleep_time)
        
        # the compute method submits a result object to the aggregator
        result = ScalarResult(sleep_time)
        self.aggregator.submit_result(result)
