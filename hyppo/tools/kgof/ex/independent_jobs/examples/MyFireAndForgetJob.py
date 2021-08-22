from abc import abstractmethod
from time import sleep

from independent_jobs.jobs.FireAndForgetJob import FireAndForgetJob
from independent_jobs.tools.Log import logger
import numpy as np


class MyFireAndForgetJob(FireAndForgetJob):
    """
    Minimal fire and forget job that takes two parameters and returns
    a noisy sum of their squares after a random delay.
    """
    def __init__(self, db_fname, x, y, result_name="result", **param_dict):
        FireAndForgetJob.__init__(self, db_fname, result_name,
                                  x=x, y=y, **param_dict)
        
        self.x = x
        self.y = y
    
    @abstractmethod
    def compute_result(self):
        """
        Note that this method directly computes and returns the result itself.
        There is no aggregators and no result instances being passed around at
        this point.
        """
        sleep_time = np.random.randint(3)
        logger.info("sleeping for %d seconds" % sleep_time)
        sleep(sleep_time)

        return self.x**2 + self.y**2 + np.random.randn()*0.1