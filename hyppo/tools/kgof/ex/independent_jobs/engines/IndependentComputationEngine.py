
from abc import abstractmethod

class IndependentComputationEngine(object):
    def __init__(self):
        pass
    
    @abstractmethod
    def submit_job(self, job):
        raise NotImplementedError()
    
    @abstractmethod
    def wait_for_all(self):
        raise NotImplementedError()
