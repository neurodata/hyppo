
from independent_jobs.engines.IndependentComputationEngine import IndependentComputationEngine


class SerialComputationEngine(IndependentComputationEngine):
    def __init__(self):
        IndependentComputationEngine.__init__(self)
    
    def submit_job(self, job):
        job.compute()
        return job.aggregator
    
    def wait_for_all(self):
        pass
