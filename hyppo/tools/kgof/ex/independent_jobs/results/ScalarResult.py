
from independent_jobs.results.JobResult import JobResult


class ScalarResult(JobResult):
    """
    Class to store a single scalar result
    """
    def __init__(self, result):
        JobResult.__init__(self)
        self.result = result
    
