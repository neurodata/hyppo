
from independent_jobs.results.JobResult import JobResult


class SingleResult(JobResult):
    def __init__(self, result):
        JobResult.__init__(self)
        self.result = result
    
