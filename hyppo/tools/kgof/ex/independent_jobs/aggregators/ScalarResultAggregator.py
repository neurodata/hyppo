
from independent_jobs.aggregators.JobResultAggregator import JobResultAggregator

class ScalarResultAggregator(JobResultAggregator):
    def __init__(self):
        JobResultAggregator.__init__(self, 1)
    
    def finalize(self):
        pass
    
    def submit_result(self, result):
        self.result = result
    
    def get_final_result(self):
        return self.result
    
    def clean_up(self):
        pass

