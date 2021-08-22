
from abc import abstractmethod


class JobResultAggregator(object):
    def __init__(self, expected_num_results):
        self.expected_num_results = expected_num_results
    
    @abstractmethod
    def finalize(self):
        raise NotImplementedError()
    
    @abstractmethod
    def submit_result(self, result):
        raise NotImplementedError()
    
    @abstractmethod
    def get_final_result(self):
        raise NotImplementedError()
    
    @abstractmethod
    def clean_up(self):
        raise NotImplementedError()

    @abstractmethod
    def store_fire_and_forget_result(self, folder, job_name):
        pass
