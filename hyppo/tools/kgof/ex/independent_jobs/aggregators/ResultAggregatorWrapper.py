
import os
from pickle import dump, load

from independent_jobs.aggregators.JobResultAggregator import JobResultAggregator
from independent_jobs.tools.FileSystem import FileSystem


class ResultAggregatorWrapper(JobResultAggregator):
    def __init__(self, wrapped_aggregator, filename, job_name, do_clean_up = False, store_fire_and_forget=False):
        self.wrapped_aggregator = wrapped_aggregator
        self.filename = filename
        self.job_name = job_name
        
        # to keep track of all submitted results
        self.result_counter = 0
        
        # whether to delete job output
        self.do_clean_up = do_clean_up
        
        # whether to store all aggregators in the current working dir
        self.store_fire_and_forget = store_fire_and_forget
    
    def submit_result(self, result):
        # NOTE: this happens on the PBS

        # pass on result to wrapper wrapped_aggregator
        self.wrapped_aggregator.submit_result(result)
        self.result_counter += 1
        
        # if all results received, dump wrapped_aggregator to disc
        # this has to happen on the PBS
        if self.result_counter == self.wrapped_aggregator.expected_num_results:
            f = open(self.filename, 'w')
            dump(self.wrapped_aggregator, f)
            f.close()
        
            # store copy in working dir, this only happens if aggregator implements the
            # store_fire_and_forget_result method
            if self.store_fire_and_forget:
                folder = "fire_and_forget_results"
                try:
                    os.makedirs(folder)
                except OSError:
                    pass
                
                self.wrapped_aggregator.store_fire_and_forget_result(folder, self.job_name)
            
        
    def finalize(self):
        # NOTE: This happens in the PBS engine, so not on the PBS
        
        # load the previously dumped aggregator to this instance, which is empty
        # since the filled one is on the PBS
        f = open(self.filename, 'r')
        self.wrapped_aggregator = load(f)
        f.close()
    
    def get_final_result(self):
        # NOTE: This happens in the PBS engine, so not on the PBS
        
        # return previously loaded finalised result
        return self.wrapped_aggregator.get_final_result()

    def clean_up(self):
        if self.do_clean_up:
            FileSystem.delete_dir_failsafe(os.path.split(self.filename)[0])
