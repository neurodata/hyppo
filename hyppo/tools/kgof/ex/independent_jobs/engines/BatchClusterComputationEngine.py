from abc import abstractmethod
from os import makedirs
import os
import pickle
from popen2 import popen2
import time

from independent_jobs.aggregators.ResultAggregatorWrapper import ResultAggregatorWrapper
from independent_jobs.engines.IndependentComputationEngine import IndependentComputationEngine
from independent_jobs.tools.FileSystem import FileSystem
from independent_jobs.tools.Log import logger
from independent_jobs.tools.Serialization import Serialization
from independent_jobs.jobs.FireAndForgetJob import FireAndForgetJob


class Dispatcher(object):
    @staticmethod
    def dispatch(filename):
        # wait until FS says that the file exists
        while not FileSystem.file_exists_new_shell(filename):
            time.sleep(1)
        
        job = Serialization.deserialize_object(filename)
        job.compute()

class BatchClusterComputationEngine(IndependentComputationEngine):
    job_filename_ending = "job.bin"
    error_filename = "error.txt"
    output_filename = "output.txt"
    batch_script_filename = "batch_script"
    aggregator_filename = "aggregator.bin"
    job_id_filename = "job_id"
    self_serialisation_fname = "serialised_engine.pkl"
    
    def __init__(self, batch_parameters, submission_cmd,
                 check_interval=10, do_clean_up=False, submission_delay=0.5,
                 max_jobs_in_queue=0):
        IndependentComputationEngine.__init__(self)
        
        self.batch_parameters = batch_parameters
        self.check_interval = check_interval
        self.do_clean_up = do_clean_up
        self.submission_cmd = submission_cmd
        self.submission_delay = submission_delay
        self.max_jobs_in_queue = max_jobs_in_queue
        # make sure submission command executable is in path
        if not FileSystem.cmd_exists(submission_cmd):
            raise ValueError("Submission command executable \"%s\" not found" % submission_cmd)
        
        # list of tuples of (job_name, submission_time), which is kept in sorted
        # order by the time, only unfinished jobs
        self.submitted_jobs = []
        
        # list of all jobs ever submitted
        self.all_jobs = []
        
        # whether to also store all aggregators in current working dir
        self.store_fire_and_forget = False
    
    def get_aggregator_filename(self, job_name):
        job_folder = self.get_job_foldername(job_name)
        return os.sep.join([job_folder, BatchClusterComputationEngine.aggregator_filename])
    
    def get_job_foldername(self, job_name):
        return os.sep.join([self.batch_parameters.foldername, job_name])
    
    def get_job_filename(self, job_name):
        return os.sep.join([self.get_job_foldername(job_name), self.job_filename_ending])
    
    @abstractmethod
    def create_batch_script(self, job_name, dispatcher_string, walltime, memory, nodes):
        raise NotImplementedError()
    
    def _get_num_unfinished_jobs(self):
        return len(self.submitted_jobs)
    
    def _insert_job_time_sorted(self, job_name, job_id):
        self.all_jobs += [job_name]
        self.submitted_jobs.append((job_name, time.time(), job_id))
        
        # sort list by second element (in place)
        self.submitted_jobs.sort(key=lambda tup: tup[1])
    
    def _get_oldest_job_in_queue(self):
        return self.submitted_jobs[0][0] if len(self.submitted_jobs) > 0 else None
    
    def _get_dispatcher_string(self, job_filename):
        lines = []
        lines.append("import socket")
        lines.append("from independent_jobs.engines.BatchClusterComputationEngine import Dispatcher")
        lines.append("from independent_jobs.tools.Log import Log, logger")
        lines.append("Log.set_loglevel(%d)" % self.batch_parameters.loglevel)
        lines.append("logger.info(\"Job running on host \" + socket.gethostname())")
        lines.append("filename=\"%s\"" % job_filename)
        lines.append("Dispatcher.dispatch(filename)")
        
        dispatcher_string = "python -c '" + os.linesep.join(lines) + "'"
        
        return dispatcher_string
    
    def submit_wrapped_pbs_job(self, wrapped_job, job_name):
        job_folder = self.get_job_foldername(job_name)
        
        # try to create folder if not yet exists
        job_filename = self.get_job_filename(job_name)
        logger.info("Creating job with file %s" % job_filename)
        try:
            makedirs(job_folder)
        except OSError:
            pass
        
        Serialization.serialize_object(wrapped_job, job_filename)
        
        # allow the queue to process things        
        time.sleep(self.submission_delay)
        
        dispatcher_string = self._get_dispatcher_string(job_filename)
        
        # get computing ressource constraints from job
        walltime, memory, nodes = wrapped_job.get_walltime_mem_nodes()
        job_string = self.create_batch_script(job_name, dispatcher_string, walltime, memory, nodes)
        
        # put the custom parameter string in front if existing
        # but not as first line to avoid problems with #/bin/bash things
        if self.batch_parameters.parameter_prefix != "":
            lines = job_string.split(os.linesep)
            job_string = os.linesep.join([lines[0],
                                                 self.batch_parameters.parameter_prefix] + lines[1:])
        
        f = open(job_folder + os.sep + BatchClusterComputationEngine.batch_script_filename, "w")
        f.write(job_string)
        f.close()
        
        job_id = self.submit_to_batch_system(job_string)
        
        if job_id == "":
            raise RuntimeError("Could not parse job_id. Something went wrong with the job submission")
        
        f = open(job_folder + os.sep + BatchClusterComputationEngine.job_id_filename, 'w')
        f.write(job_id + os.linesep)
        f.close()
        
        if not isinstance(wrapped_job, FireAndForgetJob):
            # track submitted (and unfinished) jobs and their start time
            self._insert_job_time_sorted(job_name, job_id)
    
    @abstractmethod
    def submit_to_batch_system(self, job_string):
        # send job_string to batch command
        outpipe, inpipe = popen2(self.submission_cmd)
        inpipe.write(job_string + os.linesep)
        inpipe.close()
        
        job_id = outpipe.read().strip()
        outpipe.close()
        
        return job_id
    
    def create_job_name(self):
        return FileSystem.get_unique_filename(self.batch_parameters.job_name_base)
    
    def save_all_job_list(self):
        with open(self.self_serialisation_fname, "w+") as f:
            pickle.dump(self, f)
    
    def submit_job(self, job):
        # first step: check how many jobs are there in the (internal, not cluster) queue, and if we
        # should wait for submission until this has dropped under a certain value
        if self.max_jobs_in_queue > 0 and \
           self._get_num_unfinished_jobs() >= self.max_jobs_in_queue and \
           not isinstance(job, FireAndForgetJob): # never block for fire and forget jobs
            logger.info("Reached maximum number of %d unfinished jobs in queue." % 
                        self.max_jobs_in_queue)
            self._wait_until_n_unfinished(self.max_jobs_in_queue)
        
        # save myself every few submissions (also done one wait_for_all is called)
        if len(self.all_jobs) % 100 == 0:
            self.save_all_job_list()
        
        # replace job's wrapped_aggregator by PBS wrapped_aggregator to allow
        # FS based communication
        
        # use a unique job name, but check that this folder doesnt yet exist
        job_name = self.create_job_name()
        
        aggregator_filename = self.get_aggregator_filename(job_name)
        job.aggregator = ResultAggregatorWrapper(job.aggregator,
                                                    aggregator_filename,
                                                    job_name,
                                                    self.do_clean_up,
                                                    self.store_fire_and_forget)
        
        self.submit_wrapped_pbs_job(job, job_name)
        
        return job.aggregator
    
    def _check_job_done(self, job_name):
        # race condition is fine here, but use a new python shell
        # due to NFS cache problems otherwise
        filename = self.get_aggregator_filename(job_name)
        return FileSystem.file_exists_new_shell(filename)
    
    def _get_max_wait_time_exceed_jobs(self):
        names = []
        current_time = time.time()
        for job_name, job_time, _ in self.submitted_jobs:
            # load job ressources
            job_filename = self.get_job_filename(job_name)
            job = Serialization.deserialize_object(job_filename)
            
            if abs(current_time - job_time) > job.walltime:
                names += [job_name]
        return names
    
    def _resubmit(self, job_name):
        new_job_name = self.create_job_name()
        logger.info("Re-submitting under name %s" % new_job_name)
        
        # remove from unfinished jobs list
        for i in range(len(self.submitted_jobs)):
            if self.submitted_jobs[i][0] == job_name:
                del self.submitted_jobs[i]
                break
        
        # remove from all jobs list
        for i in range(len(self.all_jobs)):
            if self.all_jobs[i] == job_name:
                del self.all_jobs[i]
                break
        
        # load job from disc and re-submit under new name
        job_filename = self.get_job_filename(job_name)
        wrapped_job = Serialization.deserialize_object(job_filename)
        self.submit_wrapped_pbs_job(wrapped_job, new_job_name)
    
    def _wait_until_n_unfinished(self, desired_num_unfinished):
        """
        Iteratively checks all non-finished jobs and updates whether they are
        finished. Blocks until there are less or exactly desired_num_unfinished
        unfinished jobs in the queue. Messages a "waiting for" info message
        for the oldest job in the queue.
        """
        
        # save all job list to file for reconstructing results later
        self.save_all_job_list()
        
        last_printed = self._get_oldest_job_in_queue()
        logger.info("Waiting for %s and %d other jobs" % (last_printed,
                                                          self._get_num_unfinished_jobs() - 1))
        while self._get_num_unfinished_jobs() > desired_num_unfinished:
            
            oldest = self._get_oldest_job_in_queue()
            if oldest != last_printed:
                last_printed = oldest
                logger.info("Waiting for %s and %d other jobs" % (last_printed,
                                                                  self._get_num_unfinished_jobs() - 1))
                
            
            # delete all finished jobs from internal list
            i = 0
            while i < len(self.submitted_jobs):
                job_name = self.submitted_jobs[i][0]
                if self._check_job_done(job_name):
                    del self.submitted_jobs[i]
                    # dont change i as it is now the index of the next element
                else:
                    i += 1
                        
            # check for re-submissions
            if self.batch_parameters.resubmit_on_timeout:
                for job_name in self._get_max_wait_time_exceed_jobs():
                    # load job ressources
                    job_filename = self.get_job_filename(job_name)
                    job = Serialization.deserialize_object(job_filename)
                    logger.info("%s exceeded maximum waiting time of %dh" 
                                % (job_name, job.walltime))
                    self._resubmit(job_name)
                    
            time.sleep(self.check_interval)

    def wait_for_all(self):
        self._wait_until_n_unfinished(0)
        logger.info("All jobs finished.")
