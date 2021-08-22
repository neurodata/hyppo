
from nose import SkipTest
from numpy.random import randint
import os
from os.path import expanduser
import shutil
import unittest

from independent_jobs.aggregators.ScalarResultAggregator import ScalarResultAggregator
from independent_jobs.engines.BatchClusterParameters import BatchClusterParameters
from independent_jobs.engines.SlurmComputationEngine import SlurmComputationEngine
from independent_jobs.jobs.DummyJob import DummyJob
from independent_jobs.tools.FileSystem import FileSystem


class DummyComputation(object):
    def __init__(self, engine):
        self.engine = engine
    
    def go_to_bed(self, sleep_time):
        job = DummyJob(ScalarResultAggregator(), sleep_time, 
                       walltime=60 * 60, memory=1, nodes=1)
        agg = self.engine.submit_job(job)
        return agg

class DummyJobTests(unittest.TestCase):
    def engine_helper(self, engine, sleep_times):
        dc = DummyComputation(engine)
        
        aggregators = []
        num_submissions = len(sleep_times)
        for i in range(num_submissions):
            aggregators.append(dc.go_to_bed(sleep_times[i]))
            
        self.assertEqual(len(aggregators), num_submissions)
        
        engine.wait_for_all()
        
        results = []
        for i in range(num_submissions):
            aggregators[i].finalize()
            results.append(aggregators[i].get_final_result().result)
            aggregators[i].clean_up()
            
        for i in range(num_submissions):
            self.assertEqual(results[i], sleep_times[i])
        
        if engine.do_clean_up:
            for i in range(num_submissions):
                self.assertFalse(FileSystem.file_exists_new_shell(aggregators[i].filename))

    def test_slurm_engine_max_waiting_time(self):
        if not FileSystem.cmd_exists("sbatch"):
            raise SkipTest
        
        home = expanduser("~")
        folder = os.sep.join([home, "unit_test_dummy_slurm_result_max_wait"])
        
        try:
            shutil.rmtree(folder)
        except OSError:
            pass
        batch_parameters = BatchClusterParameters(foldername=folder)
        engine = SlurmComputationEngine(batch_parameters, check_interval=1)
        sleep_times = [2, -1]
        self.engine_helper(engine, sleep_times)
        
    def test_slurm_engine_clean_up(self):
        if not FileSystem.cmd_exists("sbatch"):
            raise SkipTest
        
        home = expanduser("~")
        folder = os.sep.join([home, "unit_test_slurm_dummy_result"])
        try:
            shutil.rmtree(folder)
        except OSError:
            pass
        batch_parameters = BatchClusterParameters(foldername=folder)
        engine = SlurmComputationEngine(batch_parameters, check_interval=1,
                                        do_clean_up=True)
        num_submissions = 3
        sleep_times = randint(0, 3, num_submissions)
        self.engine_helper(engine, sleep_times)

    def test_slurm_engine_no_clean_up(self):
        if not FileSystem.cmd_exists("sbatch"):
            raise SkipTest
        
        home = expanduser("~")
        folder = os.sep.join([home, "unit_test_slurm_dummy_result"])
        try:
            shutil.rmtree(folder)
        except OSError:
            pass
        batch_parameters = BatchClusterParameters(foldername=folder)
        engine = SlurmComputationEngine(batch_parameters, check_interval=1,
                                        do_clean_up=False)
        num_submissions = 3
        sleep_times = randint(0, 3, num_submissions)
        self.engine_helper(engine, sleep_times)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
