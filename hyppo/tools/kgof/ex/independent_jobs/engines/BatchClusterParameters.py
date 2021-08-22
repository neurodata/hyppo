import logging
from tempfile import mkdtemp

from independent_jobs.tools.Log import logger


class BatchClusterParameters(object):
    def __init__(self, foldername=None, job_name_base="job_", \
                 loglevel=logging.INFO,
                 parameter_prefix="", resubmit_on_timeout=True):
        
        if foldername is None:
            foldername = mkdtemp()
            logger.debug("Creating temp directory for batch job: %s" % foldername)

        self.foldername = foldername
        self.job_name_base = job_name_base
        self.loglevel = loglevel
        self.parameter_prefix = parameter_prefix
        self.resubmit_on_timeout = resubmit_on_timeout
