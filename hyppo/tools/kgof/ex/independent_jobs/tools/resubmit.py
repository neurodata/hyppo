import os

from independent_jobs.engines.BatchClusterComputationEngine import BatchClusterComputationEngine


def resubmit(job_dir, batch_engine):
    if not job_dir[-1] == os.sep:
        job_dir += os.sep
    
    fname = job_dir + "batch_script"
    with open(fname, "r") as f:
        job_string = "".join(f.readlines())
    
    # delete old output to not confuse user
    output_fname = job_dir + BatchClusterComputationEngine.output_filename
    error_fname = job_dir + BatchClusterComputationEngine.error_filename

    try:
        os.remove(output_fname)
    except Exception:
        pass
    
    try:
        os.remove(error_fname)
    except Exception:
        pass
    
    batch_engine.submit_to_batch_system(job_string)

def rebuild_batch_script(directory, job_name, engine):
    if directory[-1] != os.sep:
        directory += os.sep
    
    job_filename = directory + BatchClusterComputationEngine.job_filename_ending
    dispatcher_string = engine._get_dispatcher_string(job_filename)
    script = engine.create_batch_script(job_name, dispatcher_string)
    
    # overwrite old
    with open(directory + BatchClusterComputationEngine.batch_script_filename, "w+") as f:
        f.write(script)