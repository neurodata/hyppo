import itertools
import os
from os.path import expanduser
from random import shuffle

from independent_jobs.engines.BatchClusterParameters import BatchClusterParameters
from independent_jobs.engines.SerialComputationEngine import SerialComputationEngine
from independent_jobs.examples.MyFireAndForgetJob import MyFireAndForgetJob
from independent_jobs.jobs.FireAndForgetJob import extract_array, \
    best_parameters
from independent_jobs.tools.Log import Log
import numpy as np


def visualise_array_2d(Xs, Ys, A, samples=None, ax=None):
    import matplotlib.pyplot as plt
    """
    Simple visualisation method for illustration purposes
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    vmin = np.nanmin(A)
    vmax = np.nanmax(A)
    heatmap = ax.pcolor(Xs, Ys, A.T, cmap='gray', vmin=vmin, vmax=vmax)
    heatmap.cmap.set_under('magenta')
    
    colorbar = plt.colorbar(heatmap, ax=ax)
    colorbar.set_clim(vmin=vmin, vmax=vmax)
    
    if samples is not None:
        ax.scatter(samples[:, 0], samples[:, 1], c='r', s=1);

if __name__ == '__main__':
    """
    Example that just sends out jobs that store their result to a file when done;
    there is no control over the job after it has been submitted.
    No aggregators are stored and results can be picked up from disc when ready.
    
    This script also illustrates a typical use case in scientific computing:
    Run the same function with different parameters a certain number of times.
    
    Make sure to read the minimal example first.
    """
    Log.set_loglevel(10)

    # filename of the result database
    home = expanduser("~")
    foldername = os.path.join(home, "test")
    db_fname = os.path.join(foldername, "test.txt")
    
    batch_parameters = BatchClusterParameters(foldername=foldername)
    engine = SerialComputationEngine()
#     engine = SlurmComputationEngine(batch_parameters)
    
    # here are some example parameters for jobs
    # we here create all combinations and then shuffle them
    # this randomizes the runs over the parameter space
    params_x = np.linspace(-3, 3, num=25)
    params_y = np.linspace(-2, 2, num=12)
    all_parameters = itertools.product(params_x, params_y)
    all_parameters = list(all_parameters)
    shuffle(all_parameters)
    print "Number of parameter combinations:", len(all_parameters)
    
    for params in all_parameters[:len(all_parameters) / 300]:
        x = params[0]
        y = params[1]
        # note there are no aggregators and no result instances
        job = MyFireAndForgetJob(db_fname, result_name="my_result_name",
                                    x=x, y=y, other="some_other_parameter")
        engine.submit_job(job)
    
    # The following parts should be in a separate file
    # Furthermore, they could be executed while the above (subset of)
    # jobs are still running

    # extract and plot an array over the parameters
    # automatically find best parameters
    results, param_values = extract_array(db_fname, param_names=["x", "y"],
                  result_name="my_result_name", redux_funs=[np.nanmean])
    best_params = best_parameters(db_fname, param_names=["x", "y"], result_name="my_result_name",
                    selector=np.nanmin, redux_fun=np.nanmean)
    print "best parameters:", best_params
    
    # plot stuff
    try:
        import matplotlib.pyplot as plt
        visualise_array_2d(param_values["x"], param_values["y"], results[0])
        plt.plot(best_params[0]["x"], best_params[0]["y"], "bo", markersize=15)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
    except ImportError:
        print results
