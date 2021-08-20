"""Module containing convenient functions for plotting"""

from builtins import range
from builtins import object
__author__ = 'wittawat'

import kgof.glo as glo
import matplotlib
import matplotlib.pyplot as plt
import autograd.numpy as np


def get_func_tuples():
    """
    Return a list of tuples where each tuple is of the form
        (func_name used in the experiments, label name, plot line style)
    """
    func_tuples = [
            ('job_fssdJ1q_med', 'FSSD-rand J1', 'r--^'),
            ('job_fssdJ5q_med', 'FSSD-rand', 'r--^'),
            ('job_fssdq_med', 'FSSD-rand', 'r--^'),

            ('job_fssdJ1q_opt', 'FSSD-opt J1', 'r-s'),
            ('job_fssdq_opt', 'FSSD-opt', 'r-s'),
            ('job_fssdJ5q_opt', 'FSSD-opt', 'r-s'),
            ('job_fssdJ5q_imq_optv', 'FSSD-IMQv', 'k-h'),
            ('job_fssdJ5q_imqb1_optv', 'FSSD-IMQ-1', 'k--s'),
            ('job_fssdJ5q_imqb2_optv', 'FSSD-IMQ-2', 'k-->'),
            ('job_fssdJ5q_imqb3_optv', 'FSSD-IMQ-3', 'k-.*'),
            ('job_fssdJ5q_imq_opt', 'FSSD-IMQ', 'y-x'),
            ('job_fssdJ5q_imq_optbv', 'FSSD-IMQ-bv', 'y--d'),
            ('job_fssdJ10q_opt', 'FSSD-opt', 'k-s'),

            ('job_fssdJ5p_opt', 'FSSD-opt J5', 'm-s'),
            ('job_fssdp_opt', 'FSSDp-opt', 'm-s'),
            ('job_fssdJ10p_opt', 'FSSDp-opt J10', 'k-s'),

            ('job_fssdJ1q_opt2', 'FSSD-opt2 J1', 'b-^'),
            ('job_fssdJ5q_opt2', 'FSSD-opt2 J5', 'r-^'),
            ('job_me_opt', 'ME-opt', 'b-d'),

            ('job_kstein_med', 'KSD', 'g-o'),
            ('job_kstein_imq', 'KSD-IMQ', 'c-*'),
            ('job_lin_kstein_med', 'LKS', 'g-.h'),
            ('job_mmd_med', 'MMD', 'm--^'),
            ('job_mmd_opt', 'MMD-opt', 'm-<'),
            ('job_mmd_dgauss_opt', 'MMD-dopt', 'y-<'),
            ]
    return func_tuples

def set_default_matplotlib_options():
    # font options
    font = {
    #     'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 30
    }
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


    # matplotlib.use('cairo')
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.usetex'] = True
    plt.rc('font', **font)
    plt.rc('lines', linewidth=3, markersize=10)
    # matplotlib.rcParams['ps.useafm'] = True
    # matplotlib.rcParams['pdf.use14corefonts'] = True

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

def get_func2label_map():
    # map: job_func_name |-> plot label
    func_tuples = get_func_tuples()
    #M = {k:v for (k,v) in zip(func_names, labels)}
    M = {k:v for (k,v,_) in func_tuples}
    return M


def func_plot_fmt_map():
    """
    Return a map from job function names to matplotlib plot styles 
    """
    # line_styles = ['o-', 'x-',  '*-', '-_', 'D-', 'h-', '+-', 's-', 'v-', 
    #               ',-', '1-']
    func_tuples = get_func_tuples()
    M = {k:v for (k, _, v) in func_tuples}
    return M


class PlotValues(object):
    """
    An object encapsulating values of a plot where there are many curves, 
    each corresponding to one method, with common x-axis values.
    """
    def __init__(self, xvalues, methods, plot_matrix):
        """
        xvalues: 1d numpy array of x-axis values
        methods: a list of method names
        plot_matrix: len(methods) x len(xvalues) 2d numpy array containing 
            values that can be used to plot
        """
        self.xvalues = xvalues
        self.methods = methods
        self.plot_matrix = plot_matrix

    def ascii_table(self, tablefmt="pipe"):
        """
        Return an ASCII string representation of the table.

        tablefmt: "plain", "fancy_grid", "grid", "simple" might be useful.
        """
        methods = self.methods
        xvalues = self.xvalues
        plot_matrix = self.plot_matrix

        import tabulate
        # https://pypi.python.org/pypi/tabulate
        aug_table = np.hstack((np.array(methods)[:, np.newaxis], plot_matrix))
        return tabulate.tabulate(aug_table, xvalues, tablefmt=tablefmt)

# end of class PlotValues

def plot_prob_reject(ex, fname, func_xvalues, xlabel, func_title=None, 
        return_plot_values=False):
    """
    plot the empirical probability that the statistic is above the threshold.
    This can be interpreted as type-1 error (when H0 is true) or test power 
    (when H1 is true). The plot is against the specified x-axis.

    - ex: experiment number 
    - fname: file name of the aggregated result
    - func_xvalues: function taking aggregated results dictionary and return the values 
        to be used for the x-axis values.            
    - xlabel: label of the x-axis. 
    - func_title: a function: results dictionary -> title of the plot
    - return_plot_values: if true, also return a PlotValues as the second
      output value.

    Return loaded results
    """
    #from IPython.core.debugger import Tracer 
    #Tracer()()

    results = glo.ex_load_result(ex, fname)

    def rej_accessor(jr):
        rej = jr['test_result']['h0_rejected']
        # When used with vectorize(), making the value float will make the resulting 
        # numpy array to be of float. nan values can be stored.
        return float(rej)

    #value_accessor = lambda job_results: job_results['test_result']['h0_rejected']
    vf_pval = np.vectorize(rej_accessor)
    # results['job_results'] is a dictionary: 
    # {'test_result': (dict from running perform_test(te) '...':..., }
    rejs = vf_pval(results['job_results'])
    repeats, _, n_methods = results['job_results'].shape

    # yvalues (corresponding to xvalues) x #methods
    mean_rejs = np.mean(rejs, axis=0)
    #print mean_rejs
    #std_pvals = np.std(rejs, axis=0)
    #std_pvals = np.sqrt(mean_rejs*(1.0-mean_rejs))

    xvalues = func_xvalues(results)

    #ns = np.array(results[xkey])
    #te_proportion = 1.0 - results['tr_proportion']
    #test_sizes = ns*te_proportion
    line_styles = func_plot_fmt_map()
    method_labels = get_func2label_map()
    
    func_names = [f.__name__ for f in results['method_job_funcs'] ]
    plotted_methods = []
    for i in range(n_methods):    
        te_proportion = 1.0 - results['tr_proportion']
        fmt = line_styles[func_names[i]]
        #plt.errorbar(ns*te_proportion, mean_rejs[:, i], std_pvals[:, i])
        method_label = method_labels[func_names[i]]
        plotted_methods.append(method_label)
        plt.plot(xvalues, mean_rejs[:, i], fmt, label=method_label)
    '''
    else:
        # h0 is true 
        z = stats.norm.isf( (1-confidence)/2.0)
        for i in range(n_methods):
            phat = mean_rejs[:, i]
            conf_iv = z*(phat*(1-phat)/repeats)**0.5
            #plt.errorbar(test_sizes, phat, conf_iv, fmt=line_styles[i], label=method_labels[i])
            plt.plot(test_sizes, mean_rejs[:, i], line_styles[i], label=method_labels[i])
    '''
            
    ylabel = 'Rejection rate'
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(np.hstack((xvalues) ))
    
    alpha = results['alpha']
    plt.legend(loc='best')
    title = '%s. %d trials. $\\alpha$ = %.2g.'%( results['prob_label'],
            repeats, alpha) if func_title is None else func_title(results)
    plt.title(title)
    plt.grid()
    if return_plot_values:
        return results, PlotValues(xvalues=xvalues, methods=plotted_methods,
                plot_matrix=mean_rejs.T)
    else:
        return results
        

def plot_runtime(ex, fname, func_xvalues, xlabel, func_title=None):
    results = glo.ex_load_result(ex, fname)
    value_accessor = lambda job_results: job_results['time_secs']
    vf_pval = np.vectorize(value_accessor)
    # results['job_results'] is a dictionary: 
    # {'test_result': (dict from running perform_test(te) '...':..., }
    times = vf_pval(results['job_results'])
    repeats, _, n_methods = results['job_results'].shape
    time_avg = np.mean(times, axis=0)
    time_std = np.std(times, axis=0)

    xvalues = func_xvalues(results)

    #ns = np.array(results[xkey])
    #te_proportion = 1.0 - results['tr_proportion']
    #test_sizes = ns*te_proportion
    line_styles = func_plot_fmt_map()
    method_labels = get_func2label_map()
    
    func_names = [f.__name__ for f in results['method_job_funcs'] ]
    for i in range(n_methods):    
        te_proportion = 1.0 - results['tr_proportion']
        fmt = line_styles[func_names[i]]
        #plt.errorbar(ns*te_proportion, mean_rejs[:, i], std_pvals[:, i])
        method_label = method_labels[func_names[i]]
        plt.errorbar(xvalues, time_avg[:, i], yerr=time_std[:,i], fmt=fmt,
                label=method_label)
            
    ylabel = 'Time (s)'
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xlim([np.min(xvalues), np.max(xvalues)])
    plt.xticks( xvalues, xvalues )
    plt.legend(loc='best')
    plt.gca().set_yscale('log')
    title = '%s. %d trials. '%( results['prob_label'],
            repeats ) if func_title is None else func_title(results)
    plt.title(title)
    #plt.grid()
    return results


def box_meshgrid(func, xbound, ybound, nx=50, ny=50):
    """
    Form a meshed grid (to be used with a contour plot) on a box
    specified by xbound, ybound. Evaluate the grid with [func]: (n x 2) -> n.
    
    - xbound: a tuple (xmin, xmax)
    - ybound: a tuple (ymin, ymax)
    - nx: number of points to evluate in the x direction
    
    return XX, YY, ZZ where XX is a 2D nd-array of size nx x ny
    """
    
    # form a test location grid to try 
    minx, maxx = xbound
    miny, maxy = ybound
    loc0_cands = np.linspace(minx, maxx, nx)
    loc1_cands = np.linspace(miny, maxy, ny)
    lloc0, lloc1 = np.meshgrid(loc0_cands, loc1_cands)
    # nd1 x nd0 x 2
    loc3d = np.dstack((lloc0, lloc1))
    # #candidates x 2
    all_loc2s = np.reshape(loc3d, (-1, 2) )
    # evaluate the function
    func_grid = func(all_loc2s)
    func_grid = np.reshape(func_grid, (ny, nx))
    
    assert lloc0.shape[0] == ny
    assert lloc0.shape[1] == nx
    assert np.all(lloc0.shape == lloc1.shape)
    
    return lloc0, lloc1, func_grid

def get_density_cmap():
    """
    Return a colormap for plotting the model density p.
    Red = high density 
    white = very low density.
    Varying from white (low) to red (high).
    """
    # Add completely white color to Reds colormap in Matplotlib
    list_colors = plt.cm.datad['Reds']
    list_colors = list(list_colors)
    list_colors.insert(0, (1, 1, 1))
    list_colors.insert(0, (1, 1, 1))
    lscm = matplotlib.colors.LinearSegmentedColormap.from_list("my_Reds", list_colors)
    return lscm
