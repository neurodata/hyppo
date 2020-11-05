#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm
from matplotlib.legend import Legend

import sys, os
from joblib import Parallel, delayed

sys.path.append(os.path.realpath('..'))
from benchmarks import power_4samp_2way_epsweight
from hyppo.sims import gaussian_4samp_2way
from hyppo.independence import MGC, Dcorr

import seaborn as sns
sns.set(color_codes=True, style='white', context='talk', font_scale=2)
PALETTE = sns.color_palette("Set1")
sns.set_palette(PALETTE[3:])


# In[2]:

from rpy2.robjects import Formula, numpy2ri
from rpy2.robjects.packages import importr


class Manova:
    r"""
    Wrapper of statsmodels MANOVA
    """
    def __init__(self):
        self.stats = importr('stats')
        self.r_base = importr('base')
        
        numpy2ri.activate()

        self.formula = Formula('X ~ Y')
        self.env = self.formula.environment

    def _statistic(self, x, y):
        r"""
        Helper function to calculate the test statistic
        """
        self.env['Y'] = y
        self.env['X'] = x

        stat = self.r_base.summary(self.stats.manova(self.formula), test="Pillai")[3][4]

        return stat

# In[2]:


MAX_EPSILON1 = 1
MAX_EPSILON2 = 1
STEP_SIZE = 0.05
EPSILONS1 = np.arange(0, MAX_EPSILON1 + STEP_SIZE, STEP_SIZE)
EPSILONS2 = [0,0.1,0.2,0.3]#np.arange(0, MAX_EPSILON2 + STEP_SIZE, STEP_SIZE)
WEIGHTS = EPSILONS1
POWER_REPS = 10
REPS = 1000
n_jobs = 50
workers = 50

tests = [ # Second arg is multiway flag
    # (Dcorr, True),
    # (Dcorr, False),
    # (Manova, False),
    (MGC, True),
]

diag = True

FONTSIZE = 12

run = True
plot = False


# In[71]:

def _estimate_power(test, epsilon1, epsilon2, multiway):
    return np.mean([power_4samp_2way_epsweight(
        test, workers=1, epsilon1=epsilon1, epsilon2=epsilon2,
        reps=REPS, multiway=multiway, compute_distance=None, sim_kwargs={'diag':diag})
        for _ in range(POWER_REPS)]) 


def estimate_power(test, multiway):
    est_power = np.array([
        [
            np.mean([power_4samp_2way_epsweight(test, workers=workers, epsilon1=i, epsilon2=j, reps=REPS, multiway=multiway, sim_kwargs={'diag':diag})
                for _ in range(POWER_REPS)
            ]) 
            for i in EPSILONS1
        ]
        for j in EPSILONS2
    ])
    np.savetxt('../benchmarks/4samp_2way_vs_epsilon/{}_{}_diag={}.csv'.format(multiway, test.__name__, diag),
               est_power, delimiter=',')
    
    return est_power


# In[72]:


# for test in tests:
#     for multiway in multiways:
#         est_power = Parallel(n_jobs=n_jobs, verbose=100)(
#             [delayed(_estimate_power)(test, epsilon1=i, epsilon2=j, multiway=multiway) for i in EPSILONS1 for j in EPSILONS2]
#         )

#         np.savetxt('../benchmarks/4samp_2way_vs_epsilon/{}_{}_diag={}_124.csv'.format(multiway, test.__name__, diag),
#                est_power, delimiter=',')

if run:
    outputs = Parallel(n_jobs=n_jobs, verbose=100)(
        [delayed(estimate_power)(test, multiway) for test, multiway in tests]
    )


# In[3]:


FONTSIZE = 12

def plot_power():
    fig, ax = plt.subplots(nrows=2, ncols=len(EPSILONS2), figsize=(16,8))
    
    sim_title = [
        f"Four Gaussians (off-diag {ep2})" for ep2 in EPSILONS2
    ]
    ax = np.array([ax]).reshape((2,-1))
    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            if i == 0:
                sims = gaussian_4samp_2way(100, epsilon1=4, epsilon2=EPSILONS2[j]*4, diag=True)
                
                sim_markers = [
                    "1",
                    "+",
                    "x",
                    '_',
                ]
                custom_color = [
                    "#d9d9d9",
                    "#969696",
                    "#525252",
                    "#747474",
                ]
                
                count = 0
                scatters = []
                for sim in sims:
                    x, y = np.hsplit(sim, 2)
                    scatters.append(col.scatter(x, y, marker=sim_markers[count], color=custom_color[count]))

                    #col.set_xlim(-5, 5)
                    #if case not in [2, 4]:
                    #    col.set_ylim(-5, 5)
                    col.set_xticks([])
                    col.set_yticks([])
                    col.set_title(sim_title[j], fontsize=FONTSIZE)
                    count += 1
            else:
                for test, multiway in tests:
                    power = np.genfromtxt(
                        '../benchmarks/4samp_2way_vs_epsilon/{}_{}_diag={}.csv'.format(multiway, test.__name__, diag),
                        delimiter=','
                        )

                    custom_color = {
                        "Dcorr" : "#377eb8",
                        "Hsic" : "#4daf4a",
                        "MGC" : "#e41a1c",
                    }
                    if multiway:
                            label = f'Multiway {test.__name__}'
                    else:
                        label = f'{test.__name__}'
                    if test.__name__ in custom_color.keys():
                        if multiway:
                            col.plot(EPSILONS1, power[j], custom_color["MGC"], label=label, lw=2)
                        else:
                            col.plot(EPSILONS1, power[j], custom_color[test.__name__], label=label, ls='-', lw=2)
                    else:
                        col.plot(EPSILONS1, power[j], label=label, lw=2)
                    col.tick_params(labelsize=FONTSIZE)
                    col.set_xticks([EPSILONS1[0], EPSILONS1[-1]])
                    col.set_ylim(0, 1.05)
                    col.set_yticks([])
                    if j == 0:
                        col.set_yticks([0, 1])
    
    if len(row) > 1:
        fig.text(0.5, 0.05, 'Cluster Separation', ha='center', fontsize=FONTSIZE)
    #     fig.text(0.75, 0, 'Increasing Weight', ha='center')
        fig.text(0.1, 0.3, 'Power', va='center', rotation='vertical', fontsize=FONTSIZE)
        fig.text(0.1, 0.7, 'Scatter Plots', va='center', rotation='vertical', fontsize=FONTSIZE)

        leg = plt.legend(bbox_to_anchor=(0.97, 0.45), bbox_transform=plt.gcf().transFigure,
                         ncol=1, loc='upper center', fontsize=FONTSIZE)
    else:
        fig.text(0.5, 0, 'Cluster Separation', ha='center', fontsize=FONTSIZE)
    #     fig.text(0.75, 0, 'Increasing Weight', ha='center')
        fig.text(-0.05, 0.3, 'Power', va='center', rotation='vertical', fontsize=FONTSIZE)
        fig.text(-0.05, 0.7, 'Scatter Plots', va='center', rotation='vertical', fontsize=FONTSIZE)

        leg = plt.legend(bbox_to_anchor=(1.5, 0.45), bbox_transform=plt.gcf().transFigure,
                         ncol=1, loc='upper center', fontsize=FONTSIZE)
    leg.get_frame().set_linewidth(0.0)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(5.0)
    plt.subplots_adjust(hspace=.20)
    if len(row) > 1:
        leg = Legend(fig, scatters, ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'], loc='upper left', frameon=False, ncol=1,
                bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(0.9, 0.9), fontsize=FONTSIZE)
    else:
        leg = Legend(fig, scatters, ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'], loc='upper center', frameon=False, ncol=1,
                    bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(1.3, 0.9), fontsize=FONTSIZE)
    fig.add_artist(leg);
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3)
    plt.savefig('../benchmarks/figs/4samp_power_epsilon_diag.pdf', transparent=True, bbox_inches='tight')


# In[74]:

if plot:
    plot_power()

