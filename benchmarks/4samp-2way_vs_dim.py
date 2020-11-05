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
sns.set(color_codes=True, style='white', context='talk', font_scale=1)
PALETTE = sns.color_palette("Set1")
sns.set_palette(PALETTE[3:])
np.set_printoptions(precision=3)


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

NAME = '4samp-2way_vs_dim'
MAX_EPSILON1 = 1
MAX_EPSILON2 = 1
STEP_SIZE = 0.05
EPSILON1 = 0.75
EPSILONS2 = [None]
WEIGHTS = EPSILON1
DIMENSIONS = [2, 5, 10, 25, 50, 100]
POWER_REPS = 5
REPS = 1000
n_jobs = 45
workers = 45

tests = [ # Second arg is multiway flag
    (Dcorr, True),
    (Dcorr, False),
    (Manova, False),
    (MGC, False)
]

effect_masks = [
    [0,1,0,0],
    [0,0,1,0],
    [0,1,0,1],
    [0,1,1,0],
    [0,1,1,1]
]

FONTSIZE = 12

run = False
plot = True

# In[15]:


def estimate_power(test, multiway, effect_mask):
    if test == Manova:
        ws = 1
    else:
        ws = workers
    est_power = np.array([
        np.mean([power_4samp_2way_epsweight(test, workers=ws, epsilon1=EPSILON1, effect_mask=effect_mask, epsilon2=None, reps=REPS, multiway=multiway, d=d)
            for _ in range(POWER_REPS)
        ]) 
        for d in DIMENSIONS
    ])
    if not os.path.exists(f'../benchmarks/{NAME}/'):
        os.makedirs(f'../benchmarks/{NAME}/')
    np.savetxt('../benchmarks/{}/{}_{}_{}.csv'.format(NAME, multiway, test.__name__, "".join([str(em) for em in effect_mask])),
               est_power, delimiter=',')
    
    return est_power


# In[16]:


if run:
    outputs = Parallel(n_jobs=n_jobs, verbose=100)(
        [delayed(estimate_power)(test, multiway, effect_mask)
            for test, multiway in tests
            for effect_mask in effect_masks]
    )


# In[5]:


FONTSIZE = 12

def plot_power():
    fig, ax = plt.subplots(nrows=2, ncols=len(effect_masks), figsize=(16,10))
    
    sim_title = [
        f"Four Gaussians {''.join([str(em) for em in effect_mask])}" for effect_mask in effect_masks
    ]
    
    ax = np.array([ax]).reshape((2,-1))
    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            if i == 0:
                sims = gaussian_4samp_2way(100, epsilon1=4, epsilon2=None, effect_mask=effect_masks[j])
                
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
                        '../benchmarks/{}/{}_{}_{}.csv'.format(
                            NAME,
                            multiway,
                            test.__name__,
                            "".join([str(em) for em in effect_masks[j]])
                            ),
                        delimiter=','
                        )

                    custom_color = {
                        "Dcorr" : "#377eb8",
                        "Hsic" : "#4daf4a",
                        "MGC" : "#e41a1c",
                    }
                    if multiway:
                            label = f'Multiway {test.__name__}'
                            ls = '--'
                    else:
                        label = f'{test.__name__}'
                        ls = '-'
                    if test.__name__ in custom_color.keys():
                        col.plot(DIMENSIONS, power, custom_color[test.__name__], label=label, ls=ls, lw=3)
                    else:
                        col.plot(DIMENSIONS, power, label=label, lw=2, ls=ls)
                    col.tick_params(labelsize=FONTSIZE)
                    col.set_xticks([DIMENSIONS[0], DIMENSIONS[-1]])
                    col.set_ylim(0, 1.05)
                    col.set_yticks([])
                    if j == 0:
                        col.set_yticks([0, 1])
    
    fig.text(0.5, 0.05, 'Cluster Separation', ha='center', fontsize=FONTSIZE)
#     fig.text(0.75, 0, 'Increasing Weight', ha='center')
    fig.text(0.1, 0.3, 'Power', va='center', rotation='vertical', fontsize=FONTSIZE)
    fig.text(0.1, 0.7, 'Scatter Plots', va='center', rotation='vertical', fontsize=FONTSIZE)
    
    leg = plt.legend(bbox_to_anchor=(0.97, 0.45), bbox_transform=plt.gcf().transFigure,
                     ncol=1, loc='upper center', fontsize=FONTSIZE)
    leg.get_frame().set_linewidth(0.0)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(5.0)
    plt.subplots_adjust(hspace=.20)
    leg = Legend(fig, scatters, ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'], loc='upper left', frameon=False, ncol=1,
                bbox_transform=plt.gcf().transFigure, bbox_to_anchor=(0.9, 0.9), fontsize=FONTSIZE)
    fig.add_artist(leg);
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3)
    plt.savefig(f'../benchmarks/figs/{NAME}.pdf', transparent=True, bbox_inches='tight')


# In[6]:

if plot:
    plot_power()
