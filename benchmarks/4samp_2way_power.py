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


MAX_EPSILON1 = 1
MAX_EPSILON2 = 1
STEP_SIZE = 0.05
EPSILONS1 = np.arange(0, MAX_EPSILON1 + STEP_SIZE, STEP_SIZE)
EPSILONS2 = [None]#np.arange(0, MAX_EPSILON2 + STEP_SIZE, STEP_SIZE)#
WEIGHTS = EPSILONS1
POWER_REPS = 5
REPS = 1000

tests = [
    Dcorr,
]

multiways = [
    True,
    False,
]

effect_masks = [
    [0,1,0,0],
    [0,0,1,0],
    [0,1,0,1],
    [0,1,1,0],
    [0,1,1,1]
]

FONTSIZE = 12


# In[15]:


def estimate_power(test, multiway, effect_mask):
    est_power = np.array([
        [
            np.mean([power_4samp_2way_epsweight(test, workers=-1, epsilon1=i, effect_mask=effect_mask, epsilon2=j, reps=REPS, multiway=multiway, compute_distance=None)
                for _ in range(POWER_REPS)
            ]) 
            for i in EPSILONS1
        ]
        for j in EPSILONS2
    ])
    np.savetxt('../benchmarks/4samp_2way_vs_epsilon/{}_{}_{}.csv'.format(multiway, test.__name__, "".join([str(em) for em in effect_mask])),
               est_power, delimiter=',')
    
    return est_power


# In[16]:


outputs = []
for effect_mask in effect_masks:
    output = Parallel(n_jobs=-1, verbose=100)(
        [delayed(estimate_power)(test, multiway, effect_mask) for test in tests for multiway in multiways]
    )
    outputs.append(output)


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
                for test in tests:
                    for multiway in multiways:
                        power = np.genfromtxt(
                            '../benchmarks/4samp_2way_vs_epsilon/{}_{}_{}.csv'.format(
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
                        else:
                            label = f'{test.__name__}'
                        if test.__name__ in custom_color.keys():
                            if multiway:#test.__name__ == "MGC":
                                col.plot(EPSILONS1, power, "#e41a1c", label=label, lw=3)
                            else:
                                col.plot(EPSILONS1, power, custom_color[test.__name__], label=label, ls='-', lw=3)
                        else:
                            col.plot(EPSILONS1, power, label=label, lw=2)
                        col.tick_params(labelsize=FONTSIZE)
                        col.set_xticks([EPSILONS1[0], EPSILONS1[-1]])
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
    #plt.savefig('../benchmarks/figs/4samp_power_epsilon_014.pdf', transparent=True, bbox_inches='tight')


# In[6]:


plot_power()


# In[ ]:




