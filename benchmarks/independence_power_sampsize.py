#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from benchmarks import power
from mgc.independence import *
from mgc.sims import *


# In[3]:


MIN_SAMP_SIZE = 5
MAX_SAMP_SIZE = 100
STEP_SIZE = 5


# In[4]:


simulations = {
    "linear" : linear
}

tests = {
    "CCA" : CCA,
    "Dcorr" : Dcorr,
    "HHG" : HHG,
    "Hsic" : Hsic,
    "Kendall" : Kendall,
    "Pearson" : Pearson,
    "RV" : RV,
    "Spearman" : Spearman
}


# In[5]:


def estimate_power(test, sim):
    est_power = []

    for i in tqdm(range(MIN_SAMP_SIZE, MAX_SAMP_SIZE, STEP_SIZE)):
        est_power.append(power(test, sim, n=i))
        
    return est_power


# In[7]:


est_power = [[estimate_power(test, sim) for test in tests.values()] for sim in simulations.values()]


# In[ ]:


def plot():
    tests = [
        "CCA", "Dcorr", "HHG", "Hsic",
        "Kendall", "Pearson", "RV", "Spearman"
    ]
    
    fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(28,24))
    for i, _ in enumerate(ax):
        for j, col in enumerate(row):
            sim_name = simulations[i+j]
            
            for test in tests:
                power = est_power[i+j]
                x_axis = np.arange(MIN_SAMP_SIZE, MAX_SAMP_SIZE, STEP_SIZE)
                
                col.plot(x_axis, power, label=tests)
                col.set_xticks([x_axis[0], x_axis[-1]])
                col.set_ylim(0, 1.05)
                col.set_yticks([0, 1])
                
    leg = plt.legend(bbox_to_anchor=(0.5, 0.1), bbox_transform=plt.gcf().transFigure,
                     ncol=5, loc='upper center')
    leg.get_frame().set_linewidth(0.0)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(5.0)
    plt.subplots_adjust(hspace=.75)


# In[ ]:


plot()

