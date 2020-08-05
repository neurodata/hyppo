# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from hyppo.independence import Dcorr
from scipy.stats import multiscale_graphcorr
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import sys

# %%
def simulate_2sample_null(n1, n2, d, group_structure, label_structure, std=1, seed=None):
    """
    n1,n2 : number of groups in each of the 2 samples
    m1,m2 : size of groups in each sample. If group_structure is 'within', must be
            a length 2 tuple giving the number of observations in correpsondingly indexed sample
            and the number of observations in the other sample, respectively.
    d : dimension of observations
    group_structure : one of {'similar', 'distant'}
    label_structure : one of {'within', 'across'}
    std : standard deviation of normal distribution for each group
    """
    np.random.seed(seed)
    
    # Means for each observation
    mus = np.random.normal(0,std,(n1+n2,d))
    
    # Paired observations
    if group_structure == 'similar':
        X = np.vstack((
            np.random.normal(mus,0.1,(n1+n2,d)),
            np.random.normal(mus,0.1,(n1+n2,d))
        ))
    elif group_structure == 'distant':
        X = np.vstack((mus, -mus))
    else:
        raise ValueError(f'Invalid group_structure: {group_structure}')

    if label_structure == 'across':
        Y = np.hstack([[0]*n1, [1]*n2]*2)
    elif label_structure == 'within':
        Y = np.hstack([[0]*n1, [1]*n2, [1]*n1, [0]*n2])
    else:
        raise ValueError(f'Invalid label_structure: {label_structure}')
        
    groups = np.hstack(
        [[f'{i}'] for i in range(n1+n2)]*2
    )

    return X, Y, groups

# %%
n1 = 25
n2 = 25
d = 50
n_runs = 500
n_perms = 200

test_labels = ['Restricted, group centering', 'Restricted, group mask', 'Restricted', 'Unrestricted']
test_name = 'MGC'
test_params = ['mgc_groups', 'mgc_restricted', 'mgc', 'mgc']
# test_name = 'Dcorr
# test_params = [{'groups':True}, {'groups':False}, {'groups':None}, {'groups':None}]
group_structures = ['similar', 'distant']
label_structures = ['within', 'across']

run_list = [
    (test_label, test_param, group_structure, label_structure, iterate) \
            for (test_label, test_param) in zip(test_labels, test_params) \
            for group_structure in group_structures \
            for label_structure in label_structures \
            for iterate in range(1, n_runs + 1)
]

result_dict = defaultdict(list)

for test_label, test_param, group_structure, label_structure, iterate in run_list:
    X, Y, groups = simulate_2sample_null(n1, n2, d, group_structure, label_structure, std=1, seed=iterate)
    result_key = (test_label, group_structure, label_structure)
    
    if test_label == 'Unrestricted':
        label_structure = None
        groups = None
    if test_name == 'Dcorr':
        _,pval = Dcorr(**test_param).test(
            X,Y,
            reps=n_perms,
            workers=-1,
            permute_groups=groups,
            permute_structure=label_structure,
        )
    elif test_name == 'MGC':
        _,pval,_ = multiscale_graphcorr(
            X,Y,
            reps=n_perms,
            workers=-1,
            random_state=1,
            permute_groups=groups,
            permute_structure=label_structure,
            global_corr=test_param
        )
    else:
        print('Test name invalid')
        sys.exit(0)

    result_dict[result_key].append(pval)

with open(f"/home/rflperry/hyppo/benchmarks/restricted_perm_centering/{test_name}_4_centering_tests.pkl", "wb") as f:
    pickle.dump(result_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
