# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:20:33 2021

@author: Estefany Suarez
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import itertools as itr
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

#%%
RESOLUTION = '100'
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami', f'conn_{RESOLUTION}')
INFO_DIR = os.path.join(DATA_DIR, 'info')
RAW_DIR  = os.path.join(PROJ_DIR, 'raw_results', f'res_{RESOLUTION}')

#%%
info = pd.read_csv(os.path.join(INFO_DIR, 'info.csv'))

#%%
order_labels = [
                'Chiroptera',
                'Rodentia',
                'Cetartiodactyla',
                'Carnivora',
                'Perissodactyla',
                'Primates',
               ]

#%%
def u_test(x,y):
    U, pval = stats.mannwhitneyu(x,
                                  y,
                                  alternative='two-sided',
                                  )
    U = U/(len(x)*len(y))

    return U, pval

def cohen_d_2samp(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2

    # 2 independent sample t test
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def welch_test(x,y):
    _, pval = stats.ttest_ind(x,
                              y,
                              equal_var=False,
                              alternative='two-sided'
                              )
    es = cohen_d_2samp(x,y)

    return es, pval

def get_effect_size(distance, flag):

    dist = distance.copy()

    order_pairs = list(itr.combinations(info.Order[flag==1], 2))
    idx_pairs   = list(itr.combinations(range(len(info.Order[flag==1])), 2))

    # within-orders species
    within_pairs = np.array([idx_pairs[i] for i, pair in enumerate(order_pairs) if pair[0] == pair[1]])
    i_within,j_within = zip(*within_pairs)
    within = dist[i_within,j_within]

    within_label = np.array(['within' for _ in range(len(within))])
    df_within = pd.DataFrame(np.column_stack([within,within_label]), columns=['distance','label'] )

    # between-orders species
    between_pairs = np.array([idx_pairs[i] for i, pair in enumerate(order_pairs) if pair[0] != pair[1]])
    i_between,j_between = zip(*between_pairs)
    between = dist[i_between, j_between]

    between_label = np.array(['between' for _ in range(len(between))])
    df_between = pd.DataFrame(np.column_stack([between,between_label]), columns=['distance','label'] )

    df = pd.concat([df_within, df_between])
    df['distance'] = df['distance'].astype(float)
    df['distance'] = (df['distance']-np.min(df['distance']))/(np.max(df['distance'])-np.min(df['distance']))

    # ----------- statistical significance and effect size
    es_welch, pval_welch = welch_test(df_between['distance'].astype(float).values, df_within['distance'].astype(float).values)
    es_u, pval_u = u_test(df_between['distance'].astype(float).values, df_within['distance'].astype(float).values)

    return es_welch, es_u


#%%
def get_order_flag():
    return np.array([1 if label in order_labels else 0 for label in info.Order])

def get_name_flag():
    name_flag = np.zeros_like(info.Id, dtype=int)
    for name in np.unique(info.Name):
        idx_name = np.where(info.Name == name)[0]
        name_flag[idx_name[0]] = 1
    return name_flag

    #%%
distances = [
            'spectral_distance',
            'topological_distance',
            ]

for distance in distances:
    
    print(f'\n----------- average {distance} ------------')
    distance = f'avg_{distance}'
    avg_dist = np.load(os.path.join(RAW_DIR, f'{distance}.npy'))
    order_flag = get_order_flag()
    name_flag  = get_name_flag()
    flag = np.logical_and(order_flag, name_flag)
    es_welch, es_u = get_effect_size(avg_dist, flag)

    # print(f'\n----------- {distance} ------------')
    # dist = np.load(os.path.join(RAW_DIR, f'{distance}.npy'))
    # flag = get_order_flag()
    # es_welch, es_u = get_effect_size(dist[np.ix_(flag==1,flag==1)], flag)
    
    es_welch_nulls = np.load(os.path.join(RAW_DIR, 'label_perm', f'label_perm_effect_size_welch_{distance}.npy'))
    mean_welch = np.mean(np.hstack((es_welch_nulls.copy(), es_welch.copy())))
    std_welch = np.std(np.hstack((es_welch_nulls.copy(), es_welch.copy())))
    
    es_welch = (es_welch-mean_welch)/std_welch
    es_welch_nulls = (es_welch_nulls-mean_welch)/std_welch
    pval_welch = (np.count_nonzero(np.abs(es_welch) <= np.abs(es_welch_nulls))+1)/(len(es_welch_nulls)+1)


    es_u_nulls = np.load(os.path.join(RAW_DIR, 'label_perm', f'label_perm_effect_size_u_{distance}.npy'))
    mean_u = np.mean(np.hstack((es_u_nulls.copy(), es_u.copy())))
    std_u = np.std(np.hstack((es_u_nulls.copy(), es_u.copy())))

    es_u = (es_u-mean_u)/std_u  
    es_u_nulls = (es_u_nulls-mean_u)/std_u
    pval_u = (np.count_nonzero(np.abs(es_u) <= np.abs(es_u_nulls))+1)/(len(es_u_nulls)+1)
    
    
    # figure
    sns.set(style="ticks", font_scale=2.0)
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    ax = ax.ravel()
    
    # Welch Student's T test
    sns.kdeplot(x=es_welch_nulls,
                  fill=True,
                  ax=ax[0],
                  label='null',
                  color='darkcyan'
                  )
    
    ax[0].axvline(x=es_welch, color='dodgerblue', ls='--', label=f'empirical pval={np.round(pval_welch,3)}')
    
    # ax[0].set_xlim(-10,30)
    # ax[0].legend()
    ax[0].set_xlabel('effect size - Welch T-test')
    
    # Mann–Whitney U test
    sns.kdeplot(x=es_u_nulls,
                  fill=True,
                  ax=ax[1],
                  label='null',
                  color='darkcyan'
                  )
    
    ax[1].axvline(x=es_u, color='dodgerblue', ls='--', label=f'empirical pval={np.round(pval_u,3)}')

    # ax[1].set_xlim(-10,30)
    # ax[1].legend()
    ax[1].set_xlabel('effect size - Mann-Whitney U test')
     
    sns.despine(offset=10, trim=False)
    # fig.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs', f'label_permutation_{distance}.eps'), transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

