# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 11:03:57 2022

@author: Estefany Suarez
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import itertools as itr
import numpy as np
import pandas as pd
from scipy import stats

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

def order_label_permutation(distance, title, flag):

    dist = distance.copy()
    order = info.Order[flag==1]

    es_welch_distribution = []
    pval_welch_distribution = []
    es_u_distribution = []
    pval_u_distribution = []
    for i in range(1000):
        
        permuted_order = order.copy().values[np.random.permutation(len(order))]
        
        order_pairs = list(itr.combinations(permuted_order, 2))
        idx_pairs   = list(itr.combinations(range(len(permuted_order)), 2))
    
        # within-orders species
        within_pairs = np.array([idx_pairs[i] for i, pair in enumerate(order_pairs) if pair[0] == pair[1]])
        i_within,j_within = zip(*within_pairs)
        
        # between-orders species
        between_pairs = np.array([idx_pairs[i] for i, pair in enumerate(order_pairs) if pair[0] != pair[1]])
        i_between,j_between = zip(*between_pairs)

        
        # within cells
        within = dist[i_within,j_within]
        within_label = np.array(['within' for _ in range(len(within))])
        df_within = pd.DataFrame(np.column_stack([within,within_label]), columns=['distance','label'] )
    
        # between cells
        between = dist[i_between, j_between]
        between_label = np.array(['between' for _ in range(len(between))])
        df_between = pd.DataFrame(np.column_stack([between,between_label]), columns=['distance','label'] )
    
        df = pd.concat([df_within, df_between])
        df['distance'] = df['distance'].astype(float)
        df['distance'] = (df['distance']-np.min(df['distance']))/(np.max(df['distance'])-np.min(df['distance']))
    
        # ----------- statistical significance and effect size
        es_welch, pval_welch = welch_test(df_between['distance'].astype(float).values, df_within['distance'].astype(float).values)
        es_welch_distribution.append(es_welch)
        pval_welch_distribution.append(pval_welch)

        es_u, pval_u = u_test(df_between['distance'].astype(float).values, df_within['distance'].astype(float).values)
        es_u_distribution.append(es_u)
        pval_u_distribution.append(pval_u)

        # # kde plot
        # within = df.loc[df['label'] == 'within',:]
        # between = df.loc[df['label'] == 'between',:]
    
        # sns.set(style="ticks", font_scale=2.0)
        # fig, ax = plt.subplots(1,1, figsize=(5,5))
        # sns.kdeplot(x=within['distance'],
        #               fill=True,
        #               ax=ax,
        #               label='within',
        #               cut=0,
        #               # clip=(0, 1.0)
        #               )
        # sns.kdeplot(x=between['distance'],
        #               fill=True,
        #               ax=ax,
        #               label='between',
        #               cut=0,
        #               # clip=(0, 1.0)
        #               )
    
        # ax.xaxis.set_major_locator(MultipleLocator(0.5))
        # ax.set_ylim(0,2.5)
        # plt.legend()
        # sns.despine(offset=10, trim=True)
        # # fig.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs', f'intra_vs_inter_{title}.eps'), transparent=True, bbox_inches='tight', dpi=300)
        # plt.show()
        # plt.close()

        # print(f'Welch t-test - effect size = {es_welch} and pval: {pval_welch}')
        # print(f'\tmean intra: {np.mean(within.distance)}')
        # print(f'\tmean inter: {np.mean(between.distance)}')
    
        # print(f'Wil. Mann. Whithn. U test - effect size = {es_u} and pval: {pval_u}')
        # print(f'\tmedian intra: {np.median(within.distance)}')
        # print(f'\tmedian inter: {np.median(between.distance)}')
    
        # print(f'\n\tvariance intra: {np.var(within.distance)}')
        # print(f'\tvariance inter: {np.var(between.distance)}')
    
    np.save(os.path.join(RAW_DIR, f'label_perm_effect_size_welch_{title}'), es_welch_distribution)
    np.save(os.path.join(RAW_DIR, f'label_perm_pvals_welch_{title}'), pval_welch_distribution)
    np.save(os.path.join(RAW_DIR, f'label_perm_effect_size_u_{title}'), es_u_distribution)
    np.save(os.path.join(RAW_DIR, f'label_perm_pvals_u_{title}'), pval_u_distribution)


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
            'top_bin_dist',
            'top_wei_dist',
            'top_local_dist',
            'top_global_dist',
            'top_local_bin_dist',
            'top_local_wei_dist',
            'top_global_bin_dist',
            'top_global_wei_dist',
            ]


for distance in distances:
    
    # print(f'\n----------- {distance} ------------')
    # dist = np.load(os.path.join(RAW_DIR, f'{distance}.npy'))
    # flag = get_order_flag()

    # order_label_permutation(dist[np.ix_(flag==1,flag==1)], distance, flag)
    
    
    print(f'\n----------- average {distance} ------------')
    avg_dist = np.load(os.path.join(RAW_DIR, f'avg_{distance}.npy'))
    order_flag = get_order_flag()
    name_flag  = get_name_flag()
    flag = np.logical_and(order_flag, name_flag)

    order_label_permutation(avg_dist, f'avg_{distance}', flag)



