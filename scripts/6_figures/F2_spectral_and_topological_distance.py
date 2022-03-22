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
from matplotlib.ticker import MultipleLocator
import seaborn as sns

from netneurotools import plotting

#%%
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami', 'conn')
INFO_DIR = os.path.join(DATA_DIR, 'info')
RAW_DIR  = os.path.join(PROJ_DIR, 'raw_results')

#%%
info = pd.read_csv(os.path.join(RAW_DIR, 'info.csv'))

#%%
order_labels = {
                'Chiroptera':1,
                'Rodentia':2,
                'Artiodactyla':3,
                'Carnivora':4,
                'Perissodactyla':5,
                'Primates':6,
                }

#%% distance between orders
def fig2_pa(distance, title, flag):

    int_communities = np.array([order_labels[o] for o in info.Order[flag==1]])

    distance_ = distance.copy()
    distance_ = (distance_-np.min(distance_))/(np.max(distance_)-np.min(distance_))

    # plot
    sns.set(style="ticks", font_scale=2.0)
    ax = plotting.plot_mod_heatmap(data=distance_,
                              communities=int_communities,
                              inds=None,
                              edgecolor='white',
                              ax=None,
                              figsize=(2*6.4, 2*4.8),
                              xlabels=order_labels.keys(),
                              ylabels=order_labels.keys(),
                              xlabelrotation=90,
                              ylabelrotation=0,
                              cbar=True,
                              cmap=sns.cubehelix_palette(as_cmap=True),
                              vmin=0.0,
                              vmax=1.0,
                              rasterized=True
                              )
    # ax.figure.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs', f'{title}.eps'), transparent=True, bbox_inches='tight', dpi=300)

#%% summary distance between orders
def fig2_pb(distance, title, flag):

    distance_ = distance.copy()
    distance_ = (distance_-np.min(distance_))/(np.max(distance_)-np.min(distance_))

    avg_distance = np.zeros((len(order_labels),len(order_labels)))
    for cluster_a,cluster_b in list(itr.combinations_with_replacement(order_labels.keys(), 2)):

        idx_a = np.where(info.Order[flag==1] == cluster_a)[0]
        idx_b = np.where(info.Order[flag==1] == cluster_b)[0]

        i = order_labels[cluster_a]-1
        j = order_labels[cluster_b]-1

        avg_distance[i,j] = np.median(distance_[np.ix_(idx_a, idx_b)])
        avg_distance[j,i] = avg_distance[i,j]


    # global scaling
    avg_distance = (avg_distance-np.min(avg_distance))/(np.max(avg_distance)-np.min(avg_distance))

    mask = np.zeros_like(avg_distance).astype(bool)
    mask[np.tril_indices_from(mask, -1)] = True

    # plot
    sns.set(style="ticks", font_scale=2.0)
    fig = plt.figure(figsize=(12,10))
    ax = plt.subplot(111)
    sns.heatmap(avg_distance,
                square=True,
                annot=True,
                xticklabels=order_labels.keys(),
                yticklabels=order_labels.keys(),
                annot_kws={'fontsize':20},
                cmap=sns.cubehelix_palette(as_cmap=True),
                linewidth=0.7,
                linecolor='white',
                cbar_kws={"shrink":0.985},
                # rasterized=True,
                mask=mask
                )

    plt.xticks(rotation=45)
    # fig.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs', f'median_{title}.eps'), transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


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

def fig2_pc(distance, title, flag):

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

    # kde plot
    within = df.loc[df['label'] == 'within',:]
    between = df.loc[df['label'] == 'between',:]

    sns.set(style="ticks", font_scale=2.0)
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    sns.kdeplot(x=within['distance'],
                 fill=True,
                 ax=ax,
                 label='within',
                 cut=0,
                 # clip=(0, 1.0)
                 )
    sns.kdeplot(x=between['distance'],
                 fill=True,
                 ax=ax,
                 label='between',
                 cut=0,
                 # clip=(0, 1.0)
                 )

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_ylim(0,2.5)
    plt.legend()
    sns.despine(offset=10, trim=True)
    # fig.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs', f'intra_vs_inter_{title}.eps'), transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


    print(f'\nWelch t-test - effect size = {es_welch} and pval: {pval_welch}')
    print(f'\tmean intra: {np.mean(within.distance)}')
    print(f'\tmean inter: {np.mean(between.distance)}')

    print(f'\nWil. Mann. Whithn. U test - effect size = {es_u} and pval: {pval_u}')
    print(f'\tmedian intra: {np.median(within.distance)}')
    print(f'\tmedian inter: {np.median(between.distance)}')

    # print(f'variance intra: {np.var(within.distance)}')
    # print(f'variance inter: {np.var(between.distance)}')


#%%
def get_order_flag():
    return np.array([1 if label in order_labels.keys() else 0 for label in info.Order])

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
            # 'top_bin_dist',#*
            # 'top_wei_dist',
            # 'top_local_dist',
            # 'top_global_dist',
            # 'top_local_bin_dist',#*
            # 'top_local_wei_dist',
            # 'top_global_bin_dist',#*
            # 'top_global_wei_dist',
            # 'reg_topological_distance',
            # 'reg_top_bin_dist',#*
            # 'reg_top_wei_dist',
            # 'reg_top_local_dist',
            # 'reg_top_global_dist',
            # 'reg_top_local_bin_dist',#*
            # 'reg_top_local_wei_dist',
            # 'reg_top_global_bin_dist',#*
            # 'reg_top_global_wei_dist',
            ]


for distance in distances:

    print(f'\n----------- {distance} ------------')
    avg_dist = np.load(os.path.join(RAW_DIR, f'avg_{distance}.npy'))
    order_flag = get_order_flag()
    name_flag  = get_name_flag()
    flag = np.logical_and(order_flag, name_flag)

    fig2_pa(avg_dist, distance, flag)
    fig2_pb(avg_dist, distance, flag)
    fig2_pc(avg_dist, distance, flag)
