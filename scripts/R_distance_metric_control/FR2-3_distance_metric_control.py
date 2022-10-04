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
from sklearn.manifold import MDS

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns

from netneurotools import plotting

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
order_labels = {
                'Chiroptera':1,
                'Rodentia':2,
                'Cetartiodactyla':3,
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
                              # vmin=0.0,
                              # vmax=1.0,
                              # robust=True,
                              rasterized=True
                              )
    # ax.figure.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs', f'{title}.eps'), transparent=True, bbox_inches='tight', dpi=300)

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

    sns.histplot(x=within['distance'],
                 fill=True,
                 ax=ax,
                 label='within',
                 stat='probability',
                 color='#d379b1',
                 # cut=0,
                 # clip=(0, 1.0)
                 )

    sns.histplot(x=between['distance'],
                 fill=True,
                 ax=ax,
                 label='between',
                 stat='probability',
                  color='#7f2062',
                 # cut=0,
                 # clip=(0, 1.0)
                 )

    ax.yaxis.set_minor_locator(MultipleLocator(0.025))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.set_ylim(0,0.16)
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

    print(f'variance intra: {np.var(within.distance)}')
    print(f'variance inter: {np.var(between.distance)}')

#%%
def perform_embedding(distance, title=None, flag=None, **kwargs):
    
    def mds(data, n_components=2, **kwargs):
        reducer = MDS(n_components=n_components,
                      **kwargs)
        embedding = reducer.fit_transform(data)
        return embedding

    embedding = mds(distance, dissimilarity='precomputed', **kwargs)
    c1, c2 = embedding[:,0], embedding[:,1]

    # -------------------------------------------------------
    if flag is None:
        info_ = info.copy()
    else:
        info_ = info.copy()[flag==1]

    info_['c1'] = c1
    info_['c2'] = c2
        
    sns.set(style="ticks", font_scale=1.5)  
    fig, axs = plt.subplots(1, 1, figsize=(5,5))

    sns.scatterplot(data=info_,
                    x='c1', 
                    y='c2', 
                    hue = 'Order',
                    hue_order=order_labels, #*******
                    ci=None, 
                    legend=False, 
                    # color=COLORS[label], 
                    palette=sns.color_palette("Set3", len(order_labels)),#*******
                    s=80,
                    ax=axs,
                    )
        
    # axs.legend(fontsize=15)
    fig.suptitle(title)
    # axs.set_xlim(-0.2,1.2)
    # axs.set_ylim(-0.2,1.2)

    sns.despine(offset=10, trim=False)
    # fig.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs', f'{title}.eps'), transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()



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
            'spec_dist_eigkde_cosine',
            'spec_dist_eig_cosine',
            'spec_dist_eigkde_euclid',
            'spec_dist_eig_euclid',
            ]

#%%
RES_DIR = os.path.join(RAW_DIR, 'distance_metric_control', 'zscored')
for distance in distances:

    print(f'\n----------- z-scored average {distance} ------------')
    avg_dist = np.load(os.path.join(RES_DIR, f'avg_{distance}.npy'))
    order_flag = get_order_flag()
    name_flag  = get_name_flag()
    flag = np.logical_and(order_flag, name_flag)

    # perform_embedding(avg_dist, f'avg_{distance}', flag)
    fig2_pa(avg_dist, f'avg_{distance}', flag)
    fig2_pc(avg_dist, f'avg_{distance}', flag)


    # print(f'\n----------- z-scored {distance} ------------')
    dist = np.load(os.path.join(RES_DIR, f'{distance}.npy'))
    flag = get_order_flag()

    perform_embedding(dist[np.ix_(flag==1,flag==1)], distance, flag)
    # fig2_pa(dist[np.ix_(flag==1,flag==1)], distance, flag)
    # fig2_pc(dist[np.ix_(flag==1,flag==1)], distance, flag)

#%%
RES_DIR = os.path.join(RAW_DIR, 'distance_metric_control', 'nonzscored')
for distance in distances:

    print(f'\n----------- non - z-scored average {distance} ------------')
    avg_dist = np.load(os.path.join(RES_DIR, f'avg_{distance}.npy'))
    order_flag = get_order_flag()
    name_flag  = get_name_flag()
    flag = np.logical_and(order_flag, name_flag)

    # perform_embedding(avg_dist, f'avg_{distance}', flag)
    fig2_pa(avg_dist, f'avg_{distance}', flag)
    fig2_pc(avg_dist, f'avg_{distance}', flag)


    # print(f'\n----------- non - z-scored {distance} ------------')
    dist = np.load(os.path.join(RES_DIR, f'{distance}.npy'))
    flag = get_order_flag()

    perform_embedding(dist[np.ix_(flag==1,flag==1)], distance, flag)
    # fig2_pa(dist[np.ix_(flag==1,flag==1)], distance, flag)
    # fig2_pc(dist[np.ix_(flag==1,flag==1)], distance, flag)
