# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 09:10:37 2021

@author: Estefany Suarez
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import re
import itertools as itr
import numpy as np
import pandas as pd

from scipy import stats
from scipy.spatial.distance import (pdist,squareform)

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator

import seaborn as sns

from netneurotools import plotting


#%%
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami', 'conn')
INFO_DIR = os.path.join(DATA_DIR, 'info')
RAW_DIR = os.path.join(PROJ_DIR, 'raw_results')

#%%
df = pd.read_csv(os.path.join(RAW_DIR, 'df_props.csv'))
C = (200*199)/2 # total number of connections

#%%
order_labels = [
                'Chiroptera',
                'Rodentia',
                'Artiodactyla',
                'Carnivora',
                'Perissodactyla',
                'Primates',
                ]

df = pd.concat([df.loc[df['Order'] == o] for o in order_labels]).reset_index(drop=True)

#%%
df_nulls  = []
for filename in list(df['Filename'].values):
    props_ = pd.read_csv(os.path.join(RAW_DIR, 'nulls', f'{filename}_null_props.csv'))
    df_nulls.append(np.mean(props_, axis=0))

df_nulls = pd.concat(df_nulls, axis=1).T

#%%
include_properties = [
                        'avg_node_strength',
                        'avg_node_degree',
                        'avg_wei_clustering_coeff',
                        'avg_bin_clustering_coeff',
                        'avg_wei_node_centrality',
                        'avg_bin_node_centrality',
                        'avg_wei_efficiency',
                        'avg_bin_efficiency',
                        # 'std_node_strength',
                        # 'std_node_degree',
                        # 'std_wei_clustering_coeff',
                        # 'std_bin_clustering_coeff',
                        # 'std_wei_node_centrality',
                        # 'std_bin_node_centrality',
                        # 'std_wei_efficiency',
                        # 'std_bin_efficiency',
                        'wei_char_path',
                        'bin_char_path',
                        'wei_transitivity',
                        'bin_transitivity',
                        'wei_assortativity',
                        'bin_assortativity',
                        # 'density',
                     ]

df = df.loc[:,['Id','Name', 'Filename', 'Order'] + include_properties]

#%% distribution of topological properties
for i, p in enumerate(include_properties):

    df[p] = df[p].astype(float)
    df[p] = df[p]/df_nulls[p]

    sns.set(style="ticks", font_scale=1.5, palette=sns.color_palette('Set3', len(order_labels)))  #palette='mako',
    fig, ax = plt.subplots(1,1,figsize=(10,4))

    plot = sns.boxplot(data=df, x='Order', y=p,
                            palette=sns.color_palette('Set3', len(order_labels)),
                            width=0.5,
                          # hue='model',
                            showfliers=False
                          )

    # plot._legend.remove()
    sns.despine(offset=10, trim=True)
    # fig.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs', f'boxplt_{p}.eps'), transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
