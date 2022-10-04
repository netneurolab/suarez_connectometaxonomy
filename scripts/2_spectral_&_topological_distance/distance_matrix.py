# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 16:04:13 2021

@author: Estefany Suarez
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os

import numpy as np
import pandas as pd

from scipy import stats
from scipy.spatial.distance import (pdist,squareform)

#%%
RESOLUTION = '100'
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami', f'conn_{RESOLUTION}')
INFO_DIR = os.path.join(DATA_DIR, 'info')
RAW_DIR  = os.path.join(PROJ_DIR, 'raw_results', f'res_{RESOLUTION}')

#%% topological distance
def distance(df, name, include_properties=None, metric='cosine'):

    if include_properties is not None: X = df[include_properties].values
    else: X = df.values
        
    X = stats.zscore(X, axis=0)
    distance = squareform(pdist(X, metric=metric), force='tomatrix')
    np.save(os.path.join(RAW_DIR, name), distance)
    

#%%
local_bin = [
             'avg_node_degree',
             'avg_bin_clustering_coeff',
             'avg_bin_node_centrality',
             'avg_bin_efficiency',
             'std_node_degree',
             'std_bin_clustering_coeff',
             'std_bin_node_centrality',
             'std_bin_efficiency',
            ]

local_wei = [
             'avg_node_strength',
             'avg_wei_clustering_coeff',
             'avg_wei_node_centrality',
             'avg_wei_efficiency',
             'std_node_strength',
             'std_wei_clustering_coeff',
             'std_wei_node_centrality',
             'std_wei_efficiency',
             ]

global_bin = [
             'bin_char_path',
             'bin_transitivity',
             'bin_assortativity',
            ]

global_wei = [
              'wei_char_path',
              'wei_transitivity',
              'wei_assortativity',
              ]

local_ = local_bin + local_wei
global_ = global_bin + global_wei

bin_ = global_bin + local_bin
wei_ = global_wei + local_wei

all_ = local_ + global_

#%% topological distance
df_top = pd.read_csv(os.path.join(RAW_DIR, 'df_props.csv'))

options = {
            'topological_distance':all_,
            'top_bin_dist':bin_,
            'top_wei_dist':wei_,
            'top_local_dist':local_,
            'top_global_dist':global_,
            'top_local_bin_dist':local_bin,
            'top_global_bin_dist':global_bin,
            'top_local_wei_dist':local_wei,
            'top_global_wei_dist':global_wei 
            }

for k,v in options.items():
    X = distance(df_top.copy(), k, v, metric='cosine')

#%% spectral distance with eig
N = 2*int(RESOLUTION)
df_eig = pd.read_csv(os.path.join(RAW_DIR, 'eig.csv'))
distance(df_eig.copy(), 'spectral_distance', df_eig.columns[-N:])

#%% spectral distance with eig_kde
N = 2*int(RESOLUTION)*10
df_eig = pd.read_csv(os.path.join(RAW_DIR, 'eig_kde.csv'))
distance(df_eig.copy(), 'spectral_distance', df_eig.columns[-N:])
