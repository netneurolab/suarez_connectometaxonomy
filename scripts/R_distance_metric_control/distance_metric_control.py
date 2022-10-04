# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 12:49:37 2022

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
def distance(df, name, zscore, metric, include_properties=None):

    if include_properties is not None: X = df[include_properties].values
    else: X = df.values
    
    if zscore:
        X = stats.zscore(X, axis=0)
        path = os.path.join(RAW_DIR, 'distance_metric_control', 'zscored', name)
    else:
        path = os.path.join(RAW_DIR, 'distance_metric_control', 'nonzscored', name)

    distance = squareform(pdist(X, metric=metric), force='tomatrix')
    np.save(path, distance)


#%% spectral distance - eig_kde
N = 2*int(RESOLUTION)*10

df_eig_kde = pd.read_csv(os.path.join(RAW_DIR, 'eig_kde.csv'))

# cosine distance - zscored
distance(df=df_eig_kde.copy(), 
          name='spec_dist_eigkde_cosine',
          zscore=True,
          metric='cosine',
          include_properties=df_eig_kde.columns[-N:],
          )

# cosine distance - non-zscored
distance(df=df_eig_kde.copy(), 
          name='spec_dist_eigkde_cosine',
          zscore=False,
          metric='cosine',
          include_properties=df_eig_kde.columns[-N:],
          )

# Euclidean distance - zscored
distance(df=df_eig_kde.copy(), 
          name='spec_dist_eigkde_euclid',
          zscore=True,
          metric='euclidean',
          include_properties=df_eig_kde.columns[-N:],
          )

# Euclidean distance - non-zscored
distance(df=df_eig_kde.copy(), 
          name='spec_dist_eigkde_euclid',
          zscore=False,
          metric='euclidean',
          include_properties=df_eig_kde.columns[-N:],
          )

#%% spectral distance - eig
N = 2*int(RESOLUTION)

df_eig = pd.read_csv(os.path.join(RAW_DIR, 'eig.csv'))

# cosine distance - zscored
distance(df=df_eig.copy(), 
          name='spec_dist_eig_cosine',
          zscore=True,
          metric='cosine',
          include_properties=df_eig.columns[-200:],
          )

# cosine distance - non-zscored
distance(df=df_eig.copy(), 
          name='spec_dist_eig_cosine',
          zscore=False,
          metric='cosine',
          include_properties=df_eig.columns[-200:],
          )

# Euclidean distance - zscored
distance(df=df_eig.copy(), 
          name='spec_dist_eig_euclid',
          zscore=True,
          metric='euclidean',
          include_properties=df_eig.columns[-200:],
          )

# Euclidean distance - non-zscored
distance(df=df_eig.copy(), 
          name='spec_dist_eig_euclid',
          zscore=False,
          metric='euclidean',
          include_properties=df_eig.columns[-200:],
          )


