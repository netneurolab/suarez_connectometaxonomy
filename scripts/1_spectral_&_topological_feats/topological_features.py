# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 16:04:13 2021

@author: Estefany Suarez
"""

import os
import re
import itertools as itr

import numpy as np
import pandas as pd

from scipy import stats
from scipy.spatial.distance import (pdist,squareform)
from sklearn.preprocessing import (MinMaxScaler, LabelEncoder)

from netneurotools import plotting

import seaborn as sns
import matplotlib.pyplot as plt
from rnns import topology

#%%
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami_v2', 'conn')
INFO_DIR = os.path.join(DATA_DIR, 'info')
RAW_DIR = os.path.join(PROJ_DIR, 'raw_results')

#%% connectivity matrix
data = pd.read_csv(os.path.join(INFO_DIR, 'list2.csv'), dtype={'Name':str})


#%%
filenames = []
names = []
order = []
superorder = []
family = []
conn = []
for file in os.listdir(CONN_DIR):
    filename = file.split('.')[0]
    name = ''.join([i for i in filename if not i.isdigit()])
    name = ' '.join(re.findall('[A-Z][^A-Z]*', name))

    conn.append(np.load(os.path.join(CONN_DIR, file)))

    names.append(name)
    filenames.append(filename)

    try:
        order.append(str(data.loc[data.Name == name]['Order'].values[0]))
        superorder.append(str(data.loc[data.Name == name]['Superorder'].values[0]))
        family.append(str(data.loc[data.Name == name]['Family'].values[0]))

    except:
        order.append('')
        superorder.append('')
        family.append('')

filenames = np.array(filenames)
names = np.array(names)
order = np.array(order)
superorder = np.array(superorder)
family = np.array(family)
conn = np.dstack(conn)


#%%
df_info = pd.DataFrame(data=np.column_stack([np.arange(len(filenames)), names, filenames, order, superorder, family]),
                       columns=['Id', 'Name', 'Filename', 'Order', 'Superorder', 'Family'],
                       index=None)

# df_info.to_csv('E:/P9_EIG/raw_results/info.csv')

#%% local properties
local_node_props = [
                    'node_strength',
                    'node_degree',
                    'wei_clustering_coeff',
                    'bin_clustering_coeff',
                    'wei_node_centrality',
                    'bin_node_centrality',
                    'wei_efficiency',
                    'bin_efficiency'
                    ]


avg_local_p = []
std_local_p = []
for prop in local_node_props:
    p = topology.local_topology(w=conn, property=prop)
    # np.save(os.path.join(RAW_DIR, f'{prop}.npy'), p)
    
    avg_local_p.append(np.mean(p, axis=1))
    std_local_p.append(np.std(p, axis=1))


avg_local_p = np.column_stack(avg_local_p)
avg_col_names = ['avg_'+name for name in local_node_props]

std_local_p = np.column_stack(std_local_p)
std_col_names = ['std_'+name for name in local_node_props]

df_local = pd.DataFrame(data=np.column_stack([avg_local_p, std_local_p]),
                        columns=avg_col_names+std_col_names
                        )


#%% global properties
global_props = [
                'wei_char_path',
                'bin_char_path',
                'wei_transitivity',
                'bin_transitivity',
                'wei_assortativity',
                'bin_assortativity',
                ]

global_p = []
for prop in global_props:
    p = topology.global_topology(w=conn, property=prop)
    global_p.append(p)

df_global = pd.DataFrame(data=np.column_stack(global_p),
                          columns=global_props
                        )


#%% mesoscale properties - density
idx = np.tril_indices(200, -1)
idx_h = np.tril_indices(100, -1)

idx_lh = np.arange(100)
idx_rh = np.arange(100,200)

global_density = []
lh_conns = []
rh_conns = []
intrah_conns = []
interh_conns = []
total_conns = []
for i in range(len(df_info)):
    w = conn[:,:,i].astype(bool).astype(int)
    C = np.sum(w[idx]) # total number of existent connections

    global_density.append(C/len(w[idx]))

    w_lh = w[np.ix_(idx_lh, idx_lh)][idx_h]
    lh_conns.append(np.sum(w_lh))

    w_rh = w[np.ix_(idx_rh, idx_rh)][idx_h]
    rh_conns.append(np.sum(w_rh))

    w_inter = w[np.ix_(idx_rh, idx_lh)].ravel()

    intrah_conns.append(np.sum(w_rh)+np.sum(w_lh))
    interh_conns.append(np.sum(w_inter))
    total_conns.append(C)

df_density = pd.DataFrame(np.column_stack([global_density, lh_conns, rh_conns, intrah_conns, interh_conns, total_conns]), columns=['density', 'lh_conns', 'rh_conns', 'intrah_conns', 'interh_conns', 'total_conns'], dtype=float)


#%%
df = pd.concat([df_info, df_local, df_global, df_density], axis=1)
# df.to_csv(os.path.join(RAW_DIR, 'df_props.csv'), index=False)
# df = pd.read_csv(os.path.join(RAW_DIR, 'df_props.csv'))
