# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 16:04:13 2021

@author: Estefany Suarez
"""

import os
import re
import time
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

import multiprocessing as mp

#%%
PROJ_DIR = 'E:/P9_EIG'
DATA_DIR = os.path.join(PROJ_DIR, 'data')
CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami_v2', 'conn')
INFO_DIR = os.path.join(DATA_DIR, 'info')
RAW_DIR = os.path.join(PROJ_DIR, 'raw_results')

#%% connectivity matrix
def topological_properties(file):

    filename = file.split('.')[0]
    if not os.path.exists(os.path.join(RAW_DIR, 'nulls', f'{filename}_null_props.csv')):

        conn = np.load(os.path.join(RAW_DIR, 'nulls', f'nulls_{filename}.npy'))

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

        #%%
        df = pd.concat([df_local, df_global], axis=1)
        df.to_csv(os.path.join(RAW_DIR, 'nulls', f'{filename}_null_props.csv'), index=False)


#%%
def main():

    params = [{'file':file} for file in os.listdir(CONN_DIR)]

    # hubs
    print ('\nINITIATING PROCESSING TIME - NULLS TOPOLOGICAL PROPERTIES')
    t0 = time.perf_counter()

    # run iterations for all subjects in CONN_DIR
    pool = mp.Pool(processes=16)
    res = [pool.apply_async(topological_properties, (), p) for p in params]
    for r in res: r.get()
    pool.close()

    print ('\nTOTAL PROCESSING TIME - NULLS TOPOLOGICAL PROPERTIES')
    print (time.perf_counter()-t0, "seconds process time")
    print ('END')


if __name__ == '__main__':
    main()
