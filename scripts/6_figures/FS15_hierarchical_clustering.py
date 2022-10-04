# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 12:38:22 2021

@author: Estefany Suarez
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage


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

COLORS = {k:v for k,v in zip(order_labels, sns.color_palette("Set3", len(order_labels)))} 

#%%
def cluster_map(distance, flag, title):
    
    distance_ = distance.copy()
    distance_ = (distance_-np.min(distance_))/(np.max(distance_)-np.min(distance_))

    # perform hierarchical clustersing        
    linked = linkage(
                        y = squareform(distance_, force='tovector'), 
                        method='complete',
                        # metric=metric, 
                        optimal_ordering=True,
                        )
    
    
    # # plot cluster map
    clustermap = sns.clustermap(data=distance_, 
                                cmap=sns.cubehelix_palette(as_cmap=True), #'vlag',
                                # cbar=False,
                                figsize=(10,10),
                                row_linkage=linked,
                                col_linkage=linked,
                                row_cluster=True,
                                col_cluster=True,
                                row_colors=[COLORS[i] for i in info.Order[flag==1]],
                                col_colors=[COLORS[i] for i in info.Order[flag==1]],
    #                            linewidths=0.1,
    #                            vmin=0.0,
    #                            vmax=1.0,                                  
                                rasterized=True
                                )
    
    linked_idx = clustermap.dendrogram_row.reordered_ind
        
    plt.show()
    plt.close()

    # clustermap.savefig(os.path.join('C:/Users/User/OneDrive - McGill University/Figs', f'hierarch_clust_{title}.eps'), transparent=True, bbox_inches='tight', dpi=300)
    
    return linked, linked_idx


#%% distance between labels
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

    # print(f'\n----------- average {distance} ------------')
    # avg_dist = np.load(os.path.join(RAW_DIR, f'avg_{distance}.npy'))
    # order_flag = get_order_flag()
    # name_flag  = get_name_flag()
    # flag = np.logical_and(order_flag, name_flag)

    # hierarchy = cluster_map(avg_dist, flag, f'avg_{distance}')


    print(f'\n----------- {distance} ------------')
    dist = np.load(os.path.join(RAW_DIR, f'{distance}.npy'))
    flag = get_order_flag()

    hierarchy = cluster_map(dist[np.ix_(flag==1,flag==1)], flag, distance)