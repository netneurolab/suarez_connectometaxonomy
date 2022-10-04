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

import seaborn as sns
import matplotlib.pyplot as plt 

from sklearn.manifold import MDS

#%%
RESOLUTION = '100'
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami', f'conn_{RESOLUTION}')
INFO_DIR = os.path.join(DATA_DIR, 'info')
RAW_DIR  = os.path.join(PROJ_DIR, 'raw_results', f'res_{RESOLUTION}')

#%%
info = pd.read_csv(os.path.join(INFO_DIR, 'info.csv'), index_col=False)
 
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
def perform_embedding(distance=None, title=None, flag=None):
    
    def mds(data, n_components=2, **kwargs):
        reducer = MDS(n_components=n_components,
                      **kwargs)
        embedding = reducer.fit_transform(data)
        return embedding

    embedding = mds(distance, dissimilarity='precomputed')
    c1, c2 = embedding[:,0], embedding[:,1]

    # -------------------------------------------------------
    if flag is None:
        info_ = info.copy()
    else:
        info_ = info.copy()[flag==1]

    c1 =(c1-np.min(c1))/(np.max(c1)-np.min(c1))
    c2 =(c2-np.min(c2))/(np.max(c2)-np.min(c2))

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
    # fig.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs', f'{emb_type}_{title}.eps'), transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

    return (c1, c2)

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

    # C = perform_embedding(
    #                       distance=avg_dist,
    #                       emb_type='MDS',
    #                       matrix_type='case 3',
    #                       n_components=2, 
    #                       flag=flag,
    #                       title=f'avg_{distance}',
    #                       )


    print(f'\n----------- {distance} ------------')
    dist = np.load(os.path.join(RAW_DIR, f'{distance}.npy'))
    flag = get_order_flag()

    C = perform_embedding(
                          distance=dist[np.ix_(flag==1,flag==1)],
                          title=distance,
                          flag=flag,
                          )


