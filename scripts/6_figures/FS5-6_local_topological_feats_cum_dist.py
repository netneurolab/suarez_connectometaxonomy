# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 11:31:50 2022

@author: Estefany Suarez
"""

import os
import re

import numpy as np
np.seterr(all="ignore")

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

#%%
RESOLUTION = '100'
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami', f'conn_{RESOLUTION}')
INFO_DIR = os.path.join(DATA_DIR, 'info')
RAW_DIR  = os.path.join(PROJ_DIR, 'raw_results', f'res_{RESOLUTION}')

#%%
info = pd.read_csv(os.path.join(INFO_DIR, 'info.csv'), dtype={'Name':str})

#%%
local_bin_node_props = [
                    'node_degree',
                    'bin_clustering_coeff',
                    'bin_node_centrality',
                    'bin_efficiency'
                    ]

local_wei_node_props = [
                    'node_strength',
                    'wei_clustering_coeff',
                    'wei_node_centrality',
                    'wei_efficiency',
                    ]

order_labels = [
                'Chiroptera',
                'Rodentia',
                'Cetartiodactyla',
                'Carnivora',
                'Perissodactyla',
                'Primates',
                ]

#%%
print("Binary local features - Cumulative distribution")

palette=sns.color_palette('Set3', len(order_labels))
fig, axs = plt.subplots(len(order_labels),
                        len(local_bin_node_props),
                        figsize=(20,20),
                        # sharey=True,
                        # sharex=True
                        )
axs = axs.ravel()

cont_col = 0
for p in local_bin_node_props:
    prop = np.load(os.path.join(RAW_DIR, 'local_props', f'{p}.npy'))

    cont_row = 0
    for o in order_labels:
        p_tmp = prop[np.where(info.Order == o)[0]]

        for i,j in enumerate(p_tmp):
            sns.ecdfplot(x=j, color=palette[np.where(np.array(order_labels) == o)[0][0]], ax=axs[cont_row+cont_col])

        axs[cont_row+cont_col].set_xlim(0,np.percentile(prop, 99))
        axs[cont_row+cont_col].set_xlim(0,np.percentile(prop, 99))
        cont_row += len(local_bin_node_props)

    cont_col += 1
    sns.despine(offset=10, trim=False)

# fig.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs', 'cum_local_bin_props_.eps'), transparent=True, bbox_inches='tight', dpi=300)
plt.show()
plt.close()

#%%
print("Weighted local features - Cumulative distribution")

palette=sns.color_palette('Set3', len(order_labels))
fig, axs = plt.subplots(len(order_labels),
                        len(local_wei_node_props),
                        figsize=(20,20),
                        # sharey=True,
                        # sharex=True
                        )
axs = axs.ravel()

cont_col = 0
for p in local_wei_node_props:
    prop = np.load(os.path.join(RAW_DIR, 'local_props', f'{p}.npy'))

    cont_row = 0
    for o in order_labels:
        p_tmp = prop[np.where(info.Order == o)[0]]

        for i,j in enumerate(p_tmp):
            sns.ecdfplot(x=j, color=palette[np.where(np.array(order_labels) == o)[0][0]], ax=axs[cont_row+cont_col])

        axs[cont_row+cont_col].set_xlim(0,np.percentile(prop, 99))
        axs[cont_row+cont_col].set_xlim(0,np.percentile(prop, 99))
        cont_row += len(local_bin_node_props)

    cont_col += 1
    sns.despine(offset=10, trim=False)

# fig.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs', 'cum_local_wei_props.eps'), transparent=True, bbox_inches='tight', dpi=300)
plt.show()
plt.close()
