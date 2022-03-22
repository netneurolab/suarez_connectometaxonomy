# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 11:31:50 2022

@author: Estefany Suarez
"""

import os
import re
import itertools as itr

import numpy as np
np.seterr(all="ignore")

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
CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami', 'conn')
INFO_DIR = os.path.join(DATA_DIR, 'info')
RAW_DIR = os.path.join(PROJ_DIR, 'raw_results')

#%% connectivity matrix
data = pd.read_csv(os.path.join(INFO_DIR, 'list.csv'), dtype={'Name':str})


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


names = np.array(names)
filenames = np.array(filenames)
order = np.array(order)
superorder = np.array(superorder)
family = np.array(family)
conn = np.dstack(conn)


#%%
df_info = pd.DataFrame(data=np.column_stack([np.arange(len(filenames)), names, filenames, order, superorder, family]),
                       columns=['Id', 'Name', 'Filename', 'Order', 'Superorder', 'Family'],
                       index=None)

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
                'Artiodactyla',
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
    prop = np.load(os.path.join(RAW_DIR, f'{p}.npy'))

    cont_row = 0
    for o in order_labels:
        p_tmp = prop[np.where(order == o)[0]]

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
    prop = np.load(os.path.join(RAW_DIR, f'{p}.npy'))

    cont_row = 0
    for o in order_labels:
        p_tmp = prop[np.where(order == o)[0]]

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
