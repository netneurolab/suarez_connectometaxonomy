# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 14:49:08 2021

@author: Estefany Suarez
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import seaborn as sns

#%%
RESOLUTION = '100'
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami', f'conn_{RESOLUTION}')
COOR_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami', 'coords')
INFO_DIR = os.path.join(DATA_DIR, 'info')
RAW_DIR  = os.path.join(PROJ_DIR, 'raw_results', f'res_{RESOLUTION}')

#%%
df_info = pd.read_csv(os.path.join(INFO_DIR, 'info.csv'), dtype={'Name':str})

#%%
conn = []
length     = []
max_length = []

for file in os.listdir(CONN_DIR):
    bin_conn = np.load(os.path.join(CONN_DIR, file)).astype(bool).astype(int)
    conn.append(bin_conn)

    coords = np.load(os.path.join(COOR_DIR, file))
    dist = cdist(coords, coords, metric='euclidean')
    max_length.append(np.max(dist))
    length.append(dist)

conn       = np.dstack(conn)
length     = np.dstack(length)
max_length = np.array(max_length)

#%% classification of edges based on length distribution
N = 2*int(RESOLUTION)
idx = np.tril_indices(N, -1)

short  = []
medium = []
long   = []
total  = []
percentage_dist = []
for i, _ in enumerate(df_info['Filename']):

    nonzero = np.nonzero(conn[:,:,i][idx])
    dist_ = length[:,:,i][idx][nonzero] # vector of distances for existent connections
    dist_ = dist_/max_length[i] # vector of distances/max_distance
    C = np.sum(conn[:,:,i][idx]) # total number of existent connections

    percentage_dist.append(dist_)
    short.append(len(np.where(dist_ <= 0.25)[0])/C)
    medium.append(len(np.where((dist_ >  0.25) & (dist_ <= 0.75))[0])/C)
    long.append(len(np.where(dist_ > 0.75)[0])/C)

    total.append(C)

df_dist = pd.DataFrame(np.column_stack([total, short, medium, long]), columns=['total', 'short', 'medium', 'long'])
df_dist = pd.concat((df_info, df_dist), axis=1)
df_dist = df_dist.astype({'total': float,
                          'short': float,
                          'medium': float,
                          'long': float})

#%%
order_labels = [
                'Chiroptera',
                'Rodentia',
                'Cetartiodactyla',
                'Carnivora',
                'Perissodactyla',
                'Primates',
                ]

df_dist = pd.concat([df_dist.loc[df_dist['Order'] == o] for o in order_labels]).reset_index(drop=True)
COLORS = {k:v for k,v in zip(order_labels, sns.color_palette("Set3", len(order_labels)))} #, desat=0.45

#%%
df_ = pd.melt(df_dist, id_vars=['Id','Order'],
              value_vars=['short', 'medium', 'long'],
              var_name='conn_type',
              value_name='conn_density')

sns.set(style="ticks", font_scale=2.0)
fig, ax = plt.subplots(1,1,figsize=(15,5))
sns.barplot(x="conn_type",
            y='conn_density',
            data=df_, hue="Order",
            ax=ax,
            palette=sns.color_palette('Set3', len(order_labels))
            )

ax.get_legend().remove()
ax.set_ylabel('avg. proportion of connections')
sns.despine(offset=10, trim=True)
# fig.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs', 'length_conns.eps'), transparent=True, bbox_inches='tight', dpi=300)
plt.show()
plt.close()
