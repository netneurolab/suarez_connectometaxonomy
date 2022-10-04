# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 17:52:52 2021

@author: Estefany Suarez
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import re
import itertools as itr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

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
order_labels = [
                'Chiroptera',
                'Rodentia',
                'Cetartiodactyla',
                'Carnivora',
                'Perissodactyla',
                'Primates',
                ]

#%%
flag = np.array([1 if label in order_labels else 0 for label in info.Order])

#%%
df_conns = []
for i, filename in enumerate(info['Filename']):

    conn = np.load(os.path.join(CONN_DIR, f'{filename}.npy')).astype(bool).astype(int)
    communities = np.load(os.path.join(RAW_DIR, 'rich_club', f'{filename}_communities.npy'))

    C = np.sum(conn[np.tril_indices_from(conn,-1)])

    comm_pairs = list(itr.combinations(communities, 2))
    idx_pairs  = list(itr.combinations(range(len(communities)), 2))

    # within-module conns
    within_pairs = np.array([idx_pairs[i] for i, pair in enumerate(comm_pairs) if pair[0] == pair[1]])
    i_within,j_within = zip(*within_pairs)
    within = [np.sum(conn[i_within,j_within])/C]

    within_label = np.array(['intra' for _ in range(len(within))])
    df_within = pd.DataFrame(np.column_stack([within,within_label]), columns=['conn_density','conn_type'] )

    # between-orders species
    between_pairs = np.array([idx_pairs[i] for i, pair in enumerate(comm_pairs) if pair[0] != pair[1]])
    i_between,j_between = zip(*between_pairs)
    between =[np.sum(conn[i_between, j_between])/C]

    between_label = np.array(['inter' for _ in range(len(between))])
    df_between = pd.DataFrame(np.column_stack([between,between_label]), columns=['conn_density','conn_type'] )

    df_ = pd.concat([df_within, df_between])
    df_['Order'] = info.Order.values[i]

    df_conns.append(df_)

df_conns = pd.concat(df_conns)

#%%
df_conns = df_conns.astype({'conn_density': float})
df_conns = pd.concat([df_conns.loc[df_conns['Order'] == o] for o in order_labels]).reset_index(drop=True)

sns.set(style="ticks", font_scale=2.0)
fig, ax = plt.subplots(1,1,figsize=(10,5))
sns.barplot(x="conn_type",
            y='conn_density',
            data=df_conns,
            hue="Order",
            ax=ax,
            palette=sns.color_palette('Set3', len(order_labels))
            )

ax.get_legend().remove()
# ax.set_ylim(0,1.0)
ax.set_ylabel('avg. proportion of connections')
sns.despine(offset=10, trim=True)
# fig.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs','intra_vs_inter_mod_conns.eps'), transparent=True, bbox_inches='tight', dpi=300)
plt.show()
plt.close()
