# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 09:43:46 2022

@author: Estefany Suarez
"""

import os

import numpy as np
np.seterr(all="ignore")

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

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
filenames = info.Filename.values
order = info.Order.values
palette=sns.color_palette('Set3', len(order_labels))
gammas = np.linspace(0.5,1.5,11)
df = []
for o in order_labels[:]:
    names_tmp = filenames[np.where(order == o)[0]]
    for i in names_tmp:
        q_scores = np.load(os.path.join(RAW_DIR, 'community_detection', f'{i}_modularity_scores.npy'))
        q_scores = np.mean(q_scores, axis=1)

        df_ = pd.DataFrame(data=np.column_stack([gammas, q_scores]),
                           columns=['gamma', 'modularity'],
                           index=None)
        df_['Order'] = o

        df.append(df_)

df = pd.concat(df)

#%%
palette=sns.color_palette('Set3', len(order_labels))
sns.set(style="ticks", font_scale=2.0)
fig, ax = plt.subplots(1,1,figsize=(20,10))
sns.lineplot(x='gamma', y='modularity',
              data=df, hue='Order',
              palette=sns.color_palette('Set3', len(order_labels)),
              linewidth=3,
              marker='o',
              markersize=10,
              )

ax.set_xlim(0.5, 1.5)
ax.xaxis.set_major_locator(MultipleLocator(0.1))
sns.despine(offset=10, trim=True)
# fig.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs', 'modularity.eps'), transparent=True, bbox_inches='tight', dpi=300)

plt.show()
plt.close()

#%%
df_ = df.loc[df['gamma'] == 1, :]
sns.set(style="ticks", font_scale=1.5, palette=sns.color_palette('Set3', len(order_labels)))  #palette='mako',
fig, ax = plt.subplots(1,1,figsize=(10,4))

plot = sns.boxplot(data=df_, x='Order', y='modularity',
                      # palette=sns.color_palette('Set3', len(order_labels)),
                      width=0.5,
                      # hue='model',
                      showfliers=False
                      )

# plot._legend.remove()
sns.despine(offset=10, trim=True)
# fig.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs', f'modularity_gamma.eps'), transparent=True, bbox_inches='tight', dpi=300)
plt.show()
plt.close()
