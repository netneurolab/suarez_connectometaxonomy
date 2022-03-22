# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 09:43:46 2022

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
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from rnns import topology

#%%
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami', 'conn')
INFO_DIR = os.path.join(DATA_DIR, 'info')
RAW_RES_DIR = os.path.join(PROJ_DIR, 'raw_results')

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
order_labels = [
                'Chiroptera',
                'Rodentia',
                'Artiodactyla',
                'Carnivora',
                'Perissodactyla',
                'Primates',
                ]


#%%
palette=sns.color_palette('Set3', len(order_labels))
gammas = np.linspace(0.5,1.5,11)
df = []
for o in order_labels[:]:

    names_tmp = filenames[np.where(order == o)[0]]
    for i in names_tmp:
        q_scores = np.load(os.path.join(RAW_RES_DIR, 'community_detection', f'{i}_modularity_scores.npy'))
        q_scores = np.mean(q_scores, axis=1)

        df_ = pd.DataFrame(data=np.column_stack([gammas, q_scores]),
                           columns=['gamma', 'modularity'],
                           index=None)
        df_['order'] = o

        df.append(df_)

df = pd.concat(df)

#%%
palette=sns.color_palette('Set3', len(order_labels))
sns.set(style="ticks", font_scale=2.0)
fig, ax = plt.subplots(1,1,figsize=(20,10))
sns.lineplot(x='gamma', y='modularity',
              data=df, hue='order',
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

plot = sns.boxplot(data=df_, x='order', y='modularity',
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
