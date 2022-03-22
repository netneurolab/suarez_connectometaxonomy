# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 09:10:37 2021

@author: Estefany Suarez
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator

import seaborn as sns

#%%
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami', 'conn')
INFO_DIR = os.path.join(DATA_DIR, 'info')
RAW_DIR = os.path.join(PROJ_DIR, 'raw_results')

#%%
df = pd.read_csv(os.path.join(RAW_DIR, 'df_props.csv'))
C = (200*199)/2

#%%
order_labels = [
                'Chiroptera',
                'Rodentia',
                'Artiodactyla',
                'Carnivora',
                'Perissodactyla',
                'Primates',
                ]

df = pd.concat([df.loc[df['Order'] == o] for o in order_labels]).reset_index(drop=True)

#%% intra vs inter connections
df['interh_conns_p'] = df['interh_conns']/df['total_conns']
df['intrah_conns_p'] = df['intrah_conns']/df['total_conns']

# df['interh_conns_p'] = df['interh_conns']/C
# df['intrah_conns_p'] = df['intrah_conns']/C

df_ = df[['Id', 'Order', 'interh_conns_p', 'intrah_conns_p']]
df_ = pd.melt(df_, id_vars=['Id','Order'],
              value_vars=['interh_conns_p', 'intrah_conns_p'],
              var_name='conn_type',
              value_name='conn_density')


sns.set(style="ticks", font_scale=2.0, palette=sns.color_palette('Set3', len(order_labels)))
fig, ax = plt.subplots(1,1,figsize=(10,5))
sns.barplot(x="conn_type",
            y='conn_density',
            data=df_,
            hue="Order",
            ax=ax,
            # width=.5,
            # kwargs={'width':0.5},
            # palette=sns.color_palette('Set3', len(order_labels))
            )
ax.get_legend().remove()
# ax.set_ylim(0,0.8)
ax.set_ylabel('avg. proportion of connections')
sns.despine(offset=10, trim=True)
# fig.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs', 'intra_vs_inter_hm_conns.eps'), transparent=True, bbox_inches='tight', dpi=300)
plt.show()
plt.close()


#%%
df['lh_density'] = df['lh_conns']/df['total_conns']
df['rh_density'] = df['rh_conns']/df['total_conns']

# df['lh_density'] = df['lh_conns']/C
# df['rh_density'] = df['rh_conns']/C

df_ = df[['Id', 'Order', 'lh_density', 'rh_density']]
df_ = pd.melt(df_, id_vars=['Id','Order'],
              value_vars=['lh_density', 'rh_density'],
              var_name='conn_type',
              value_name='conn_density')


sns.set(style="ticks", font_scale=2.0, palette=sns.color_palette('Set3', len(order_labels)))
fig, ax = plt.subplots(1,1,figsize=(10,5))
sns.barplot(x="conn_type",
            y='conn_density',
            data=df_,
            hue="Order",
            ax=ax,
            # palette=sns.color_palette('Set3', len(order_labels))
            )
ax.get_legend().remove()
ax.yaxis.set_major_locator(MultipleLocator(0.2))
# ax.set_ylim(0,0.7)
ax.set_ylabel('avg. proportion of connections')
sns.despine(offset=10, trim=True)
# fig.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs', 'lh_vs_rh_conns.eps'), transparent=True, bbox_inches='tight', dpi=300)
plt.show()
plt.close()
