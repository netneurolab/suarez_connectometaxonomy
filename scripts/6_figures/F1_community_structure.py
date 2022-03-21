# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 17:52:52 2021

@author: Estefany Suarez
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import re
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from netneurotools import plotting

#%%
PROJ_DIR = 'E:/P9_EIG'
DATA_DIR = os.path.join(PROJ_DIR, 'data')
CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami_v2', 'conn')
INFO_DIR = os.path.join(DATA_DIR, 'info')

RAW_RES_DIR = os.path.join(PROJ_DIR, 'raw_results', 'rich_club_gamma_1.0')


#%%
data = pd.read_csv(os.path.join(INFO_DIR, 'list2.csv'), dtype={'Name':str})
selected = pd.read_csv(os.path.join(INFO_DIR, 'selected.csv'), dtype={'Name':str})

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
order      = []
filenames  = []
names = []
for file in os.listdir(CONN_DIR):
    filename = file.split('.')[0]
    name = ''.join([i for i in filename if not i.isdigit()])
    name = ' '.join(re.findall('[A-Z][^A-Z]*', name))

    names.append(name)
    filenames.append(filename)
    order.append(str(data.loc[data.Name == name]['Order'].values[0]))
    
filenames  = np.array(filenames)
order      = np.array(order)

#%%
flag = np.array([1 if label in order_labels else 0 for label in order])

#%%
sns.set(style="ticks", font_scale=2.0) 
fig, axs = plt.subplots(6, 5, figsize=(25,25))
fig.subplots_adjust(hspace=0.2, wspace=0.1)
axs= axs.ravel()

cont = 0
for order in order_labels:
    
    tmp = selected.loc[selected['Order'] == order, 'Filename'].values
        
    for filename in tmp: 
        idx = np.where(filenames == filename)[0][0]

        conn = np.load(os.path.join(CONN_DIR, f'{filename}.npy')).astype(bool).astype(int)
        communities = np.load(os.path.join(RAW_RES_DIR, f'{filename}_communities.npy'))
        
        plotting.plot_mod_heatmap(data=0.5*conn,  
                                  communities=communities, 
                                  inds=None, 
                                  edgecolor='Purple', 
                                  linewidth=0.1,
                                  ax=axs[cont], 
                                  figsize=(8,8), 
                                  cbar=False, 
                                  square=True,
                                   cmap=sns.cubehelix_palette(10, 
                                                              # start=2, 
                                                              # gamma=1.1, 
                                                              # hue=0.5, 
                                                              # dark=0.7, 
                                                              # light=0.7, 
                                                              as_cmap=True),
                                  # cmap='Purples', #sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True),
                                  # cmap=sns.color_palette("rocket", 6, as_cmap=True),
                                  vmin=0.0,
                                  vmax=1.0,
                                  rasterized=True
                                  )
        
        axs[cont].set_yticklabels([])
        axs[cont].set_xticklabels([])
        axs[cont].set_yticks([])
        axs[cont].set_xticks([])
        axs[cont].set_title(names[idx])
        
        cont += 1
    
sns.despine(offset=10, left=True, bottom=True, trim=True)
# fig.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs', 'communities.eps'), transparent=True, bbox_inches='tight', dpi=300)
plt.show()
plt.close()