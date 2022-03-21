# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 14:49:08 2021

@author: Estefany Suarez
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import re
import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import seaborn as sns


#%%
PROJ_DIR = 'E:/P9_EIG'
DATA_DIR = os.path.join(PROJ_DIR, 'data')
CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami_v2', 'conn')
COOR_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami_v2', 'coords')
INFO_DIR = os.path.join(DATA_DIR, 'info')

RAW_RES_DIR = os.path.join(PROJ_DIR, 'raw_results')
PROC_RES_DIR = os.path.join(PROJ_DIR, 'proc_results')

#%%
data = pd.read_csv(os.path.join(INFO_DIR, 'list2.csv'), dtype={'Name':str})


#%%
conn = []
names      = [] 
order      = []
filenames  = []
superorder = []
family     = []

length     = []
max_length = []

for file in os.listdir(CONN_DIR):
    filename = file.split('.')[0]
    name = ''.join([i for i in filename if not i.isdigit()])
    name = ' '.join(re.findall('[A-Z][^A-Z]*', name))
     
    bin_conn = np.load(os.path.join(CONN_DIR, file)).astype(bool).astype(int)
    conn.append(bin_conn)
        
    coords = np.load(os.path.join(COOR_DIR, file))
    dist = cdist(coords, coords, metric='euclidean')
    max_length.append(np.max(dist))#*bin_conn)) 
    length.append(dist)
    
    names.append(name)
    filenames.append(filename)
    order.append(str(data.loc[data.Name == name]['Order'].values[0]))
    superorder.append(str(data.loc[data.Name == name]['Superorder'].values[0]))
    family.append(str(data.loc[data.Name == name]['Family'].values[0]))
   
length     = np.dstack(length)
max_length = np.array(max_length)
conn       = np.dstack(conn)
filenames  = np.array(filenames)
names      = np.array(names)
order      = np.array(order)
superorder = np.array(superorder)
family     = np.array(family)


#%% classification of edges based on length distribution
idx = np.tril_indices(200, -1)
# C = (200*199)/2

short  = []
medium = []
long   = []
total  = []
percentage_dist = []
for i in range(225):
    
    nonzero = np.nonzero(conn[:,:,i][idx])
    dist_ = length[:,:,i][idx][nonzero] # vector of distances for existent connections 
    dist_ = dist_/max_length[i] # vector of distances/max_distance
    C = np.sum(conn[:,:,i][idx]) # total number of existent connections 

    percentage_dist.append(dist_)
    short.append(len(np.where(dist_ <= 0.25)[0])/C)
    medium.append(len(np.where((dist_ >  0.25) & (dist_ <= 0.75))[0])/C)
    long.append(len(np.where(dist_ > 0.75)[0])/C)
    
    total.append(C)
    
df_dist = pd.DataFrame(np.column_stack([np.arange(225), names, order, family, total, short, medium, long]), columns=['id', 'name', 'order', 'family', 'total', 'short', 'medium', 'long'])
df_dist = df_dist.astype({'total': float,
                          'short': float, 
                          'medium': float,
                          'long': float})


#%%                
order_labels = [
                'Chiroptera',
                'Rodentia',
                'Artiodactyla',
                'Carnivora',
                'Perissodactyla',
                'Primates',
                ]

df_dist = pd.concat([df_dist.loc[df_dist['order'] == o] for o in order_labels]).reset_index(drop=True)
COLORS = {k:v for k,v in zip(order_labels, sns.color_palette("Set3", len(order_labels)))} #, desat=0.45


#%% connection length distribution
idx = np.tril_indices(200, -1)

sns.set(style="ticks", font_scale=2.0)  #palette='mako', 
fig, axs = plt.subplots(1,len(order_labels),figsize=(5*len(order_labels),5))
axs = axs.ravel()
for i in range(225):
    
    if order[i] in order_labels:
        nonzero = np.nonzero(length[:,:,i][idx])
        dist = length[:,:,i][idx][nonzero]
        # dist = dist/np.max(dist)
                
        sns.kdeplot(dist, color=COLORS[order[i]], ax=axs[np.where(np.array(order_labels) == order[i])[0][0]])
        # axs[np.where(np.array(order_labels) == order[i])[0][0]].get_yaxis().set_visible(False)    
        axs[np.where(np.array(order_labels) == order[i])[0][0]].set_xlabel('connection length')
        axs[np.where(np.array(order_labels) == order[i])[0][0]].set_xlim(0,150)
        axs[np.where(np.array(order_labels) == order[i])[0][0]].set_title(order[i])
        axs[np.where(np.array(order_labels) == order[i])[0][0]].set_ylim(0,0.04)

        
# plt.legend()
sns.despine(offset=10, trim=True)
# fig.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs', 'conn_length_dist.eps'), transparent=True, bbox_inches='tight', dpi=300)
plt.show()
plt.close()

