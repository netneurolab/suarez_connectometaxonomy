# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 13:35:44 2021

@author: Estefany Suarez
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import itertools as itr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# import eigenfunctions as fn


#%%
PROJ_DIR = 'E:/P9_EIG'
DATA_DIR = os.path.join(PROJ_DIR, 'data')
CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami_v2', 'conn')
INFO_DIR = os.path.join(DATA_DIR, 'info')
RAW_DIR  = os.path.join(PROJ_DIR, 'raw_results') 


#%%
info = pd.read_csv(os.path.join(RAW_DIR, 'info.csv'))

order = info['Order']
order_labels = [
                'Chiroptera',
                'Rodentia',
                'Artiodactyla',
                'Carnivora',
                'Perissodactyla',
                'Primates',
                ]

#%%
# eig = np.load(os.path.join(RAW_DIR, 'eig.npy')) 
# x_d, dx_d, eig_kde = fn.get_eigen_kde(eigenspectra=eig)

eig_kde = pd.read_csv(os.path.join(RAW_DIR, 'eig_kde.csv')).values[:,6:]
x_d, dx_d = np.linspace(0, 2, 2000, retstep=True)


#%%
sns.set(style="ticks", font_scale=1.00) 
fig, axs = plt.subplots(len(order_labels[:]), 
                        len(order_labels[:]), 
                        figsize=(5*len(order_labels[:]),5*len(order_labels[:])), 
                        sharex=True, 
                        sharey=True,
                        subplot_kw={'xlim':(0, 2)}
                        )
COLORS = {k:v for k,v in zip(order_labels, sns.color_palette("Set3", len(order_labels)))} #, desat=0.45

fig.subplots_adjust(hspace=0.2, wspace=0.1)
axs = axs.ravel()
for i, (cluster_a, cluster_b) in enumerate(list(itr.product(order_labels[:], order_labels[:]))):
         
    idx_a = np.where(order == cluster_a)[0]
    idx_b = np.where(order == cluster_b)[0]
    
    # PDF
    eig_kdea = np.vstack(eig_kde)[idx_a] #np.mean(np.vstack(eig_kde)[idx_a], axis=0)
    eig_kdeb = np.vstack(eig_kde)[idx_b] #np.mean(np.vstack(eig_kde)[idx_b], axis=0)

    for density in eig_kdea:
        axs[i].plot(x_d, density, '-', linewidth=0.5, color=COLORS[cluster_a]) #label=f'{name} - N = {len(ew)} nodes', 

    for density in eig_kdeb:
        axs[i].plot(x_d, density, '-', linewidth=0.5, color=COLORS[cluster_b]) #label=f'{name} - N = {len(ew)} nodes',
        
    axs[i].get_yaxis().set_visible(False)    
    axs[i].set_title(f'{cluster_a} - {cluster_b}')

    sns.despine(offset=10, left=True, trim=True)

# fig.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs', 'kdes.eps'), transparent=True, bbox_inches='tight', dpi=300)
plt.show() 
plt.close()    
