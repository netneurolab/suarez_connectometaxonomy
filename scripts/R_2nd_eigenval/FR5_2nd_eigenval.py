# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 12:38:22 2021

@author: Estefany Suarez
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
import pandas as pd

import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import (squareform)
from scipy.cluster.hierarchy import dendrogram, linkage

#%%
RESOLUTION = '100'
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami', f'conn_{RESOLUTION}')
INFO_DIR = os.path.join(DATA_DIR, 'info')
RAW_DIR  = os.path.join(PROJ_DIR, 'raw_results', f'res_{RESOLUTION}')

#%%
info = pd.read_csv(os.path.join(INFO_DIR, 'info.csv'))

#%%
order_labels = [
                'Chiroptera',
                'Rodentia',
                'Cetartiodactyla',
                'Carnivora',
                'Perissodactyla',
                'Primates',
                ]

COLORS = {k:v for k,v in zip(order_labels, sns.color_palette("Set3", len(order_labels)))} 

#%%
df_eig = pd.read_csv(os.path.join(RAW_DIR, 'eig.csv'))
eig = df_eig.values[:,-200:].astype(float)
eig_round = np.round(eig, 3)

#%%
lambda_1 = []
lambda_2 = []
first_nonzer0_lambda = []
optimal_n_comunnities = []
frequency_lambda_eq2_1 = []
lambda_n = []
for i in range(len(eig)):
    
    eig_ = eig[i]
    eig_round_ = eig_round[i]
    
    # value of first eigenvalue
    lambda_1.append(eig_round_[0])
    
    # value of second eigenvalue
    lambda_2.append(eig_round_[1])
    
    # index of first nonzero lambda
    first_nonzer0_lambda.append(np.where(eig_round_ > 0)[0][0])
    
    # optimal number of communities
    eigen_gap = [eig_round_[i+1]-eig_round_[i] for i in range(len(eig_round_)-1)]
    optimal_n_comunnities.append(np.argmax(eigen_gap))
    
    # frequency of eigenvalues ~ 1.0
    hist, bin_edges = np.histogram(eig_round_, density=False)
    diff = np.abs(bin_edges-1)
    bin_idx = np.argmin(diff)-1
    frequency_lambda_eq2_1.append(hist[bin_idx])
    
    # value of last eigenvalue
    lambda_n.append(eig_round_[-1])
    

#%%
lambda_1 = np.array(lambda_1)[:, np.newaxis]
lambda_2 = np.array(lambda_2)[:, np.newaxis]
first_nonzer0_lambda = np.array(first_nonzer0_lambda)[:, np.newaxis]
optimal_n_comunnities = np.array(optimal_n_comunnities)[:, np.newaxis]
frequency_lambda_eq2_1 = np.array(frequency_lambda_eq2_1)[:, np.newaxis]
lambda_n = np.array(lambda_n)[:, np.newaxis]


df = pd.DataFrame(data=np.column_stack([lambda_1, lambda_2, first_nonzer0_lambda, optimal_n_comunnities, frequency_lambda_eq2_1, lambda_n]),
                  columns=['eig 1', 'eig 2', 'first_nonzero_eig', 'optimal communities', 'frequency around one', 'eig n'],
                       index=None)

df = pd.concat([info, df], axis=1)

#%%
order_flag = np.array([1 if label in order_labels else 0 for label in df.Order])
df = df[order_flag == 1]

#%%
sns.set(style="ticks", font_scale=1.5, palette=sns.color_palette('Set3', len(order_labels)))  #palette='mako',
fig, ax = plt.subplots(1,1,figsize=(10,4))

y = 'eig 2'
df[y] = df[y].astype(float)

plot = sns.boxplot(data=df, x='Order', y=y,
                   order=order_labels, 
                            palette=sns.color_palette('Set3', len(order_labels)),
                            width=0.5,
                            # hue='model',
                            showfliers=True,
                          )

ax.set_ylim(0,0.6)
# plot._legend.remove()
sns.despine(offset=10, trim=False)
# fig.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs', f'boxplt_{y}.eps'), transparent=True, bbox_inches='tight', dpi=300)
plt.show()
plt.close()












