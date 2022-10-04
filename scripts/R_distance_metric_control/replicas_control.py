# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:20:33 2021

@author: Estefany Suarez
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
import pandas as pd

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

#%%
def get_order_flag():
    return np.array([1 if label in order_labels else 0 for label in info.Order])

def get_name_flag():

    name_flag = np.zeros_like(info.Id, dtype=int)
    for name in np.unique(info.Name):
        idx_name = np.where(info.Name == name)[0]
        name_flag[np.random.choice(idx_name, 1)] = 1

    return name_flag

#%%
distances = [
            'spec_dist_eigkde_cosine',
            'spec_dist_eigkde_euclid',
            'spec_dist_eig_cosine',
            'spec_dist_eig_euclid',
            ]

#%%
RES_DIR = os.path.join(RAW_DIR, 'distance_metric_control', 'zscored')
for distance in distances:

    print(f'\n----------- {distance} ------------')

    dist = np.load(os.path.join(RES_DIR,  f'{distance}.npy'))
    order_flag = get_order_flag()

    avg_dist = []
    name_flags = []
    for i in range(10000):
        print(i)
        name_flag  = get_name_flag()
        name_flags.append(name_flag[np.newaxis,:])
        flag = np.logical_and(order_flag, name_flag)
        avg_dist.append(dist.copy()[np.ix_(flag==1, flag==1)])

    avg_dist = np.dstack(avg_dist)
    avg_dist = np.mean(avg_dist, axis=2)
    np.save(os.path.join(RES_DIR, f'avg_{distance}'), avg_dist)

    name_flags = np.vstack(name_flags)
    np.save(os.path.join(RES_DIR, f'avg_{distance}_name_flags'), name_flags)

#%%
RES_DIR = os.path.join(RAW_DIR, 'distance_metric_control', 'nonzscored')
for distance in distances:

    print(f'\n----------- {distance} ------------')

    dist = np.load(os.path.join(RES_DIR,  f'{distance}.npy'))
    order_flag = get_order_flag()

    avg_dist = []
    name_flags = []
    for i in range(10000):
        print(i)
        name_flag  = get_name_flag()
        name_flags.append(name_flag[np.newaxis,:])
        flag = np.logical_and(order_flag, name_flag)
        avg_dist.append(dist.copy()[np.ix_(flag==1, flag==1)])

    avg_dist = np.dstack(avg_dist)
    avg_dist = np.mean(avg_dist, axis=2)
    np.save(os.path.join(RES_DIR, f'avg_{distance}'), avg_dist)

    name_flags = np.vstack(name_flags)
    np.save(os.path.join(RES_DIR, f'avg_{distance}_name_flags'), name_flags)
