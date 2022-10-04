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

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns


#%%
RESOLUTION = '100'
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami', f'conn_{RESOLUTION}')
INFO_DIR = os.path.join(DATA_DIR, 'info')
RAW_DIR  = os.path.join(PROJ_DIR, 'raw_results', f'res_{RESOLUTION}')

#%%
info = pd.read_csv(os.path.join(INFO_DIR, 'info.csv'))
order = info['Order']

#%%
def fig4(distance):

    spectral_distance = np.load(os.path.join(RAW_DIR, 'avg_spectral_distance.npy'))
    topological_distance = np.load(os.path.join(RAW_DIR, f'avg_{distance}.npy'))

    # spectral_distance = np.load(os.path.join(RAW_DIR, 'spectral_distance.npy'))
    # topological_distance = np.load(os.path.join(RAW_DIR, f'{distance}.npy'))

    idx = np.tril_indices_from(spectral_distance, -1)

    spectral_distance = spectral_distance[idx]
    topological_distance = topological_distance[idx]

    topological_distance = (topological_distance-np.min(topological_distance))/(np.max(topological_distance)-np.min(topological_distance))
    spectral_distance = (spectral_distance-np.min(spectral_distance))/(np.max(spectral_distance)-np.min(spectral_distance))

    # plot
    sns.set(style="ticks", font_scale=2.0)
    # fig = plt.figure(figsize=(12,10))

    r = np.round(np.corrcoef(topological_distance, spectral_distance)[0][1], 5)
    print(r)

    ax = sns.jointplot(x=topological_distance,
                  y=spectral_distance,
                  label=f'R: {r}',
                  kind="hex",
                  color="#AE7895",
                  rasterized=True,
                  # palette=sns.color_palette("flare", as_cmap=True),
                  )


    ax.ax_joint.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.ax_joint.set_xlabel(distance)
    ax.ax_joint.set_ylabel('spectral distance')
    # plt.xticks(rotation=45)
    # ax.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs', f'spectral_vs_{distance}.eps'), transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

#%%
distances = [
            'topological_distance',
            'top_bin_dist',
            'top_wei_dist',
            'top_local_dist',
            'top_global_dist',
            'top_local_bin_dist',
            'top_local_wei_dist',
            'top_global_bin_dist',
            'top_global_wei_dist',
            ]


for distance in distances:

    print(f'\n----------- {distance} ------------')

    fig4(distance)
