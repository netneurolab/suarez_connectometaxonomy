# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 09:10:37 2021

@author: Estefany Suarez
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator

import seaborn as sns


#%%
PROJ_DIR = 'E:/P9_EIG'
DATA_DIR = os.path.join(PROJ_DIR, 'data')
CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami_v2', 'conn')
INFO_DIR = os.path.join(DATA_DIR, 'info')
RAW_DIR = os.path.join(PROJ_DIR, 'raw_results')

#%% network properties empirical networks
include_properties = [
                        'avg_wei_clustering_coeff',
                        'avg_bin_clustering_coeff',
                        'wei_transitivity',
                        'bin_transitivity',
                        'wei_char_path',
                        'bin_char_path',
                     ]

df = pd.read_csv(os.path.join(RAW_DIR, 'df_props.csv'))
df = df.loc[:,['Id','Name', 'Filename', 'Order'] + include_properties]

#%% network properties null models
df_null = []
for file in os.listdir(CONN_DIR):
    filename = file.split('.')[0] 
    props = pd.read_csv(os.path.join(RAW_DIR, 'nulls', f'{filename}_null_props.csv'))
    props = props.loc[:,include_properties]
        
    df_null.append(np.mean(props, axis=0))
        
df_null = pd.concat(df_null, axis=1).T
df_null = pd.concat([df.iloc[:,:4], df_null], axis=1)


#%% SW index

# C/Crandom ratio
df['C_ratio_wei'] = df['wei_transitivity']/df_null['wei_transitivity']
df['C_ratio_bin'] = df['bin_transitivity']/df_null['bin_transitivity']

# L/Lrandom ratio
df['L_ratio_wei'] = df['wei_char_path']/df_null['wei_char_path']
df['L_ratio_bin'] = df['bin_char_path']/df_null['bin_char_path']

# small-world index
df['sw_bin'] = df['C_ratio_bin']/df['L_ratio_bin']
df['sw_wei'] = df['C_ratio_wei']/df['L_ratio_wei']

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

#%%
def error_bars(df, title):
    
    C_avg = []
    L_avg = []
    C_std = []
    L_std = []
    for o in order_labels:
        df_tmp = df.loc[df['Order'] == o, :]
        
        C_avg.append(np.median(df_tmp[f'C_ratio_{title}'].values))
        L_avg.append(np.median(df_tmp[f'L_ratio_{title}'].values))
    
        C_std.append(df_tmp[f'C_ratio_{title}'].values.std())
        L_std.append(df_tmp[f'L_ratio_{title}'].values.std())
        
        
    sns.set(style="ticks", font_scale=1.5)
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    
    for i, _ in enumerate(order_labels):
        plt.errorbar(x=L_avg[i],y=C_avg[i], 
                     yerr=C_std[i], xerr=L_std[i], 
                     fmt="o", c=sns.color_palette('Set3', len(order_labels)).as_hex()[i]
                     )
        
        plt.scatter(x=L_avg[i], y=C_avg[i], 
                    c=sns.color_palette('Set3', len(order_labels)).as_hex()[i], 
                    s=190, 
                    )
    
    plt.plot([0,2], [0,2], c='k', linestyle='--')
    
    plt.xlim(1,1.25)
    plt.ylim(1,2)
    
    plt.ylabel(f'C_ratio_{title}')
    plt.xlabel(f'L_ratio_{title}')
    sns.despine(offset=10, trim=True)
    # fig.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs', f'sw_error_{title}.eps'), transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

error_bars(df, title='bin')
# error_bars(df, title='wei')


#%%
def inset(df, title):
    sns.set(style="ticks", font_scale=1.5, palette=sns.color_palette('Set3', len(order_labels)))
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    plot = sns.scatterplot(data=df, x=f'L_ratio_{title}', y=f'C_ratio_{title}',
                          palette=sns.color_palette('Set3', len(order_labels)),
                          hue='Order',
                          s=250,
                          )
    
    plt.xlim(1,1.25)
    plt.ylim(1,2)
    
    plt.plot([0,2], [0,2], c='k', linestyle='--')
    # plot.legend().remove()
    
    plt.ylabel(f'C_ratio_{title}')
    plt.xlabel(f'L_ratio_{title}')

    sns.despine(offset=10, trim=True)
    # fig.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs', f'sw_inset_{title}.eps'), transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

inset(df, title='bin')
# inset(df, title='wei')
# 
#%%
def scatterplot(df, title):
    sns.set(style="ticks", font_scale=1.5, palette=sns.color_palette('Set3', len(order_labels)))
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    plot = sns.scatterplot(data=df, x=f'L_ratio_{title}', y=f'C_ratio_{title}',
                          palette=sns.color_palette('Set3', len(order_labels)),
                          hue='Order',
                          s=115,
                          )
    plt.xlim(1,2)
    plt.ylim(1,2)
    
    plt.plot([0,2], [0,2], c='k', linestyle='--')
    # plot.legend().remove()

    plt.ylabel(f'C_ratio_{title}')
    plt.xlabel(f'L_ratio_{title}')

    sns.despine(offset=10, trim=True)
    # fig.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs', f'sw_{title}.eps'), transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

scatterplot(df, title='bin')
# scatterplot(df, title='wei')

