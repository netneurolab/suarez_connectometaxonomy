# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 13:26:52 2021

@author: Estefany Suarez
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os

import numpy as np
import pandas as pd

import eigenfunctions as fn

#%%
RESOLUTION = '100'
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami', f'conn_{RESOLUTION}')
INFO_DIR = os.path.join(DATA_DIR, 'info')
RAW_DIR  = os.path.join(PROJ_DIR, 'raw_results', f'res_{RESOLUTION}')

try:
    os.mkdir(RAW_DIR)
except:
    pass

#%%
df_info = pd.read_csv(os.path.join(INFO_DIR, 'info.csv'), dtype={'Name':str})

#%%
eig = []
for file in os.listdir(CONN_DIR):
    
    # weighted network
    wei_conn = np.load(os.path.join(CONN_DIR, file)) 

    # binary network
    bin_conn = wei_conn.astype(bool).astype(int)

    # convert directed graph to undirected graph
    if not fn.check_symmetric(bin_conn): bin_conn = fn.dir2und(bin_conn)

    # normalized Laplacian
    l = fn.norm_laplacian(bin_conn, degree='out')

    # Laplacian eigenspectrum
    ew = np.real(fn.eigen_spectrum(l))
    eig.append(ew)


# %% eigen spectra
eig = np.sort(np.array(eig), axis=1)
df_eig = pd.DataFrame(eig, columns=[f'eig_{i}' for i in range(eig.shape[1])], dtype=float)
df_eig = pd.concat([df_info, df_eig], axis=1)
df_eig.to_csv(os.path.join(RAW_DIR, 'eig.csv'), index=False)


# %% kernel density approximation
x_d, dx_d, eig_kde = fn.get_eigen_kde(eigenspectra=eig)
eig_kde = np.array(eig_kde)
df_eig_kde = pd.DataFrame(eig_kde, columns=[f'eig_kde_{i}' for i in range(eig_kde.shape[1])], dtype=float)
df_eig_kde = pd.concat([df_info, df_eig_kde], axis=1)
df_eig_kde.to_csv(os.path.join(RAW_DIR, 'eig_kde.csv'), index=False)
