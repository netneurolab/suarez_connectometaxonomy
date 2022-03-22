# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 13:26:52 2021

@author: Estefany Suarez
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import re
import itertools as itr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns

from scipy import stats
from scipy.spatial.distance import (pdist,squareform)
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.preprocessing import (StandardScaler,MinMaxScaler,LabelEncoder)

import eigenfunctions as fn


#%%
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami', 'conn')
INFO_DIR = os.path.join(DATA_DIR, 'info')
RAW_DIR  = os.path.join(PROJ_DIR, 'raw_results')

#%%
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

filenames = np.array(filenames)
names = np.array(names)
order = np.array(order)
superorder = np.array(superorder)
family = np.array(family)
conn = np.dstack(conn)

#%%
df_info = pd.DataFrame(data=np.column_stack([np.arange(len(filenames)), names, filenames, order, superorder, family]),
                       columns=['Id', 'Name', 'Filename', 'Order', 'Superorder', 'Family'],
                       index=None)

df_info.to_csv(os.path.join(INFO_DIR, 'info.csv'), index=False)

#%%
eig = []
for i, filename in enumerate(filenames):

    # weighted network
    wei_conn = conn[:,:,i]

    # binary network
    bin_conn = wei_conn.astype(bool).astype(int)

    # convert directed graph to undirected graph
    if not fn.check_symmetric(bin_conn): bin_conn = fn.dir2und(bin_conn)

    # normalized Laplacian
    l = fn.norm_laplacian(bin_conn, degree='out')

    # Laplacian eigenspectrum
    ew = np.real(fn.eigen_spectrum(l))
    ew_diff = [ew[i+1]-ew[i] for i in range(0, len(ew)-1)]

    eig.append(ew)

#%% eigen spectra
eig = np.array(eig)
df_eig = pd.DataFrame(eig, columns=[f'eig_{i}' for i in range(eig.shape[1])], dtype=float)
df_eig = pd.concat([df_info, df_eig], axis=1)
df_eig.to_csv(os.path.join(RAW_DIR, 'eig.csv'), index=False)


#%% kernel density approximation
x_d, dx_d, eig_kde = fn.get_eigen_kde(eigenspectra=eig)
eig_kde = np.array(eig_kde)
df_eig_kde = pd.DataFrame(eig_kde, columns=[f'eig_kde_{i}' for i in range(eig_kde.shape[1])], dtype=float)
df_eig_kde = pd.concat([df_info, df_eig_kde], axis=1)
df_eig_kde.to_csv(os.path.join(RAW_DIR, 'eig_kde.csv'), index=False)


#%% z-score kernel density approximation
zeig_kde = stats.zscore(eig_kde, axis=0)
df_zeig_kde = pd.DataFrame(zeig_kde, columns=[f'zeig_kde_{i}' for i in range(zeig_kde.shape[1])], dtype=float)
df_zeig_kde = pd.concat([df_info, df_zeig_kde], axis=1)
df_eig_kde.to_csv(os.path.join(RAW_DIR, 'zeig_kde.csv'), index=False)
