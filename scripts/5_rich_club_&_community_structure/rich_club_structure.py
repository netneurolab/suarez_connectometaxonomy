# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 14:49:08 2021

@author: Estefany Suarez
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import re
import time
import itertools as itr

import numpy as np
import pandas as pd
import bct
from scipy import stats
import statsmodels.stats.multitest as multi
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
import seaborn as sns

from rnns import topology

from netneurotools import modularity, cluster

import multiprocessing as mp


#%%
PROJ_DIR = 'E:/P9_EIG'
DATA_DIR = os.path.join(PROJ_DIR, 'data')
CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami_v2', 'conn')
COOR_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami_v2', 'coords')
INFO_DIR = os.path.join(DATA_DIR, 'info')
RAW_RES_DIR = os.path.join(PROJ_DIR, 'raw_results')

RND_SEED = 1234

#%% ---------------------------------------------------------------------------
# Identification of Hubs
# -----------------------------------------------------------------------------
def participation_index(file):

        print(f'-------{file}--------')

        filename = file.split('.')[0]
        if not os.path.exists(os.path.join(RAW_RES_DIR, 'rich_club', f'{filename}_deg.npy')):

            # binary connectivity matrix
            w = np.load(os.path.join(CONN_DIR, file)).astype(bool).astype(int)

            # consensus community detection
            np.random.default_rng(RND_SEED)
            ci, q, _ = modularity.consensus_modularity(w, gamma=1.0)
            cis = np.unique(ci)
            # _, q = bct.modularity.modularity_und(w, kci=ci)
            print('------- end community detection -------')

            # node degree
            ki = np.sum(w, axis=1)

            # participation coefficient
            pi = []
            for node in range(len(w)):
                ratio = []
                for c in cis:
                    kis = np.sum(w[node][np.where(ci == c)])
                    ratio.append((kis/ki[node])**2)

                pi.append(1-np.sum(ratio))

            np.save(os.path.join(RAW_RES_DIR, 'rich_club', f'{filename}_deg'), ki)
            np.save(os.path.join(RAW_RES_DIR, 'rich_club', f'{filename}_participation_idx'), pi)
            np.save(os.path.join(RAW_RES_DIR, 'rich_club', f'{filename}_modularity_scores'), q)
            np.save(os.path.join(RAW_RES_DIR, 'rich_club', f'{filename}_communities'), ci)


#%% ---------------------------------------------------------------------------
# Rich-club coefficient
# -----------------------------------------------------------------------------
def rich_club(file):

    print(f'-------{file}--------')
    filename = file.split('.')[0]

    # binary connectivity matrix
    G = np.load(os.path.join(CONN_DIR, file)).astype(bool).astype(int)

    # rich club coefficent of network
    rc, knodes, _ = bct.core.rich_club_bu(G)

    # rich club coefficient nulls
    random = np.load(os.path.join(RAW_RES_DIR, 'nulls', f'nulls_{filename}.npy')).astype(bool).astype(int)
    rc_nulls = []
    for i in range(random.shape[-1]):
        rc_, _, _ = bct.core.rich_club_bu(random[:,:,i])
        rc_nulls.append(rc_)

    # normalized rich club coefficient
    rc_random = np.mean(rc_nulls, axis=0)
    rc_norm = rc/rc_random

    # statistical significance
    zrc_tmp = stats.zscore(np.row_stack([rc, rc_nulls]), axis=0)

    p_vals = []
    for k in range(zrc_tmp.shape[-1]):
        upper = len(np.where(zrc_tmp[1:,k] > zrc_tmp[0,k])[0])
        p_vals.append(upper/len(rc_nulls))

    np.save(os.path.join(RAW_RES_DIR, 'rich_club', f'{filename}_phi'), rc)
    np.save(os.path.join(RAW_RES_DIR, 'rich_club', f'{filename}_knodes'), knodes)
    np.save(os.path.join(RAW_RES_DIR, 'rich_club', f'{filename}_phi_nulls'), rc_nulls)
    np.save(os.path.join(RAW_RES_DIR, 'rich_club', f'{filename}_phi_random'), rc_random)
    np.save(os.path.join(RAW_RES_DIR, 'rich_club', f'{filename}_phi_norm'), rc_norm)
    np.save(os.path.join(RAW_RES_DIR, 'rich_club', f'{filename}_unc_pvals'), p_vals)


#%%
def main():


    # hubs
    print ('\nINITIATING PROCESSING TIME - PARTICIPATION INDEX')
    t0 = time.perf_counter()

    params = [{'file':file} for file in os.listdir(CONN_DIR)]

    # run iterations for all subjects in CONN_DIR
    pool1 = mp.Pool(processes=5)
    res1 = [pool1.apply_async(participation_index, (), p) for p in params]
    for r in res1: r.get()
    pool1.close()

    print ('\nTOTAL PROCESSING TIME - PARTICIPATION INDEX')
    print (time.perf_counter()-t0, "seconds process time")
    print ('END')


    # rich club organization
    print ('\nINITIATING PROCESSING TIME - RICH CLUB')
    t0 = time.perf_counter()

    params = [{'file':file} for file in os.listdir(CONN_DIR)]

    # run iterations for all subjects in CONN_DIR
    pool2 = mp.Pool(processes=24)
    res2 = [pool2.apply_async(rich_club, (), p) for p in params]
    for r in res2: r.get()
    pool2.close()

    print ('\nTOTAL PROCESSING TIME - RICH CLUB')
    print (time.perf_counter()-t0, "seconds process time")
    print ('END')


if __name__ == '__main__':
    main()
