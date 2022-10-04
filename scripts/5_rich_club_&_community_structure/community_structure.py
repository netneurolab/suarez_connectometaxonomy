# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 14:49:08 2021

@author: Estefany Suarez
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import time

import numpy as np
from netneurotools import modularity

import multiprocessing as mp

#%%
RESOLUTION = '100'
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami', f'conn_{RESOLUTION}')
COOR_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami', 'coords')
INFO_DIR = os.path.join(DATA_DIR, 'info')
RAW_DIR  = os.path.join(PROJ_DIR, 'raw_results', f'res_{RESOLUTION}')

RND_SEED = 1234

#%% ---------------------------------------------------------------------------
# Identification of Hubs
# -----------------------------------------------------------------------------
def community_detection(file, gammas):

        print(f'-------{file}--------')

        filename = file.split('.')[0]
        if not os.path.exists(os.path.join(RAW_DIR, 'community_detection', f'{filename}_communities.npy')):

            # binary connectivity matrix
            w = np.load(os.path.join(CONN_DIR, file)).astype(bool).astype(int)

            # consensus community detection
            np.random.default_rng(RND_SEED)

            ci = []
            q  = []
            for gamma in gammas:
                ci_, q_, _ = modularity.consensus_modularity(w, gamma=gamma)
                ci.append(ci_)
                q.append(q_)

            np.save(os.path.join(RAW_DIR, 'community_detection', f'{filename}_modularity_scores'), q)
            np.save(os.path.join(RAW_DIR, 'community_detection', f'{filename}_communities'), ci)


#%%
def main():

    # hubs
    print ('\nINITIATING PROCESSING TIME - PARTICIPATION INDEX')
    t0 = time.perf_counter()

    params = [{'file':file, 'gammas':np.linspace(0.5,1.5,11)} for file in os.listdir(CONN_DIR)]

    # run iterations for all subjects in CONN_DIR
    pool1 = mp.Pool(processes=5)
    res1 = [pool1.apply_async(community_detection, (), p) for p in params]
    for r in res1: r.get()
    pool1.close()

    print ('\nTOTAL PROCESSING TIME - PARTICIPATION INDEX')
    print (time.perf_counter()-t0, "seconds process time")
    print ('END')

if __name__ == '__main__':
    main()
