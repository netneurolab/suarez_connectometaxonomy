# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 13:20:43 2021

@author: Estefany Suarez
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import time
import numpy as np
import bct

import multiprocessing as mp

#%%
RESOLUTION = '100'
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami', f'conn_{RESOLUTION}')
COOR_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami', 'coords')
INFO_DIR = os.path.join(DATA_DIR, 'info')
RAW_DIR  = os.path.join(PROJ_DIR, 'raw_results', f'res_{RESOLUTION}')

if not os.path.exists(os.path.join(RAW_DIR, 'nulls')):
    os.mkdir(os.path.join(RAW_DIR, 'nulls'))

RND_SEED = 1234

#%% ---------------------------------------------------------------------------
# Generate rewired nulls
# -----------------------------------------------------------------------------
def generate_rewired_nulls(file):

    print(f'-------{file}--------')

    # rich club coefficent of network
    G = np.load(os.path.join(CONN_DIR, file))

    # rich club coefficient nulls
    np.random.default_rng(RND_SEED)
    nulls = []
    for _ in range(1000):
        try:
            random, _ = bct.reference.randmio_und_connected(G, itr=10)

        except:
            random, _ = bct.reference.randmio_und(G, itr=10)

        nulls.append(random)

    nulls = np.dstack(nulls)

    filename = file.split('.')[0]
    np.save(os.path.join(RAW_DIR, 'nulls', f'nulls_{filename}'), nulls)


#%%
def main():

    print ('\nINITIATING PROCESSING TIME - NULLS')
    t0 = time.perf_counter()

    # run iterations for all subjects in CONN_DIR
    params = [{'file':file} for file in os.listdir(CONN_DIR)]

    pool = mp.Pool(processes=24)
    res = [pool.apply_async(generate_rewired_nulls, (), p) for p in params]
    for r in res: r.get()
    pool.close()

    print ('\nTOTAL PROCESSING TIME - NULLS')
    print (time.perf_counter()-t0, "seconds process time")
    print ('END')


if __name__ == '__main__':
    main()
