# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:30:00 2022

@author: Estefany Suarez
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import re
import random

import numpy as np
import pandas as pd
import scipy.io as sio


#%%
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
INFO_DIR = os.path.join(DATA_DIR, 'info')
CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami')
RAW_DATA_DIR = 'C:/Users/User/OneDrive - McGill University/xnodes/'


#%%
info = pd.read_csv(os.path.join(INFO_DIR, 'info.csv'))

#%%
for file in os.listdir(RAW_DATA_DIR)[:]:
    
    conn = sio.loadmat(os.path.join(RAW_DATA_DIR, file))['Tconmat']
    
    filename = file.split('_')[0][:-1]
    resolution = ''.join([i for i in file.split('_')[-1] if i.isdigit()])
    foldername = f'conn_{resolution}'
    
    directory = os.path.join(CONN_DIR, foldername)
    
    try: 
        os.mkdir(directory)
    except:
        pass
    
    np.save(os.path.join(directory, filename), conn)
