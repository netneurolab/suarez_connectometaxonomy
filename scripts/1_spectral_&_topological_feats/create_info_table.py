# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 13:26:52 2021

@author: Estefany Suarez
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import re
import numpy as np
import pandas as pd
from scipy import stats

import eigenfunctions as fn


#%%
RESOLUTION = '100'
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami', f'conn_{RESOLUTION}')
INFO_DIR = os.path.join(DATA_DIR, 'info')
RAW_DIR  = os.path.join(PROJ_DIR, 'raw_results', f'res_{RESOLUTION}')

#%%
data1 = pd.read_csv(os.path.join(INFO_DIR, 'list1.csv'), dtype={'Name':str})
data2 = pd.read_csv(os.path.join(INFO_DIR, 'list2.csv'), dtype={'Name':str})

#%%
superorder = []
order = []
suborder = []
family = []
subfamily=[]
genus = []
species = []
names = []
common_names = []
filenames = []

for i, file in enumerate(os.listdir(CONN_DIR)):
        
    filename = file.split('.')[0]
    name = ''.join([i for i in filename if not i.isdigit()])
    name = ' '.join(re.findall('[A-Z][^A-Z]*', name))

    tmp_data1 = data1.loc[data1.Name == name]
    tmp_data2 = data2.loc[data2.Filename == filename]
    
    try:
        suborder_  = str(tmp_data2['Sub-Order'].values[0])
    except:        
        suborder_  = ''
    
    try:
        subfamily_ = str(tmp_data2['Sub-Family'].values[0])
    except:
        subfamily_ = ''
    
    try:
        genus_     = str(tmp_data2['Genus'].values[0])
    except:        
        genus_     = ''
    
    try:
        species_   = str(tmp_data2['Species'].values[0])
    except:
        species_   = ''
    
    superorder.append(str(tmp_data1['Super-Order'].values[0]))
    order.append(str(tmp_data1['Order'].values[0]))
    suborder.append(suborder_)
    family.append(str(tmp_data1['Family'].values[0]))
    subfamily.append(subfamily_)
    genus.append(genus_)
    species.append(species_)
    names.append(name)
    filenames.append(filename)
    common_names.append(str(tmp_data1['Common-Name'].values[0]))

#%%
columns = [
    'Id',
    'Super-Order',
    'Order',
    'Sub-Order',
    'Family',
    'Sub-Family',
    'Genus',
    'Species',
    'Name',
    'Common-Name',
    'Filename'
    ]

col_values = np.column_stack([np.arange(len(filenames)), 
                              superorder, 
                              order, 
                              suborder, 
                              family, 
                              subfamily,
                              genus,
                              species,
                              names, 
                              common_names, 
                              filenames, 
                              ])

df_info = pd.DataFrame(data=col_values,
                        columns=columns,
                        index=None)

df_info.to_csv(os.path.join(INFO_DIR, 'info.csv'), index=False)
