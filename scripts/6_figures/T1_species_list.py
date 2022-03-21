# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 16:39:13 2022

@author: Estefany Suarez
"""


import os

import numpy as np
import pandas as pd


order_labels = [
                'Chiroptera',
                'Rodentia',
                'Artiodactyla',
                'Carnivora',
                'Perissodactyla',
                'Primates',
               ]

df_info = pd.read_csv('E:/P9_EIG/raw_results/info.csv', index_col='Id')
names, counts = np.unique(df_info['Name'], return_counts=True)

flag = np.array([1 if df_info.loc[df_info.Name == n]['Order'].values[0] in order_labels else 0 for n in names])

names = names[flag == 1]
counts = counts[flag == 1]

#%%
order = []
superorder = []
index = []
for i, n in enumerate(names):
    index.append(i+1)
    order.append(str(df_info.loc[df_info.Name == n]['Order'].values[0]))
    superorder.append(str(df_info.loc[df_info.Name == n]['Superorder'].values[0]))

#%%
new_df = pd.DataFrame(data=np.column_stack([index, names, order, superorder, counts]),
                       columns=['Id','Name', 'Order', 'Superorder', 'Qty'],
                       index=None)

table = new_df.to_latex(index=False, longtable=True)