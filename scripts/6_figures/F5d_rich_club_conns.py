# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 09:34:22 2021

@author: Estefany Suarez
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import re
import numpy as np
import pandas as pd
import statsmodels.stats.multitest as multi

import matplotlib.pyplot as plt
import seaborn as sns

#%%
PROJ_DIR = 'E:/P9_EIG'
DATA_DIR = os.path.join(PROJ_DIR, 'data')
CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami_v2', 'conn')
COOR_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami_v2', 'coords')
INFO_DIR = os.path.join(DATA_DIR, 'info')

RAW_RES_DIR = os.path.join(PROJ_DIR, 'raw_results', 'rich_club_gamma_1.0')

#%%
data = pd.read_csv(os.path.join(INFO_DIR, 'list2.csv'), dtype={'Name':str})
C = (200*199)/2

#%%
filenames  = []
names      = []
order      = []
superorder = []
family     = []
conn = []
for file in os.listdir(CONN_DIR):
    filename = file.split('.')[0]
    name = ''.join([i for i in filename if not i.isdigit()])
    name = ' '.join(re.findall('[A-Z][^A-Z]*', name))
    
    conn.append(np.load(os.path.join(CONN_DIR, file)))

    names.append(name)
    filenames.append(filename)
    order.append(str(data.loc[data.Name == name]['Order'].values[0]))
    superorder.append(str(data.loc[data.Name == name]['Superorder'].values[0]))
    family.append(str(data.loc[data.Name == name]['Family'].values[0]))

filenames  = np.array(filenames)
names      = np.array(names)
order      = np.array(order)
superorder = np.array(superorder)
family     = np.array(family)
conn = np.dstack(conn)

#%%
order_labels = [
                'Chiroptera',
                'Rodentia',
                'Artiodactyla',
                'Carnivora',
                'Perissodactyla',
                'Primates',
                ]

COLORS = {k:v for k,v in zip(order_labels, sns.color_palette("Set3", len(order_labels)))}


#%%
idx = np.tril_indices(200, -1)
n_modules = []
hub_nodes       = []
# provincial_hubs = []
# connector_hubs  = []
rich_club_nodes = []
rich_club_conns = []
feeder_conns    = []
local_conns     = []
total_conns     = []
max_rc_coeff = []
for i, filename in enumerate(filenames):

    # identification and classification of hubs
    deg = np.load(os.path.join(RAW_RES_DIR, f'{filename}_deg.npy'))
    # pi  = np.load(os.path.join(RAW_RES_DIR, f'{filename}_participation_idx.npy'))
    
    mean_deg = np.mean(deg)
    std_deg = np.std(deg)
    hub_threshold = mean_deg + std_deg
    hubs = deg >= hub_threshold 
    # connector = pi > 0.5 
    # provincial = pi <= 0.5
    
    # connector  = np.logical_and(hubs, connector)
    # provincial = np.logical_and(hubs, provincial)
    
               
    # rich-club coefficient as a function of k
    knodes = np.load(os.path.join(RAW_RES_DIR, f'{filename}_knodes.npy'))
    phi = np.load(os.path.join(RAW_RES_DIR, f'{filename}_phi.npy'))
    phi_nulls = np.load(os.path.join(RAW_RES_DIR, f'{filename}_phi_nulls.npy'))
    phi_random = np.mean(phi_nulls, axis=0)
    phi_norm = phi/phi_random
    
    
    # significance
    pvals = np.load(os.path.join(RAW_RES_DIR, f'{filename}_unc_pvals.npy'))
    _, cpvals = multi.fdrcorrection(pvals[phi_norm > 1.00], alpha=0.05, method='indep', is_sorted=False)
    pvals[phi_norm > 1.00] = cpvals
    
    # variables to plot 
    y = phi_norm.copy() # normalized rich-club coefficient
    x = np.arange(len(phi_norm))+1 # klevels 
            
    # filter k-levels according to rich-club organization, significance 
    # and number of nodes with degree > k 
    rich_club = phi_norm > 1.00
    significant = pvals < 0.001
    # at_least_10nodes = knodes > 10
    # mask = np.logical_and(np.logical_and(rich_club, significant), at_least_10nodes)
    mask = np.logical_and(rich_club, significant)
    
    # values to plot
    x_plot = x[mask]
    y_plot = y[mask]
    
    # identify k-level with highest rich-club coefficient
    k = x_plot[np.argmax(y_plot)]
    max_phi = np.max(y_plot)
    max_rc_coeff.append(max_phi)

    # identify nodes in k-core
    nodes_kcore = np.zeros((200))
    nodes_kcore[np.where(deg > k)[0]] = 1
    nodes_kcore = nodes_kcore.astype(int).astype(bool)
    
    # identify rich-club nodes
    rc_nodes = np.logical_and(hubs, nodes_kcore)
    rich_club_nodes.append(np.sum(rc_nodes.astype(int)))      
   
    # hubs: provincial and connector
    cis = np.load(os.path.join(RAW_RES_DIR, f'{filename}_communities.npy'))
    n_modules.append(len(np.unique(cis)))
    hub_nodes.append(np.sum(hubs.astype(int)))
    # provincial_hubs.append(np.sum(provincial.astype(int)))
    # connector_hubs.append(np.sum(connector.astype(int)))

    # rich-club connections
    conn_ = conn[:,:,i][np.ix_(np.where(rc_nodes)[0],np.where(rc_nodes)[0])]
    rich_club_conns.append(np.sum(conn_[np.tril_indices_from(conn_, -1)]))
    
    # feeder connections
    conn_ = conn[:,:,i][np.ix_(np.where(rc_nodes)[0],np.where(np.logical_not(rc_nodes))[0])]
    feeder_conns.append(np.sum(conn_))
    
    # local connections
    conn_ = conn[:,:,i][np.ix_(np.where(np.logical_not(rc_nodes))[0],np.where(np.logical_not(rc_nodes))[0])]
    local_conns.append(np.sum(conn_[np.tril_indices_from(conn_, -1)]))
   
    total_conns.append(np.sum(conn[:,:,i][np.tril_indices(200,-1)]))
   

#%%
df_rc = pd.DataFrame(np.column_stack([np.arange(225), names, filenames, order, max_rc_coeff]), \
                       columns=['Id', 'Name', 'Filename', 'Order', 'RC'])
df_rc = df_rc.astype({'RC': float})
df_rc = pd.concat([df_rc.loc[df_rc['Order'] == o] for o in order_labels]).reset_index(drop=True)

sns.set(style="ticks", font_scale=2.0)
fig, axs = plt.subplots(1,1,figsize=(14,8))
sns.violinplot(data=df_rc,
                y='RC',
                x='Order',
                # s=300,
                # hue='Order',
                palette=sns.color_palette("Set3", len(np.unique(df_rc['Order']))),
                ax=axs
                )
plt.ylim(0.8,1.8)
sns.despine(offset=10, trim=True)
# fig.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs', f'RC.eps'), transparent=True, bbox_inches='tight', dpi=300)
plt.show()
plt.close()

#%%
df_conns = pd.DataFrame(np.column_stack([np.arange(225), names, filenames, order, rich_club_conns, feeder_conns, local_conns, total_conns]), \
                        columns=['Id', 'Name', 'Filename', 'Order', 'rich club', 'feeder', 'local', 'total'])

df_conns = df_conns.astype({'rich club': float})
df_conns = df_conns.astype({'feeder': float})
df_conns = df_conns.astype({'local': float})
df_conns = df_conns.astype({'total': float})

df_conns = pd.concat([df_conns.loc[df_conns['Order'] == o] for o in order_labels]).reset_index(drop=True)

#%%
df_conns['rich club'] = df_conns['rich club']/df_conns['total'] 
df_conns['feeder']    = df_conns['feeder']/df_conns['total']  
df_conns['local']     = df_conns['local']/df_conns['total']

df_ = pd.melt(df_conns, id_vars=['Id','Order'], 
              value_vars=['rich club', 'feeder', 'local'], 
              var_name='conn_type',
              value_name='conn_density')

# plot
sns.set(style="ticks", font_scale=2.0) 
fig, ax = plt.subplots(1,1,figsize=(16,5))
sns.barplot(x="conn_type", 
            y='conn_density', 
            data=df_, 
            hue="Order", 
            ax=ax, 
            palette=sns.color_palette('Set3', len(order_labels))
            )

ax.get_legend().remove()
# ax.set_ylim(0,1.0)
ax.set_ylabel('avg. proportion of connections')
sns.despine(offset=10, trim=True)
# fig.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs','rich_club_conns.eps'), transparent=True, bbox_inches='tight', dpi=300)
plt.show()
plt.close()


#%%
df_modules = pd.DataFrame(np.column_stack([np.arange(225), names, filenames, order, n_modules]), \
                       columns=['Id', 'Name', 'Filenames',  'Order', 'modules'])
df_modules = df_modules.astype({'modules': int})
df_modules = pd.concat([df_modules.loc[df_modules['Order'] == o] for o in order_labels]).reset_index(drop=True)


sns.set(style="ticks", font_scale=2.0) 
fig, ax = plt.subplots(1,1,figsize=(16,5))
sns.barplot(x='Order', 
            y='modules', 
            data=df_modules, 
            palette=sns.color_palette('Set3', len(order_labels)), 
            ax=ax)
sns.despine(offset=10, trim=True)
# fig.savefig(fname=os.path.join('C:/Users/User/Dropbox/figs', f'l_{pa}-{pb}_split.png'), transparent=True, bbox_inches='tight', dpi=300)
plt.show()
plt.close()
