# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 16:00:59 2022

@author: Estefany Suarez
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import random
import numbers
random.seed(1111)

import numpy as np
import pandas as pd
import scipy.stats as stats

import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as mc
 
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import cm

#%%
RESOLUTION = '100'
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami', f'conn_{RESOLUTION}')
COOR_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami', 'coords')
INFO_DIR = os.path.join(DATA_DIR, 'info')
RAW_DIR  = os.path.join(PROJ_DIR, 'raw_results', f'res_{RESOLUTION}')

#%%
info = pd.read_csv(os.path.join(INFO_DIR, 'info.csv'))
extra_info = pd.read_csv(os.path.join(INFO_DIR, 'extra_info.csv'))

#%%
order_labels = [
                'Chiroptera',
                'Rodentia',
                'Cetartiodactyla',
                'Carnivora',
                'Perissodactyla',
                'Primates',
                ]

COLORS = {k:v for k,v in zip(order_labels, sns.color_palette("Set3", len(order_labels)))} 
flag = np.array([1 if label in order_labels else 0 for label in info.Order])
    
#%%
def rotate_z(data, teta=180):
    teta = teta*np.pi/180.
    data = data.T
    
    rot_matrix = [[np.cos(teta), -np.sin(teta), 0], 
                  [np.sin(teta), np.cos(teta), 0],
                  [0, 0, 1]]
    
    return np.dot(rot_matrix, data).T

def center_coords(coords):
    return (coords-coords.mean(axis=0))

def anova_table(aov):
    aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])
    cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
    aov = aov[cols]
    return aov

def get_color_distribution(scores, cmap="viridis", vmin=None, vmax=None):

    '''
    Function to get a color for individual values of a distribution of scores.
    '''

    n = len(scores)

    if vmin is None:
        vmin = np.amin(scores)
    if vmax is None:
        vmax = np.amax(scores)

    cmap = cm.get_cmap(cmap, 256)
    new_colors = cmap(np.linspace(0, 1, 256))

    if vmin != vmax:
        scaled = (scores - vmin)/(vmax - vmin) * 255
        scaled[scaled < 0] = 0
        scaled[scaled > 255] = 255
    else:
        scaled = np.zeros((n)) + 128

    c = np.zeros((n, 4))
    for i in range(n):
        c[i] = new_colors[int(scaled[i]), :]

    return c


def plot_network(G, coords, edge_scores, node_scores, edge_cmap="Viridis",
                 edge_alpha=0.25, edge_vmin=None, edge_vmax=None,
                 ignore_edges=[], 
                 node_cmap="viridis", node_alpha=1, node_vmin=None,
                 nodes_color='black', node_vmax=None, linewidth=0.25, s=100,
                 directed=False, projection=None, view="sagittal", view_edge=True,
                 axis=False, figsize=None,
                 node_order=None):
    '''
    Function to draw (plot) a network of nodes and edges.
    Parameters
    ----------
    G : dict or (n,n) ndarray
        Dictionary storing general information about the network we wish to
        plot or an (n,n) ndarray storing the adjacency matrix of the network.
        Where 'n' is the number of nodes in the network.
    coords : (n, 3) ndarray
        Coordinates of the network's nodes.
    edge_scores: (n,n) ndarray
        ndarray storing edge scores for individual edges in the network. These
        scores will be used to color the edges.
    node_scores : (n,) ndarray
        ndarray storing node scores for individual nodes in the network. These
        scores will be used to color the nodes.
    node_vmin, node_vmax: float, default: None
        Minimal and maximal values of the nodes colors. If None, the min and
        max of the node_scores array will be used.
    Returns
    -------
    '''

    if isinstance(G, dict):
        G = G['adj']

    if not np.all(G == G.T) and not directed:
        warnings.warn(("network appears to be directed, yet 'directed' "
                       "parameter was set to 'False'. The values of the edges "
                       "may be wrong."))

    if figsize is None:
        figsize = (10, 10)

    fig = plt.figure(figsize=figsize, 
                     # facecolor='#606161'
                     )
    ax = fig.add_subplot(111, projection=projection)

    # Identify all the edges in the network
    Edges = np.where(G > 0)

    # Get the color of the edges
    if edge_scores is None:
        edge_colors = np.full((len(Edges[0])), "black", dtype="<U10")
    else:
        edge_colors = get_color_distribution(edge_scores[Edges],
                                                    cmap=edge_cmap,
                                                    vmin=edge_vmin,
                                                    vmax=edge_vmax)
        
    # Get the color of the nodes
    if node_scores is None:
        node_scores = nodes_color

    if projection is None:

        # plot the edges
        if view_edge:
            for edge_i, edge_j, c in zip(Edges[0], Edges[1], edge_colors):

                if (edge_i, edge_j) in ignore_edges:
                    continue
                
                else:
                    x1 = coords[edge_i, 0]
                    x2 = coords[edge_j, 0]
                    y1 = coords[edge_i, 1]
                    y2 = coords[edge_j, 1]
    
                    ax.plot([x1, x2],
                            [y1, y2],
                            c=c,
                            linewidth=linewidth*edge_scores[edge_i, edge_j],
                            alpha=edge_alpha,
                            zorder=0)

        # plot the nodes
        ax.scatter(coords[:, 0],
                   coords[:, 1],
                   c=node_scores,
                   cmap=node_cmap,
                   vmin=node_vmin,
                   vmax=node_vmax,
                   clip_on=False,
                   alpha=node_alpha,
                   s=s,
                   zorder=1)

    if not axis:
        ax.axis('off')

    return fig, ax
    

#%%
local_props = [
            'node_degree',
            'bin_clustering_coeff',
            'bin_node_centrality',
            'bin_efficiency',
            ]

plot_brains = [
                'VampireBat1',
                'BlindMole1',
                'Deer1',
                'CommonFox',
                'Tapir1',
                'MacaquePT'
                ]


#%%
n = 20
for j, prop_name in enumerate(local_props):
    
    print(f'\n--------------{prop_name}-----------------------')
    local_prop = np.load(os.path.join(RAW_DIR, 'local_props', f'{prop_name}.npy'))

    front = []
    back  = []
    for i, file in enumerate(os.listdir(COOR_DIR)[:]):
        
        filename = file.split('.')[0]
        order = info['Order'][i]
        orientation = extra_info['Brain Orientation'][i]
    
        if order in order_labels:

            # conn 
            conn = np.load(os.path.join(CONN_DIR, file))
            
            # coords
            coords = np.load(os.path.join(COOR_DIR, file))
            coords = center_coords(coords)
            if orientation == 'LR':
                coords = rotate_z(coords)
    
            # node selection
            x = coords[:,0]
            x_sort = np.argsort(x)
            front_nodes = list(x_sort[-n:])
            back_nodes  = list(x_sort[:n]) 
            
            prop = local_prop[i]
            front.append(np.mean(prop[front_nodes]).astype(float))
            back.append(np.mean(prop[back_nodes]).astype(float))
   
            # # -------------------- plot brains
            # if (j == 0) and (filename in plot_brains):
                
            #     # node scores = colors 
            #     node_scores = np.array(['#000000' for j in range(len(coords))])
            #     node_scores[front_nodes] = '#A2142F' #red
            #     node_scores[back_nodes]  = '#4DBEEE' #blue
                
            #     # edge scores + create list
            #     # of edges to ignore
            #     edges = np.where(conn != 0)
            #     edge_scores = conn.copy()
            #     ignore_edges = []
            #     for e_i, e_j in zip(edges[0], edges[1]):

            #         if (e_i not in (front_nodes+back_nodes)) and (e_j not in (front_nodes+back_nodes)):
            #             ignore_edges.append((e_i, e_j))
            #             edge_scores[e_i, e_j] = 0
                    
            #         if conn[e_i,e_j] < np.percentile(conn, 95):
            #             ignore_edges.append((e_i, e_j))
            #             edge_scores[e_i, e_j] = 0

            #     # scale edge scores
            #     edge_scores = (edge_scores-np.min(edge_scores))/(np.max(edge_scores)-np.min(edge_scores))

            #     fig, ax = plot_network(G=conn, 
            #                   coords=coords, 
            #                   edge_scores=edge_scores, 
            #                   node_scores=node_scores, 
            #                   edge_cmap="Greys",
            #                   edge_alpha=0.25, #0.1
            #                   edge_vmin=np.percentile(edge_scores[np.nonzero(edge_scores)], 2), 
            #                   edge_vmax=np.percentile(edge_scores[np.nonzero(edge_scores)], 98),
            #                   ignore_edges=ignore_edges,
            #                   node_cmap="viridis", node_alpha=1, node_vmin=None, 
            #                   nodes_color='black', node_vmax=None, linewidth=1, s=100, #0.1, 50
            #                   projection=None, view="sagittal", view_edge=True,
            #                   axis=False, directed=False, figsize=None,
            #                   node_order=None
            #                   )
                
            #     ax.set_xlim(-55,55)
            #     ax.set_ylim(-55,55)
            #     plt.suptitle(filename)
            #     # fig.savefig(fname=os.path.join(f'C:/Users/User/OneDrive - McGill University/Figs/{order}', f'{filename}.png'), transparent=True, bbox_inches='tight', dpi=300)
                
            #     plt.show()
            #     plt.close()
    
    # boxplot -------  
    df = info.copy()[flag ==1]                
    df['front'] = front
    df['back']  = back
    df = df[['Id', 'Order', 'Filename', 'front', 'back']]
    df['diff'] = df['front'] - df['back']
    
    y = 'diff'  #'front' #'back' #'diff'

    sns.set(style="ticks", font_scale=1.5)
    fig, ax = plt.subplots(1,1,figsize=(8,4))
    sns.boxplot(x='Order',
                y=y,
                data=df,
                order=order_labels,
                ax=ax,
                palette=sns.color_palette('Set3', len(order_labels)),
                width=0.5,
                )
    
    sns.swarmplot(x='Order',
                y=y,
                data=df,
                order=order_labels,
                ax=ax,
                color='grey',
                size=3
                )

    if y == 'diff': 
        ax.set_ylabel('fronto-occipital')
    elif y == 'front': 
        ax.set_ylabel('frontal')
    elif y == 'back': 
        ax.set_ylabel('occipital')

    sns.despine(offset=10, trim=False)
    # fig.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs', f'{y}_{prop_name}.eps'), transparent=True, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

    # ANOVA model
    model = ols(f'{y} ~ C(Order)', data=df).fit()
    aov_table = sm.stats.anova_lm(model, typ=2)
    print('\n')
    print(anova_table(aov_table))

    # # UNIVARIATE multiple comparisons
    # comp = mc.MultiComparison(df['diff'], df['Order'])
    # tbl, a1, a2 = comp.allpairtest(stats.ttest_ind, method= "bonf")
    # print('\n')
    # print(tbl)

#%%
n = 20
print(\n'-------- Brain Plots --------')
for j, prop_name in enumerate(local_props):
    
    front = []
    back  = []
    for i, file in enumerate(os.listdir(COOR_DIR)[:]):
        
        filename = file.split('.')[0]
        order = info['Order'][i]
        orientation = extra_info['Brain Orientation'][i]
    
        if order in order_labels:

            # conn 
            conn = np.load(os.path.join(CONN_DIR, file))
            
            # coords
            coords = np.load(os.path.join(COOR_DIR, file))
            coords = center_coords(coords)
            if orientation == 'LR':
                coords = rotate_z(coords)
    
            # node selection
            x = coords[:,0]
            x_sort = np.argsort(x)
            front_nodes = list(x_sort[-n:])
            back_nodes  = list(x_sort[:n]) 
            
   
            # -------------------- plot brains
            if filename in plot_brains:
                
                # node scores = colors 
                node_scores = np.array(['#000000' for j in range(len(coords))])
                node_scores[front_nodes] = '#A2142F' #red
                node_scores[back_nodes]  = '#4DBEEE' #blue
                
                # edge scores + create list
                # of edges to ignore
                edges = np.where(conn != 0)
                edge_scores = conn.copy()
                ignore_edges = []
                for e_i, e_j in zip(edges[0], edges[1]):

                    if (e_i not in (front_nodes+back_nodes)) and (e_j not in (front_nodes+back_nodes)):
                        ignore_edges.append((e_i, e_j))
                        edge_scores[e_i, e_j] = 0
                    
                    if conn[e_i,e_j] < np.percentile(conn, 95):
                        ignore_edges.append((e_i, e_j))
                        edge_scores[e_i, e_j] = 0

                # scale edge scores
                edge_scores = (edge_scores-np.min(edge_scores))/(np.max(edge_scores)-np.min(edge_scores))

                fig, ax = plot_network(G=conn, 
                              coords=coords, 
                              edge_scores=edge_scores, 
                              node_scores=node_scores, 
                              edge_cmap="Greys",
                              edge_alpha=0.25, #0.1
                              edge_vmin=np.percentile(edge_scores[np.nonzero(edge_scores)], 2), 
                              edge_vmax=np.percentile(edge_scores[np.nonzero(edge_scores)], 98),
                              ignore_edges=ignore_edges,
                              node_cmap="viridis", node_alpha=1, node_vmin=None, 
                              nodes_color='black', node_vmax=None, linewidth=1, s=100, #0.1, 50
                              projection=None, view="sagittal", view_edge=True,
                              axis=False, directed=False, figsize=None,
                              node_order=None
                              )
                
                ax.set_xlim(-55,55)
                ax.set_ylim(-55,55)
                plt.suptitle(filename)
                # fig.savefig(fname=os.path.join(f'C:/Users/User/OneDrive - McGill University/Figs/{order}', f'{filename}.png'), transparent=True, bbox_inches='tight', dpi=300)
                
                plt.show()
                plt.close()
    
