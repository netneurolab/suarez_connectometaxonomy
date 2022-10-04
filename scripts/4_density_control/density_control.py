# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 16:04:13 2021

@author: Estefany Suarez
"""
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import curve_fitting as cf

#%%
RESOLUTION = '100'
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'mami', f'conn_{RESOLUTION}')
INFO_DIR = os.path.join(DATA_DIR, 'info')
RAW_DIR  = os.path.join(PROJ_DIR, 'raw_results', f'res_{RESOLUTION}')

#%%
order_labels = [
                'Chiroptera',
                'Rodentia',
                'Cetartiodactyla',
                'Carnivora',
                'Perissodactyla',
                'Primates',
                ]

#%%
def select_model(data, variable, transform_variables):
    """
        Regress out variable from transform_variables
        data: DataFrame
        variable: str
        transform_variables: dictionary whose keys correspond to variables and
        values correspond to the fitting model
    """

    for p in transform_variables:

        print(f'\n---------{p}----------')

        model, r2, y_pred, params = cf.select_model(x=data[variable].values.astype(float),
                                                    y=data[p].values.astype(float),
                                                    score=cf.r2
                                                    )

        if model == 'linear': eq = '$y = %5.3f x + %5.3f$' % tuple(params)
        elif model == 'exponential': eq = '$y = %5.3f e^{%5.3f x} + %5.3f$' % tuple(params)

        try:
            textstr = '\n'.join((
                            f'{model} fit ($R^2 = %5.2f$)' % (r2, ),
                            f'{eq}'
                            ))
        except:
            textstr = ''

        sns.set(style="ticks", font_scale=2.00)
        fig, axs = plt.subplots(1,1,figsize=(8,8))
        sns.scatterplot(data=data,
                        x=variable,
                        y=p,
                        s=300,
                        hue='Order',
                        hue_order=order_labels,
                        palette=sns.color_palette("Set3", len(order_labels)),
                        ax=axs
                        )

        if model is not None:
            sns.lineplot(data[variable].values.astype(float),
                         y_pred,
                         color='grey',
                         ax=axs,
                         )

            # replace feature by model residuals
            if 'assortativity' not in p:
                data[p] = data[p].values.astype(float) - y_pred


        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        axs.text(0.05, 0.95, textstr, transform=axs.transAxes, fontsize=14,
                  verticalalignment='top', bbox=props)

        axs.get_legend().remove()
        axs.set_xlim(0.10, 0.35)

        sns.despine(offset=10, trim=True)
        # fig.savefig(fname=os.path.join('C:/Users/User/OneDrive - McGill University/Figs', f'reg_density_{p}.eps'), transparent=True, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()

    return data


#%%
local_bin = [
             'avg_node_degree',
             'avg_bin_clustering_coeff',
             'avg_bin_node_centrality',
             'avg_bin_efficiency',
              'std_node_degree',
              'std_bin_clustering_coeff',
              'std_bin_node_centrality',
              'std_bin_efficiency',
            ]

local_wei = [
             'avg_node_strength',
             'avg_wei_clustering_coeff',
             'avg_wei_node_centrality',
             'avg_wei_efficiency',
              'std_node_strength',
              'std_wei_clustering_coeff',
              'std_wei_node_centrality',
              'std_wei_efficiency',
             ]

global_bin = [
             'bin_char_path',
             'bin_transitivity',
             'bin_assortativity',
            ]

global_wei = [
              'wei_char_path',
              'wei_transitivity',
              'wei_assortativity',
              ]

local_ = local_bin + local_wei
global_ = global_bin + global_wei

bin_ = global_bin + local_bin
wei_ = global_wei + local_wei

all_ = local_ + global_

#%%
df_props = pd.read_csv(os.path.join(RAW_DIR, 'df_props.csv'))
drop_idx = [idx for idx in df_props['Id']  if df_props.loc[df_props['Id'] == idx, 'Order'].values not in order_labels]
df_props = df_props.drop(drop_idx)

#%%
df_reg_props = select_model(df_props.copy(), 'density', all_)
df_reg_props.to_csv(os.path.join(RAW_DIR, 'density_reg',' df_props_reg.csv'), index=False)
