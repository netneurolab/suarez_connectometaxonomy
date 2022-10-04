# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 13:58:36 2020

@author: Estefany Suarez
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from scipy.linalg import eig

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut

#%%
def check_symmetric(a, tol=1e-16):
    return np.allclose(a, a.T, atol=tol)


def dir2und(M):
    "M has to be binary"

    # connections in the upper and lower diagonals; same order
    upper = M[np.triu_indices_from(M, 1)]
    lower = M.T[np.triu_indices_from(M, 1)]

    or_ = np.logical_or(upper, lower).astype(int)

    und_conn = np.zeros_like(M).astype(int)
    und_conn[np.triu_indices_from(M, 1)] = or_
    und_conn = und_conn.T
    und_conn[np.triu_indices_from(M, 1)] = or_

    return und_conn


def norm_laplacian(A, degree='out'):

    D = np.zeros_like(A)

    if degree == 'in':
        np.fill_diagonal(D, np.sum(A, axis=0))
    else:
        np.fill_diagonal(D, np.sum(A, axis=1))

    D_inv = 1/D
    D_inv[np.isinf(D_inv)] = 0

    I = np.identity(len(D), dtype='int')

    L = I-np.dot(D_inv,A)

    return L


def eigen_spectrum(M):

    return np.abs(eig(M, left=False, right=False))


def gaussian_filter(x, x_d, sigma=0.015):

   def get_filter(x_d):
       gauss_filter = lambda xi: 1/np.sqrt(2*np.pi*(sigma**2)) * np.exp(-((x_d-xi)**2)/(2*(sigma**2)))
       return np.sum([gauss_filter(xi) for xi in x])

   return list(map(get_filter, x_d))


def gaussian_kernel(x, x_d, bandwidth=None):

    # CV to determine optimal bandwidth CV
    if bandwidth is None:
        bandwidths = 10 ** np.linspace(-1, 1, 100)
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidths},
                            cv=LeaveOneOut())
        grid.fit(x[:, None])
        bandwidth = grid.best_params_['bandwidth']

        print(f'\nOptimal bandwidth = {bandwidth}')

    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(x[:, None])
    log_density = kde.score_samples(x_d[:, None])

    return log_density

#%%
def get_eigen_kde(eigenspectra, bandwidth=0.015):

    x_d, dx_d = np.linspace(0, 2, 2000, retstep=True)

    eigen_kde = []
    for ew in eigenspectra:
        
        # convolve with Gaussian filter
        
        # # method 1
        # density = gaussian_filter(ew, x_d)
        # density = density/auc(x_d, density)

        # # method 2
        density = np.exp(gaussian_kernel(ew, x_d, bandwidth))
        
        eigen_kde.append(density)

        # print(f'AUC = {auc(x_d, density)}')

    return (x_d, dx_d, eigen_kde)
