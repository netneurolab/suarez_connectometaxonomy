# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 13:58:36 2020

@author: Estefany Suarez
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from numpy.linalg import inv
from scipy.linalg import eig

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut

from scipy.stats import (entropy, wasserstein_distance, energy_distance)
from scipy.spatial.distance import (pdist, squareform)
from sklearn.metrics.pairwise import cosine_similarity

from math import (log, log2)

import matplotlib.pyplot as plt
import seaborn as sns

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
#    D_inv = inv(D)
    D_inv[np.isinf(D_inv)] = 0
    
    I = np.identity(len(D), dtype='int')

    L = I-np.dot(D_inv,A)
    
    return L
    

def eigen_spectrum(M):  
    
    return eig(M, left=False, right=False)

            
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
    
    eigen_kde = []
    for ew in eigenspectra:
        
        # convolve with Gaussian filter
        x_d, dx_d = np.linspace(0, 2, 2000, retstep=True)
#        density = fn.gaussian_filter(ew, x_d)
#        density = density/auc(x_d, density)
        density = np.exp(gaussian_kernel(ew, x_d, bandwidth))#, 0.015)) 

        eigen_kde.append(density)

#        print(f'AUC = {auc(x_d, density)}')
   
    return (x_d, dx_d, eigen_kde)


#%%      
def distance(p, q=None, dmetric='euclidean', **kwargs):
    
    if dmetric == 'jensen_shannon':
        distance = jensen_shannon(p, q, **kwargs)
    
    elif dmetric == 'kolmorogov_smirnov':
        """
            p, q can be the empirical distributions
        """
        distance = kolmorogov_smirnov(p,q)
    
    elif dmetric == 'wasserstein':
        """
            p, q are the empirical distributions
        """
        distance = wasserstein_distance(p,q)
    
    elif dmetric == 'cramer_von_mises':
        """
            p, q are the empirical distributions
        """
        distance = energy_distance(p, q)
    
    else:
        """
            p is a matrix of samples by features
        """
        distance = squareform(pdist(p, metric=dmetric), force='tomatrix')
            
      
    return distance
    

def kolmorogov_smirnov(p, q):
    """
    Computes the maximal vertical distance between cumulative distributions
    (this is the statistic for KS tests). Code mostly copied from
    scipy.stats.ks_twosamp

    Parameters
    ----------
    p : array_like
        First data set
    q : array_like
        Second data set
    Returns
    -------
    d : float
        Max distance, i.e. value of the Kolmogorov Smirnov test. Sign is + if
        the cumulative of p < the one of q at that location, else -.
    x : float
        Value of x where maximal distance d is reached.
    """
    from numpy import ma

    (p, q) = (ma.asarray(p), ma.asarray(q))
    (n1, n2) = (p.count(), q.count())
    mix = ma.concatenate((p.compressed(), q.compressed()))
    mixsort = mix.argsort(kind='mergesort')
    csum = np.where(mixsort < n1, 1./n1, -1./n2).cumsum()

    # Check for ties
    if len(np.unique(mix)) < (n1+n2):
        ind = np.r_[np.diff(mix[mixsort]).nonzero()[0], -1]
        csum = csum[ind]
        mixsort = mixsort[ind]

    csumabs = ma.abs(csum)
    i = csumabs.argmax()

    d = csum[i]
    # mixsort[i] contains the index of mix with the max distance
    # x = mix[mixsort[i]]

    return np.abs(d) #(d, x)


colors = sns.color_palette("tab10")
def jensen_shannon(p, q, base=2):
    """
    method to compute the Jenson-Shannon Distance 
    between two probability distributions
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)
    
    # calculate m
    m = (p + q) / 2
    
    # e_p =  entropy(p, m, base=base)
    # e_q =  entropy(q, m, base=base)
    # if np.isinf(e_p) or np.isinf(e_q):
        
    #     print(f'\nbase = {base}')
    #     print(f'\tentropy p = {e_p}')
    #     print(f'\tentropy q = {e_q}')
        
    #     fig = plt.figure(figsize=(10,10))
    #     sns.kdeplot(p, label='p', c=colors[0])
    #     sns.kdeplot(q, label='q', c=colors[1])
    #     sns.kdeplot(m, label='m', c=colors[2])
    #     plt.show()
    #     plt.close()

    # compute Jensen Shannon Divergence            
    divergence = (entropy(p, m, base=base) + entropy(q, m, base=base)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance
