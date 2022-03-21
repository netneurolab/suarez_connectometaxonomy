# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 11:34:14 2021

@author: Estefany Suarez
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import numpy as np
from scipy.optimize import curve_fit

from sklearn.metrics import (explained_variance_score)


def exponential(x, a, b, c):
        return a * np.exp(b * x) + c
    
# def quadratic(x, a, b, c):
#     return a * x**2 + b * x + c
    
def linear(x, a, b):
    return a * x + b
    
def select_model(x, y, score, **kwargs):
    s = []
    models = [linear,  exponential] #quadratic,
    for model in models:
        popt, _ = curve_fit(model, x, y, maxfev=1000000)
        y_hat   = model(x, *popt)
        s.append(score(y, y_hat, len(popt)))
        
        print(f'{str(model.__name__)} - R2 : {r2(y, y_hat)}')
        print(f'{str(model.__name__)} - AIC : {aic(y, y_hat, len(popt))}')
        print(f'{str(model.__name__)} - BIC : {bic(y, y_hat, len(popt))}')
    
    model = models[np.argmax(s)]
    popt, _ = curve_fit(model, x, y, maxfev=1000000)
    y_hat   = model(x, *popt)

    if r2(y, y_hat) > 0.01:
        return str(model.__name__), r2(y, y_hat), y_hat, popt
    else:
        return None, None, None, None
    
def bic(y, y_hat, k, *args, **kwargs):
    resid = y - y_hat
    sse = sum(resid**2)
    return np.log(len(y))*k - 2*np.log(sse)

def aic(y, y_hat, k, *args, **kwargs):
    resid = y - y_hat
    sse = sum(resid**2)
    return 2*k - 2*np.log(sse)

def r2(y, y_hat, *args, **kwargs):
    return explained_variance_score(y, y_hat)


def fit_curve(x, y, model):
    popt, _ = curve_fit(model, x, y, maxfev=1000000)
    y_hat   = model(x, *popt)
    
    return popt, y_hat, r2(y, y_hat)
