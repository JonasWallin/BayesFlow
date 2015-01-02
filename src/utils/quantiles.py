# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 14:38:03 2015

@author: johnsson
"""
from __future__ import division
import numpy as np

def get_quantiles(Y,classif_freq,alpha):
    '''
        Returns alpha quantile(s) in each dimension of data Y for each weighting
        of the data given by each column of classif_freq.
    '''
    KK = classif_freq.shape[1]
    weights_all = classif_freq/sum(classif_freq[0,:])
    d = Y.shape[1]
    quantiles = np.zeros((KK,len(alpha),d))
    for k in range(KK):
        for dd in range(d):
            quantiles[k,:,dd] = get_quantile(Y[:,dd],weights_all[:,k],alpha)
    return quantiles

def get_quantile(y,w,alpha):
    '''
        Returns alpha quantile(s) (alpha can be a vector) of empirical probability distribution of data y weighted by w
    '''
    alpha = np.array(alpha)
    alpha_ord = np.argsort(alpha)
    alpha_sort = alpha[alpha_ord]
    y_ord = np.argsort(y)
    y_sort = y[y_ord]
    cumdistr = np.cumsum(w[y_ord])
    q = np.empty(alpha.shape)
    if cumdistr[-1] == 0:
        q[:] = float('nan')
        return q
    cumdistr /= cumdistr[-1]
    j = 0
    i = 0
    while j < len(alpha) and i < len(cumdistr):
        if alpha_sort[j] < cumdistr[i]:
            q[j] = y_sort[i]
            j += 1
        else:
            i += 1
    if j < len(alpha):
        q[j:] = y_sort[-1]
    return q[np.argsort(alpha_ord)]
