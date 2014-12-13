# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 11:01:11 2014

@author: jonaswallin
"""

from sklearn import mixture as skmixture
import numpy as np


if __name__ == '__main__':
    data = np.loadtxt('../data/faithful.dat',skiprows=1,usecols=(0,1))
    g = skmixture.GMM(n_components=2, covariance_type='full')
    g.fit(data) 
    print('weights =')
    print(np.round(g.weights_, 2))
    print('mu = ')
    print(np.round(g.means_, 2))
    print('covar = ')
    print(np.round(g.covars_, 2))