# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 09:58:01 2015

@author: jonaswallin
"""

from __future__ import division
import numpy as np
import numpy.random as npr
import scipy.linalg as spl
from BayesFlow.PurePython.GMM import mixture as mixP
import BayesFlow.GMM as GMM
import copy as cp
import scipy.spatial as ss
import matplotlib.pyplot as plt

n  =10**4
sim = 30

##
#simulating data
##
npr.seed(123456)
nClass = 3
dim    = 3
P = [0.4, 0.3, 0.3]
Thetas = [np.array([0.,0, 0]), np.array([0., -2, 1]), np.array([1., 2, 0])]
Sigmas = [ 0.1*spl.toeplitz([2.,0.5,0]),0.1* spl.toeplitz([2.,-0.5,1]),
			  0.1*spl.toeplitz([1.,.3,.3]) ] 
		
mix_obj       = mixP(K = nClass)
mix_obj.mu    = cp.deepcopy(Thetas)
mix_obj.sigma = cp.deepcopy(Sigmas)
mix_obj.p     = cp.deepcopy(P)
mix_obj.d     = dim
Y = mix_obj.simulate_data(n)



mix = mixP(K = nClass, high_memory=False)
mix.set_data(Y)
for i in range(10):#np.int(np.ceil(0.1*self.sim))):  # @UnusedVariable
    mix.sample()
    
plt.scatter(Y[:,1],Y[:,2],s=80,c = mix.x)

#plt.figure()


mix.set_AMCMC(4000)

mu_sample = list()
for k in range(nClass):
    mu_sample.append(np.zeros_like(Thetas[k])) 

for i in range(sim):#np.int(np.ceil(0.1*self.sim))):  # @UnusedVariable
    mix.sample()
    for k in range(nClass):
        mu_sample[k] += mix.mu[k]/np.double(sim)
#plt.scatter(Y[:,1],Y[:,2],s=80,c = mix.x)
print mu_sample