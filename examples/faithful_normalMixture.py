# -*- coding: utf-8 -*-
"""
Running the Gibbs sampler on old faithfull data
to profile use:
%prun main():
     1000    0.048    0.000    0.067    0.000 GMM.py:40(sample_mu)
     1000    0.034    0.000    0.079    0.000 GMM.py:83(compute_ProbX)
     1000    0.033    0.000    0.062    0.000 GMM.py:56(sample_sigma)
     ****************************************************************************
     1000    0.045    0.000    0.062    0.000 GMM.py:40(sample_mu)
     1000    0.031    0.000    0.057    0.000 GMM.py:56(sample_sigma)
     2000    0.026    0.000    0.026    0.000 {bayesianmixture.distributions.rng_cython.sample_sigma_zero_mean}
     2000    0.021    0.000    0.021    0.000 {bayesianmixture.distributions.rng_cython.calc_lik}
     2000    0.014    0.000    0.014    0.000 {bayesianmixture.distributions.rng_cython.sample_mu}
     1000    0.012    0.000    0.012    0.000 {bayesianmixture.distributions.rng_cython.calc_exp_normalize}
     1000    0.012    0.000    0.030    0.000 GMM.py:208(sample_p)

Created on Fri Jun 20 16:52:31 2014

working on probX 0.2
@author: jonaswallin
"""

from __future__ import division
import numpy as np

from BayesFlow import mixture
import BayesFlow.PurePython.GMM as GMM
from matplotlib import pyplot as plt
import time

def main():
    sim = 1000
    data = np.ascontiguousarray(np.loadtxt('../data/faithful.dat',skiprows=1,usecols=(0,1)))
    mix = mixture(data, 2)
    mus = np.zeros((sim,4))
    t0 = time.time()
    for i in range(sim):
        mix.sample()
        mus[i,:2] = mix.mu[0]
        mus[i,2:] = mix.mu[1]
    t1 = time.time()

    
    print("mixture took %.4f sec"%(t1-t0))

    
if __name__ == '__main__':
    sim = 1000
    data = np.loadtxt('../data/faithful.dat',skiprows=1,usecols=(0,1))
    mix = mixture(data, 2)
    mus = np.zeros((sim,4))
    t0 = time.time()
    
    for i in range(sim):
        mix.sample()
        mus[i,:2] = mix.mu[0]
        mus[i,2:] = mix.mu[1]
    t1 = time.time()
    if 0:
        for k in range(mix.K):
            plt.plot(mix.data[mix.x==k,0],mix.data[mix.x==k,1],'o')
        
        plt.figure()
        for k in range(mix.K):
            plt.plot(mus[:,(2*k):(2*(k+1))])
            
        plt.show()
    
    print("mixture took %.4f sec"%(t1-t0))
    mix2 = GMM.mixture(data,2)
    mus = np.zeros((sim,4))
    t0 = time.time()
    for i in range(sim):
        mix2.sample()
        mus[i,:2] = mix2.mu[0]
        mus[i,2:] = mix2.mu[1]
    t1 = time.time()
    print("Python mixture took %.4f sec"%(t1-t0))