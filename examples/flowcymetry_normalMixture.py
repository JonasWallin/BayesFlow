# -*- coding: utf-8 -*-
"""
Running the Gibbs sampler on flowcymetry data
http://www.physics.orst.edu/~rubin/nacphy/lapack/linear.html


matlab time:  Elapsed time is 1.563538 seconds.

improve sample_mu:
python:                0.544    0.005    0.664  
cython_admi:           0.469    0.005    0.493 
moved_index_in_cython: 0.148    0.002    0.217 (most time is highmem)
changed_index          0.136    0.001    0.174

removed_higmem:        0.048    0.000    0.048



improve sample_sigma:
python:                0.544    0.005    0.664    
cython_admi:           0.313    0.003    0.364
moved_index_in_cython: 0.145    0.001    0.199 
changed_index        : 0.074    0.000    0.081 (used BLAS matrix calc)
changed to syrk      : 0.060    0.000    0.067
to profile use:
%prun main(K=5):
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      500    0.358    0.001    0.358    0.001 rng_cython.pyx:262(calc_lik)
      100    0.297    0.003    0.297    0.003 rng_cython.pyx:291(calc_exp_normalize)
      500    0.159    0.000    0.167    0.000 rng_cython.pyx:129(sample_mix_sigma_zero_mean)
      100    0.145    0.001    0.199    0.002 GMM.py:40(sample_mu)
        1    0.099    0.099    0.218    0.218 npyio.py:628(loadtxt)
      500    0.053    0.000    0.053    0.000 rng_cython.pyx:169(sample_mu)
      100    0.052    0.001    0.052    0.001 rng_cython.pyx:238(draw_x)
      100    0.045    0.000    0.700    0.007 GMM.py:90(compute_ProbX)
59998/29999    0.037    0.000    0.040    0.000 npyio.py:772(pack_items)
    30000    0.026    0.000    0.048    0.000 npyio.py:788(split_line)
      507    0.018    0.000    0.018    0.000 {method 'reduce' of 'numpy.ufunc' objects}
    60000    0.017    0.000    0.017    0.000 {method 'split' of 'str' objects}
      100    0.015    0.000    0.034    0.000 GMM.py:208(sample_p)
       12    0.014    0.001    0.014    0.001 {numpy.core.multiarray.array}
    29999    0.012    0.000    0.012    0.000 {zip}






%prun main_python(K=5)
  ncalls  tottime  percall  cumtime  percall filename:lineno(function)
  10707    0.584    0.000    0.584    0.000 {method 'reduce' of 'numpy.ufunc' objects}
      100    0.574    0.006    2.195    0.022 GMM.py:149(sample_x)
      100    0.544    0.005    0.664    0.007 GMM.py:176(sample_mu)
      100    0.499    0.005    1.295    0.013 GMM.py:219(compute_ProbX)
      100    0.334    0.003    0.549    0.005 GMM.py:189(sample_sigma)
     3501    0.310    0.000    0.310    0.000 {numpy.core._dotblas.dot}
    16112    0.252    0.000    0.252    0.000 {numpy.core.multiarray.array}
        1    0.101    0.101    0.223    0.223 npyio.py:628(loadtxt)
      100    0.048    0.000    0.048    0.000 {method 'cumsum' of 'numpy.ndarray' objects}
59998/29999    0.038    0.000    0.041    0.000 npyio.py:772(pack_items)



Created on Fri Jun 20 16:52:31 2014

@author: jonaswallin
"""

from __future__ import division
import numpy as np

from bayesianmixture import mixture
import bayesianmixture.PurePython.GMM as GMM
from matplotlib import pyplot as plt
import numpy.random as npr
import time

K = 5
def main(K= 5):
    sim = 100
    data = np.ascontiguousarray(np.loadtxt('../data/flowcym.dat',skiprows=1,usecols=(1,2,3,4,5,6)))
    mix = mixture(data,K,high_memory=True)
    t0 = time.time()
    for i in range(sim):  # @UnusedVariable
        mix.sample()
    t1 = time.time()
    print("mixture took %.4f sec"%(t1-t0))
    
def main_python(K = 5):
    sim = 100
    data = np.ascontiguousarray(np.loadtxt('../data/flowcym.dat',skiprows=1,usecols=(1,2,3,4,5,6)))
    mix = GMM.mixture(data,K)
    t0 = time.time()
    for i in range(sim):  # @UnusedVariable
        mix.sample()
    t1 = time.time()

    
    print("mixture took %.4f sec"%(t1-t0))
    
if __name__ == '__main__':
    sim = 10
    data = np.ascontiguousarray(np.loadtxt('../data/flowcym.dat',skiprows=1,usecols=(1,2,3,4,5,6)))
    mix = mixture(data, K)
    mus = np.zeros((sim,2*data.shape[1]))
    t0 = time.time()
    for i in range(sim):
        mix.sample()
        mus[i,:data.shape[1]] = mix.mu[0]
        mus[i,data.shape[1]:] = mix.mu[1]
    t1 = time.time()
    if 1:
        for k in range(mix.K):
            plt.plot(mix.data[mix.x==k,0],mix.data[mix.x==k,1],'o')
        
        plt.figure()
        for k in range(mix.K):
            plt.plot(mus[:,(2*k):(2*(k+1))])
            
        plt.show()
    
    print("mixture took %.4f sec"%(t1-t0))
    mix2 = GMM.mixture(data,K)
    mus = np.zeros((sim,4))
    t0 = time.time()
    for i in range(sim):
        mix2.sample()
    t1 = time.time()
    print("Python mixture took %.4f sec"%(t1-t0))
    if 0:
        import pstats, cProfile
    
        import pyximport
        pyximport.install()
        
        import bayesianmixture.distributions.rng_cython as rng_cython
    
        #cProfile.runctx("rng_cython.sample_mu_rep(np.sum(mix.data[mix.x == 0 ,:],1),mix.sigma[0],mix.prior[0]['mu']['theta'].reshape(mix.d),mix.prior[0]['mu']['sigma'],npr.rand(mix.d),10000)", globals(), locals(), "Profile.prof")
         
        cProfile.runctx("for k in range(100): mix.sample_mu()", globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()