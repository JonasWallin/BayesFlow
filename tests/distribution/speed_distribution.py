'''
Created on Jul 1, 2014

@author: jonaswallin


fist cython program
took:
**** MV *****
cython took  0.0834 msec/sim (sim, n ,d ) = (1000 20000,1) 
cython took  0.3410 msec/sim (sim, n ,d ) = (1000 20000,5) 
moved to blas - lapack
cython took  0.0592 msec/sim (sim, n ,d ) = (1000 20000,1) 
cython took  0.1033 msec/sim (sim, n ,d ) = (1000 20000,5) 


**** IW *****
cython took  0.1641 msec/sim (sim, n ,d ) = (1000 20000,1) 
cython took  0.7898 msec/sim (sim, n ,d ) = (1000 20000,5) 
sumY to C 
cython took  0.1918 msec/sim (sim, n ,d ) = (1000 20000,1) 
cython took  0.5829 msec/sim (sim, n ,d ) = (1000 20000,5) 
outer to C 
cython took  0.2347 msec/sim (sim, n ,d ) = (1000 20000,1) 
cython took  0.4569 msec/sim (sim, n ,d ) = (1000 20000,5) 
sampling wishart 
cython took  0.1498 msec/sim (sim, n ,d ) = (1000 20000,1) 
cython took  0.2697 msec/sim (sim, n ,d ) = (1000 20000,5) 
'''

import numpy as np
import scipy.linalg as spl
from bayesianmixture.PurePython.distribution import multivariatenormal as mv_python
from bayesianmixture.distribution import multivariatenormal as mv, Wishart  # @UnresolvedImport
from bayesianmixture.PurePython.distribution import invWishart as invWis_python, Wishart as Wis_python
from bayesianmixture.distribution import invWishart as invWis
from bayesianmixture.PurePython.distribution.wishart import  invwishartrand
import time

def test_mv(sim, d, n, mv, string):
	row = np.zeros(d)
	row2 = np.zeros(d)
	row[0] = 4.
	row2[0] = 2.
	if d > 1:
		row[1] = -1
		row2[1] = 1
	
	prior = {'mu':-10 * np.ones(d),'Sigma':spl.toeplitz(row)}
	param = {'Sigma': spl.toeplitz(row2)}
	Y = np.empty((n,d))
	L = np.linalg.cholesky(param['Sigma'])
	for i in range(n):  # @UnusedVariable
		Y[i,:] = np.dot(L,np.random.randn(d,1)).reshape((d,))
	dist = mv(prior = prior, param = param)  
	mu_est = np.zeros((sim,dist.mu_p.shape[0]))
	t0 = time.time()
	dist.set_data(Y)
	for i in range(sim):
		dist.set_parameter(param)
		dist.set_prior(prior)
		dist.set_data(Y)
		mu_est[i,:] = dist.sample()
	
	t1 = time.time()
	string += " %.4f msec/sim (sim, n ,d ) = (%d %d,%d) "%(1000*np.double(t1-t0)/sim, sim, n, d )
	print(string)

def test_invW(sim, d, n, invW, string):
	row = np.zeros(d)
	row2 = np.zeros(d)
	row[0] = 4.
	row2[0] = 2.
	if d > 1:
		row[1] = -1
		row2[1] = 1
	
	prior = {'nu': d ,'Q': np.eye(d)}
	param = {'theta': np.ones(d)}
	Y = np.empty((n,d))
	L = np.linalg.cholesky(spl.toeplitz(row))
	for i in range(n):  # @UnusedVariable
		Y[i,:] = np.dot(L,np.random.randn(d,1)).reshape((d,)) + param['theta']
	dist = invW(prior = prior, param = param)  
	mu_est = np.zeros((sim,dist.theta.shape[0],dist.theta.shape[0]))
	t0 = time.time()
	dist.set_data(Y)
	for i in range(sim):
		dist.set_parameter(param)
		dist.set_prior(prior)
		dist.set_data(Y)
		mu_est[i,:,:] = dist.sample()
	
	t1 = time.time()
	string += " %.4f msec/sim (sim, n ,d ) = (%d %d,%d) "%(1000*np.double(t1-t0)/sim, sim, n, d )
	print(string)

def test_W(sim, d, n, Wis, string):
	nu = 5
	row = np.zeros(d)
	row2 = np.zeros(d)
	row[0] = 4.
	row2[0] = 2.
	if d > 1:
		row[1] = -1
		row2[1] = 1
	
	prior = {'nus': d ,'Qs': np.eye(d)}
	param = {'nu': d+2}
	Y = []
	Sigma = spl.toeplitz(row)/ nu
	for i in range(sim):  # @UnusedVariable
		Y.append( invwishartrand(nu, Sigma))
		
	dist = Wis(prior = prior, param = param)  
	mu_est = np.zeros((sim,dist.Q_s.shape[0], dist.Q_s.shape[0]))
	t0 = time.time()
	dist.set_data(Y)
	for i in range(sim):
		dist.set_parameter(param)
		dist.set_prior(prior)
		dist.set_data(Y)
		mu_est[i,:,:] = dist.sample()
	
	t1 = time.time()
	string += " %.4f msec/sim (sim, n ,d ) = (%d %d,%d) "%(1000*np.double(t1-t0)/sim, sim, n, d )
	print(string)
	
if __name__ == '__main__':
	print("**** MV *****")
	string = "pure python took "
	test_mv(1000, 1, 20000, mv_python, string)
	test_mv(1000, 5, 20000, mv_python, string)
	string = "cython took "
	test_mv(1000, 1, 20000, mv, string)
	test_mv(1000, 5, 20000, mv, string)
	print("**** IW *****")
	string = "pure python took "
	test_invW(1000, 1, 20000, invWis_python, string)
	test_invW(1000, 5, 20000, invWis_python, string)
	string = "cython took "
	test_invW(1000, 1, 20000, invWis, string)
	test_invW(1000, 5, 20000, invWis, string)
	print("**** WIS *****")
	string = "pure python took "
	test_W(100, 1, 2000, Wis_python, string)
	test_W(100, 5, 2000, Wis_python, string)
	string = "cython took "
	test_W(100, 1, 2000, Wishart, string)
	test_W(100, 5, 2000, Wishart, string)