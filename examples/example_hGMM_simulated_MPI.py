'''
Created on Jul 19, 2014

@author: jonaswallin
'''

import bayesianmixture as bm
import numpy.random as npr
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as spl
from bayesianmixture.PurePython.distribution.wishart import  invwishartrand
from bayesianmixture.PurePython.GMM import mixture
from mpi4py import MPI
from bayesianmixture import distance_sort_MPI
import copy as cp

def generate_data( n_obs, K):
	d = 4
	n_y = 20
	mu = [np.repeat(np.double(k),d) for k in range(K)]
	r1 =  np.zeros(d)
	r1[0] = 3.
	r1[1] = -1
	r1[2] = 1
	r1[2] = .5
	r2 =  np.zeros(d)
	r2[0] = 2.
	r2[1] = -1.5
	r2[2] = 0
	r2[2] = 0.		
	sigma =[np.eye(d), spl.toeplitz(r1), spl.toeplitz(r1), 0.5*np.eye(d)]
	p = np.array([1.,1.,1.,0.1])
	p2 =  np.array([1.,1.,1.])
	p /= np.sum(p)
	p2 /= np.sum(p2)
	data_sigma = [ [invwishartrand(10+d, sigma[k]) for k in range(K)] for n in range(n_y)]  # @UnusedVariable
	data_mu    = [ [mu[k]+ np.random.randn(d)*0.005 for k in range(K)] for n in range(n_y)]  # @UnusedVariable
		
	Y = []
	for i in range(n_y):
		
		if i < 17:
			mix = mixture(K = K)
			mix.mu = data_mu[i]
			mix.sigma = data_sigma[i]
			mix.p = p
		else:
			mix = mixture(K = K-1)
			mix.mu = data_mu[i][:-1]
			mix.sigma = data_sigma[i][:-1]
			mix.p = p2
		mix.d = d
		Y.append(mix.simulate_data(n_obs))
	
	return Y, data_sigma, data_mu

def simulate(hGMM, sim):


	for i in range(sim):
		if np.mod(i,20) == 0:
			print "i =%d"%i
		hGMM.sample()
		
	return hGMM

def simulate_pull(hGMM, sim, c, labels):
	hGMM.set_nuss(c)
	hGMM.set_nu_mus(c)	
	hGMM,labels = simulate(hGMM, sim, labels)
	return hGMM, labels




def simulate_push(hGMM, sim):
	
	

	c = 1. #np.max([np.max(np.linalg.eig(hGMM.normal_p_wisharts[k].param['Sigma'])[0]) for k in range(hGMM.K)])
	for npw, wpnu in zip(hGMM.normal_p_wisharts, hGMM.wishart_p_nus):
		npw.param['Sigma'] += np.eye(hGMM.d)*c
		wpnu.param['Q'] = 10**-6 * np.eye(hGMM.d)
		wpnu.param['nu'] = hGMM.d + 4
	hGMM.update_GMM()
	hGMM.set_nuss(0 )
	hGMM.set_nu_mus(0 )
	for i in range(sim):
		c = 1. #np.max([np.max(np.linalg.eig(hGMM.normal_p_wisharts[k].param['Sigma'])[0]) for k in range(hGMM.K)])
		for npw, wpnu in zip(hGMM.normal_p_wisharts, hGMM.wishart_p_nus):
			npw.param['Sigma'] += np.eye(hGMM.d)*c
			wpnu.param['Q'] = 10**-6 * np.eye(hGMM.d)
			wpnu.param['nu'] = hGMM.d + 4
		hGMM.update_GMM()
		hGMM.sample()
	return hGMM




if __name__ == "__main__":
	#npr.seed(1234567891)
	#plt.close('all')
	#plt.ion()
	n_obs = 1000
	K = 4
	sim  = 500
	hGMM = bm.hierarical_mixture_mpi(K = K)
	if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable
		Y, data_sigma, data_mu = generate_data(n_obs, K)
	else:
		Y = None
	hGMM.set_data(Y)
	hGMM.set_prior_param0()
	hGMM.set_p_labelswitch(1)

	hGMM = simulate(hGMM,500)
	mus = distance_sort_MPI(hGMM)
	#simulate_push2(hGMM, 0)
	hGMM = simulate(hGMM,500)
	mus = distance_sort_MPI(hGMM)
	hGMM.set_p_activation([0.2,0.2])
	hGMM = simulate(hGMM,sim)
	f,ax = hGMM.plot_mus([0,1], None)
	plt.show()