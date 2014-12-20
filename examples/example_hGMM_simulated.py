'''
Created on Jul 19, 2014

@author: jonaswallin
'''

import BayesFlow as bm
import numpy.random as npr
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as spl
from BayesFlow.PurePython.distribution.wishart import  invwishartrand
from BayesFlow.PurePython.GMM import mixture

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

def simulate(hGMM, sim,labels):


	for i in range(sim):
		if np.mod(i,20) == 0:
			print "i =%d"%i
		hGMM.sample()
		for k in range(len(hGMM.GMMs)):
			label_k = hGMM.GMMs[k].sample_labelswitch()
			if label_k != None:
				labels[k].append(label_k)
		
	return hGMM, labels

def simulate_pull(hGMM, sim, c, labels):
	hGMM.set_nuss(c)
	hGMM.set_nu_mus(c)	
	hGMM,labels = simulate(hGMM, sim, labels)
	return hGMM, labels



def simulate_push(hGMM, sim, c):
	c = 1. #np.max([np.max(np.linalg.eig(hGMM.normal_p_wisharts[k].param['Sigma'])[0]) for k in range(hGMM.K)])

	for npw, wpnu in zip(hGMM.normal_p_wisharts, hGMM.wishart_p_nus):
		npw.param['Sigma'] += np.eye(hGMM.d)*c
		wpnu.param['Q'] = 10**-6 * np.eye(hGMM.d)
		wpnu.param['nu'] = hGMM.d + 4
	hGMM.update_GMM()
	hGMM.set_nuss(hGMM.d )
	hGMM.set_nu_mus(hGMM.d )
	hGMM = simulate(hGMM, sim)
	return hGMM


def simulate_push2(hGMM, sim):
	
	

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

def distance_sort(hGMM):
	
	mus  = [hGMM.GMMs[0].mu[k].reshape((1, hGMM.d)) for k in range(hGMM.K)]
	
	
	for GMM in hGMM.GMMs[1:]:
		index_k = np.argsort(GMM.p)[::-1] #sort by probabilility
		mus_t = np.array([np.mean(mu,axis=0) for mu in  mus])
		list_temp = [None for k in range(GMM.K)]
		for index in index_k:
			#print mus_t
			#print mus_t - GMM.mu[index]
			dist = np.linalg.norm(mus_t - GMM.mu[index],axis=1)
			i = np.argsort(dist)[0]
			mus_t[i,:] = np.inf
			list_temp[index] = i 
		list_temp = np.argsort(np.array(list_temp))
		mus = [np.vstack((mu,GMM.mu[list_temp[i]])) for i,mu in enumerate(mus) ]
		GMM.mu = [GMM.mu[i] for i in list_temp]
		GMM.sigma = [GMM.sigma[i] for i in list_temp]
		GMM.p = np.array([GMM.p[i] for i in list_temp])
	hGMM.update_prior()
	[GMM.updata_mudata() for GMM in hGMM.GMMs]
	return mus


if __name__ == "__main__":
	
	plt.close('all')
	plt.ion()
	n_obs = 1000
	K = 4
	sim  = 500
	hGMM = bm.hierarical_mixture(K = K)
	Y, data_sigma, data_mu = generate_data(n_obs, K)
	hGMM.set_data(Y)
	hGMM.set_prior_param0()
	hGMM.set_p_labelswitch(1)
	labels2 = [[] for k in range(len(hGMM.GMMs))]

	hGMM,labels2 = simulate(hGMM,100, labels2)
	mus = distance_sort(hGMM)
	simulate_push2(hGMM, 0)
	hGMM,labels2 = simulate(hGMM,100, labels2)
	mus = distance_sort(hGMM)
	hGMM.set_p_activation([0.2,0.2])
	hGMM,labels2 = simulate(hGMM,sim, labels2)
	#mus = distance_sort(hGMM)
	#hGMM.set_p_labelswitch(1)
	#hGMM,labels2 = simulate(hGMM,sim, labels2)
	
	f,ax = hGMM.plot_mus([0,1], None)
	ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	f = hGMM.plot_thetas([0, 1], ax)
	f,ax = hGMM.GMMs[0].plot_scatter([0,1], None)
	hGMM.GMMs[-2].plot_scatter([0,1], None)
	plt.show()
	
	#hGMM = simulate_push2(hGMM, 0, 10.)
	#hGMM.GMMs[0].mu
	#hGMM.GMMs[0].sample()
	#hGMM.GMMs[0].mu
	#np.mean(hGMM.GMMs[0].data[hGMM.GMMs[0].x==1,:],0)
	#hGMM.GMMs[0].plot_scatter([0,1], None)
	#np.array([GMM.mu[3] for GMM in hGMM.GMMs])
	#np.var(mu,0)
	#hGMM.normal_p_wisharts[3].Sigma_class.Y_outer
	#hGMM.normal_p_wisharts[3].Sigma_class.n
	
	##
	#
	#hGMM.GMMs[0].mu
	# np.mean(hGMM.GMMs[0].data[hGMM.GMMs[0].x==1,:],0)
	# hGMM.GMMs[0].sample()
	# np.mean(hGMM.GMMs[0].data[hGMM.GMMs[0].x==1,:],0)
	# hGMM.GMMs[0].mu -> wrong mu
	# hGMM.GMMs[0].sample_mu() mu get correct
	# hGMM.GMMs[0].mu 