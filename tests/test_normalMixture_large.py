# -*- coding: utf-8 -*-
"""
Testing on real problem so this test should take longer time
Created on Thu Jun 19 21:17:04 2014

@author: jonaswallin
"""
from __future__ import division

import unittest
import sys
import os
import numpy as np
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path + '/../data')


from bayesianmixture import GMM
from bayesianmixture import PurePython
#TODO: there is a bug in GMM that missed, the sampling of sample_P
#TODO: need to find a test that discover the
#TODO: most compute_P given the true parameter should be fairly similar to
#TODO: the sum should be similar to prior probabilites!

class oldFaithful(object):
	sim = 10**3
	def setUp(self):
		self.data = np.loadtxt('../data/faithful.dat',skiprows=1,usecols=(0,1))
	def set_param(self):
		self.set_mu = [np.array([  4.29,  79.97]), np.array([  2.04,  54.48])]
		self.set_sigma = [np.array([[  0.17 ,  0.94],[  0.94,  36.  ]]),
						  np.array([[ 0.07  ,  0.44] ,[ 0.44, 33.72 ]])]
		self.p = np.array([ 0.64 , 0.36])
	
	
	def test_sample_x(self):	
		self.set_param()  
		self.mix.sample_x()
		P = self.mix.prob_X.copy()
		x = self.mix.x.copy()
		x.dtype = np.double
		for i in range(self.sim-1):  # @UnusedVariable
			self.mix.sample_x()
			x += self.mix.x 
		x /= self.sim
		np.testing.assert_array_almost_equal(P[:,1],x,decimal=1)

	def test_sample_p(self):
		"""
			The distribution should be approximatly the prior
		""" 
		self.mix.sample_x()
		self.mix.alpha_vec = np.array([5*10**5,1*10**5])
		self.mix.sample_p()
		np.testing.assert_array_almost_equal(self.mix.p, self.mix.alpha_vec/np.sum(self.mix.alpha_vec),decimal=2)
	
	def test_sample_p2(self):  
		self.set_param()  
		self.mix.sample_x()
		alpha_mean = np.zeros(self.mix.K)
		for k in range(self.mix.K):
			alpha_mean[k] = self.mix.prior[k]['p'] + np.sum(self.mix.x==k)
		alpha_mean /= np.sum(alpha_mean)
		p = np.zeros(self.mix.K)
		for i in range(self.sim):  # @UnusedVariable
			self.mix.sample_p()
			p += self.mix.p
		p /= self.sim
		
		np.testing.assert_array_almost_equal(p, alpha_mean,decimal=2)
	
	def test_sample_mu(self):   
		"""
			The distribution should be approximatly the prior
		"""	 
		
		self.mix.sample_x()
		self.mix.prior[0]["mu"]["theta"]  = np.array([100,100.])
		self.mix.prior[0]["mu"]["Sigma"]  = np.diag([1.,1.])*10**(-10)
		self.mix.sample_mu()
		np.testing.assert_array_almost_equal(self.mix.mu[0],self.mix.prior[0]["mu"]["theta"],decimal=2)
	
	def test_sample_mu2(self):	
		self.set_param()
		self.mix.sample_x()
		mu = [np.zeros((2,)),np.zeros((2,))]
		for k in range(2):
			x_sum = sum(self.data[self.mix.x==k,:],0)
			inv_sigma	= np.linalg.inv(self.mix.sigma[k])
			inv_sigma_mu = np.linalg.inv(self.mix.prior[k]['mu']['Sigma'])
			theta		= self.mix.prior[k]['mu']['theta']
			mu_s  = np.dot(inv_sigma_mu, theta).reshape(theta.shape[0]) + np.dot(inv_sigma, x_sum)
			Q	 = np.sum(self.mix.x==k) * inv_sigma  + inv_sigma_mu
			mu[k] = np.linalg.solve(Q,mu_s)
		sample = [np.zeros((2,)),np.zeros((2,))]
		for i in range(self.sim):  # @UnusedVariable
			self.mix.sample_mu()
			for k in range(self.mix.K):
				sample[k] += self.mix.mu[k]
		for k in range(self.mix.K):
			sample[k] /= self.sim
			np.testing.assert_array_almost_equal(np.array([1,0.1])*sample[k],np.array([1,0.1])*mu[k],decimal = 1)			
			
	def test_sample_sigma(self):
		"""
			The distribution should be approximatly the prior
		"""
		self.mix.sample_x()
		self.mix.prior[0]["sigma"]["nu"]  = 10**7
		self.mix.prior[0]["sigma"]["Q"]   = np.diag([1.,1.]) * (self.mix.prior[0]["sigma"]["nu"] -2 -1)
		self.mix.sample_sigma()
		np.testing.assert_array_almost_equal(self.mix.sigma[0],np.diag([1.,1.]),decimal=1)

	def test_sample_sigma2(self):
		self.set_param()
		self.mix.sample_x()
		Sigma_mean = []
		for k in range(self.mix.K):
			if self.mix.high_memory == True:
				X = self.mix.data_mu[k][self.mix.x == k ,:]
			else:
				X = self.mix.data[self.mix.x == k ,:] - self.mix.mu[k]
			Q = self.mix.prior[k]["sigma"]["Q"] + np.dot(X.transpose(), X)
			Q /= self.mix.prior[k]["sigma"]["nu"] + X.shape[0] - 2 - 1
			Sigma_mean.append(Q)
		sample = [np.zeros((2,2)),np.zeros((2,2))]
		for i in range(self.sim):  # @UnusedVariable
			self.mix.sample_sigma()
			
			for k in range(self.mix.K):
				sample[k] += self.mix.sigma[k]

		for k in range(self.mix.K):
			sample[k] /= self.sim
			np.testing.assert_array_almost_equal(np.dot(np.diag([1,0.1]),np.dot(sample[k],np.diag([1,0.1]))),
												 np.dot(np.diag([1,0.1]),np.dot(Sigma_mean[k],np.diag([1,0.1]))),
												 decimal=1) 
	def test_update_mu_prior(self):
		prior = [{} for k in range(self.mix.K)]
		for k in range(self.mix.K):
			prior[k]['theta']    = np.random.rand(self.data.shape[1],1)
			prior[k]['Sigma'] = np.random.rand()*np.eye(self.data.shape[1])
			
		self.mix.set_prior_mu(prior)
		for k in range(self.mix.K):
			np.testing.assert_array_equal(self.mix.prior[k]['mu']['theta'], prior[k]['theta'])
			np.testing.assert_array_equal(self.mix.prior[k]['mu']['Sigma'], prior[k]['Sigma'])


		prior = [{} for k in range(self.mix.K)]
		for k in range(self.mix.K):
			prior[k]['theta']    = np.random.rand(self.data.shape[1],1)
			prior[k]['Sigma'] = np.random.rand()*np.eye(self.data.shape[1])			
		mu =  np.zeros((self.mix.K,self.data.shape[1]))
		Sigma = np.zeros((self.mix.K, self.data.shape[1],self.data.shape[1]))
		for k in range(self.mix.K):
			mu[k,:] = prior[k]['theta'].reshape(mu[k,:].shape)
			Sigma[k,:,:] = prior[k]['Sigma']
		self.mix.set_prior_mu_np(mu, Sigma)
		for k in range(self.mix.K):
			np.testing.assert_array_equal(self.mix.prior[k]['mu']['theta'], prior[k]['theta'])
			np.testing.assert_array_equal(self.mix.prior[k]['mu']['Sigma'], prior[k]['Sigma'])		

	def test_update_sigma_prior(self):
		prior = [{} for k in range(self.mix.K)]
		for k in range(self.mix.K):
			prior[k]['nu']    = np.random.randint(10)
			prior[k]['Q'] = np.random.rand()*np.eye(self.data.shape[1])
			
		self.mix.set_prior_sigma(prior)
		for k in range(self.mix.K):
			np.testing.assert_array_equal(self.mix.prior[k]['sigma']['nu'], prior[k]['nu'])
			np.testing.assert_array_equal(self.mix.prior[k]['sigma']['Q'], prior[k]['Q'])


		prior = [{} for k in range(self.mix.K)]
		for k in range(self.mix.K):
			prior[k]['nu']    = np.random.randint(10)
			prior[k]['Q'] = np.random.rand()*np.eye(self.data.shape[1])
			
		nu = np.zeros(self.mix.K)
		Q = np.zeros((self.mix.K, self.data.shape[1],self.data.shape[1]))
		for k in range(self.mix.K):
			nu[k] = prior[k]['nu']
			Q[k,:,:] = prior[k]['Q']
		self.mix.set_prior_sigma_np(nu, Q)
		for k in range(self.mix.K):
			np.testing.assert_array_equal(self.mix.prior[k]['sigma']['nu'], prior[k]['nu'])
			np.testing.assert_array_equal(self.mix.prior[k]['sigma']['Q'], prior[k]['Q'])				
			
			
class test_Python_mixture_on_OldF(oldFaithful, unittest.TestCase):
	"""
		test
	"""
	
	def setUp(self):
		super(test_Python_mixture_on_OldF,self).setUp()
		self.K = 2
		self.mix = PurePython.GMM.mixture(self.data, self.K)

class test_mixture_on_OldF(oldFaithful, unittest.TestCase):
	"""
		test
	"""
	
	def setUp(self):
		super(test_mixture_on_OldF,self).setUp()
		self.K = 2
		self.mix = GMM.mixture(self.data, self.K)	
def main():
	unittest.main()	
	
if __name__ == '__main__':
	main()