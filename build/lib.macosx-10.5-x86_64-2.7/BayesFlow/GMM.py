# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 23:33:57 2014

@author: jonaswallin
"""

import PurePython.GMM
import numpy as np
#import bayesianmixture.distributions.rng_cython as rng_cython
import BayesFlow.mixture_util.GMM_util as GMM_util



class mixture(PurePython.GMM.mixture):
	"""
		The main Gaussian mixture model
	"""
	def __init__(self, data = None, K = None,  prior = None, high_memory=True , name = None):
		super(mixture,self).__init__(None,K = K, prior = prior, high_memory = high_memory, name = name)
		#self.rng = rng_cython.rng_class()
		self.set_data(data)
		self.x_count = np.empty(K,dtype=np.int)
		
	def load_param(self, params):	
		
		super(mixture,self).load_param(params)
		if self.noise_class == 1:
			self.x_index = np.empty((self.n, self.K + 1),dtype=np.int) 
			self.x_count = np.empty(self.K + 1,dtype=np.int)
			
		
		
	def add_noiseclass(self, Sigma_scale = 5., mu = None, Sigma = None):
		"""
			adds a class that does not update and cant be deactiveted or label switch
			the data need to be loaded first!
			
			Sigma_scale  - (double)  the scaling constants time the covariance matrix
		"""
		super(mixture,self).add_noiseclass(Sigma_scale,mu,Sigma)
		self.x_index = np.empty((self.n, self.K + 1),dtype=np.int) 
		self.x_count = np.empty(self.K + 1,dtype=np.int)
		
	def set_data(self, data):
		if not data is None:
			super(mixture,self).set_data(data)
			self.x_index = np.empty((self.n, self.K),dtype=np.int)
		
	def sample_mu(self):
		"""
			Draws the mean parameters
			self.mu[k] - (d,) np.array
		"""   
		
		
		
		for k in range(self.K):  
			if self.active_komp[k] == True:
				theta = self.prior[k]['mu']['theta'].reshape(self.d)
				self.mu[k] = GMM_util.sample_mu(self.data, self.x_index, self.x_count,
											self.sigma[k], 
											theta,
											self.prior[k]['mu']['Sigma'],
											np.int(k))
			else:
				self.mu[k] = np.NAN * np.ones(self.d)
			
		self.updata_mudata()
		
		
		
	def likelihood_prior(self, mu, Sigma, k, R_S_mu = None, log_det_Q = None, R_S = None):
			"""
					Computes the prior that is 
					\pi( \mu | \theta[k], \Sigma[k]) \pi(\Sigma| Q[k], \nu[k]) = 
					N(\mu; \theta[k], \Sigma[k]) IW(\Sigma; Q[k], \nu[k])
			"""
			
			if np.isnan(mu[0]) == 1:
					return 0, None, None, None
			
			if R_S_mu is None:
					R_S_mu = GMM_util.cholesky(self.prior[k]['mu']['Sigma'])
					#R_S_mu = sla.cho_factor(self.prior[k]['mu']['Sigma'],check_finite = False)
					
			
			
			if log_det_Q is None:
					log_det_Q = GMM_util.log_det(self.prior[k]['sigma']['Q'])
			
			if R_S is None:
					R_S = GMM_util.cholesky(Sigma)
					#R_S = sla.cho_factor(Sigma,check_finite = False)
			
			
			
			lik = GMM_util.likelihood_prior(mu.reshape((self.d,1)),  self.prior[k]['mu']['theta'],  self.prior[k]['mu']['theta'], R_S_mu, R_S, self.prior[k]['sigma']['nu'],
										self.prior[k]['sigma']['Q'])
			lik = lik +  (self.prior[k]['sigma']['nu'] * 0.5) * log_det_Q
			lik = lik - self.ln_gamma_d(0.5 * self.prior[k]['sigma']['nu']) - 0.5 * np.log(2) * (self.prior[k]['sigma']['nu'] * self.d)
			
			return lik, R_S_mu, log_det_Q, R_S
		
						
	def sample_sigma(self):
		"""
			Draws the covariance parameters
		
		"""
		
		if self.high_memory == True:
			
			for k in range(self.K):  
				if self.active_komp[k] == True: 
					self.sigma[k] =  GMM_util.sample_mix_sigma_zero_mean(self.data_mu[k],self.x_index,self.x_count, k,
					 self.prior[k]["sigma"]["Q"],
					 self.prior[k]["sigma"]["nu"])
				else:
					self.sigma[k] = np.NAN * np.ones((self.d, self.d))
		else:
			for k in range(self.K):  
				if self.active_komp[k] == True: 
					X_mu = self.data - self.mu[k]
					self.sigma[k] =  GMM_util.sample_mix_sigma_zero_mean(X_mu,self.x_index,self.x_count, k,
					 self.prior[k]["sigma"]["Q"],
					 self.prior[k]["sigma"]["nu"])
				else:
					self.sigma[k] = np.NAN * np.ones((self.d, self.d))
				
	def sample_x(self):
		"""
			Draws the label of the observations
		
		"""
		self.compute_ProbX()
		GMM_util.draw_x(self.x,self.x_index,self.x_count, self.prob_X)

	def calc_lik(self, mu = None, sigma = None, p = None, active_komp = None):
		
		return np.sum(np.log(self.calc_lik_vec(mu, sigma, p, active_komp)))

	def calc_lik_vec(self, mu = None, sigma = None, p = None, active_komp = None):
		
		self.compute_ProbX(norm=False, mu = mu, sigma = sigma,p = p, active_komp =  active_komp)
		if p is None:
			p = self.p
		
		if active_komp is None:
			active_komp = self.active_komp
			
		
		l = np.zeros(self.n)
		for k in range(self.K + self.noise_class): 
			if active_komp[k] == True:
				l += np.exp(self.prob_X[:,k])*p[k]
		
		return l
	
			
	def compute_ProbX(self,norm =True, mu = None, sigma = None, p =None, active_komp = None):
		"""
			Computes the E[x=i|\mu,\Sigma,p,Y] 
		"""
		if mu is None:
			mu = self.mu
			sigma = self.sigma
			high_memory = self.high_memory
			p = self.p
		else:
			high_memory = False
			
		if active_komp is None:
			active_komp = self.active_komp
		l = np.empty(self.n, order='C' )
		for k in range(self.K):
			
			if active_komp[k] == True:
				if high_memory == True:
					GMM_util.calc_lik(l, self.data_mu[k], self.sigma[k])
					
				else:
					X_mu = self.data - mu[k].reshape(self.d)
					
					GMM_util.calc_lik(l,   X_mu, sigma[k])
				
				self.prob_X[:,k] = l
			else:
				self.prob_X[:,k] = 0.
		
		
		if self.noise_class:
			self.prob_X[:,self.K] = self.l_noise
		
		if norm==True:
			GMM_util.calc_exp_normalize(self.prob_X, p, np.array(range(self.K + self.noise_class), dtype = np.int )[active_komp])

