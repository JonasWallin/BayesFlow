'''
Created on Jul 10, 2014

@author: jonaswallin
'''
from __future__ import division
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import numpy.random as npr

from . import GMM
#from BayesFlow.distribution import normal_p_wishart, Wishart_p_nu
from .distribution import normal_p_wishart, Wishart_p_nu



class hierarical_mixture(object):
	
	def __init__(self, K ):
		"""
			starting up the class and defning number of classes
		
		"""
		self.d  = 0
		self.K = K
		self.normal_p_wisharts = [ normal_p_wishart() for k in range(self.K)]  # @UnusedVariable
		self.wishart_p_nus     = [Wishart_p_nu() for k in range(self.K) ]  # @UnusedVariable
		self.GMMs = []
		self.noise_class = 0 
		
		
	def add_noise_class(self,Sigma_scale = 5.):
		
		[GMM.add_noiseclass(Sigma_scale) for GMM in self.GMMs ]
		
	def set_prior(self, prior):
		
		pass
	
	
	def set_prior_param0(self):
		
		if self.d == 0:
			raise ValueError('have not set d need for prior0')
		
		for npw in self.normal_p_wisharts:
			npw.set_prior_param0(self.d)

		for wpn in self.wishart_p_nus:
			wpn.set_prior_param0(self.d)
			
		#TODO: def set param -> GMM function!!!
		if len(self.GMMs) > 0:
			self.update_GMM()

	def update_GMM(self):
		"""
			Transforms the data from the priors to the GMM
		"""
		mu_param = [npw.param for npw in self.normal_p_wisharts]
		sigma_param = [wpn.param for wpn in self.wishart_p_nus]
		for i in range(self.n):
			self.GMMs[i].set_prior_mu(mu_param)
			self.GMMs[i].set_prior_sigma(sigma_param)
	
	def update_prior(self):
		"""
			transforms the data from the GMM to the prior
		
		"""
		for k in range(self.K):
			mus    = np.array([GMM.mu[k] for GMM in self.GMMs])
			index = np.isnan(mus[:,0])==False
			Sigmas = np.array([GMM.sigma[k] for GMM in self.GMMs])
			self.normal_p_wisharts[k].set_data(mus[index,:])
			self.wishart_p_nus[k].set_data(Sigmas[index,:,:])
			
			
	def reset_nus(self, nu, Q =None):
		"""
			reseting the values of the latent parameters of the covariance 
			
		"""
		
		for wpn in self.wishart_p_nus:
			Q_ = np.zeros_like(wpn.param['Q'].shape[0])
			if Q == None:
				Q_ = 10**-10*np.eye(wpn.param['Q'].shape[0]) 
			else:
				Q_  = Q[:]
			param = {'nu':nu, 'Q': Q_}
			wpn.set_val(param)
	
	
	def set_nu_MH_param(self, sigma = 5, iteration = 5):
		"""
			setting the parametet for the MH algorithm
		"""
		for wpn in self.wishart_p_nus:
			wpn.set_MH_param( sigma , iteration)
	
	def reset_Sigma_theta(self, Sigma = None):
		"""
			reseting the values of the latent parameters of the mean
		"""

		for npw in self.normal_p_wisharts:
			if Sigma ==  None:
				npw.param['Sigma']  = 10**10*np.eye(npw.param['Sigma'].shape[0])
			else:
				npw.param['Sigma'][:]  = Sigma[:]
		
		
	def reset_prior(self,nu = 10):
		"""
			reseting the values for the latent layer
		"""
		
		self.reset_nus(nu)
		self.reset_Sigma_theta()	
		self.update_GMM()
		
	def set_nuss(self, nu):
		"""
			increase to force the mean to move together
		"""
		for k in range(self.K):
			self.normal_p_wisharts[k].Sigma_class.nu = nu
			
	def set_nu_mus(self, nu):
		"""
			increase to force the covariance to move together
		
		"""
		for k in range(self.K):
			self.wishart_p_nus[k].Q_class.nu_s = nu

	def set_p_activation(self, p):
		
		for GMM in self.GMMs:
			GMM.p_act   = p[0]
			GMM.p_inact = p[1]
	
	def set_p_labelswitch(self,p):
		
		
		for GMM in self.GMMs:
			GMM.p_switch   = p
	

	def set_data(self, data):
		"""
			List of np.arrays
		
		"""
		
		self.d = data[0].shape[1]
		self.n = len(data)
		for Y in data:
			if self.d != Y.shape[1]:
				raise ValueError('dimension missmatch in thet data')
			self.GMMs.append(GMM.mixture(data= Y, K = self.K))
	
	
	def get_thetas(self):
		
		return [cp.deepcopy(npw.param['theta']) for npw in self.normal_p_wisharts]
		
	def get_mus(self):
		
		mus = np.array([[GMM.mu[k] for k in range(self.K) ]  for GMM in self.GMMs ],dtype='d')
		return mus

	def get_ps(self):
		
		ps = np.array([GMM.p.flatten()   for GMM in self.GMMs ],dtype='d')
		return ps
	
	def get_sigmas(self):
	
	
		sigmas = np.array([[GMM.sigma[k]for k in range(self.K) ]  for GMM in self.GMMs ],dtype='d')	
		return sigmas
	
	
	def sample(self):
		
		for GMM in self.GMMs:
			GMM.sample() 
		self.update_prior()
		for k in range(self.K):
			self.normal_p_wisharts[k].sample()
			self.wishart_p_nus[k].sample()
		self.update_GMM()
		
	def sampleY(self):
		"""
			draws a sample from the joint distribution of all persons
		"""
		Y = [GMM.simulate_one_obs() for GMM in self.GMMs]
		prob = [GMM.n for GMM in self.GMMs]
		return Y[npr.choice(range(len(self.GMMs)), p = prob/np.sum(prob))]
	
	def plot_mus(self,dim, ax = None, cm = plt.get_cmap('Dark2'), size_point = 1., colors = None):
		"""
			plots all the posteriror mu's dimension dim into ax
		
		"""
		if colors != None:
			if len(colors) != self.K:
				print "in hier_GMM.plot_mus: can't use colors aurgmen with length not equal to K"
				return
		if ax == None:
			f = plt.figure()
			ax = f.add_subplot(111)
		else:
			f = None
		
		if len(dim) == 1:
			
			print("one dimension not implimented yet")
			pass
		
		elif len(dim) == 2:
			
			
			
			
			
			
			for k in range(self.K):
				mus = np.array([GMM.mu[k] for GMM in self.GMMs])
				if colors != None:
					ax.plot(mus[:,dim[0]],mus[:,dim[1]],'.',color=cm(k/self.K), s = size_point)
				else:
					ax.plot(mus[:,dim[0]],mus[:,dim[1]],'.',color=colors[k], s = size_point)
			return f, ax
			
		elif len(dim) == 3:
			
			print("three dimension not implimented yet")
			pass	
	
		else:
			print("more then three dimensions thats magic!")