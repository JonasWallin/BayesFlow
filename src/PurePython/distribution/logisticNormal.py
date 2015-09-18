# -*- coding: utf-8 -*-
"""
Created on Sat May 23 11:36:32 2015

@author: jonaswallin
"""
from __future__ import division

import numpy as np
import numpy.random as npr
import cPickle as pickle

#DONE: simulerings test som visar att samplingen funkar (Done)
#DONE: AMCMC version (Done)
#TODO: cython version (Not needed)
#TODO: make adjustment so n can contain NaN in which case
#	   we sample alpha with missing value!

class logisticMNormal(object):
	"""
		Class for sampling and storing multilogit normal distirbution:

		\alpha \sim N(\mu, \Sigma)
		p = [1 \alpha]/(1 + sum(exp(\alpha))
		n \sim \Multinomial(p)

		The sampling is using Adjusted MALA adaptive
	"""
	def pickle(self, filename):
		"""
			store object in file
		"""
		f = file(filename, 'wb')
		pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
		f.close()

	@staticmethod
	def unpickle(filename):
		"""
			load object from file
			use:

			object = multivariatenormal.unpickle(filename)
		"""
		with file(filename, 'rb') as f:
			return pickle.load(f)

	def __init__(self, prior = None):
		'''
			prior:
			prior['mu']	- np.array(dim=1) (d-1)
			prior['Sigma'] = np.array(dim=2) (d-1) x (d-1)
		'''
		self.prior = None
		self.n_count = 0
		self.n = None
		self.alpha = None
		self.mu  = None
		self.Q = None
		if prior != None:
			self.set_prior(prior)
			
		self.AMCMC = False
		self.sigma_MCMC = 0.99 #not equal to one
		self.count_mcmc = 0.
		self.accept_mcmc = 0.
		self.amcmc_count  = 0.
		self.amcmc_accept = 0.


	def set_mu(self, mu):
		"""
			Setting the mean prior
		"""
		self.mu	      = np.zeros_like(mu)
		self.mu[:]	  = mu[:]
		
		
	def set_prior(self, prior):
		"""

				prior:
					prior['mu']	- np.array(dim=1) (d-1)
					prior['Sigma'] = np.array(dim=2) (d-1) x (d-1)
		"""
		self.set_mu(prior['mu'])
		self.Sigma	  = np.zeros_like(prior['Sigma'])
		self.Sigma[:] = prior['Sigma'][:]
		self.Q		  = np.linalg.inv(self.Sigma)
		self.Q_mu	  = np.dot(self.Q,self.mu)
		self.d		  = self.Q_mu.shape[0] + 1


	def set_data(self, n):
		"""
			n - (Kx1) numpy vector of observation count
		"""
		shape = (np.prod(n.shape),1)
		self.n	   = np.zeros(shape)
		self.n[:]  = n.reshape(shape)[:]
		self.sum_n = np.sum(n)
		
		if self.alpha != None and self.mu != None:
			self.update_llik()

	def get_llik_grad_hess(self, alpha = None):
		"""
			gradient and Hessian of the log likelihood component of the model
			
			alpha - (d-1 , ) vector of probability components 
		"""
		if alpha == None:
			alpha = self.alpha


		exp_alpha = np.exp(alpha)
		c_alpha = (1 + np.sum(exp_alpha))
		grad = - self.sum_n * exp_alpha / c_alpha

		
		exp_alpha *= np.sqrt(self.sum_n)/c_alpha
		Hessian    = np.outer(exp_alpha.T, exp_alpha) 
		Hessian   += np.diag(grad.flatten())
		n_alpha    = self.n[1:]
		grad      += n_alpha
		
		llik =  np.sum(n_alpha * alpha) - self.sum_n * np.log(c_alpha)


		return llik, grad, Hessian
		
	def get_lprior_grad_hess(self, alpha = None):
	
		if alpha == None:
			alpha = self.alpha
		
		alpha_mu = alpha - self.mu
		grad = - np.dot(self.Q, alpha_mu)
		grad.reshape(np.prod(grad.shape))
		llik =   np.dot(alpha_mu.T, grad)/2.
		Hessian = np.zeros_like(self.Q)
		Hessian[:] = - self.Q[:]
		return llik, grad, Hessian
	
	
	def get_f_grad_hess(self, alpha = None):
		"""
			Calculating value, the gradient , Heissan of the density
			
			alpha - (d-1) values
		"""
		
		
		llik, grad, Hessian	= self.get_llik_grad_hess(alpha)
		llik2, grad2, Hessian2 = self.get_lprior_grad_hess(alpha)
		llik += llik2
		grad += grad2
		Hessian += Hessian2
		return llik, grad, Hessian
	
	
	def get_p(self, alpha = None):
		"""
			get the probabilities from the object 
		"""
		if alpha == None:
			alpha = self.alpha
			
		p = np.vstack((1.,np.exp(alpha)))
		p /= np.sum(p)
		p.reshape(np.prod(p.shape))
		return p
		
	def set_alpha(self, alpha): 
		"""
			setting alpha parameter
		
		"""
		self.alpha    = np.zeros((np.prod(alpha.shape),1))
		self.alpha[:] = alpha.reshape((np.prod(alpha.shape),1))[:]
		if self.n != None and self.mu != None:
			self.update_llik()
			
			
	def set_alpha_p(self, p):
		"""
			setting alpha through p
			
			p - (dx1) simplex vector
		"""
		
		alpha   = np.zeros(len(p)-1)
		sum_exp = 1./ p[0]
		for i in range(1,len(p)):
			alpha[i-1] = np.log(p[i] * sum_exp)
		
		self.set_alpha(alpha)		   
			
	def update_llik(self, alpha = None):
		"""
			Update components of the likelihood used in MALA
		"""
		
		store = False
		if alpha == None:
			store = True
			alpha = self.alpha
			
		llik, grad,  neg_Hessian = self.get_f_grad_hess(alpha)
		neg_Hessian *= -1
		L	= np.linalg.cholesky(neg_Hessian)
		Lg   = np.linalg.solve(L, grad)
		LtLg = np.linalg.solve(L.T, 0.5 * Lg)
		
		if store:
			self.llik, self.grad, self.neg_Hessian = llik, grad, neg_Hessian
			self.L	= L
			self.Lg   = Lg
			self.LtLg = LtLg
		
		return llik, grad, neg_Hessian, L, Lg, LtLg
	

		
	def sample(self, z = None):
		"""
			Sampling using AMCMC MALA with preconditioner as Hessian
		"""
		self.count_mcmc   += 1
		self.amcmc_count  += 1
		if z == None:
			z =npr.randn(self.d-1,1)
		
		mu = self.alpha  + self.LtLg *self.sigma_MCMC**2
		alpha_s = np.linalg.solve(self.L.T, self.sigma_MCMC*z) + mu 
		llik_s, grad_s, neg_Hessian_s, L_s, Lg_s, LtLg_s = self.update_llik(alpha_s)
		
		mu_s = alpha_s + LtLg_s * self.sigma_MCMC**2
		a_mu_s = self.alpha - mu_s
		a_s_mu = alpha_s - mu
		q_s = -(0.5/self.sigma_MCMC**2) *  np.dot( a_s_mu.T, np.dot(self.neg_Hessian, a_s_mu) )   
		q_o = -(0.5/self.sigma_MCMC**2) *  np.dot( a_mu_s.T, np.dot(neg_Hessian_s, a_mu_s) )
		
		U = np.random.rand(1)

		if np.log(U) < llik_s - self.llik + q_o - q_s:	   
			self.accept_mcmc  += 1
			self.amcmc_accept += 1
			self.llik, self.grad, self.neg_Hessian, self.L, self.Lg, self.LtLg   =  llik_s, grad_s, neg_Hessian_s, L_s, Lg_s, LtLg_s
			self.alpha = alpha_s
		
		
			
		if self.AMCMC:
			
			self.update_AMCMC()
		
	def set_AMCMC(self, batch = 50, accpate = 0.574, delta_rate = 1.):
		"""
			Using AMCMC
			
			batch	  - (int) how often to update sigma_MCMC
			accpate	- [0,1] desired accpance rate (0.574)
			delta_rate - [0,1] updating ratio for the amcmc
		"""
		
		
		
		self.amcmc_delta_max	= 0.1
		self.amcmc_desired_accept = accpate
		self.amcmc_batch		= batch
		self.amcmc_delta_rate   =  delta_rate
		self.AMCMC = True


	def update_AMCMC(self):
		"""
		Using roberts and rosenthal method for tunning the acceptance rate
		"""
	
	
		if (self.amcmc_count +1) % self.amcmc_batch == 0:

			delta = np.min([self.amcmc_delta_max, (self.count_mcmc/self.amcmc_batch)**(-self.amcmc_delta_rate)])
			
			if self.amcmc_accept / self.amcmc_count > self.amcmc_desired_accept:
				self.sigma_MCMC *= np.exp(delta) 
			else:
				self.sigma_MCMC /= np.exp(delta)
			
			self.amcmc_count  = 0.
			self.amcmc_accept = 0.