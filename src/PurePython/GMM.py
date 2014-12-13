# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 15:51:51 2014

@author: jonaswallin
"""
from __future__ import division
import numpy as np
import numpy.random as npr
import copy as cp
from bayesianmixture.PurePython.distribution import wishart
import scipy.special as sps
import matplotlib.pyplot as plt
import scipy.linalg as sla
from bayesianmixture.utils.gammad import ln_gamma_d
import cPickle as pickle
#import bayesianmixture.mixture_util.GMM_util as GMM_util


#TODO: remove noise class, should be fairly easy to add an usefull

def log_betapdf(p, a, b):
	
	pdf = - sps.betaln(a, b)
	pdf += (a - 1.) * np.log(p) + (b - 1.) * np.log( 1 - p)
	return pdf

def log_dir(p, alphas):
	'''log pdf dirichlet function'''
	alpha_m = alphas - 1.
	c = sps.gammaln(alphas.sum()) - sps.gammaln(alphas).sum()
	return c + (alpha_m * np.log(p)).sum()

def sample_mu(X, Sigma, theta, Sigma_mu):
	"""
		Sampling the posterior mean given:
		X		 - (nxd)  the data
		Sigma	 - (dxd)  the covariance of the data
		theta	 - (dx1)  the prior mean of mu
		Sigma_mu  - (dx1)  the prior cov of mu
		
	"""
	
	x_sum = np.sum(X,0)
	inv_Sigma	= np.linalg.inv(Sigma)
	inv_Sigma_mu = np.linalg.inv(Sigma_mu)
	
	mu_s		 = np.dot(inv_Sigma_mu, theta).reshape(theta.shape[0]) + np.dot(inv_Sigma, x_sum)
	Q			= X.shape[0] * inv_Sigma  + inv_Sigma_mu
	L = np.linalg.cholesky(Q)
	mu_t = np.linalg.solve(L, mu_s)
	mu = np.linalg.solve(L.transpose(), mu_t + np.random.randn(X.shape[1]))
	return mu.reshape(mu.shape[0])
	
def sample_sigma_zero_mean(X, Q, nu):
	
	Q_star = Q +  np.dot(X.transpose(), X)
	nu_star = nu + X.shape[0]
	return wishart.invwishartrand(nu_star, Q_star)	

def sample_sigma(X, mu, Q, nu):
	"""
		parameter for sampling the posterior distribution
		of the covariance matrix given:
		X   - (nxd)	the data
		mu  - (d)	  the mean
		Q   - (dxd)	the covariance
		nu  - (double) the observation parameter for the Inverse Wishart prior
	
	"""
	
	X_mu = X - mu
	return sample_sigma_zero_mean(X_mu, Q, nu)

class mixture(object):
	"""
		Regular Bayesian mixture model
		with multivariate normal clusters
	
		model:
		
			f(Y) = \sum_{i=1}^K p_i f(Y; \mu_i, \Sigma_i).
	
	"""
	
	def __init__(self, data = None, K = None,  prior = None, high_memory=True , name=None):
		"""
			Startup
		
			data	-  nxd np.array
			K	   -  the number of classes
			prior   -  The prior parameters for mu, Sigma, prob
					   list one element per class
					   each element is dictionary with
					   "mu", "sigma", "p"
			high_memoty - have large amount of memory avialble
		"""
		
		self.mu  = []	 
		self.sigma = []		
		self.K = K
			
		self.high_memory = high_memory
		self.prior = cp.deepcopy(prior)
		if data != None:
			self.set_data(data)


		# components for activation deactivation of cluster
		self.p_act = 0. #probability of activating cluster
		self.p_inact = 0. #probability of inactive incluster
		self.active_komp = np.ones(self.K,dtype='bool') 	
		self.beta_act_param = [2, 60.]
		self.komp_prior = 5. # the prior probability is  exp(- np.sum(self.active_kmp)*self.komp_prior) * np.sum(self.active_kmp)
		# label switchin
		self.p_switch = 0
		self.noise_class = 0
		self.lab =np.array([-1,-1])
		
		self.name=  name
	
	
	
	def load_param(self, params):
		"""
			loads the object from previosuly stored (though not the data)
		
		"""
		self.p_act       = params["p_act"]
		self.p_inact     = params["p_inact"]
		self.active_komp = params["active_komp"]
		self.komp_prior  = params["komp_prior"]
		self.p_switch    = params["p_switch"]
		self.noise_class = params["noise_class"]
		if self.noise_class:
			self.prob_X = np.zeros((self.n, self.K + 1))
			self.noise_sigma = params["noise_sigma"]
			self.noise_mean  = params["noise_mean"]
			self.update_noiseclass()
		self.lab		 = params["lab"]
		self.set_mu(params["mu"])
		self.set_sigma(params["sigma"])
		self.p           = params["p"]
		self.alpha_vec   = params["alpha_vec"]
		self.high_memory = params["high_memory"]
		
		
	def write_param(self):
	
		params = {}
		params["p_act"]   		= cp.deepcopy(self.p_act )      
		params["p_inact"] 		= cp.deepcopy(self.p_inact     )
		params["active_komp"] 	= cp.deepcopy(self.active_komp )
		params["komp_prior"] 	= cp.deepcopy(self.komp_prior  )
		params["p_switch"] 		= cp.deepcopy(self.p_switch    )
		params["noise_class"] 	= cp.deepcopy(self.noise_class )
		if self.noise_class:
			params["noise_sigma"] = cp.deepcopy(self.noise_sigma) 
			params["noise_mean"]  = cp.deepcopy(self.noise_mean )
			
		params["lab"]   		= cp.deepcopy(self.lab		  )
		params["mu"]   			= cp.deepcopy(self.mu  		)
		params["sigma"] 		= cp.deepcopy(self.sigma  )
		params["p"]     		= cp.deepcopy(self.p        )    
		params["alpha_vec"] 	= cp.deepcopy(self.alpha_vec )
		params["high_memory"]	= cp.deepcopy(self.high_memory)
		
		return params
	
	
	def set_name(self,name):
		"""
			setting the name
		"""
		self.name = name
		
	def add_noiseclass(self, Sigma_scale  = 5.):
		"""
			adds a class that does not update and cant be deactiveted or label switch
			the data need to be loaded first!
			
			Sigma_scale  - (double)  the scaling constants time the covariance matrix
		"""
		
		
		if self.data  == None:
				raise ValueError, 'need data to be loaded first'
		
		self.noise_class = 1
		self.active_komp = np.hstack((self.active_komp,True))
		self.p = np.hstack((self.p * (1- 0.01), 0.01))
		self.alpha_vec =  np.hstack((self.alpha_vec,1/2.)) 
		Sigma  = Sigma_scale *  np.cov(self.data.T)*10.
		mu     = np.mean(self.data,0)
			
		self.prob_X = np.zeros((self.n, self.K + 1))
		self.noise_sigma = Sigma
		self.noise_mean  = mu
		self.update_noiseclass()
	
	def update_noiseclass(self):
		"""
			Run this if change noise_sigma or noise_mean
		"""
		
		Q = np.linalg.inv(self.noise_sigma)
		X_mu = self.data - self.noise_mean
		self.l_noise  = np.log(np.linalg.det(Q))/2. - (self.d/2.)* np.log(2 * np.pi)
		self.l_noise -= np.sum(X_mu * np.dot(X_mu,Q),1)/2.
		
	def sample_active_komp(self):
		"""
			tries to turn off or on an active or inactive component
		
		"""

		U = npr.rand()
		if U < self.p_act:
			# active_cluster
			self.sample_activate()
		
		U = npr.rand()
		if U < self.p_inact:
			#in active cluster
			self.sample_inactivate()
		
		
		
	def sample_inactivate(self):
		"""
			try to inactivate a component using RJMCMC
		"""
				
				
				
		K_s = np.array(range(self.K+self.noise_class))
		self.act_index    = K_s[self.active_komp==True]
		if self.noise_class == 1:
			self.act_index = self.act_index[:-1]
		self.in_act_index = K_s[self.active_komp==False]
		if len(self.act_index)< 2: 
			return
		
		q_in = self.p_inact /  len(self.act_index)
		q_ac = self.p_act   / (1 + len(self.in_act_index))
		


		k_off       = npr.choice(self.act_index) # the new sample index
		active_komp = cp.deepcopy(self.active_komp)
		mu		    = cp.deepcopy(self.mu)
		sigma	    = cp.deepcopy(self.sigma)
		p		    = cp.deepcopy(self.p)
		
		active_komp[k_off] = False
		p_off		       = p[k_off]
		p				   = p/(1. - p_off)
		p[k_off]	 	   = 0.
		mu[k_off]		   = np.NAN * np.ones(self.d )
		sigma[k_off] 	   = np.NAN * np.ones((self.d, self.d))
		
		
		
		log_Jacobian = len(self.active_komp) * np.log(1 - p_off)
		piy_vec_star = self.calc_lik_vec(mu = mu, sigma = sigma, p = p, active_komp = active_komp)
		log_b_star   = log_betapdf(p_off, self.beta_act_param[0], self.beta_act_param[1])
		log_d_star   = log_dir(p[active_komp == True], self.alpha_vec[active_komp == True] )  


		# pi(y, \theta) 
		piy_vec	  = self.calc_lik_vec()
		log_d	   = log_dir(self.p[self.active_komp == True], self.alpha_vec[self.active_komp == True] )   		
		log_piy_div_piy_star = np.sum(np.log(piy_vec/piy_vec_star) )

		
		#pi(y, \theta_star)
		log_piy_div_piy_star -= log_d_star - log_d + self.komp_prior
		alpha = -log_piy_div_piy_star - log_Jacobian  - np.log(q_in) + np.log(q_ac) + log_b_star
		if np.isnan(alpha):
			return # 0 divided by zero then change nothing
		if np.log(npr.rand()) < alpha:
			self.p 			 = p
			self.mu 		 = mu
			self.sigma	   = sigma
			self.active_komp = active_komp
			
	
	def sample_activate(self):
		"""
			try to activate a component using RJMCMC
			
			q_in - probability of choosing to incative a spesific component
			q_ac - probability of choosing to active a spesific component
			log_piy	- log likelihood f(Y; \mu, \Sigma)
			log_d_star - log likelihood dirchelet process
			komp_prior - the value of the prior see __init__
		"""
		
		K_s = np.array(range(self.K+self.noise_class))
		self.act_index = K_s[self.active_komp==True]
		if self.noise_class == 1:
			self.act_index = self.act_index[:-1]
		self.in_act_index = K_s[self.active_komp==False]		
		
		
		if len(self.in_act_index) == 0: # no inactive component -1
			return
		
		
		q_in = self.p_inact / ( 1+ len(self.act_index)) #probability of a speifisc component targeted for activation
		q_ac = self.p_act / len(self.in_act_index) #probability of a speifisc component targeted for activation
		

		
		
		k_in = npr.choice(self.in_act_index) # the new sample index
		active_komp = cp.deepcopy(self.active_komp)
		mu		  = cp.deepcopy(self.mu)
		sigma	   = cp.deepcopy(self.sigma)
		p		   = cp.deepcopy(self.p)
		
		active_komp[k_in] = True
		
		p_in = npr.beta(self.beta_act_param[0], self.beta_act_param[1])
		p		   = (1.-p_in) * p
		p[k_in]	 = p_in
		mu[k_in]	= npr.multivariate_normal(self.prior[k_in]['mu']['theta'].reshape(self.d)	, self.prior[k_in]['mu']['Sigma'] ) 
		sigma[k_in] = wishart.invwishartrand_prec(self.prior[k_in]['sigma']['nu'], self.prior[k_in]['sigma']['Q'])	
		
		
		log_Jacobian = len(self.active_komp) * np.log(1 - p_in)
		piy_vec_star = self.calc_lik_vec(mu = mu, sigma = sigma, p = p, active_komp = active_komp)
		log_b_star   = log_betapdf(p_in, self.beta_act_param[0], self.beta_act_param[1])
		log_d_star   = log_dir(p[active_komp == True], self.alpha_vec[active_komp == True] )   
		
		
		
		# pi(y) - current
		piy_vec	  = self.calc_lik_vec()
		log_d		= log_dir(self.p[self.active_komp == True], self.alpha_vec[self.active_komp == True] )   		
		
		
		
		pi_div_pi_star = np.sum(np.log(piy_vec/piy_vec_star)) - log_d_star + log_d + self.komp_prior
		alpha = - pi_div_pi_star + log_Jacobian  + np.log(q_in) - np.log(q_ac) - log_b_star
		if np.isnan(alpha):
			return # 0 divided by zero then change nothing
		if np.log(npr.rand()) < alpha:
			self.p 			 = p 
			self.mu 		 = mu
			self.sigma	   = sigma
			self.active_komp = active_komp
		
		
	def set_data(self, data):
		
		if data.shape[0] <= data.shape[1]:
				raise ValueError, 'the number of observations must be larger then the dimenstion'
		self.data = np.empty_like(data)
		self.data[:] = data[:]
		self.n = self.data.shape[0]
		self.d  = self.data.shape[1]
		#just a stupied inital guess!!!
		cov_data  = np.cov(data.transpose())
		if len(self.mu) == 0:
			self.mu  = []	 
			self.sigma = []
			mean_data = np.mean(data,0)
			for i in range(self.K):  # @UnusedVariable
				self.mu.append(npr.multivariate_normal(mean_data,cov_data*0.1))
				self.sigma.append(0.1*cov_data)
		

		#creating non-informative priors
		mu_prior = {"theta":np.zeros((self.d ,1)),"Sigma":np.diag(np.diag(cov_data))*10**4 }
		sigma_prior = {"nu":self.d, "Q":np.eye(self.d )*10**-6}
		self.alpha_vec = 0.5*np.ones(self.K) 
		if self.prior == None:
			self.prior =[]
			for i in range(self.K):  # @UnusedVariable
				self.prior.append({"mu":cp.deepcopy(mu_prior), "sigma": cp.deepcopy(sigma_prior),"p": 1/2.})
		
		self.p = np.ones(self.K, dtype=np.double)/self.K 
		self.prob_X = np.zeros((self.n, self.K))
		self.x = -np.ones(shape=self.n,dtype = np.int, order='C' )
		if self.high_memory == True:
			self.data_mu = []
			for k in range(self.K):
				self.data_mu.append( self.data - self.mu[k] )
		
		self.ln_gamma_d = ln_gamma_d(self.d)
		
	def sample(self):
		
		self.sample_x()
		self.sample_mu()
		self.sample_sigma()
		self.sample_p()
		self.sample_active_komp()
		self.lab = self.sample_labelswitch()
		
	def updata_mudata(self):
		
		if self.high_memory == True:
			for k in range(self.K):
				if self.active_komp[k] == True:
					self.data_mu[k] = self.data - self.mu[k]	
		
	def set_mu(self, mu):
		"""
			setting value for the mean parameter
		
		"""
		for k in range(self.K):
			self.mu[k][:] = mu[k][:]
			
		self.updata_mudata()

	def set_sigma(self, sigma):
		
		for k in range(self.K):
			self.sigma[k][:] = sigma[k][:]
	
	def set_prior(self, prior):  
		self.prior = cp.deepcopy(prior)
		for k in range(self.K):
			self.alpha_vec[k] = self.prior[k]['p']
	
	def set_prior_sigma(self, prior):
		for k in range(self.K):
			self.prior[k]['sigma']['nu']   = prior[k]['nu']
			self.prior[k]['sigma']['Q'][:] = prior[k]['Q'][:] 
			
	def set_prior_sigma_np(self, nu, Q):
		
		for k in range(self.K):
			self.prior[k]['sigma']['nu']   = nu[k]	
			self.prior[k]['sigma']['Q'][:] = Q[k,:,:]
			
	def set_prior_mu_np(self, mu, Sigma):
		"""
			when mu is np.array 2D
			when Sigma is np.arry 3D
		"""
		for k in range(self.K):
			self.prior[k]['mu']['theta'][:]   = mu[k,:].reshape(self.prior[k]['mu']['theta'].shape)
			self.prior[k]['mu']['Sigma'][:]   = Sigma[k,:,:] 
			
			
	def set_prior_mu(self, prior):
		for k in range(self.K):
			self.prior[k]['mu']['theta'][:]   = prior[k]['theta'].reshape(self.prior[k]['mu']['theta'].shape)
			self.prior[k]['mu']['Sigma'][:]   = prior[k]['Sigma'][:] 
				
	def set_param(self, param):
		
		if len(self.mu) == 0 :
			for k in range(self.K):
				self.mu.append(np.empty_like(param[k]['mu']))
				self.sigma.append((np.empty_like(param[k]['sigma'])))
			
		for k in range(self.K):
			self.mu[k][:] = param[k]['mu'][:]
			self.sigma[k][:] = param[k]['sigma'][:]
			
		self.updata_mudata()

			
	def sample_x(self):
		"""
			Draws the label of the observations
		
		"""
		self.compute_ProbX()
		P = np.cumsum(self.prob_X,1)
		U = npr.rand(self.n)
		index_n = np.array(range(self.n),dtype=np.int)
		for i in range(self.K + self.noise_class): 
			index = U < P[:,i]
			self.x[index_n[index]] = i
			
			P = P[index == False]
			U = U[index == False]
			
			
			index_n = index_n[index == False] 
			
			if U.shape[0] == 0:
				break
		else:
			self.x[index_n] = self.K - 1 + self.noise_class
			
		

	def sample_labelswitch(self):
		"""
			Tries to switch two random labels
		"""	
	
		if npr.rand() < self.p_switch:
				labels = npr.choice(self.K,2,replace=False)
				if np.sum(self.active_komp[labels]) == 0:
						return np.array([-1,-1])
					
				lik_old, R_S_mu0, log_det_Q0, R_S0  = self.likelihood_prior(self.mu[labels[0]],self.sigma[labels[0]], labels[0])
				lik_oldt, R_S_mu1, log_det_Q1, R_S1 = self.likelihood_prior(self.mu[labels[1]],self.sigma[labels[1]], labels[1])
				lik_old += lik_oldt
				lik_star = self.likelihood_prior(self.mu[labels[1]],self.sigma[labels[1]], labels[0], R_S_mu0, log_det_Q0, R_S1)[0]
				lik_star += self.likelihood_prior(self.mu[labels[0]],self.sigma[labels[0]], labels[1], R_S_mu1,log_det_Q1, R_S0)[0]
				if np.log(npr.rand()) < lik_star - lik_old:
						self.active_komp[labels[0]], self.active_komp[labels[1]] = self.active_komp[labels[1]], self.active_komp[labels[0]]
						self.mu[labels[0]], self.mu[labels[1]] = self.mu[labels[1]], self.mu[labels[0]]
						self.sigma[labels[0]], self.sigma[labels[1]] = self.sigma[labels[1]], self.sigma[labels[0]]
						self.p[labels[0]], self.p[labels[1]] = self.p[labels[1]], self.p[labels[0]]
						self.updata_mudata()
						return labels
		
		return np.array([-1,-1])
		

	def likelihood_prior(self, mu, Sigma, k, R_S_mu = None, log_det_Q = None, R_S = None):
			"""
					Computes the prior that is 
					\pi( \mu | \theta[k], \Sigma[k]) \pi(\Sigma| Q[k], \nu[k]) = 
					N(\mu; \theta[k], \Sigma[k]) IW(\Sigma; Q[k], \nu[k])
			"""
			if np.isnan(mu[0]) == 1:
					return 0, None, None, None
			
			if R_S_mu == None:
					R_S_mu = sla.cho_factor(self.prior[k]['mu']['Sigma'],check_finite = False)
			log_det_Sigma_mu = 2 * np.sum(np.log(np.diag(R_S_mu[0])))
			
			if log_det_Q == None:
					R_Q = sla.cho_factor(self.prior[k]['sigma']['Q'],check_finite = False)
					log_det_Q = 2 * np.sum(np.log(np.diag(R_Q[0])))
			
			if R_S == None:
					R_S = sla.cho_factor(Sigma,check_finite = False)
			log_det_Sigma	= 2 * np.sum(np.log(np.diag(R_S[0])))
			
			
			
			mu_theta = mu - self.prior[k]['mu']['theta'].reshape(self.d)
			# N(\mu; \theta[k], \Sigma[k])
			
			lik = - np.dot(mu_theta.T, sla.cho_solve(R_S_mu, mu_theta, check_finite = False))  /2
			lik = lik - 0.5 * (self.prior[k]['sigma']['nu'] + self.d + 1.) * log_det_Sigma
			lik = lik +  (self.prior[k]['sigma']['nu'] * 0.5) * log_det_Q
			lik = lik - 0.5 * log_det_Sigma_mu
			lik = lik - self.ln_gamma_d(0.5 * self.prior[k]['sigma']['nu']) - 0.5 * np.log(2) * (self.prior[k]['sigma']['nu'] * self.d)
			lik = lik - 0.5 * np.sum(np.diag(sla.cho_solve(R_S, self.prior[k]['sigma']['Q'])))
			return lik, R_S_mu, log_det_Q, R_S

			
	
	def sample_mu(self):
		"""
			Draws the mean parameters
			self.mu[k] - 
		"""
		for k in range(self.K):  
			if self.active_komp[k] == True:
				self.mu[k] = sample_mu(self.data[self.x == k ,:], self.sigma[k], self.prior[k]['mu']['theta'], self.prior[k]['mu']['Sigma'])
			else:
				self.mu[k] = np.NAN * np.ones(self.d)
				
		self.updata_mudata()
		
		
	def sample_sigma(self):
		"""
			Draws the covariance parameters
		
		"""
		
		if self.high_memory == True:
			for k in range(self.K): 
				if self.active_komp[k] == True: 
					self.sigma[k] = sample_sigma_zero_mean(self.data_mu[k][self.x == k ,:], 
					 							self.prior[k]["sigma"]["Q"],
					 					 		self.prior[k]["sigma"]["nu"])
				else:
					self.sigma[k] = np.NAN * np.ones((self.d, self.d))
		else:
			for k in range(self.K):  
				if self.active_komp[k] == True: 
					self.sigma[k] = sample_sigma(self.data[self.x == k ,:], self.mu[k], 
					 					 self.prior[k]["sigma"]["Q"],
					 					 self.prior[k]["sigma"]["nu"])
				
				else:
					self.sigma[k] = np.NAN * np.ones((self.d, self.d))
	
	
	def sample_p(self):
		"""
			Draws the posterior distribution for
			the probabilities of class belonings
		"""
		alpha = self.alpha_vec.copy()
		for k in range(self.K + self.noise_class):
			if self.active_komp[k]:
				alpha[k] += np.sum(self.x==k)
				
		
		alpha = alpha[self.active_komp]
		self.p[self.active_komp] = np.random.dirichlet(alpha)
		self.p[self.active_komp == False] = np.NAN

	
	
	def calc_lik(self, mu = None, sigma = None, p = None, active_komp = None):
		
		return np.sum(np.log(self.calc_lik_vec(mu, sigma, p, active_komp)))
	
	def calc_lik_vec(self, mu = None, sigma = None, p = None, active_komp = None):
		
		
		self.compute_ProbX(norm=False, mu = mu, sigma = sigma,p = p, active_komp =  active_komp)
		if active_komp == None:
			active_komp = self.active_komp
		l = np.zeros(self.n)
		for k in range(self.K + self.noise_class): 
			if active_komp[k] == True:
				l += np.exp(self.prob_X[:,k])
		return l
		
				
	
	def compute_ProbX(self, norm =True, mu = None, sigma = None, p =None, active_komp = None):
		"""
			Computes the E[x=i|\mu,\Sigma,p,Y] 
		"""
		if mu == None:
			mu = self.mu
			sigma = self.sigma
			high_memory = self.high_memory
			p = self.p
		else:
			high_memory = False
		if active_komp == None:
			active_komp = self.active_komp
		
		for k in range(self.K):
			if active_komp[k] == True:
				Q = np.linalg.inv(sigma[k])
				if high_memory == True:
					self.prob_X[:,k] = np.log(np.linalg.det(Q))/2. - (self.d/2.)* np.log(2 * np.pi)
					self.prob_X[:,k] -= np.sum(self.data_mu[k] * np.dot(self.data_mu[k],Q),1)/2.
				else:
					X_mu = self.data - mu[k]
					self.prob_X[:,k] = np.log(np.linalg.det(Q))/2. - (self.d/2.)* np.log(2 * np.pi)
					self.prob_X[:,k] -= np.sum(X_mu * np.dot(X_mu,Q),1)/2.
				
				self.prob_X[:,k] += np.log(p[k])
		
		
		
		if self.noise_class:
			self.prob_X[:,self.K] = self.l_noise + np.log(p[self.K])
		if norm:
			self.prob_X -= np.reshape(np.max(self.prob_X,1),(self.prob_X.shape[0],1))
			self.prob_X = np.exp(self.prob_X)
			self.prob_X /= np.reshape(np.sum(self.prob_X,1),(self.prob_X.shape[0],1))
		
		for k in range(self.K):
			if active_komp[k] == False:
				self.prob_X[:,k] = 0.

	def simulate_data(self,n):
		"""
			simulates data using current Sigma, mu, p
			if there exists noise class it will not be used
		"""
		
		p = np.zeros_like(self.p)
		p[:] = self.p[:]
		p[np.isnan(p)] = 0 
		if self.noise_class:
			p = p[:-1]
			p /= np.sum(p)		
		x_ = npr.multinomial(1, p, size=n)
		_, x = np.where(x_)  
		X =np.zeros((n,self.d))
		for k in range(self.K):
			n_count = np.sum(x == k)
			x_ = npr.multivariate_normal(self.mu[k], self.sigma[k],size = n_count)
			X[x == k,:] = x_
		return X
	
	def simulate_one_obs(self):
		"""
			if there exists noise class it will not be used
		"""
		p = np.zeros_like(self.p)
		p[:] = self.p[:]
		p[np.isnan(p)] = 0 
		if self.noise_class:
			p = p[:-1]
			p /= np.sum(p)
		x = npr.choice(range(self.K),p = p)
		return npr.multivariate_normal(self.mu[x], self.sigma[x],size = 1)
	
	
	def plot_scatter(self, dim, ax=None):
		'''
			Plots the scatter plot of the data over dim
			and assigning each class a different color
		'''
		
		
		if ax == None:
			f = plt.figure()
			ax = f.add_subplot(111)
		else:
			f = None
			
		data= self.data[:,dim]
		cm = plt.get_cmap('gist_rainbow')
		if len(dim) == 2:
			for k in range(self.K):
				ax.plot(data[self.x==k,dim[0]],data[self.x==k, dim[1]],'+',label='k = %d'%(k+1),color=cm(k/self.K))
				
		return f, ax

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
			
			object = mixture.unpickle(filename)
		"""
		with file(filename, 'rb') as f:
			return pickle.load(f)	

if __name__ == "__main__":
	N = 100
	K = 2
	ncomps=  3
	import  datetime  # @UnusedImport
	#import test_help
	#npr.seed(datetime.datetime.now().microsecond)
	#true_labels, data = test_help.generate_data(n=N, k=K, ncomps=ncomps)
	mix = mixture(np.random.randn(N,2),K=ncomps)
	mix.sample_x()