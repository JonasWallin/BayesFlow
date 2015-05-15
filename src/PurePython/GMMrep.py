'''
Created on May 12, 2015

@author: jonaswallin
'''
from __future__ import division
import numpy as np
import numpy.random as npr
import copy as cp
from BayesFlow.PurePython.distribution import wishart
from BayesFlow.PurePython.GMM import sample_mu_Xbar, sample_sigma_zero_mean, sample_mu, log_dir, log_betapdf
import scipy.special as sps
#import matplotlib.pyplot as plt
import scipy.linalg as sla
from BayesFlow.utils.gammad import ln_gamma_d
import cPickle as pickle

def sample_mu_integrate(n, Y_sum, Sigma_eps, Sigma):
	"""
		samples \mu in the following model:
		f(\mu_{i,l} | \mu_{i})  \sim N(\mu_{i}, \Sigma_{\epsilon,i})
		f(Y) = \sum_{i=1}^K p_i f(Y; \mu_{i,l}, \Sigma_i)
		and then one integrates out all the \mu_{i,l}
		
		n         - (mx1) number of observations in each group
		Y_sum     - (mxd) the sum of all observations for each group
		Sigma_eps - (dxd) covariance of mu_{i,l} | \mu_l
		Sigma     - (dxd) covariance of Y | \mu_{i,l} 
		
		
		returns:
		
		mu 		  - (dx1) sample of mu
		mu_mu     - (dx1) mean of mu
		mu_sigma  - (dxd) covariance of mu
	"""
	
	if len(n) > 1:
		
		L = np.linalg.cholesky(Sigma)
		Sigma_inv    = np.linalg.solve(L ,np.eye(Sigma.shape[0]))
		Sigma_inv    = np.linalg.solve(L.transpose() ,Sigma_inv)
		S            = np.dot(Sigma, Sigma_eps)
		S_eig, S_vec = np.linalg.eig(S)
		S_vec_T = S_vec.T
		D  = np.zeros(Sigma.shape[0])
		mu = np.zeros(Sigma.shape[0])
		for i in range(len(n)):
			D_i = 1./(S_eig + 1./n[i])
			mu += D_i * np.dot(S_vec_T, Y_sum[i]/n[i])
			D += D_i 
		Sigma_mu = np.dot(S_vec,np.diag(1./D)) 
		mu_mu    = np.dot(Sigma_mu, mu)
		Sigma_mu = np.dot( np.dot(Sigma_mu, S_vec_T), Sigma )
	else:
		
		Sigma_mu = Sigma/n[0] + Sigma_eps
		mu_mu    = Y_sum[0]/n[0]
		
		
	L = np.linalg.cholesky(Sigma_mu)	
	mu = np.dot(L,npr.randn(L.shape[0])) + mu_mu
	#Sigma_mu = 
	#np.linalg.eig(Sigma[0])
	return mu, mu_mu, Sigma_mu



def underconstruct(func):
	print("%s is not implimented yet"%func.__name__)
	def decorotar(*args, **kwargs):
		
		return func(*args,**kwargs)
	
	return decorotar

#TODO: remove noise class, should be fairly easy to add an usefull

#TODO: ALOT
#TODO: add logit link interface!
#TODO: check AMCMC

class mixture_repeat_measurements(object):
	"""
		Bayesian mixture model with multivariate normal clusters,
		where the clustered have warying mean coming from measurment error
	
	
		model:
			f(\mu_{i,l} | \mu_{i})  \sim N(\mu_{i}, \Sigma_{\epsilon,i})
			f(Y) = \sum_{i=1}^K p_i f(Y; \mu_{i,l}, \Sigma_i).
	
	"""
	
	
	
	
	def __init__(self, data = None, measurement =None, K = None,  prior = None, high_memory=True , name=None, AMCMC = False):
		"""
			Startup
		
			data	   		-   nxd np.array with observations
			measurement		- nx1 inidicating which measurement the observations belongs to
								 
			K	       -  the number of classes
			
			
			prior      -  The prior parameters for mu, Sigma, prob
					   	  list one element per class
					      each element is dictionary with
					      "mu", "sigma", "p"
			
			
			high_memory - have large amount of memory avialble
			
			AMCMC 		- changes the Gibbs sample of x_i so that with probability underP[i]
						  samples x_i otherwise keeps x_i 
		"""
		
		
		self.mu  = []	 
		self.sigma = []		
		self.K = K
			
		self.high_memory = high_memory
		self.prior = cp.deepcopy(prior)
		if not data is None:
			self.set_data(data, measurement)


		# components for activation deactivation of cluster
		self.p_act = 0. #probability of activating cluster
		self.p_inact = 0. #probability of inactive incluster
		self.active_komp = [np.ones(self.K,dtype='bool') for i in range(self.n_measurements)] 	
		self.beta_act_param = [2, 60.]
		self.komp_prior = 5. # the prior probability is  exp(- np.sum(self.active_kmp)*self.komp_prior) * np.sum(self.active_kmp)
		# label switchin
		self.p_switch = 0
		self.noise_class = 0
		self.lab =np.array([-1,-1])
		
		self.name   =  name
		self.AMCMC  = AMCMC
	
	
		self.sample_mu_given_mu_eps = False #used in sample mu
	
	@underconstruct
	def load_param(self, params):
		"""
			loads the object from previosuly stored (though not the data)
		
		"""
		pass
		
	@underconstruct
	def write_param(self):
	
		pass
	
	def set_name(self,name):
		"""
			setting the name
		"""
		self.name = name
	
	@underconstruct
	def add_noiseclass(self, Sigma_scale  = 5., mu = None, Sigma = None,a = 1):
		"""
			adds a class that does not update and cant be deactiveted or label switch
			the data need to be loaded first!
			
			Sigma_scale  - (double)  a scaling constant for the the covariance matrix (not used if Sigma supplied)
			mu           - (d x 1 vector) mean value for the noise. If not supplied, the mean of the data is used.
			Sigma        - (d x d matrix) covariance matrix fo the noise
			a 			 - (double) Dirichlet distribution parameter corresponding to noise cluster
		"""
		
		
		pass
	
	@underconstruct
	def update_noiseclass(self):
		"""
			Run this if change noise_sigma or noise_mean
		"""
		
		pass
	
	
	@underconstruct	
	def sample_active_komp(self):
		"""
			tries to turn off or on an active or inactive component
		
		"""

		pass
		
		
	@underconstruct
	def sample_inactivate(self):
		"""
			try to inactivate a component using RJMCMC
		"""
				
				
				
		pass
	
	
	@underconstruct
	def sample_activate(self):
		"""
			try to activate a component using RJMCMC
			
			q_in - probability of choosing to incative a spesific component
			q_ac - probability of choosing to active a spesific component
			log_piy	- log likelihood f(Y; \mu, \Sigma)
			log_d_star - log likelihood dirchelet process
			komp_prior - the value of the prior see __init__
		"""
		
		pass
		
		
		
	def set_data(self, data, measurement):
		"""
			data        - nxd the data
			measurement - nx1 the index indicating which data belongs to which measurement
		
		"""
		if data.shape[0] <= data.shape[1]:
				raise ValueError, 'the number of observations must be larger then the dimenstion'
		self.data = np.empty_like(data)
		self.data[:] = data[:]
		self.measurement = np.empty_like(measurement)
		self.measurement[:] = measurement[:]
		self.unique_meas    = np.unique(self.measurement)
		self.n_measurements = len(self.unique_meas)
		self.n = np.array([np.sum(measurement == i) for i in self.unique_meas])
		#TODO: index_n is depricated now
		#self.index_n = np.array(range(self.n),dtype=np.int)
		
		
		self.data_split = []
		for i in range(self.n_measurements):
			self.data_split.append(self.data[self.measurement == self.unique_meas[i],:])
		
		
		self.d  = self.data.shape[1]
		#just a stupied inital guess!!!
		self.xbar = np.zeros((self.n_measurements,self.K,self.d))
		self.n_x  = np.zeros((self.n_measurements,self.K,1))
		self.mu_mean = np.zeros((self.K,self.d))
		self.mu_cov  = np.zeros((self.K,self.d,self.d))
		if len(self.mu) == 0:
			self.mu_eps  = []	 
			self.sigma = []
			self.sigma_eps = []
			
			for i in self.unique_meas:
				index = self.measurement == i
				mean_data = np.mean(data[index,:],0)
				cov_data  = np.cov(data[index,:].transpose())
				mu_i = []
				
				for j in range(self.K):  # @UnusedVariable
					mu_i.append(npr.multivariate_normal(mean_data,cov_data*0.1))
					
				self.mu_eps.append(mu_i)
				
			
			mean_data = np.mean(data,0)
			cov_data  = np.cov(data.transpose())
				
			for i in range(self.K):
				self.mu.append(npr.multivariate_normal(mean_data,cov_data*0.1))
				self.sigma.append(cov_data)
				self.sigma_eps.append(cov_data*0.01)

			self.mu_eps = np.array(self.mu_eps)
			self.sigma = np.array(self.sigma)
			self.mu = np.array(self.mu)
		mu_prior = {"theta":np.zeros((self.d ,1)),"Sigma":np.diag(np.diag(cov_data))*10**4 }
		sigma_prior = {"nu":self.d, "Q":np.eye(self.d )*10**-6}
		self.alpha_vec = 0.5*np.ones(self.K) 
		if self.prior is None:
			self.prior =[]
			for i in range(self.K):  # @UnusedVariable
				self.prior.append({"mu":cp.deepcopy(mu_prior), "sigma": cp.deepcopy(sigma_prior),"p": 1/2.})
		
		self.p = np.ones((self.n_measurements, self.K), dtype=np.double)/self.K 
		
		self.x = []
		self.prob_X = [] 
		for i in range(self.n_measurements):
			self.prob_X.append(np.zeros((self.n[i], self.K)))
			self.x.append( -np.ones(shape=self.n[i],dtype = np.int, order='C' ))
			
			

		
		self.ln_gamma_d = ln_gamma_d(self.d)
	
	
	@underconstruct
	def sample(self):
		
		pass
	
	

	
	
	@underconstruct
	def set_sigma(self, sigma):
		
		pass
	
	
	@underconstruct
	def set_prior(self, prior):  
		pass
	
	
	@underconstruct
	def set_prior_sigma(self, prior):
		pass
	
	@underconstruct
	def set_prior_sigma_np(self, nu, Q):
		
		pass
	
	@underconstruct
	def set_prior_mu_np(self, mu, Sigma):
		"""
			when mu is np.array 2D
			when Sigma is np.arry 3D
		"""
		pass
			
			
	@underconstruct
	def set_prior_mu(self, prior):
		pass
	
	
	@underconstruct			
	def set_param(self, param, active_only=False):
		
		pass
	
	
	@underconstruct
	def sample_x(self):
		"""
			Draws the label of the observations
		
		"""
		
		pass
	
	@underconstruct
	def sample_labelswitch(self):
		"""
			Tries to switch two random labels
		"""	
		pass
	
	@underconstruct
	def likelihood_prior(self, mu, Sigma, k, R_S_mu = None, log_det_Q = None, R_S = None):
		
		pass


	@underconstruct	
	def updata_mudata(self):
		
		if self.high_memory == True:
			for k in range(self.K):
				if self.active_komp[k] == True:
					self.data_mu[k] = self.data - self.mu[k]	
		
	def set_mu(self, mu):
		"""
			setting value for the mean parameter
			
			this is the mu_0 parameters
 		
 		
 			mu - d x K mean parameters
		"""
		for k in range(self.K):
			self.mu[k][:] = mu[k][:]
			
		#self.updata_mudata()

	
	@underconstruct
	def sample_mu(self):
		
		"""
			Draws the mean parameters either given the repeated measurements mu_eps or integrating them out,
			self.mu[k]
			
			if integrated out we also store mu_mean[k] and mu_cov[k] which is the mean and covariance of mu[k] given sigma_eps and sigma, Y
		"""

		
		#TODO: add parametersization option is it possible?
		if self.sample_mu_given_mu_eps:
			
			for k in range(self.K):
				self.mu[k] = sample_mu(self.mu_eps[:,k], self.sigma_eps[k],  self.prior[k]['mu']['theta'], self.prior[k]['mu']['Sigma'])
	
		else:
			
			
			for k in range(self.K):
				self.mu[k],self.mu_mean[k], self.mu_cov[k] = sample_mu_integrate(self.n_x[:,k], self.xbar[:,k], self.sigma_eps[k], self.sigma[k])
	
	def sample_mu_eps(self):
		"""
			Samples each measurment mu for the sample from
			\pi(\mu_Y{k,l})  = \pi(\bf{Y}_{x==k}|\mu_eps{k,l}, \Sigma_k, \Sigma_eps)   \pi(\mu_eps{k}|\mu_{k},\sigma_{eps_{k}})
		
			this sampling allows for variation for repetead samples from the same indivual where the mean of each cluster center is moving
		"""
		#TODO: add parametersization option is it possible?
		
		if self.AMCMC  == False:
			for l in range(self.n_measurements):
				for k in range(self.K):  
					if self.active_komp[l][k] == True:
						self.mu_eps[l][k] = sample_mu(self.data_split[l][self.x[l] == k ,:], self.sigma[k], self.mu[k], self.sigma_eps[k])
					else:
						self.mu_eps[l][k] = np.NAN * np.ones(self.d)
		else:
			for l in range(self.n_measurements):
				for k in range(self.K):  
					if self.active_komp[l][k] == True:
						self.mu_eps[l][k] = sample_mu_Xbar(self.n_x[l][k], self.xbar[l][k], self.sigma[k], self.mu[k], self.sigma_eps[k])
					else:
						self.mu_eps[l][k] = np.NAN * np.ones(self.d)
			
			

	
	@underconstruct
	def sample_sigma(self):
		
		pass
	
	
	@underconstruct
	def sample_p(self):
		pass
	
	
	@underconstruct
	def calc_lik(self, mu = None, sigma = None, p = None, active_komp = None):
		
		pass
	
	@underconstruct
	def calc_lik_vec(self, mu = None, sigma = None, p = None, active_komp = None):
		
		pass
	


	@underconstruct
	def set_AMCMC(self, n_AMCMC, min_p_AMCMC = 10**-6, p_rate_AMCMC = 0.66):
		"""
			Adapative MCMC parameters
			
			n_AMCMC      - (1x1) expected number of samples in each iteration
			
			min_p_AMCMC  - (1x1) mimum probabilility of sampling a class
			
			p_rate_AMCMC - (1x1) rate on which we update the AMCMC
			
			p_AMCMC      - (nxK) integrated probability of beloning to a class for a point 
			
			p_max_AMCMC  - (nx1) the probability used to determine sampling
			 
		"""
		self.AMCMC          = True
		self.p_AMCMC        =  [np.ones((n,self.K + 1)) for n in self.n] 
		self.p_rate_AMCMC   = p_rate_AMCMC
		self.p_count_AMCMC  = [ np.zeros(n) for n in self.n]
		self.p_max_AMCMC        = [np.ones((n,1)) for n in self.n]
		self.min_p_AMCMC    = min_p_AMCMC
		self.n_AMCMC        = n_AMCMC

	@underconstruct
	def update_AMCMC(self,slice_p):
		"""
			Updates the coeffient used in the AMCMC given a new sample of x
			
			
			slice_p  - (?xl) list of probabilies of the updated samples
		
		"""
		for l in range(self.n_measurements):
			#updating the count of the data that was sampled in the Gibbs sampler
			p_count_AMCMC_vec = self.p_count_AMCMC[l].compress(self.index_AMCMC[l], axis=0)
			p_count_AMCMC_vec += 1
			self.p_count_AMCMC[l][self.index_AMCMC[l]] = p_count_AMCMC_vec 
		
			#coeffient on how much to update the parameter of the AMCMC
			weight = p_count_AMCMC_vec**(-self.p_rate_AMCMC)
			one_minus_weight = 1 - weight
		
			p_AMCMC_vec = self.p_AMCMC[l].compress(self.index_AMCMC[l],axis=0)
			p_AMCMC_vec *= one_minus_weight[:,np.newaxis]
			p_AMCMC_vec[:,:(self.K  + self.noise_class)] += weight[:,np.newaxis] * slice_p[l]
			
			p_ = np.min(1-p_AMCMC_vec,1)
			self.p_AMCMC[l][self.index_AMCMC[l],:] = p_AMCMC_vec
		
			# setting the probabilites to being sample to a minimum probability of being sampled
			p_[p_ < self.min_p_AMCMC] = self.min_p_AMCMC
			self.p_max_AMCMC[l][self.index_AMCMC[l]] = p_[:,np.newaxis]
			# making sure that the expcted number of sampled Gibbs steps is correct
			c = self.n_AMCMC / np.sum(self.p_max_AMCMC[l]) 
			self.p_max_AMCMC[l] *= c 

	@underconstruct
	def compute_ProbX(self, norm =True, mu_eps = None, sigma = None, p =None, active_komp = None):
		"""
			Computes the E[x=i|\mu,\Sigma,p,Y] 
		"""
		if mu_eps is None:
			mu_eps = self.mu_eps
			sigma = self.sigma
			p = self.p
			
		if active_komp is None:
			active_komp = self.active_komp
		
		if self.AMCMC:
			U = [None]*self.n_measurements
			self.index_AMCMC = [None] * self.n_measurements
			n_index          = [None] * self.n_measurements
			for l in range(self.n_measurements):
				U[l] = npr.rand(self.n[l],1)
				self.index_AMCMC[l] = U[l] < self.p_max_AMCMC[l]
			
		else:
			self.index_AMCMC[l] =	[np.ones((n,1)) for n in self.n]
					
		n_index  =[np.sum(index_A) for index_A in self.index_AMCMC]  
		self.index_AMCMC = [ np.reshape(self.index_AMCMC[l],self.index_AMCMC[l].shape[0]) for l in range(self.n_measurements)]
		
		slice_p = [self.prob_X[l].compress(self.index_AMCMC[l] ,axis=0) for l in range(self.n_measurements)]
		
		X_slice  =[self.data_split[l].compress(self.index_AMCMC[l],axis=0) for l in range(self.n_measurements)]
		
		Qs = [None] * self.K
		log_const = [None] * self.K
		for k in range(self.K):
			Qs[k] = np.linalg.inv(sigma[k])
			log_const[k] = np.log(np.linalg.det(Qs[k]))/2. - (self.d/2.)* np.log(2 * np.pi)
		
		for l in range(self.n_measurements):
			xbar_l = np.sum(X_slice[l],0)
			for k in range(self.K):
				if active_komp[l][k] == True:
					
					Qmu     =  np.dot(mu_eps[l,k,:],Qs[k])
					mu_T_mu = np.dot(mu_eps[l,k,:].T, Qmu)
					slice_p[l][:,k] = log_const[k] - 0.5 * mu_T_mu +  np.dot(xbar_l, Qmu)
					slice_p[l][:,k] -= 0.5 * np.sum(X_slice[l] * np.dot(X_slice[l],Qs[k]),1)
						
					slice_p[l][: ,k] += np.log(p[l,k])
		
		
		
			if self.noise_class:
				slice_p[l][:, self.K] = self.l_noise + np.log(p[l,self.K])
			if norm:
				slice_p[l] -= np.reshape(np.max(slice_p[l],1),(n_index[l] ,1))
				slice_p[l][:] = np.exp(slice_p[l])
				slice_p[l] /= np.reshape(np.sum(slice_p[l],1),(n_index[l] ,1))
		
		
			for k in range(self.K):
				if active_komp[l][k] == False:
					slice_p[l][:,k] = 0.
				
			self.prob_X[l][self.index_AMCMC[l],:] = slice_p[l]

		
		
		
		if self.AMCMC:
			self.update_AMCMC(slice_p)
	
	@underconstruct
	def simulate_one_obs(self):
		"""
			if there exists noise class it __will__ be used
		"""
		pass


	@underconstruct
	def deactivate_component(self,k_off):
		
		pass

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
