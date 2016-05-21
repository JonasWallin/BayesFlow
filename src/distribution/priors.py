'''
Created on Jul 2, 2014

@author: jonaswallin
'''
import numpy as np
import pickle

from .distribution_cython import invWishart, multivariatenormal, Wishart  # @UnresolvedImport
#from BayesFlow.PurePython.distribution.priors import nu_class 
from ..PurePython.distribution.priors import nu_class 


# make it so that that invWishart returns sigma,Q, logdet(sigma)
class normal_p_wishart(object):
	"""
		prior class for \mu with parameter:
		\mu		\sim N(\theta  , \Sigma_\mu)
		\theta	 \sim N(\theta_0,  \Sigma_\theta) 
		\Sigma_\mu \sim IW(Q_\mu, \nu_\mu)
	
		methods for sampling \theta, \Sigma_mu given \mu, theta_0, Q_\mu, \nu_\mu
	"""
	
	def __init__(self, prior = None, param = None):
		"""
			prior dict
			prior['theta'] -> dict ['mu']	- np.array(dim=1) \theta_0
								   ['Sigma'] - np.array(dim=2) \Sigma_\theta
			prior['Sigma'] -> dict ['nu']	- int 
								   ['Q']	 - np.array(dim=2)
			param dict
			param['theta'] -> dict ['Sigma'] 
			param['Sigma'] -> dict ['theta']
		"""
		self.theta_class = multivariatenormal()
		self.Sigma_class = invWishart() 
		self.param = {}
		if not prior is None:
			self.set_prior(prior)
		
		if not param is None:
			self.set_parameter(param)
			
			
	def set_prior(self, prior):
		"""
			see init
		
		"""
		self.theta_class.set_prior(prior)
		self.Sigma_class.set_prior(prior)
	
	def set_prior_param0(self, d):
		"""
			setting deafult "non informative"
			priors + starting values
		
		"""
		self.theta_class.set_prior0(d)
		self.Sigma_class.set_prior0(d)
		
		param = {}
		param['theta'] = np.zeros(d)
		param['Sigma'] = np.eye(d)
		self.set_parameter(param)
	
	def set_parameter(self, param):
		"""
			see init
		
		"""	   
		self.theta_class.set_parameter(param)
		self.Sigma_class.set_parameter(param)
		self.param['theta'] = np.zeros_like(param['theta'])
		self.param['Sigma'] = np.zeros_like(param['Sigma'])
		self.param['theta'][:] = param['theta'][:]
		self.param['Sigma'][:] = param['Sigma'][:]
	
	def set_data(self, data):
		"""
			mu obeservations
			data - np.array[dim = 2]
		"""
		self.theta_class.set_data(data)
		self.Sigma_class.set_data(data, self.theta_class.sumY)
	
		
	
	def sample(self):
		"""
			Sampling \theta, \Sigma
			
			returns:
				dict with ['theta']
						  ['Sigma']
		"""
		self.param['theta'][:] = self.theta_class.sample()[:]
		self.Sigma_class.set_parameter(self.param)
		self.param['Sigma'][:] = self.Sigma_class.sample()[:]
		self.theta_class.set_parameter(self.param)


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
			
			object = normal_p_wishart.unpickle(filename)
		"""
		with file(filename, 'rb') as f:
			return pickle.load(f)

	
class Wishart_p_nu(object):
	"""
		Wishart plus nu prior
		prior class for \Sigma with parameter:
		\Sigma		\sim IW(Q  , \nu)
		Q   	    \sim W(Q_s,  \nu_s) 
		\nu         \sim 1
	
		methods for sampling \nu, Q given Q_s, {\Sigma} \nu_s 
	"""
	
	def __init__(self, prior = None, param = None, AMCMC=False):
		"""
			prior dict
			prior['nu'] ->  None
			prior['Q']  -> dict ['nus']	- int 
								   ['Qs']	 - np.array(dim=2)
			param dict
			param['nu']    -> dict ['Q'] 
			param['Q']     -> dict ['nu']
		"""
		self.nu_class = nu_class(AMCMC=AMCMC)
		self.Q_class  = Wishart()
		self.param = {}
		if not prior is None:
			self.set_prior(prior)
		
		if not param is None:
			self.set_parameter(param)
		
	def set_MH_param(self, sigma = 5, iterations = 5):	
		"""
			setting the parametet for the MH algorithm for the nu class
			
			sigma     -  the sigma in the MH algorihm on the Natural line
			iteration -  number of time to sample using the MH algortihm  
		"""
		self.nu_class.set_MH_param(sigma ,  iterations )
		
	def set_prior(self, prior):
		
		self.nu_class.set_prior(prior)
		self.Q_class.set_prior(prior)
		
	def set_prior_param0(self, d):
		"""
			setting deafult "non informative"
			priors + starting values
		
		"""
		self.Q_class.set_prior0(d)	
		param = {}
		param['nu'] = d
		param['Q'] = np.eye(d)
		self.set_parameter(param)
		
	def set_val(self, param):
		"""
			setting the current iteration to param
			para - dict keys: ['nu'] ['Q'] 
		"""
		self.param = param
		self.Q_class.set_parameter(self.param)
		self.nu_class.set_parameter(self.param)
		self.nu_class.set_val(param['nu'])
		
	def set_parameter(self, param):
		self.nu_class.set_parameter(param)
		self.Q_class.set_parameter(param)
		self.nu_class.set_d(self.Q_class.d)
		self.param['nu'] = param['nu']
		self.param['Q']  = np.zeros_like(param['Q'])
		self.param['Q'][:] = param['Q']

	def set_data(self, Sigmas = None, Qs = None, detQ = None):
		"""
			Sigma obeservations
			data - list of np.array[dim = 2]
		"""
		
		if Qs is None:
			self.Q_class.set_data(Sigmas = Sigmas)
			self.nu_class.set_data(data = Sigmas)
	
	def sample(self):
		"""
			Sampling \nu,Q
			
			returns:
				dict with ['nu']
						  ['Q']
		"""
		self.param['nu'] = self.nu_class.sample()
		self.Q_class.set_parameter(self.param)
		self.param['Q'] = self.Q_class.sample()
		self.nu_class.set_parameter(self.param)
		
		
	def pickle(self, filename):
		"""
			store writte
		"""
		f = file(filename, 'wb')
		pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
		f.close()

	@staticmethod
	def unpickle(filename):
		"""
			load class
			
			
			object = Wishart_p_nu.unpickle(filename)
		"""
		with file(filename, 'rb') as f:
			return pickle.load(f)
	
