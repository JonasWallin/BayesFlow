'''
Created on Jun 30, 2014

@author: jonaswallin
'''
import copy as cp
import numpy as np

import cPickle as pickle


class multivariatenormal(object):
	'''
		Class for sampling from a Multivariate normal distribution on the form
		f(X| Y, \Sigma, \mu_p, \Sigma_p) \propto N(Y; X, \Sigma) N(X; \mu_p, \Sigma_p)
	'''

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
		
	def __init__(self, param = None ,prior = None):
		'''
			prior:
			prior['mu'] = np.array(dim=1)
			prior['Sigma'] = np.array(dim=2)
		'''
		self.prior = None
		self.param = None
		self.Y = None
		if prior != None:
			self.set_prior(prior)
		
		if param != None:
			self.set_parameter(param)
		
	def set_prior(self,prior):
		"""
		
		"""
		self.mu_p = prior['mu']
		self.Sigma_p = prior['Sigma']
		self.Q_p = np.linalg.inv(self.Sigma_p)
		self.Q_pmu_p = np.dot(self.Q_p,self.mu_p)
		self.d = self.Q_pmu_p.shape[0]	
		
	def set_parameter(self, parameter):
		"""
			parameter should be a dictonray with 'Sigma'
		
		"""
		self.Sigma = parameter['Sigma']
		self.Q = np.linalg.inv(self.Sigma)
		self.d = self.Q.shape[0]
		
	def set_data(self, Y):
		"""
			Y - (nxd) numpy vector
		"""
		self.Y = cp.deepcopy(Y)
		self.sumY = np.sum(Y,0)
		self.n = Y.shape[0]
	def sample(self):
		"""
			return X
		"""
		if self.Y != None:
			mu_sample = self.Q_pmu_p + np.dot(self.Sigma, self.sumY)
			Sigma_sample =  self.Q_p + self.n * self.Q
		else:
			mu_sample = self.Q_pmu_p
			Sigma_sample = self.Q_p
			
		R = np.linalg.cholesky(Sigma_sample)
		X = np.linalg.solve(R, mu_sample)
		X = np.linalg.solve(R.T, X + np.random.randn(self.d))
		return X
	
	
		
		