import numpy as np
import numpy.random as npr
from numpy.linalg import cholesky, solve
from scipy.stats import chi2

import pickle

def wishart_var(nu, phi):
	var = np.zeros_like(phi)
	for i in range(phi.shape[0]):
		for ii in range(phi.shape[0]):
			var[i, ii] = nu * ( phi[i, ii]**2 + phi[i, i] * phi[ii, ii])	
	
	return var

def inv(x):
	return solve(x, np.eye(x.shape[0]))

def invwishartrand(nu, phi):
	return inv(wishartrand(nu,phi))

def invwishartrand_prec(nu, inv_phi):
	return inv(wishartrand_prec(nu, inv_phi))


def wishartrand_prec(nu, inv_phi):
	dim = inv_phi.shape[0]
	chol = cholesky(inv_phi)
	#nu = nu+dim - 1
	#nu = nu + 1 - np.arange(1,dim+1)
	#foo = npr.randn(dim,dim )
	
	foo = np.zeros((dim, dim))
	for i in range(dim):
		for j in range(i):
			foo[i,j] = npr.normal(0,1)
		foo[i,i] = np.sqrt(chi2.rvs(nu-(i+1)+1))
	return np.dot(chol, np.dot(foo, np.dot(foo.T, chol.T)))


def wishartrand(nu, phi):
	"""
		f(X; nu, \Phi) propto |X|^(nu -p -1)/2 exp( tr( \Phi^-1 X)/2 )
	"""
	return wishartrand_prec(nu, inv(phi))



class Wishart(object):
	
	'''
		Class for sampling from a Wishart where the distribution of 
		f(Q;  \Sigma, \Sigma_s, \nu_s,\nu)  \propto W(\Sigma; Q, \nu) IW(Q; \nu_s, Q_s)
	'''

	
	def __init__(self, param = None, prior =None):
		
		self.n   = 0
		self.d   = 0
		self.nu  = 0
		self.Q_s = 0
		self.Q   = 0
		if param != None:
			self.set_parameter(param)
		
		if prior != None:
			self.set_prior(prior)


	def pickle(self, filename): # @DuplicatedSignature
		"""
			store object in file
		"""
		f = open(filename, 'wb')
		pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
		f.close()

	@staticmethod
	def unpickle(filename): # @DuplicatedSignature
		"""
			load object from file
			use:
			
			object = Wishart.unpickle(filename)
		"""
		with open(filename, 'rb') as f:
			return pickle.load(f)	
	
	def set_parameter(self, param):
		
		self.nu = param['nu']
	
	def set_prior(self, prior):
		
		self.nu_s = prior['nus']
		self.Q_s = np.empty_like(prior['Qs'])
		self.Q_s[:]  =prior['Qs'][:]
		self.d    = self.Q_s.shape[0]

	def set_data(self, Sigmas = None, Qs =None):
		"""
			Sigma is a list containg sigmas
		"""
		
		if Qs == None:
			self.n = len(Sigmas)
			self.Q = np.zeros((self.d, self.d))
			for Sigma in Sigmas:
				self.Q += np.linalg.inv(Sigma)
		else:
			self.n = len(Qs)
			self.Q = np.zeros((self.d, self.d))
			for Q in Qs:
				self.Q += Q
	
	def sample(self):
		
		X = wishartrand_prec(self.nu_s + self.n * self.nu ,np.linalg.inv(self.Q + self.Q_s)) 
		return X
	

	
class invWishart(object):
	'''
		Class for sampling from a inverse Wishart where the distribution of 
		f(\Sigma| Y, \theta, Q, \nu)  \propto N(Y; theta, \Sigma) IW(\Sigma; Q, nu)
	'''
	
	def __init__(self, param = None ,prior = None):
		'''
			prior:
			prior['Q']  = np.array(dim=2)
			prior['nu'] = int
			
			param:
			param['theta'] = np.array(dim = 1)
		'''
		
		self.n = 0 
		if prior != None:
			self.set_prior(prior)
		
		if param != None:
			self.set_parameter(param)
	
	def pickle(self, filename):
		"""
			store object in file
		"""
		f = open(filename, 'wb')
		pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
		f.close()

	@staticmethod
	def unpickle(filename):
		"""
			load object from file
			use:
			
			object = normal_p_wishart.unpickle(filename)
		"""
		with open(filename, 'rb') as f:
			return pickle.load(f)
			
	def set_prior(self,prior):
		"""
		
		"""
		self.nu = prior['nu']
		self.Q = np.empty_like(prior['Q'])
		self.Q[:] = prior['Q'][:]
		self.Q_sample = np.empty_like(self.Q)
		
	def set_parameter(self, parameter):
		"""
			parameter should be a dict with 'theta'
		
		"""
		self.theta = np.empty_like(parameter['theta'])
		self.theta[:] = parameter['theta'][:]
		self.theta_outer = np.outer(self.theta, self.theta)
		
	def set_data(self, Y = None, sumY = None):
		"""
			Y - (nxd) numpy vector
		"""
		if sumY == None:
			self.sumY = np.sum(Y,0)
		else:
			self.sumY = sumY 
		self.Y_outer = np.dot(Y.transpose(), Y)
		self.n = Y.shape[0]
		
	
	def sample(self):
		"""
			return X
		"""
		self.Q_sample[:] = self.Q[:] 
		
		if self.n != 0:
			self.Q_sample += self.n * self.theta_outer
			Temp = np.outer(self.sumY, self.theta)
			self.Q_sample -= Temp + Temp.T
			self.Q_sample += self.Y_outer
		nu_sample = self.n + self.nu 
		return invwishartrand_prec(nu_sample, np.linalg.inv(self.Q_sample))	

	
if __name__ == '__main__':
	pass
