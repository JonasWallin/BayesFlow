'''
Created on Aug 15, 2014

@author: jonaswallin
'''
import unittest
import numpy as np
import numpy.random as npr
import scipy.linalg as spl
from bayesianmixture.PurePython.GMM import mixture as mixP
import bayesianmixture.GMM as GMM
import copy as cp
import scipy.spatial as ss

class Test(unittest.TestCase):
	"""
		Simulate data see if one gets sutiable result?
	
	"""
	n  =10**3
	sim = 10**2
	
	def compare_class(self):
		ss_mat =  ss.distance.cdist(  np.array(self.Thetas), np.array(self.mix.mu), "euclidean")

		col_index = []
		for k in range(self.nClass):
			col_index.append( np.argmin(ss_mat[k,:]))
			ss_mat[:,col_index[k]] = np.inf	
			


		for k in range(self.nClass):
			np.testing.assert_array_almost_equal(self.mix.mu[col_index[k]], self.Thetas[k], 1, "cant locate true mean")  
			np.testing.assert_array_almost_equal(self.mix.p[col_index[k]], self.P[k], 1, "cant locate true prob")  
			
		#for p in self.mix.p:
		#	print p
	
	def setUp(self):
		npr.seed(123456)
		self.nClass = 4
		self.dim    = 3
		self.P = [0.4, 0.3, 0.2 ,0.1]
		self.Thetas = [np.array([0.,0, 0]), np.array([0, -2, 1]), np.array([1., 2, 0]), np.array([-2,2,3.1])]
		self.Sigmas = [0.1*np.eye(3), 0.1*spl.toeplitz([2.,0.5,0]),0.1* spl.toeplitz([2.,-0.5,1]),
			  0.1*spl.toeplitz([1.,.3,.3]) ] 
		
		mix_obj = mixP(K = self.nClass)
		mix_obj.mu    = cp.deepcopy(self.Thetas)
		mix_obj.sigma = cp.deepcopy(self.Sigmas)
		

		mix_obj.p = cp.deepcopy(self.P)
		mix_obj.d = self.dim
		self.Y = mix_obj.simulate_data(self.n)
		
		
	def test_mixP(self):
		self.mix =  mixP(K = self.nClass)
		self.mix.set_data(self.Y)
		for i in range(self.sim):  # @UnusedVariable
			self.mix.sample()
		np.set_printoptions(precision=2)
		
		
		self.compare_class()
	
	def test_mixP_noise(self):
		"""
			Does not work since the noise class makes it impossoble to detect the out lier cluster!!!!!
		"""
		self.mix =  mixP(K = self.nClass)
		self.mix.set_data(self.Y)
		self.mix.add_noiseclass()
		for i in range(self.sim):  # @UnusedVariable
			self.mix.sample()
		
	def test_mix_noise(self):
		"""
			Does not work since the noise class makes it impossoble to detect the out lier cluster!!!!!
		"""
		self.mix =  GMM.mixture(K = self.nClass)
		self.mix.set_data(self.Y)
		self.mix.add_noiseclass()
		for i in range(self.sim):  # @UnusedVariable
			self.mix.sample()

	def test_mix_noise3(self):
		"""
			Testing with non pure python noise class
		"""
		self.mix =  GMM.mixture(K = self.nClass)
		self.mix.set_data(self.Y)
		for i in range(self.sim):  # @UnusedVariable
			self.mix.sample()
		self.mix.add_noiseclass()
		for i in range(self.sim):  # @UnusedVariable
			self.mix.sample()
		self.compare_class()

	def test_mix_load_read(self):
		"""
			Testing with non pure python noise class
		"""
		self.mix =  GMM.mixture(K = self.nClass)
		self.mix.set_data(self.Y)
		for i in range(self.sim):  # @UnusedVariable
			self.mix.sample()
		self.mix.add_noiseclass()
		for i in range(self.sim):  # @UnusedVariable
			self.mix.sample()
		params = self.mix.write_param()
		self.mix2 = GMM.mixture(K = self.nClass)
		self.mix2.set_data(self.Y)
		self.mix2.load_param(params)
		
		for i in range(self.sim):  # @UnusedVariable
			self.mix2.sample()
	
	def test_likelihood_ratio(self):
		"""
			updating likelihood ratio test to cython:
			
			Profile result (from other file):
			 9888    0.665    0.000    2.078    0.000 GMM.py:64(likelihood_prior) (Pure Python)
			 9888    0.638    0.000    1.830    0.000 GMM.py:64(likelihood_prior) (moved choleksy factor to cython)
			 9888    0.758    0.000    1.340    0.000 GMM.py:64(likelihood_prior) (moved solve to cython)
			 9888    0.304    0.000    0.878    0.000 GMM.py:64(likelihood_prior) (finalised)
		
		"""
		self.mix =  GMM.mixture(K = self.nClass)
		self.mix2 =  mixP(K = self.nClass)
		self.mix.set_data(self.Y)
		self.mix2.set_data(self.Y)
		for i in range(self.sim):  # @UnusedVariable
			self.mix.sample()
			
		labels = npr.choice(self.mix2.K,2,replace=False)
		np.testing.assert_almost_equal(self.mix2.likelihood_prior(self.mix2.mu[labels[0]],self.mix2.sigma[labels[0]], labels[0])[0], 
									self.mix.likelihood_prior(self.mix2.mu[labels[0]],self.mix2.sigma[labels[0]], labels[0])[0], 8)
		np.testing.assert_almost_equal(self.mix2.likelihood_prior(self.mix2.mu[labels[0]],self.mix2.sigma[labels[0]], labels[0])[1][0], 
									self.mix.likelihood_prior(self.mix2.mu[labels[0]],self.mix2.sigma[labels[0]], labels[0])[1], 8)	
		
		np.testing.assert_almost_equal(self.mix2.likelihood_prior(self.mix2.mu[labels[0]],self.mix2.sigma[labels[0]], labels[0])[2], 
									self.mix.likelihood_prior(self.mix2.mu[labels[0]],self.mix2.sigma[labels[0]], labels[0])[2], 8)
		np.testing.assert_almost_equal(np.triu(self.mix2.likelihood_prior(self.mix2.mu[labels[0]],self.mix2.sigma[labels[0]], labels[0])[3][0]), 
									self.mix.likelihood_prior(self.mix2.mu[labels[0]],self.mix2.sigma[labels[0]], labels[0])[3], 8)
if __name__ == "__main__":
	#import sys;sys.argv = ['', 'Test.test_run']
	unittest.main()