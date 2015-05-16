'''
like test_GMM but slower since higher level of precision 
thus larger number of simulations


Created on Feb 20, 2015

@author: jonaswallin
'''
from __future__ import division
import unittest
import numpy as np
import numpy.random as npr
import scipy.linalg as spl
from BayesFlow.PurePython.GMM import mixture as mixP
import BayesFlow.GMM as GMM
import copy as cp
import scipy.spatial as ss

class Test(unittest.TestCase):


	n  =10**4
	sim = 4*10**3
	
	def compare_class(self,title = ""):
		"""
			Testing if we can detect the true mean and probabilites,
			since test is random we have low almost equal
		"""
		ss_mat =  ss.distance.cdist(  np.array(self.Thetas), np.array(self.mix.mu), "euclidean")

		col_index = []
		for k in range(self.nClass):
			col_index.append( np.argmin(ss_mat[k,:]))
			ss_mat[:,col_index[k]] = np.inf	
			


		for k in range(self.nClass):
			np.testing.assert_array_almost_equal( self.sigma_sample[col_index[k]]/(self.sim-1) , self.Sigmas[k], 2, title + "cant locate true Sigma")  
			np.testing.assert_array_almost_equal( self.mu_sample[col_index[k]]/self.sim, self.Thetas[k], 2, title +  "cant locate true mu")  
			np.testing.assert_array_almost_equal( self.p_sample[col_index[k]]/self.sim,self.P[k], 2, title + "cant locate true prob")  
			
		#for p in self.mix.p:
		#	print p
	
	def setUp(self):
		npr.seed(123456)
		self.nClass = 3
		self.dim    = 3
		self.P = [0.4, 0.3, 0.3]
		self.Thetas = [np.array([0.,0, 0]), np.array([0., -2, 1]), np.array([1., 2, 0])]
		self.Sigmas = [ 0.1*spl.toeplitz([2.,0.5,0]),0.1* spl.toeplitz([2.,-0.5,1]),
			  0.1*spl.toeplitz([1.,.3,.3]) ] 
		
		mix_obj = mixP(K = self.nClass)
		mix_obj.mu    = cp.deepcopy(self.Thetas)
		mix_obj.sigma = cp.deepcopy(self.Sigmas)
		
		mix_obj.p = cp.deepcopy(self.P)
		mix_obj.d = self.dim
		self.Y = mix_obj.simulate_data(self.n)

	def general_mix(self, mix, **kwargs):
		"""
			simple mix data test
		"""
		npr.seed(122351)
		self.mix = mix(K = self.nClass,**kwargs)
		self.mix.set_data(self.Y)
		self.mu_sample = list()
		self.sigma_sample = list()
		self.p_sample = list()
		for k in range(self.nClass):
			self.mu_sample.append(np.zeros_like(self.Thetas[k])) 
			self.sigma_sample.append(np.zeros_like(self.Sigmas[k])) 
			self.p_sample.append(np.zeros_like(self.P[k])) 
			
			
		for i in range(self.sim):  # @UnusedVariable
			self.mix.sample()
			for k in range(self.nClass):
				self.mu_sample[k] += self.mix.mu[k]
				self.sigma_sample[k] += self.mix.sigma[k]
				self.p_sample[k] += self.mix.p[k]
		np.set_printoptions(precision=2)
			
		self.compare_class("MCMC:")

	def general_mix_AMCMC(self, mix, **kwargs):
		"""
			simple mix data test using AMCMC
		"""
		self.mix = mix(K = self.nClass,**kwargs)
		self.mix.set_data(self.Y)
		for i in range(100):#np.int(np.ceil(0.1*self.sim))):  # @UnusedVariable
			self.mix.sample()
		self.mu_sample = list()
		self.sigma_sample = list()
		self.p_sample = list()
		
		for k in range(self.nClass):
			self.mu_sample.append(np.zeros_like(self.Thetas[k])) 
			self.sigma_sample.append(np.zeros_like(self.Sigmas[k])) 
			self.p_sample.append(np.zeros_like(self.P[k])) 
			
		self.mix.set_AMCMC(1200)
		sim_m = 2.
		for i in range(np.int(np.ceil(sim_m*self.sim))):  # @UnusedVariable
			self.mix.sample()
			for k in range(self.nClass):
				
				self.mu_sample[k] += self.mix.mu[k]/sim_m
				self.sigma_sample[k] += self.mix.sigma[k]/sim_m
				self.p_sample[k] += self.mix.p[k]	/sim_m	
		np.set_printoptions(precision=2)
			
		self.compare_class("AMCMC:")

	def test_mix_AMCMC(self):
		"""
			testing convergences usiong AMCMC
		"""
		self.general_mix_AMCMC(mixP, high_memory=False)
		
	def test_mix(self):
		"""
			simple test to see if data converges GMM
		"""
		#self.general_mix(mixP)
		#print "1"
		#self.general_mix(mixP, high_memory = False)
		#print "2"
		#self.general_mix(GMM.mixture, high_memory = False)
		#print "3"
		#self.general_mix(GMM.mixture)	
		#print "4"


if __name__ == "__main__":
	#import sys;sys.argv = ['', 'Test.testName']
	unittest.main()