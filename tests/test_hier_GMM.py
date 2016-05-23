'''
Created on Jul 10, 2014

@author: jonaswallin
'''
import unittest
import BayesFlow as bm
import numpy.random as npr
import numpy as np
import scipy.linalg as spl
from BayesFlow.PurePython.distribution.wishart import  invwishartrand
from BayesFlow.PurePython.GMM import mixture


class Test(unittest.TestCase):


	def setUp(self):
		
		self.n_y = 20
		self.n_obs = 1000
		self.d = 4
		self.K = 4
		self.sim  = 1000
		self.hGMM = bm.hierarical_mixture(K = self.K)
	
	
	def setdata(self):
		
		self.mu = [np.repeat(np.double(k),self.d) for k in range(self.K)]
		self.r1 =  np.zeros(self.d)
		self.r1[0] = 3.
		self.r1[1] = -1
		self.r1[2] = 1
		self.r1[2] = .5
		self.r2 =  np.zeros(self.d)
		self.r2[0] = 2.
		self.r2[1] = -1.5
		self.r2[2] = 0
		self.r2[2] = 0.		
		self.sigma =[np.eye(self.d), spl.toeplitz(self.r1), spl.toeplitz(self.r1), 0.5*np.eye(self.d)]
		self.p = np.array([1.,1.,1.,0.1])
		self.p2 =  np.array([1.,1.,1.])
		self.p /= np.sum(self.p)
		self.p2 /= np.sum(self.p2)
		
		
			
		self.data_sigma = [ [invwishartrand(5, self.sigma[k]) for k in range(self.K)] for n in range(self.n_y)]  # @UnusedVariable
		self.data_mu    = [ [self.mu[k]+ np.random.randn(self.d)*0.2 for k in range(self.K)] for n in range(self.n_y)]  # @UnusedVariable
		
		self.Y = []
		for i in range(self.n_y):
			
			if i < 17:
				mix = mixture(K = self.K)
				mix.mu = self.data_mu[i]
				mix.sigma = self.data_sigma[i]
				mix.p = self.p
			else:
				mix = mixture(K = self.K-1)
				mix.mu = self.data_mu[i][:-1]
				mix.sigma = self.data_sigma[i][:-1]
				mix.p = self.p2
			mix.d = self.d
			self.Y.append(mix.simulate_data(self.n_obs))
		self.hGMM.set_data(self.Y)
		self.hGMM.set_prior_param0()
		self.hGMM.set_p_labelswitch(0.5)
		self.hGMM.set_p_activation([0,0])
		
	def test_setdata(self):
		self.setdata()
		
	def test_sample(self):

		self.setdata()
		self.hGMM.sample()

	@unittest.skip("Fails systematically")
	def test_mean_converge(self):

		self.setdata()
		for i in range(self.sim):   # @UnusedVariable
			self.hGMM.sample()
			if i > self.sim/2:
				self.hGMM.set_p_activation([0.1,0.1])

		

if __name__ == "__main__":
	#import sys;sys.argv = ['', 'Test.testName']
	unittest.main()