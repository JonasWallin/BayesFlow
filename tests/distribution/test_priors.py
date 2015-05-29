'''
Created on Jul 3, 2014

@author: jonaswallin
'''
import unittest
import os
import numpy as np
import numpy.random as npr
from BayesFlow.distribution import normal_p_wishart, Wishart_p_nu
import scipy.linalg as spl
from BayesFlow.PurePython.distribution.wishart import  invwishartrand

class Test_wishart_p_nu(unittest.TestCase):

	sim  = 100
	
	def testStartup(self):
		
		self.startup1()
	
	def startup1(self):
		prior = {'nus':2,'Qs':np.array([[0.]])}
		self.dist = Wishart_p_nu(prior = prior)
		self.param = {'nu':8,'Q':np.array([[0.9]])}
		self.dist.set_parameter(self.param)
	
	def test_load_save(self):
		
		self.startup1()
		fileName = "test.pkl"
		self.dist.pickle(fileName)
		dist2 = Wishart_p_nu.unpickle(fileName)
		os.remove(fileName)
		
	
	def startup2(self):
		self.d = 4
		prior = {'nus':2,'Qs':np.zeros((self.d, self.d))}
		self.dist = Wishart_p_nu(prior = prior)
		self.param = {'nu':8,'Q':np.eye(self.d)}
		self.dist.set_parameter(self.param)
		
	def test_simulate(self):
		self.startup2()

		row = np.zeros(self.d)
		row[0] = 5.
		row[1] = -1
		row[2] = 2.	 
		Sigma = spl.toeplitz(row)
		Y = []
		for i in range(self.sim):  # @UnusedVariable
			Y.append( invwishartrand(10, Sigma))	

		self.dist.set_data(Y)
		self.dist.sample()
		
class Test_normal_p_wishart(unittest.TestCase):
	n   = 20000
	sim = 2000

	def setUp(self):
		pass


	def tearDown(self):
		pass


	def testStartup(self):
		
		self.startup1()
		
	def startup1(self):
		prior = {'mu':np.array([2.3]),'Sigma':np.array([[10.1**6]]),'nu':2,'Q':np.array([[0.]])}
		self.dist = normal_p_wishart(prior = prior)
		self.param = {'theta':np.array([2.]),'Sigma':np.array([[0.9]])}
		self.dist.set_parameter(self.param)

	def startup2(self):
		d = 4
		row = np.zeros(d)
		row[0] = 5.
		row[1] = -1
		row[2] = 2.	 
		Sigma = spl.toeplitz(row)
		prior = {'mu':npr.randn(d),'Sigma':10.1**6 * np.eye(d),'nu':2,'Q':np.eye(d)}
		self.dist = normal_p_wishart(prior = prior)
		self.param = {'theta':6 * npr.randn(d) ,'Sigma':Sigma}
		self.dist.set_parameter(self.param)



	def Simulate(self):
		
		self.Y = np.dot(npr.randn(self.n, self.dist.theta_class.Sigma.shape[0]), np.linalg.cholesky(self.dist.theta_class.Sigma).T) 
		self.Y += self.dist.Sigma_class.theta
		
	def testSimulate(self):
		
		self.startup1()
		self.Simulate()
		
	def testMean(self):
		
		startups = [self.startup1, self.startup2 ]
		for startup in startups:
			startup()
			self.Simulate()
			self.dist.set_data(self.Y)		
			self.dist.sample()
			mu_mean    = np.zeros_like(self.dist.param['theta'])
			mu_mean[:] = self.dist.param['theta'][:]
			Sigma_mean = np.zeros_like(self.dist.param['Sigma'])
			Sigma_mean[:] = self.dist.param['Sigma'][:]
			
			for i in range(self.sim-1):  # @UnusedVariable
				self.dist.sample()
				mu_mean    += self.dist.param['theta']
				Sigma_mean += self.dist.param['Sigma']
			mu_mean    /= self.sim
			Sigma_mean /= self.sim
			np.testing.assert_array_almost_equal(mu_mean, self.param['theta'],decimal = 1)
			np.testing.assert_array_almost_equal(Sigma_mean, self.param['Sigma'],decimal = 1)
		

if __name__ == "__main__":
	#import sys;sys.argv = ['', 'Test.testStartuo']
	unittest.main()