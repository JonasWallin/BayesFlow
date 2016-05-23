'''
Created on Jun 30, 2014

@author: jonaswallin
'''
import unittest
import numpy as np
import scipy.linalg as spl
try:
	from bayesianmixture.purepython.distribution import multivariatenormal as mv_python
	from bayesianmixture.distribution import multivariatenormal as mv  # @unresolvedimport
except ImportError:
	raise unittest.SkipTest("bayesianmixture not available")
import os

class multivariatenormal_base(object):
	sim = 5000
	
	def __init__(self):
		
		self.mv = None
	
	def test_startup(self):
		
		prior = {'mu':np.array([0]),'Sigma':np.array([[1.]])}
		self.dist = self.mv(prior = prior)
		self.assertEqual(self.dist.Q_pmu_p[0],0.)
		self.dist.set_parameter({'Sigma':np.array([[2.]])})
		self.assertEqual(self.dist.d,1)
		self.dist.set_data(np.random.rand(100, 1))
		self.assertEqual(self.dist.n,100)
		self.assertEqual(self.dist.sumY.shape[0],1)
		
	def startup_pureprior1(self):
		
		prior = {'mu':np.array([0]),'Sigma':np.array([[1.]])}
		self.dist = self.mv(prior = prior)
	
	def startup_pureprior2(self):
		
		prior = {'mu':np.array([0,2,3]),'Sigma':spl.toeplitz([4,-1,0])}
		self.dist = self.mv(prior = prior)
		
	def test_prior_dist(self):
		
		prior_startup = [self.startup_pureprior1, self.startup_pureprior2]
		for prior in prior_startup:
			prior()
			mu_est = np.zeros((self.sim,self.dist.mu_p.shape[0]))
			for i in range(self.sim):
				mu_est[i,:] = self.dist.sample()
				
			np.testing.assert_array_almost_equal(self.dist.mu_p,np.mean(mu_est,0),decimal=1)
			np.testing.assert_array_almost_equal(np.cov(mu_est.T), self.dist.Sigma_p, decimal=0)
			
			
	def startup1(self):
		prior = {'mu':np.array([0.1]),'Sigma':np.array([[10**10]])}
		param = {'Sigma': np.array([[2.]])}
		self.Y = np.linalg.cholesky(param['Sigma']) * np.random.randn(self.sim,1)
		self.dist = self.mv(prior = prior, param = param)
	
	def startup2(self):
		
		self.prior = {'mu':np.array([-10,-10]),'Sigma':10**4*np.eye(2)}
		self.param = {'Sigma': np.array([[2.,-1],[-1, 2.]])}
		self.Y = np.empty((self.sim,2))
		L = np.linalg.cholesky(self.param['Sigma'])
		for i in range(self.sim):  # @UnusedVariable
			self.Y[i,:] = np.dot(L,np.random.randn(2,1)).reshape((2,))
		self.dist = self.mv(prior = self.prior, param = self.param)
		
		
		
	def test_pickle(self):
		"""
			testing if pickling works
		
		"""
		self.startup1()
		fileName = "test.pkl"
		self.dist.pickle(fileName)
		self.dist = self.mv.unpickle(fileName)
		os.remove(fileName)
		
		self.dist.set_data(self.Y)
		mu_est = np.zeros((self.sim,self.dist.mu_p.shape[0]))
		for i in range(2):
			mu_est[i,:] = self.dist.sample()
		self.dist.pickle(fileName)
		self.dist = self.mv.unpickle(fileName)
		os.remove(fileName)
		for i in range(2):
			mu_est[i,:] = self.dist.sample()
		#pass

	def test_sample(self):
		
		startup = [self.startup1, self.startup2]
		for start in startup:
			start()
			self.dist.set_data(self.Y)
			mu_est = np.zeros((self.sim,self.dist.mu_p.shape[0]))
			
			for i in range(self.sim):
				mu_est[i,:] = self.dist.sample()
				
				
			np.testing.assert_array_almost_equal(np.mean(self.Y,0),np.mean(mu_est,0),decimal=1) 
			np.testing.assert_array_almost_equal(np.cov(self.Y.T)/self.Y.shape[0],np.cov(mu_est.T),decimal=1) 

	def test_sample2(self):
		"""
			strong prior should make the sample of X less then the mean of Y
		
		"""
		n  =2000
		prior = {'mu':np.array([-10,-10]),'Sigma':10**-2*np.eye(2)}
		param = {'Sigma': np.array([[2.,1],[1, 2.]])}
		self.Y = np.empty((n,2))
		L = np.linalg.cholesky(param['Sigma'])
		for i in range(n):  # @UnusedVariable
			self.Y[i,:] = np.dot(L,np.random.randn(2,1)).reshape((2,))
		self.dist = self.mv(prior = prior, param = param)  
		self.dist.set_data(self.Y)
		mu_est = np.zeros((self.sim, self.dist.mu_p.shape[0]))
		
		for i in range(self.sim):
			mu_est[i,:] = self.dist.sample()
		np.testing.assert_array_less(np.mean(mu_est,0), np.mean(self.Y,0), "the prior should push sample downards")
		np.testing.assert_array_less(np.cov(mu_est.T),np.cov(self.Y.T)/self.Y.shape[0], "the prior is smaller then covarance") 
		
		true_mean =  np.dot( np.linalg.inv(prior['Sigma']), prior['mu'])
		true_mean += np.dot( np.linalg.inv(param['Sigma']), np.sum(self.Y, 0))
		true_mean =  np.dot( np.linalg.inv(np.linalg.inv(prior['Sigma']) + n * np.linalg.inv(param['Sigma'])), true_mean)
		np.testing.assert_array_almost_equal(np.mean(mu_est,0), true_mean, decimal=1)


class Test_MV_python(unittest.TestCase,multivariatenormal_base):


	def setUp(self):
		self.mv = mv_python


	def tearDown(self):
		pass


	def testName(self):
		pass

class Test_MV(unittest.TestCase,multivariatenormal_base):


	def setUp(self):
		self.mv = mv


	def tearDown(self):
		pass


	def testName(self):
		pass


if __name__ == "__main__":
	#import sys;sys.argv = ['', 'Test.testName']
	unittest.main()