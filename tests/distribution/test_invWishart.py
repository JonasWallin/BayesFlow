'''
Created on Jul 2, 2014

@author: jonaswallin
'''
import unittest
import numpy as np
import numpy.random as npr
import scipy.linalg as spl
from BayesFlow.PurePython.distribution import invWishart as invWis_python
from BayesFlow.distribution import invWishart as invWis
import os


class invWishart_base(object):
	sim = 5000
	n   = 8000
	def __init__(self):
		
		self.invWishart = None
	
	def test_startup(self):
		
		prior = {'nu':5,'Q':np.array([[1.]])}
		param = {'theta': np.array([0.])}
		self.dist = self.invWishart(prior = prior,param  =param)
		

		
	def startup1(self):
		prior = {'nu':0,'Q':np.array([[0.]])}
		param = {'theta': np.array([0.])}		
		self.dist = self.invWishart(prior = prior,param  =param)
		self.Sigma = np.array([[2.]])
		self.sample_Y()

	def startup2(self):
		d = 4
		prior = {'nu':0,'Q':np.zeros((d,d))}
		param = {'theta': npr.randn(d)}   
		row = np.zeros(d)	 
		self.dist = self.invWishart(prior = prior,param  =param)
		if d > 1:
			row[0] = 4.
		if d >2:
			row[1] = -2
		self.Sigma = spl.toeplitz(row)
		self.sample_Y()
	
	def sample_Y(self):
		self.Y = np.dot(npr.randn(self.n, self.Sigma.shape[0]),np.linalg.cholesky(self.Sigma).T) 
		self.Y += self.dist.theta   
		
	def test_pickle(self):
		"""
			testing if pickling works
		
		"""
		self.startup1()
		fileName = "test.pkl"
		self.dist.pickle(fileName)
		self.dist = self.invWishart.unpickle(fileName)
		os.remove(fileName)
		
		self.dist.set_data(self.Y)
		Sigma_mean= np.zeros(self.dist.Q.shape)
		for i in range(2):  # @UnusedVariable
			Sigma_mean += self.dist.sample()
		self.dist.pickle(fileName)
		self.dist = self.invWishart.unpickle(fileName)
		os.remove(fileName)
		for i in range(2):  # @UnusedVariable
			Sigma_mean += self.dist.sample()
		#pass
	
	def test_mean(self):
		
		startups = [self.startup1,self.startup2]
		
		for startup in startups:
			startup()
			self.dist.set_data(self.Y)
			
			Sigma_mean= np.zeros(self.dist.Q.shape)
			for i in range(self.sim):  # @UnusedVariable
				Sigma_mean += self.dist.sample()
			Sigma_mean /= self.sim
			#print(self.dist.sample())
			#print(self.Sigma)
			np.testing.assert_array_almost_equal(Sigma_mean, self.Sigma, decimal=0)
	
	def test_big_prior(self):
	
		d = 4
		prior = {'nu':10**7,'Q':np.eye(d)*10.**7 }
		param = {'theta': np.zeros(d)}   
		row = np.zeros(d)	 
		self.dist = self.invWishart(prior = prior,param  =param)
		if d > 1:
			row[0] = 4.
		if d >2:
			row[1] = -2
		self.Sigma = spl.toeplitz(row)
		self.sample_Y()
		
		self.dist.set_data(self.Y)
			
		Sigma_mean= np.zeros(self.dist.Q.shape)
		for i in range(self.sim):  # @UnusedVariable
			Sigma_mean += self.dist.sample()
		Sigma_mean /= self.sim
		np.testing.assert_array_almost_equal(Sigma_mean, prior['Q']/prior['nu'], decimal=1)

	def test_prior_effect(self):
	
		d = 4
		prior = {'nu':10**3,'Q':np.eye(d)*10.**3 }
		param = {'theta': np.zeros(d)}   
		row = np.zeros(d)	 
		self.dist = self.invWishart(prior = prior,param  =param)
		if d > 1:
			row[0] = 4.
		if d >2:
			row[1] = -2
		self.Sigma = spl.toeplitz(row)
		self.sample_Y()
		
		self.dist.set_data(self.Y)
			
		Sigma_mean= np.zeros(self.dist.Q.shape)
		for i in range(self.sim):  # @UnusedVariable
			Sigma_mean += self.dist.sample()
		Sigma_mean /= self.sim
		
		np.testing.assert_array_less(np.diag(Sigma_mean),np.diag(self.Sigma)) 

		
class Test_invWis_python(unittest.TestCase,invWishart_base):


	def setUp(self):
		print('python')
		self.invWishart = invWis_python


	def tearDown(self):
		pass


	def testName(self):
		pass 
	
class Test_invWis(unittest.TestCase,invWishart_base):


	def setUp(self):
		print('cython')
		self.invWishart = invWis


	def tearDown(self):
		pass


	def testName(self):
		pass 
		
if __name__ == "__main__":
	#import sys;sys.argv = ['', 'Test.testName']
	unittest.main()