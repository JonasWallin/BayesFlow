'''
Wishart + nu
Created on Jul 7, 2014

@author: jonaswallin
'''
import unittest
import numpy as np
import scipy.linalg as spl
from BayesFlow.PurePython.distribution import Wishart as Wis_python
from BayesFlow.PurePython.distribution.wishart import  wishart_var, invwishartrand
from BayesFlow.distribution import Wishart
import os




class Wishart_base(object):
	sim = 100
	n   = 1000
	def __init__(self):
		
		self.Wishart = None
		
	def test_startup(self):
		d = 4
		prior = {'nus':10**1,'Qs':np.eye(d)*10.**2 }
		param = {'nu': 4}  
		self.d = d   
		self.dist = self.Wishart(prior = prior,param  =param)


	def sample_data(self):
		
		self.data = []
		r = np.zeros(self.d)
		nu  =10.
		r[0] = 5.
		r[1] = -0.5
		r[2] = 1.
		self.Sigma = spl.toeplitz(r)/nu
		for i in range(self.sim):  # @UnusedVariable
			self.data.append( invwishartrand(nu, self.Sigma))
		self.dist.set_data(self.data)
		Q = np.zeros((self.d, self.d))
		for i in range(self.sim):
			Q += self.data[i]
		#print(self.data)
		#print(self.Sigma * nu)
	def compare_mean_var(self):

		mean_Q = np.zeros((self.dist.d, self.dist.d))
		var_Q   = np.zeros_like(mean_Q)
		for i in range(self.sim):  # @UnusedVariable
			res = self.dist.sample()
			mean_Q += res
			var_Q  += res**2
		mean_Q /= self.sim
		var_Q  /= self.sim
		if self.dist.Q != None:
			V = np.linalg.inv(self.dist.Q_s + self.dist.Q)
		else:
			V = np.linalg.inv(self.dist.Q_s)
		nu = self.dist.n * self.dist.nu + self.dist.nu_s
		true_Var = wishart_var(nu, V)
		np.testing.assert_array_almost_equal(mean_Q, nu*V,     decimal = 2)
		np.testing.assert_array_almost_equal(var_Q,  true_Var, decimal = 2)

	
	def test_prior(self):
		self.test_startup()
		self.compare_mean_var()
		
	def stest_data(self):
		#self.test_startup()
		#self.sample_data()
		#self.compare_mean_var()
		pass 
	
	
	def test_pickle(self):
		"""
			testing if pickling works
		
		"""
		self.test_startup()
		fileName = "test.pkl"
		self.dist.pickle(fileName)
		self.dist = self.Wishart.unpickle(fileName)
		os.remove(fileName)
		
		self.sample_data()
		mean_Q = np.zeros((self.dist.d, self.dist.d))
		var_Q   = np.zeros_like(mean_Q)
		for i in range(2):  # @UnusedVariable
			res = self.dist.sample()
			mean_Q += res
			var_Q  += res**2
		self.dist.pickle(fileName)
		self.dist = self.Wishart.unpickle(fileName)
		os.remove(fileName)
		for i in range(2):  # @UnusedVariable
			mean_Q += res
			var_Q  += res**2
		#pass
	
	
class test_pythonWishart(unittest.TestCase,Wishart_base):


	def setUp(self):
		print("pyth")
		self.Wishart = Wis_python

class test_Wishart(unittest.TestCase,Wishart_base):


	def setUp(self):
		print("wish")
		self.Wishart = Wishart

if __name__ == "__main__":
	#import sys;sys.argv = ['', 'Test.testName']
	unittest.main()
