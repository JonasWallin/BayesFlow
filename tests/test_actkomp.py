'''
Created on Jul 16, 2014
program for testing activating and deacitvation of mixtures
@author: jonaswallin
'''
from __future__ import division
import unittest
import numpy as np
from BayesFlow.PurePython.GMM import mixture
from BayesFlow import PurePython
import BayesFlow as bm
import scipy.linalg as spl

class comp(object):


	def setUp(self):
		self.d = 3
		self.K = 4
		self.n_obs =1000
		self.sim = 20

	def generate_data(self, p ):
		
		self.mu = [np.repeat(np.double(k),self.d) for k in range(self.K)]
		self.sigma =[np.eye(self.d)*0.01 for k in range (self.K)]
		mix = mixture(K = self.K)
		mix.mu = self.mu
		mix.sigma = self.sigma
		mix.p = p
		mix.d = self.d
		self.Y = mix.simulate_data(self.n_obs)
		self.mix.mu = self.mu
		self.mix.sigma = self.sigma
		self.mix.p = p
		self.mix.d = self.d
		self.mix.set_data(self.Y)
		


	def set_prior(self):
		prior = [{'nu':10,'Q':(10 - self.d  - 1)*np.eye(self.d)*0.01,'theta':np.repeat(np.double(k),self.d),'Sigma':np.eye(self.d)*0.01} for k in range(self.K)]
		self.mix.set_prior_mu(prior)
		self.mix.set_prior_sigma(prior)
		
	def test_activate_komp(self):
		p = np.ones(self.K)/self.K
		self.generate_data(p)
		self.set_prior()
		self.mix.active_komp[3] = False
		#self.mix.sample()
		self.mix.p_inact = 1.
		self.mix.p_act   = 1.
		self.mix.in_act_index = [3]
		
		for i in range(self.sim):  # @UnusedVariable
			self.mix.sample_activate()
		np.testing.assert_array_equal(self.mix.active_komp, [True for k in range(self.K)])  # @UnusedVariable



	def test_activate_komp2(self):
		p = np.ones(self.K)/self.K
		p[3] = 0.01
		p = p/sum(p)
		self.generate_data(p)
		self.set_prior()
		self.mix.active_komp[3] = False
		#self.mix.sample()
		self.mix.p_inact = 1.
		self.mix.p_act   = 1.
		for i in range(self.sim):  # @UnusedVariable
			self.mix.sample_activate()
		np.testing.assert_array_equal(self.mix.active_komp, [True for k in range(self.K)])  # @UnusedVariable

	def generate_data2(self, p ):
		
		self.mu = [np.repeat(np.double(k),self.d) for k in range(self.K-1)]
		self.sigma =[np.eye(self.d)*0.01 for k in range (self.K-1)]
		mix = mixture(K = self.K-1)
		mix.mu = self.mu
		mix.sigma = self.sigma
		mix.p = p[:(self.K-1)] / np.sum(p[:(self.K-1)])
		self.p = mix.p
		mix.d = self.d
		self.Y = mix.simulate_data(self.n_obs)
		self.mix.mu = [np.repeat(np.double(k),self.d) for k in range(self.K)]
		self.mix.sigma = [np.eye(self.d)*0.01 for k in range (self.K)]
		self.mix.p = p
		self.mix.d = self.d
		self.mix.set_data(self.Y)

	def test_inactivate_komp(self):
		p = np.ones(self.K)/self.K
		self.generate_data2(p)
		self.set_prior()
		#self.mix.sample()
		self.mix.p_inact = 1.
		self.mix.p_act   = 1.
		for i in range(self.sim):  # @UnusedVariable
			self.mix.sample_inactivate()
		true_act = np.double([True for k in range(self.K)])  # @UnusedVariable
		true_act[-1] = 0.
		np.testing.assert_array_equal(np.double(self.mix.active_komp), true_act)  # 
		
		
	def test_inactivate_komp2(self):
		p = np.ones(self.K)/self.K
		p[3] = 0.01
		p = p/sum(p)
		self.generate_data2(p)
		self.set_prior()
		#self.mix.sample()
		self.mix.p_inact = 1.
		self.mix.p_act   = 1.
		for i in range(self.sim):  # @UnusedVariable
			self.mix.sample_inactivate()
		true_act = np.double([True for k in range(self.K)])  # @UnusedVariable
		true_act[-1] = 0.
		np.testing.assert_array_equal(np.double(self.mix.active_komp), true_act)  # 	

	def test_comp_komp2(self):
		p = np.ones(self.K)/self.K
		self.generate_data2(p)
		self.set_prior()
		#self.mix.sample()
		self.mix.p_inact = 1.
		self.mix.p_act   = 1.
		for i in range(self.sim):  # @UnusedVariable
			self.mix.sample_active_komp()
			
		true_act = np.double([True for k in range(self.K)])  # @UnusedVariable
		true_act[-1] = 0.
		np.testing.assert_array_equal(np.double(self.mix.active_komp), true_act)  #	
		
class test_bm(comp, unittest.TestCase):
	"""
		test
	"""
	
	def setUp(self):
		super(test_bm,self).setUp()
		self.mix = bm.mixture(K = self.K)		
		

class test_python(comp, unittest.TestCase):
	"""
		test
	"""
	
	def setUp(self):
		super(test_python,self).setUp()
		self.mix = PurePython.GMM.mixture(K = self.K)


class test_bm_vs_python(unittest.TestCase):
	
	def setUp(self):
		self.d = 3
		self.r =  np.zeros(self.d)
		self.r[0] = 3.
		self.r[1] = -1
		self.r[2] = 1
		self.K = 4
		self.n_obs =100
		self.sim = 10
		self.mix_py = PurePython.GMM.mixture(K = self.K)
		self.mix = bm.mixture(K = self.K)
		
	def set_prior(self):
		prior = [{'nu':10,'Q':(10 - self.d  - 1)*np.eye(self.d)*0.01,'theta':np.repeat(np.double(k),self.d),'Sigma':spl.toeplitz(self.r)*0.01} for k in range(self.K)]
		self.mix.set_prior_mu(prior)
		self.mix.set_prior_sigma(prior)
		self.mix_py.set_prior_mu(prior)
		self.mix_py.set_prior_sigma(prior)


	def generate_data(self, p ):
		
		self.mu = [np.repeat(np.double(k),self.d) for k in range(self.K)]
		self.sigma =[spl.toeplitz(self.r)*0.01 for k in range (self.K)]
		mix = mixture(K = self.K)
		mix.mu = self.mu
		mix.sigma = self.sigma
		mix.p = p
		self.p = p
		mix.d = self.d
		self.Y = mix.simulate_data(self.n_obs)
		self.mix.mu = self.mu
		self.mix.sigma = self.sigma
		self.mix.p = p
		self.mix.d = self.d
		self.mix.set_data(self.Y)
		self.mix_py.mu = self.mu
		self.mix_py.sigma = self.sigma
		self.mix_py.p = p
		self.mix_py.d = self.d
		self.mix_py.set_data(self.Y)

	def test_probX(self):
		p = np.ones(self.K)/self.K
		self.generate_data(p)
		self.set_prior()
		self.mix_py.compute_ProbX(norm=True)
		self.mix.compute_ProbX(norm=True)
		np.testing.assert_array_almost_equal(self.mix_py.prob_X, self.mix.prob_X, 8)
		
		
		self.mix_py.compute_ProbX(norm=True,mu = self.mu, sigma = self.sigma, p = self.p)
		self.mix.compute_ProbX(norm=True,mu = self.mu, sigma = self.sigma, p = self.p)
		np.testing.assert_array_almost_equal(self.mix_py.prob_X, self.mix.prob_X, 8)
		
		
	def test_calc_lik(self):
		p = np.ones(self.K)/self.K
		self.generate_data(p)
		self.set_prior()
		np.testing.assert_array_almost_equal(self.mix.calc_lik(), self.mix_py.calc_lik(), 8)
		np.testing.assert_array_almost_equal(self.mix.calc_lik(mu = self.mu, sigma = self.sigma, p = self.p), self.mix_py.calc_lik(mu = self.mu, sigma = self.sigma, p = self.p), 8)
		
if __name__ == "__main__":
	#import sys;sys.argv = ['', 'Test.testName']
	unittest.main()