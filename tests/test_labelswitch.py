'''
Created on Jul 18, 2014

@author: jonaswallin
'''
import unittest
import numpy as np
from bayesianmixture.PurePython.GMM import mixture
from bayesianmixture import PurePython
import bayesianmixture as bm
import copy as cp
import scipy.linalg as spl


class comp(object):


	def setUp(self):
		self.d = 3
		self.K = 4
		self.n_obs =1000
		self.sim = 20
		self.r =  np.zeros(self.d)
		self.r[0] = 3.
		self.r[1] = -1
		self.r[2] = 1
		
	def generate_data_mu(self ):
		p = np.ones(self.K)/self.K
		self.mu = [np.repeat(np.double(k),self.d) for k in range(self.K)]
		self.sigma =[np.eye(self.d)*0.01 for k in range (self.K)]
		mix = mixture(K = self.K)
		mix.mu = self.mu
		mix.sigma = self.sigma
		mix.p = p
		mix.d = self.d
		self.Y = mix.simulate_data(self.n_obs)
		self.mix.mu = cp.deepcopy(self.mu)
		self.mix.sigma = cp.deepcopy(self.sigma)
		self.mix.p =cp.deepcopy( p)
		self.mix.d = self.d
		self.mix.set_data(self.Y)
		


	def set_prior_mu(self):
		prior = [{'nu':10,'Q':(10 - self.d  - 1)*np.eye(self.d)*0.01,'theta':np.repeat(np.double(k),self.d),'Sigma':np.eye(self.d)*0.01} for k in range(self.K)]
		self.mix.set_prior_mu(prior)
		self.mix.set_prior_sigma(prior)


	def test_label_move_mu(self):
		self.generate_data_mu()
		self.set_prior_mu()
		self.mix.mu[1], self.mix.mu[0] = self.mix.mu[0], self.mix.mu[1]
		self.mix.p_switch = 1
		for i in range(self.sim):  # @UnusedVariable
			self.mix.sample_labelswitch()
		np.testing.assert_array_almost_equal(np.array(self.mix.mu), np.array(self.mu), 8)


	def set_prior_sigma(self):
		prior = [{'nu':10,'Q':(10 - self.d  - 1)*spl.toeplitz(self.r)*0.01,'theta':np.repeat(np.double(k),self.d),'Sigma':np.eye(self.d)*0.01} for k in range(self.K-1)]
		prior.append({'nu':10,'Q':(10 - self.d  - 1)*np.eye(self.d)*0.01,'theta':np.repeat(np.double(0),self.d),'Sigma':np.eye(self.d)*0.01})
		self.mix.set_prior_mu(prior)
		self.mix.set_prior_sigma(prior)

	def generate_data_sigma(self ):
		p = np.ones(self.K)/self.K
		self.mu = [np.repeat(np.double(k),self.d) for k in range(self.K-1)]
		self.mu.append(np.repeat(np.double(0),self.d) )
		self.sigma =[spl.toeplitz(self.r)*0.01 for k in range (self.K-1)]
		self.sigma.append(np.eye(self.d)*0.01 )
		mix = mixture(K = self.K)
		mix.mu = self.mu
		mix.sigma = self.sigma
		mix.p = p
		mix.d = self.d
		self.Y = mix.simulate_data(self.n_obs)
		self.mix.mu = cp.deepcopy(self.mu)
		self.mix.sigma = cp.deepcopy(self.sigma)
		self.mix.p =cp.deepcopy( p)
		self.mix.d = self.d
		self.mix.set_data(self.Y)
		
	def test_label_move_sigma(self):
		self.generate_data_sigma()
		self.set_prior_sigma()
		self.mix.sigma[3], self.mix.sigma[0] = self.mix.sigma[0], self.mix.sigma[3]
		self.mix.p_switch = 1
		for i in range(self.sim):  # @UnusedVariable
			self.mix.sample_labelswitch()
		np.testing.assert_array_almost_equal(np.array(self.mix.sigma), np.array(self.sigma), 8)

class test_python(comp, unittest.TestCase):
	"""
		test
	"""
	
	def setUp(self):
		super(test_python,self).setUp()
		self.mix = PurePython.GMM.mixture(K = self.K)

class test_bm(comp, unittest.TestCase):
	"""
		test
	"""
	
	def setUp(self):
		super(test_bm,self).setUp()
		self.mix = bm.mixture(K = self.K)	

if __name__ == "__main__":
	#import sys;sys.argv = ['', 'Test.testName']
	unittest.main()