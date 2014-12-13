'''
Built for checking nu
Created on Jul 5, 2014

@author: jonaswallin
'''


from __future__ import division
from bayesianmixture.PurePython.distribution import nu_class
from bayesianmixture.PurePython.distribution.wishart import invwishartrand
import scipy.linalg as spl
import numpy as np


import unittest

class test_nu(unittest.TestCase):
	
	
	def simulate(self, n):
		
		nu_t = 20
		r = np.array([5,-2,1,0])
		d = len(r)
		Q = spl.toeplitz(r) / nu_t
		
		self.nu = nu_class()
		self.nu.set_d(d)
		param = {'Q':Q}
		
		self.nu.set_parameter(param)
		invwishartrand
		self.nu.set_data([invwishartrand(nu_t, Q) for i in range(n)])
		iterations = 10000
		self.nus = np.zeros(iterations)
		for i in range(iterations):
			self.nus[i] = self.nu.sample(sigma = 2,iterations = 10)
		self.nu_vec = range(nu_t-10,nu_t+10)
		lik_  = np.array([self.nu.loglik(nu_) for nu_ in self.nu_vec])
		lik_  -= np.max(lik_)
		lik_ = np.exp(lik_) 
		lik_ /= np.sum(lik_)
		self.lik_ = lik_
	def test_mean(self):
		
		ns = [10,100,1000]
		
		for n in ns:
			self.simulate(n)
			np.testing.assert_almost_equal(np.sum(self.lik_*self.nu_vec), np.mean(self.nus), 1)


if __name__ == "__main__":
	unittest.main()