# -*- coding: utf-8 -*-
"""
Created on Sat May 23 23:29:46 2015

@author: jonaswallin
"""
import unittest
import numpy as np
import copy as cp
import numpy.random as npr
from BayesFlow.PurePython.distribution import logisticMNormal as logisticMNormal_python
class Test(unittest.TestCase):
	"""
		Simulate data see if one gets sutiable result?
	
	"""
	eps = 10**-6

	def setUp(self):
		npr.seed(123456)
		self.MMN_obj = logisticMNormal_python()
	
	def set_data1(self):
		self.n   = 10**3
		self.p = [0.1, 0.2, 0.3, 0.4]
		self.Y = np.random.multinomial(self.n, self.p, size=1)
		self.MMN_obj.set_data(self.Y)

	def set_data2(self):
		self.n   = 10**2
		self.p = [0.1, 0.9]
		self.Y = np.random.multinomial(self.n, self.p, size=1)
		self.MMN_obj.set_data(self.Y)


	def set_prior(self):
		
		self.mu = npr.randn(len(self.p)-1)
		A = npr.randn(len(self.p)-1, len(self.p) - 1)
		self.Sigma = np.dot(A.T,A)
		self.MMN_obj.set_prior({"mu":self.mu, "Sigma": self.Sigma})

	def test_alpha_to_p(self):
		
		self.p = [0.1, 0.2, 0.3, 0.4]
		self.MMN_obj.set_alpha_p(self.p)
		np.testing.assert_almost_equal(self.p, self.MMN_obj.get_p())


	def compare_gradient_f(self):
		
		self.MMN_obj.set_alpha_p(self.p)
		
		llik, grad, H =  self.MMN_obj.get_f_grad_hess()
		H_est = np.zeros_like(H)
		for i in range(len(self.MMN_obj.alpha)):
			alpha_eps =cp.deepcopy(self.MMN_obj.alpha)
			alpha_eps[i] += self.eps
			llik2, grad_eps, H_eps = self.MMN_obj.get_f_grad_hess(alpha_eps)  # @UnusedVariable
			np.testing.assert_almost_equal(grad[i], (llik2-llik)/self.eps, decimal = 3)
			H_est[i,:] = (grad_eps - grad) /self.eps
			
			
		np.testing.assert_almost_equal(H, (H_est + H_est.T)/2., decimal = 3)		
		
		
	def compare_gradient_lik(self):
		
		self.MMN_obj.set_alpha_p(self.p)
		llik, grad, H = self.MMN_obj.get_llik_grad_hess()
		H_est = np.zeros_like(H)
		for i in range(len(self.MMN_obj.alpha)):
			alpha_eps =cp.deepcopy(self.MMN_obj.alpha)
			alpha_eps[i] += self.eps
			llik2, grad_eps, H_eps = self.MMN_obj.get_llik_grad_hess(alpha_eps)  # @UnusedVariable
			np.testing.assert_almost_equal(grad[i], (llik2-llik)/self.eps, decimal = 3)
			H_est[i,:] = (grad_eps - grad) /self.eps
			
		np.testing.assert_almost_equal(H, (H_est + H_est.T)/2., decimal = 3)
		
	def test_gradient_prior(self):
		
		self.set_data1()		
		self.set_prior()
		self.MMN_obj.set_alpha_p(self.p)
		llik, grad, H = self.MMN_obj.get_lprior_grad_hess()
		mu_a = self.MMN_obj.alpha - self.mu
		Q = np.linalg.inv(self.Sigma)
		llik_true = - np.dot(mu_a.T, np.dot(Q,mu_a))/2
		np.testing.assert_almost_equal(llik, llik_true, decimal = 7)
		np.testing.assert_almost_equal(grad, - np.dot(Q, mu_a), decimal = 7)
		np.testing.assert_almost_equal(H, -Q , decimal = 7)
		#self.MMN_obj
	
	def test_gradient_lik(self):
		self.set_data1()
		self.compare_gradient_lik()
		self.set_data2()
		self.compare_gradient_lik()
		
	def test_gradgient(self):   
		self.set_data1()
		self.set_prior()
		self.compare_gradient_f()
		
	def test_sample(self):
		
		self.set_data1()
		self.set_prior()
		self.MMN_obj.set_alpha_p(self.p)
		self.MMN_obj.sample()
			
		
if __name__ == "__main__":
	#import sys;sys.argv = ['', 'Test.test_run']
	unittest.main()