'''
Created on Aug 16, 2014

@author: jonaswallin
'''
import unittest
import scipy.linalg as spl

import BayesFlow.mixture_util.GMM_util as GMM_util
import numpy as np
import numpy.random as npr


class Test(unittest.TestCase):


	def testCholesky(self):
		
		A = spl.toeplitz([1.,.3,.3])
		R_Q = spl.cho_factor(A,check_finite = False)
		R = GMM_util.cholesky(A)
		np.testing.assert_almost_equal(R, np.triu(R_Q[0]), 8)

	def testSolve_w_Cholesky(self):
		
		A = spl.toeplitz([1.,.3,.3])
		mu = np.ones((3,1))
		R = GMM_util.cholesky(A)
		invA_A = GMM_util.solve_R(A, R)
		np.testing.assert_almost_equal(invA_A,np.eye(3), 8)
		invA_mu = GMM_util.solve_R(mu, R)
		np.testing.assert_almost_equal(invA_mu, np.linalg.solve(A, mu), 8)
		
if __name__ == "__main__":
	#import sys;sys.argv = ['', 'Test.testName']
	unittest.main()