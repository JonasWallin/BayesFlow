'''
Created on Jul 15, 2014

@author: jonaswallin
'''
import unittest
import numpy as np
import bayesianmixture as bm
import scipy.linalg as spl
from mpi4py import MPI

from bayesianmixture.PurePython.distribution.wishart import  invwishartrand
from bayesianmixture.PurePython.GMM import mixture
class Test(unittest.TestCase):


	def setUp(self):
		self.K = 2
		self.hGMM = bm.hierarical_mixture_mpi(K = self.K)
		self.d = 2
		self.n = 5
		self.n_obs = 1000
		self.comm = MPI.COMM_WORLD  # @UndefinedVariable

	def test_generate_data(self):
		
		if self.comm.Get_rank() == 0:
			self.mu = [np.repeat(np.double(k),self.d) for k in range(self.K)]
			r = np.zeros(self.d)
			r[0] = 3.
			r[1] = -1
			self.sigma = [spl.toeplitz(r) for k in range(self.K)]
			
			self.data_sigma = [ [invwishartrand(5, self.sigma[k]) for k in range(self.K)] for n in range(self.n)]  # @UnusedVariable
			self.data_mu    = [ [self.mu[k]+ np.random.randn(self.d)*0.2 for k in range(self.K)] for n in range(self.n)]  # @UnusedVariable
			self.Y = []
			for i in range(self.n):
				mix = mixture(K = self.K)
				mix.mu = self.data_mu[i]
				mix.sigma = self.data_sigma[i]
				mix.p = np.ones(self.K) / self.K
				mix.d = self.d
				self.Y.append(mix.simulate_data(self.n_obs))
		else:
			self.Y = None
		self.hGMM.set_data(self.Y)
		self.hGMM.set_prior_param0()
				
	def test_update_prior(self):
		self.test_generate_data()
		for k in range(self.K):
			for i in range(len(self.hGMM.GMMs)):
				self.hGMM.GMMs[i].mu[k] = np.repeat(k + 10.*i + 100* self.comm.Get_rank(), self.d)
				
		
		self.comm.Barrier()
		self.hGMM.update_prior()
		


if __name__ == "__main__":
	#import sys;sys.argv = ['', 'Test.testName']
	unittest.main()