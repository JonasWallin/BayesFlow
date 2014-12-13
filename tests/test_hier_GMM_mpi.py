'''
WARNING NEED TO SET A TEMP DIRECTORY!!!!!! (dirname line 21)
RUN WITH :
mpiexec -n 4 python test_hier_GMM_mpi.py 

Created on Sep 17, 2014

@author: jonaswallin
mpiexec -n 4 python test_hier_GMM.py
'''
import unittest
import bayesianmixture as bm
import numpy as np
import scipy.linalg as spl
from bayesianmixture.PurePython.distribution.wishart import  invwishartrand
from mpi4py import MPI
from bayesianmixture.PurePython.GMM import mixture



dirname = "/Users/jonaswallin/temp/Flow/"


class Test(unittest.TestCase):

	def test_setdata(self):
		
		self.setdata()
	
	
	
	def test_save_load(self):
		self.setdata()
		hGMM = bm.hierarical_mixture_mpi(K = 4)
		hGMM.set_data(self.Y)
		hGMM.set_prior_param0()
		hGMM.update_GMM()
		hGMM.update_prior()
		hGMM.set_p_labelswitch(1.)
		hGMM.set_prior_actiavation(10)
		
		hGMM.sample()
		hGMM.save_GMMS_to_file(dirname)
		hGMM.save_prior_to_file(dirname)
		hGMM.load_GMMS_from_file(dirname)
		hGMM.load_prior_from_file(dirname)
		hGMM.sample()
		
		hGMM.save_to_file(dirname)
		hGMM.load_to_file(dirname)
		
		hGMM.noise_class = 1
		hGMM.save_to_file(dirname)
		hGMM.load_to_file(dirname)
		
		np.testing.assert_equal(1, hGMM.noise_class)
	
	def test_save_load2(self):	
		
		hGMM = bm.hierarical_mixture_mpi(K = 4)
		hGMM.load_to_file(dirname)
		
		np.testing.assert_equal(1, hGMM.noise_class)
		
	def setUp(self):
		
		if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable
			self.n_y = 20
			self.n_obs = 1000
			self.d = 4
			self.K = 4
			self.sim  = 1000
			self.hGMM = bm.hierarical_mixture(K = self.K)


	def setdata(self):
		
		if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable
			self.mu = [np.repeat(np.double(k),self.d) for k in range(self.K)]
			self.r1 =  np.zeros(self.d)
			self.r1[0] = 3.
			self.r1[1] = -1
			self.r1[2] = 1
			self.r1[2] = .5
			self.r2 =  np.zeros(self.d)
			self.r2[0] = 2.
			self.r2[1] = -1.5
			self.r2[2] = 0
			self.r2[2] = 0.		
			self.sigma =[np.eye(self.d), spl.toeplitz(self.r1), spl.toeplitz(self.r1), 0.5*np.eye(self.d)]
			self.p = np.array([1.,1.,1.,0.1])
			self.p2 =  np.array([1.,1.,1.])
			self.p /= np.sum(self.p)
			self.p2 /= np.sum(self.p2)
			
			
			
			self.data_sigma = [ [invwishartrand(5, self.sigma[k]) for k in range(self.K)] for n in range(self.n_y)]  # @UnusedVariable
			self.data_mu    = [ [self.mu[k]+ np.random.randn(self.d)*0.2 for k in range(self.K)] for n in range(self.n_y)]  # @UnusedVariable
			
			self.Y = []
			for i in range(self.n_y):
				
				if i < 17:
					mix = mixture(K = self.K)
					mix.mu = self.data_mu[i]
					mix.sigma = self.data_sigma[i]
					mix.p = self.p
				else:
					mix = mixture(K = self.K-1)
					mix.mu = self.data_mu[i][:-1]
					mix.sigma = self.data_sigma[i][:-1]
					mix.p = self.p2
				mix.d = self.d
				self.Y.append(mix.simulate_data(self.n_obs))
		else:
			self.Y = None		
		


if __name__ == "__main__":
	#import sys;sys.argv = ['', 'Test.testName']
	unittest.main()