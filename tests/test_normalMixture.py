# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 22:27:52 2014

@author: jonaswallin
"""
import unittest
import os
import numpy as np
path = os.path.dirname(os.path.realpath(__file__))


from bayesianmixture.PurePython import GMM

class test_normalMixture(unittest.TestCase):
    n = 100
    def setUp(self):
        self.K = 2
        self.data = np.random.rand(self.n, 2) + 1
        self.mix = GMM.mixture(self.data, self.K)
        self.mix2 = GMM.mixture(self.data, self.K)
        self.mix2.high_memory = False
        
    def test_simple(self):
        self.assertEqual(self.mix.K,self.K)
        self.assertEqual(self.mix2.K,self.K)
    
    def test_dim(self):
        self.assertEqual(self.mix.d,2 )
        self.assertEqual(self.mix2.d,2 )
        
    def test_error(self):
        with self.assertRaisesRegexp(ValueError, "the number of observations must be larger then the dimenstion"):
            GMM.mixture(np.ones((1,1)),2) 

    def build_test_x(self,n,mix):
        mix.p = np.ones(2)
        mix.p[0] = 0.
        mix.sample_x()
        np.testing.assert_array_equal(np.ones(n), mix.x)
    def test_x(self):
        self.build_test_x(self.n,self.mix)
        self.build_test_x(self.n,self.mix2)
    
    def build_test_x2(self,n,mix):
        
        mix.mu[0] = np.array([1,1.])
        mix.sigma[0] = np.diag([1,1.])*10**-4
        mix.mu[1] = np.array([-10,-10.])
        mix.sigma[1] = np.diag([1,1.])*10**-4
        if mix.high_memory == True:
            for k in range(mix.K):
                mix.data_mu[k] =  mix.data - mix.mu[k] 
        mix.sample_x()
        
        np.testing.assert_array_equal(np.zeros(n), mix.x) 
        
    def test_x2(self):
        self.build_test_x2(self.n,self.mix2)
        self.build_test_x2(self.n,self.mix)
        
    def test_sampling(self):
        GMM.sample_sigma(self.mix.data,
        self.mix.mu[0], self.mix.prior[0]["sigma"]["Q"], self.mix.prior[0]["sigma"]["nu"])
        self.mix.sample_x()
        self.mix.sample_sigma()
        
    def set_mu(self, mix):
        
        mu = [np.array([1,1.]), np.array([1,-11.])]
        mix.set_mu(mu)
        for k in range(mix.K):
            np.testing.assert_array_equal(mix.mu[k], mu[k]) 
        if mix.high_memory == True:
            for k in range(mix.K):
                np.testing.assert_array_equal(mix.data_mu[k] ,  mix.data - mu[k] )
                

        
        
    def test_set_mu(self):
        self.set_mu(self.mix)
        self.set_mu(self.mix2)
    
    def test_sample_mu(self):
        self.mix.sample_x()
        self.mix.sample_mu()
    
def main():
    unittest.main()

if __name__ == '__main__':
    main()