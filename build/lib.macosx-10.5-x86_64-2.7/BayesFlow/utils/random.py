'''
Created on Aug 12, 2015

@author: jonaswallin
'''

import numpy as np

def rmvn(mu, sigma):
	"""
		generates a sample from multivariate normal N(\mu, \sigma)
		Created beacuse npr.multivariate_normal seems temporary depricated
		*mu*    mean
		*sigma* covariance
	"""
	
	
	L = np.linalg.cholesky(sigma)
	return( mu.reshape(mu.shape[0]) + np.dot(L, np.random.randn(sigma.shape[0])))
	