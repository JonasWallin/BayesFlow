'''
Created on Jul 21, 2014

@author: jonaswallin
'''
from __future__ import division
import numpy as np


def mvn(mu1, mu2, Sigma1, Sigma2):
	"""
		computes the Hellinger distance between N(mu1, Sigma1), N(mu2, Sigma2)
	
		mu1,2      -  (dx1) mean
		Sigma1,2   -  (dxd) the covariance
	"""
	
	P = (Sigma1 + Sigma2) /2.
	detP = np.linalg.det(P)
	DetS12 = np.sqrt(np.linalg.det(Sigma1) * np.linalg.det(Sigma2))
	mu_mu = mu1 - mu2
	BC = np.exp( - np.dot(mu_mu, np.linalg.solve(P, mu_mu).T)/8. )
	BC *= np.sqrt(DetS12/detP)
	H = np.sqrt(1 - BC)
	return H