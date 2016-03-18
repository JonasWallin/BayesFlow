'''
Artifical example for testing if the class multivariatenormal_regression, works

Created on Mar 17, 2016

@author: jonaswallin
'''
from __future__ import division
import numpy as np
import numpy.random as npr
import time
from BayesFlow.distribution import multivariatenormal_regression
beta = np.ones((2,1))
N = 1000           # number of observations
d = beta.shape[0]  # number of covariates 
m = 5  			   # dimension of data
Bs = np.zeros((m, d, N))
Ys = np.zeros((m, N))
Sigmas = np.zeros((m, m , N))
V = np.eye(d) # prior
V[0,0] = 5
mu = 2.1*np.ones(d)
for i in range(N):
	R = 0.1 * npr.randn(m, m)
	Sigmas[:,:, i] =  R * R.transpose() + np.eye(m)
	R = np.linalg.cholesky(Sigmas[:,:, i])
	B  = np.hstack( ( np.ones((m, 1)), npr.randn(m,1) ))
	Bs[:,:,i] = B
	Ys[:,i] = np.dot(B, beta).transpose() + np.dot(R, npr.randn(m, 1)).transpose()




# Sampling posterior of beta
t0 = time.time()
mu_post = np.linalg.solve(V, mu)
Q_post  = np.linalg.inv(V)

for i in range(N):
	mu_post += np.dot(Bs[:,:,i].transpose(), np.linalg.solve(Sigmas[:,:,i], Ys[:,i]))
	Q_post  += np.dot(Bs[:,:,i].transpose(), np.linalg.solve(Sigmas[:,:,i], Bs[:,:,i]))
t1 = np.double(time.time() - t0) * 1000.
string = "(poor python: %.4f Msec/N (N = %d)  "%(t1/N, N)
print(string)

mu_post    = np.linalg.solve(Q_post, mu_post)
Sigma_post = np.linalg.inv(Q_post) 

print(mu_post)