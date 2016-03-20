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
beta = 2*npr.rand(10, 1) - 0.5
N = 1000           # number of observations
d = beta.shape[0]  # number of covariates 
m = 5  			   # dimension of data
Bs = np.zeros((N, m, d))
Ys = np.zeros((N, m))
Sigmas = np.zeros((N, m, m ))
V = np.eye(d) # prior
V[0,0] = 5
mu = 2.1*np.ones(d)
Qs = np.zeros_like(Sigmas)
for i in range(N):
	R = 0.1 * npr.randn(m, m)
	Sigmas[i, :,:] =  R * R.transpose() + np.eye(m)
	Qs[i, :, :] = np.linalg.inv(Sigmas[i, :, :])
	R = np.linalg.cholesky(Sigmas[i, :, :])
	B  = np.hstack( ( np.ones((m, 1)), npr.randn(m, d - 1) ))
	Bs[i, :,:] = B
	Ys[i, :] = np.dot(B, beta).transpose() + np.dot(R, npr.randn(m, 1)).transpose()




# Sampling posterior of beta
t0 = time.clock()
Q_post  = np.linalg.inv(V)
mu_post = np.dot(Q_post, mu)


for i in range(N):
	mu_post += np.dot(Bs[i,:,:].transpose(), np.linalg.solve(Sigmas[i,:,:], Ys[i,:]))
	Q_post  += np.dot(Bs[i,:,:].transpose(), np.linalg.solve(Sigmas[i,:,:], Bs[i,:,:]))
t1 = np.double(time.clock() - t0) 
string = "(poor python: %.4f sec (N = %d) "%(t1, N, )
print(string)

mu_post    = np.linalg.solve(Q_post, mu_post)
Sigma_post = np.linalg.inv(Q_post) 

print(mu_post)


MVNRegObj = multivariatenormal_regression({'mu': mu, 'Sigma':V})

X0 = MVNRegObj.sample()
MVNRegObj.setY(Ys)
MVNRegObj.setB(Bs)
MVNRegObj.setSigmaY(Sigmas)
t0 = time.clock()
X = MVNRegObj.sample()
t1 = np.double(time.clock() - t0) 
string = "(cython: %.4f sec (N = %d) "%(t1, N, )
print(string)
t0 = time.clock()
X = MVNRegObj.sample()
t1 = np.double(time.clock() - t0) 
string = "(cython: %.4f sec (N = %d) (pre calc Q)"%(t1, N, )
print(string)
print(X - mu_post)