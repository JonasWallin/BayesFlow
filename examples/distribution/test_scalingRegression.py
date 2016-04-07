'''
test code for scalingRegression, verifying that gradient and Hessian is correctly calculated
Created on Mar 28, 2016

@author: jonaswallin
'''
from __future__ import division
import numpy as np
import numpy.random as npr
import time
from BayesFlow.distribution import multivariatenormal_scaling

beta = - 0.01*(npr.rand(2, 1) + 1)
d = beta.shape[0]  # number of covariates
m = 5 #dimension of Y
N = 10 # number of observations
R = 0.1 * npr.randn(m, m)
Sigma =  R * R.transpose() + np.eye(m)
Q = np.linalg.inv(Sigma)
Ys = np.zeros((N, m))
Bs = np.zeros((N, m, d))	
lik_python = 0.
V = np.eye(d) # prior
V[0,0] = 5
mu     = np.zeros(d)
Second_Der = - np.linalg.inv(V)
for i in range(N):
	
	B  = np.hstack( ( np.ones((m, 1)), npr.randn(m, d - 1) ))
	Bs[i, :,:] = B
	Sigmas = np.dot(np.diagflat(np.exp( np.dot(B, beta)  )) ,
					np.dot(Sigma,
					np.diagflat(np.exp( np.dot(B, beta)  ))))
	Rs = np.linalg.cholesky(Sigmas).transpose()
	Ys[i, :] =  np.dot(Rs, npr.randn(m, 1)).transpose()
	Y_scaled = np.dot(np.diagflat(np.exp( - np.dot(B, beta))) , Ys[i,:])
	lik_python -= np.sum(  np.dot(B, beta)) + 0.5 *  np.dot(Y_scaled.transpose(),
						np.dot(Q, Y_scaled) )
	
	D_ = np.dot(np.dot(np.diagflat(Y_scaled), Q), np.diagflat(Y_scaled))
	print(Y_scaled)
	print( np.outer(Y_scaled, Y_scaled))
	D2_ = np.dot(Q, np.outer(Y_scaled, Y_scaled))
	
	D2_ = np.dot( np.dot(B.transpose(), np.diagflat(Y_scaled) * np.diagflat(np.dot(Q, Y_scaled)) ),
						   B)
	print(D2_)
	Second_Der -=  np.dot( np.dot(B.transpose(), D_ ),
						   B)
	Second_Der -= D2_

print("****")
# setting up object

MVNscaleObj = multivariatenormal_scaling({'mu': mu, 'Sigma':V})


beta = beta.flatten()
lik_python -=0.5* np.dot((beta - mu), 
				   np.linalg.solve(V, (beta - mu).transpose()))
MVNscaleObj.setY(Ys)
MVNscaleObj.setB(Bs)
MVNscaleObj.setSigmaY(Sigma)
lik = MVNscaleObj.loglik(beta.flatten())
print('lik = %.4f'%lik)
print('python lik = %.4f'%lik_python)
grad = MVNscaleObj.gradlik(beta.flatten())
print('grad = %s'%grad)
grad_python = np.zeros_like(grad)
eps = 1e-4
beta_eps    = np.zeros_like(beta)
beta_eps[:] = beta[:]
grad_num     = np.zeros_like(beta)
Delta_num    = np.zeros((d, d))
grad_eps = np.zeros_like(beta)
for i in range(beta.shape[0]):
	beta_eps[i] += eps 
	grad_num[i] = MVNscaleObj.loglik(beta_eps.flatten())
	grad_eps    = MVNscaleObj.gradlik(beta_eps.flatten())
	beta_eps[i] -= eps 
	beta_eps[i] -= eps 
	grad_num[i] -= MVNscaleObj.loglik(beta_eps.flatten())
	grad_eps    -= MVNscaleObj.gradlik(beta_eps.flatten())
	beta_eps[i] += eps 
	grad_num[i] /= 2 * eps
	grad_eps    /= 2 * eps
	Delta_num[:,i] += grad_eps
	Delta_num[i,:] += grad_eps
Delta_num /= 2
print('num grad = %s'%grad_num)

print('Delta num =%s'%Delta_num)
print('Delta     =%s'%Second_Der)
	
	
	


