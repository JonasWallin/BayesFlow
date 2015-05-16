# -*- coding: utf-8 -*-
"""
Created on Thu May  7 19:21:41 2015

@author: jonaswallin
"""
import matplotlib.pyplot as plt
import numpy as np
from BayesFlow.PurePython.GMMrep import mixture_repeat_measurements

n  = 500
m  = 200
K = 3
np.random.seed(0)
mu    = np.array([[1. , 2.], [4.,0.],[-2.,1]])
Sigma = 0.1*np.array([np.eye(2), np.array([[2.,1],[1, 2.]]), np.array([[2.,-1.],[-1., 2.]])])
prob  = np.array([0.3,0.5,0.2]) 
sigma_eps = .1
prob_eps  = 0.1


z_true = np.zeros(n*m)
x = np.zeros((n*m, len(mu[0])))
index = np.zeros(n*m)
for i in range(m):
    index[n*i:n*(i+1)] = i
    prob_i = prob + np.random.rand(3)*prob_eps
    prob_i /= sum(prob_i)
    z_i = np.argmax(np.random.multinomial(1, prob_i, size=n), 1)
    z_true[n*i:n*(i+1)] = z_i
    x_i  = np.zeros((n,len(mu[0])))
    mu_s = np.zeros_like(mu)
    mu_s[:] = mu[:] +  sigma_eps*np.random.randn(np.prod(mu.shape)).reshape(mu.shape)
    for j in range(len(mu)):
        index_i = z_i == j
        x_i[index_i,:] = np.random.multivariate_normal(mu_s[j],Sigma[j],(np.sum(index_i),))
    
    x[n*i:n*(i+1)] = x_i




mix_obj = mixture_repeat_measurements(x,measurement = index, K=K)
for i in range(m):
    mix_obj.x[i][:] = z_true[index == i] 
    x_i = x[index==i,:]
    z_i = z_true[index==i]
    for k in range(K):
        mix_obj.xxTbar[i][k][:] = np.dot(x_i[z_i==k,:].T,x_i[z_i==k,:])
        mix_obj.xbar[i][k][:] = np.sum(x_i[z_i==k,:],0)
        mix_obj.n_x[i][k]       = sum(z_i==k)

for k in range(K):
    mix_obj.sigma[k]     =  Sigma[k]    
    mix_obj.sigma_eps[k] = np.eye(Sigma[0].shape[0])*sigma_eps
    mix_obj.mu[k]        = mu[k]
 
 
#TODO: add test 
#ADD test:
# FOR n = 500, m= 200
# the first digjit of:
# np.mean(mix_obj.mu_eps,0)-mu should be (close to) zero
mix_obj.sample_mu_eps()

mix_obj.AMCMC = True

#TODO: add test
#ADD test:
# FOR n = 500, m= 200
# the first digjit of:
# np.mean(mix_obj.mu_eps,0)-mu should be (close to) zero
mix_obj.sample_mu_eps()



#ADD test:
# FOR n = 500, m= 200
# the first digjit of:
# mix_obj.mu-mu should be zero
mix_obj.sample_mu_given_mu_eps = False
mix_obj.sample_mu()
#ADD test:
# FOR n = 500, m= 200
# the first digjit of:
# mix_obj.mu-mu should be zero
mix_obj.sample_mu_given_mu_eps = True
mix_obj.sample_mu()

mix_obj.set_AMCMC(100)
mix_obj.compute_ProbX()
mix_obj.sample_x()
mix_obj.AMCMC = False
mix_obj.compute_ProbX()
mix_obj.sample_x()

#TODO: add test
#ADD test:
# FOR n = 500, m= 200
# the first digjit of:
# Sigma vs mix_obj.sigma should be (close to) zero
mix_obj.sample_sigma()

plt.scatter(x[:,0],x[:,1])
