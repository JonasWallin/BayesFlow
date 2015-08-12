'''
Test for repeated measurement test, where one has missing components of the largest class in
every second repetation
Created on Jul 2, 2015

@author: jonaswallin
'''

#TODO: run code with actibe_comp until works
#TODO: create test if one can detect very simple missing components
import numpy as np
import matplotlib.pyplot as plt
from BayesFlow.PurePython.GMMrep import mixture_repeat_measurements


np.random.seed(1)

n  = 200 # number of cells per repeated experiment
m  = 4 #number of experiment 




mu	= np.array([[1. , 2.], [4.,0.],[-2.,1]])
Sigma = 0.1*np.array([np.eye(2), np.array([[2.,1],[1, 2.]]), np.array([[2.,-1.],[-1., 2.]])])
prob  = np.array([0.3,0.5,0.2]) 
sigma_eps = .1
prob_eps  = 0.1
K = len(prob)

z_true = np.zeros(n*m)
x = np.zeros((n*m, len(mu[0])))
index = np.zeros(n*m)
# simulating the data
for i in range(m):
	index[n*i:n*(i+1)] = i
	prob_i = prob + np.random.rand(3)*prob_eps
	if i % 2 == 0:
		prob_i[1] = 0.
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
	
# setting up the bascis of the object
mix_obj = mixture_repeat_measurements(x,measurement = index, K=K)
for i in range(m):
	mix_obj.x[i][:] = z_true[index == i] 
	x_i = x[index==i,:]
	z_i = z_true[index==i]
	for k in range(K):
		mix_obj.xxTbar[i][k][:] = np.dot(x_i[z_i==k,:].T,x_i[z_i==k,:])
		mix_obj.xbar[i][k][:] = np.sum(x_i[z_i==k,:],0)
		mix_obj.n_x[i][k]	   = sum(z_i==k)

for k in range(K):
	mix_obj.sigma[k]	 =  Sigma[k]	
	mix_obj.sigma_eps[k] = np.eye(Sigma[0].shape[0])*sigma_eps
	mix_obj.mu[k]		= mu[k]
	
f, axarr = plt.subplots(2, 2)
col = ['r','g','b']
for i in range(4):
	x_i = x[index==i,:]
	z_i = z_true[index==i]
	m = np.int(i < 2)
	for k in range(K):
		axarr[m, i % 2].scatter(x_i[z_i == k,0],x_i[z_i == k,1],c = col[k])
	
