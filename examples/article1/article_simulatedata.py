'''
Created on Aug 10, 2014

@author: jonaswallin
'''
from __future__ import print_function
import sys
import numpy as np
import numpy.random as npr
import scipy.linalg as spl
import BayesFlow.PurePython.GMM as GMM
import BayesFlow.PurePython.distribution.wishart as wishart
import BayesFlow.utils as util





def simulate_data_v2(n_cells, n_persons, seed = None, silent = True):
	"""
		simulating a larger data sets for article
	"""

	sigmas  = np.load('../../data/covs_.npy')
	thetas  = np.load('../../data/means_.npy')
	weights = np.load('../../data/weights_.npy')
	weights /= np.sum(weights) 
	
	
	if not silent:
		print("preprocsseing sigma:", end  = '')
		sys.stdout.flush()  
	sigma_theta = []
	for sigma in sigmas:
		var_ = np.sort(np.linalg.eig(sigma)[0])
		z_sigma = var_ * npr.randn(*sigma.shape)
		sigma_theta.append( sigma + np.dot(z_sigma.T,z_sigma)) 

	if not silent:
		print("done")
		sys.stdout.flush()
	
	nu = 100
	ratio_act = np.array([ 1.,  0.95,  0.2,  0.1 ,  0.9,  1,  1,
        1,  1 ,  1,  0.95,  0.99])
	Y, act_Class, mus  = simulate_data_( thetas = thetas,
				 	sigma_theta = sigma_theta, 
				 	sigmas = sigmas,
					weights =  weights,
					nu = nu, 
					ratio_act = ratio_act, 
					n_cells = n_cells, 
					n_persons = n_persons,
					seed = seed,
					silent = silent)
	
	return Y, act_Class, mus , thetas, sigmas, weights

def simulate_data_v1(nCells = 5*10**4, nPersons = 40, seed = 123456, ratio_P =  [1., 1., 0.8, 0.1]):
	"""
		Simulates the data following the instruction presented in the article
	
	"""

	if seed != None:
		npr.seed(seed)
		
		
		
	P = [0.49, 0.3, 0.2 , 0.01 ]
	Thetas = [np.array([0.,0, 0]), np.array([0, -2, 1]), np.array([1., 2, 0]), np.array([-2,2,1.5])]
	Z_Sigma  = [np.array([[1.27, 0.25, 0],[0.25, 0.27, -0.001],[0., -0.001, 0.001]]),
			    np.array([[0.06, 0.04, -0.03],[0.04, 0.05, 0],[-0.03, 0., 0.09]]),
			    np.array([[0.44, 0.08, 0.08],[0.08, 0.16, 0],[0.08, 0., 0.16]]),
			    0.01*np.eye(3)]
	Sigmas = [0.1*np.eye(3), 0.1*spl.toeplitz([2.,0.5,0]),0.1* spl.toeplitz([2.,-0.5,1]),
			  0.1*spl.toeplitz([1.,.3,.3]) ] 
	
	nu = 100
		
	Y, act_Class, mus = simulate_data_(Thetas, Z_Sigma, Sigmas, P, nu = nu, ratio_act = ratio_P, n_cells = nCells, n_persons = nPersons,
				seed = seed)
	
	
	return Y, act_Class, mus, Thetas, Sigmas, P





def simulate_data_( thetas, sigma_theta, sigmas, weights, nu = 100, ratio_act = None, n_cells = 5*10**4, n_persons = 40,
					seed = None, silent = True):
	"""
		simulating data given:
		*thetas*      list of latent means
		*sigma_theta* variation between the means
		*sigmas*      list of latent covariances
		*weights*     list of probabilites
		*nu*          inverse wishart parameter
		*ratio_act*     probabilility that the cluster is active at a person
		*n_cells*     number of cells at a person
		*n_persons*   number of persons
		*seed*        random number generator
	"""
	
	if seed is None:
		npr.seed(seed)
		
		
	K = len(weights)
	dim = thetas[0].shape[0]
	if ratio_act is None:
		ratio_act = np.ones(K )
		
		
	act_class = np.zeros((n_persons, K))
	for i in range(K):
		act_class[:np.int(np.ceil(n_persons * ratio_act[i])), i] = 1.
	Y = []
	
	nu  = 100
	mus = []
	
	
	
	for i in range(n_persons):
		
		if not silent:
			print("setting up person_{i}: ".format(i = i),end = '')
			sys.stdout.flush()
			
		
		mix_obj = GMM.mixture(K = np.int(np.sum(act_class[i, :])))
		theta_temp  = []
		sigma_temp  = []
		for j in range(K):
			if act_class[i, j] == 1:
				theta_temp.append(thetas[j] +  util.rmvn( np.zeros((dim, 1)), sigma_theta[j] ))
				sigma_temp.append(wishart.invwishartrand(nu, (nu - dim - 1) * sigmas[j]))
			else:
				theta_temp.append(np.ones(dim) * np.NAN)
				sigma_temp.append(np.ones((dim,dim)) * np.NAN)
		theta_temp_ = [  theta_temp[aC] for aC in np.where(act_class[i, :] == 1)[0]]
		sigma_temp_ = [  sigma_temp[aC] for aC in np.where(act_class[i, :] == 1)[0]]

		mix_obj.mu = theta_temp_
		mus.append(theta_temp)
		mix_obj.sigma = sigma_temp_
		
		
		p_ = np.array([ (0.2*np.random.rand()+0.9) * weights[aC]  for aC in np.where(act_class[i, :] == 1)[0]]  )
		p_ /= np.sum(p_)
		mix_obj.p = p_
		mix_obj.d = dim
		Y.append(mix_obj.simulate_data(n_cells))
		
		if not silent:
			print("done")
			sys.stdout.flush()
		
	mus = np.array(mus)
	
	return Y, act_class, mus.T
	
if __name__ == "__main__":
	
	
	Y = simulate_data(nCells = 10**2, nPersons = 10)[0]
