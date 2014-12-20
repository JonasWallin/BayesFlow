'''
Created on Aug 10, 2014

@author: jonaswallin
'''
import numpy as np
import numpy.random as npr
import scipy.linalg as spl
import BayesFlow.PurePython.GMM as GMM
import BayesFlow.PurePython.distribution.wishart as wishart

def simulate_data(nCells = 5*10**4, nPersons = 40, seed = 123456, ratio_P =  [1., 1., 0.8, 0.1]):
	"""
		Simulates the data following the instruction presented in the article
	
	"""

	if seed != None:
		npr.seed(seed)
		
		
	nClass = 4
	dim    = 3
	P = [0.48, 0.3, 0.2 , 0.01 ]
	Thetas = [np.array([0.,0, 0]), np.array([0, -2, 1]), np.array([1., 2, 0]), np.array([-2,2,1.5])]
	Z_Sigma  = [np.array([[1.27, 0.25, 0],[0.25, 0.27, -0.001],[0., -0.001, 0.001]]),
			    np.array([[0.06, 0.04, -0.03],[0.04, 0.05, 0],[-0.03, 0., 0.09]]),
			    np.array([[0.44, 0.08, 0.08],[0.08, 0.16, 0],[0.08, 0., 0.16]]),
			    0.01*np.eye(3)]
	Sigmas = [0.1*np.eye(3), 0.1*spl.toeplitz([2.,0.5,0]),0.1* spl.toeplitz([2.,-0.5,1]),
			  0.1*spl.toeplitz([1.,.3,.3]) ] 
	

		
	act_Class = np.zeros((nPersons,4))
	for i in range(nClass):
		act_Class[:np.ceil(nPersons*ratio_P[i]),i] = 1.
	Y = []
	
	nu  = 100
	mus = []
	for i in range(nPersons):
		mix_obj = GMM.mixture(K = np.int(np.sum(act_Class[i, :])))
		theta_temp  = []
		sigma_temp  = []
		for j in range(nClass):
			if act_Class[i, j] == 1:
				theta_temp.append(Thetas[j] +  npr.multivariate_normal(np.zeros(3), Z_Sigma[j]))
				sigma_temp.append(wishart.invwishartrand(nu, (nu - dim - 1)* Sigmas[j]))
			else:
				theta_temp.append(np.ones(dim)*np.NAN)
				sigma_temp.append(np.ones((dim,dim))*np.NAN)
		theta_temp_ = [  theta_temp[aC] for aC in np.where(act_Class[i, :] == 1)[0]]
		sigma_temp_ = [  sigma_temp[aC] for aC in np.where(act_Class[i, :] == 1)[0]]

		mix_obj.mu = theta_temp_
		mus.append(theta_temp)
		mix_obj.sigma = sigma_temp_
		
		
		p_ = np.array([ (0.2*np.random.rand()+0.9) * P[aC]  for aC in np.where(act_Class[i, :] == 1)[0]]  )
		p_ /= np.sum(p_)
		mix_obj.p = p_
		mix_obj.d = dim
		Y.append(mix_obj.simulate_data(nCells))
	mus = np.array(mus)
	return Y, act_Class, mus.T, Thetas, Sigmas, P
		
if __name__ == "__main__":
	
	
	Y = simulate_data(nCells = 10**2, nPersons = 10)[0]
	print Y