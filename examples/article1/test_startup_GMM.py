'''
Created on Oct 3, 2015

@author: jonaswallin
'''
from __future__ import print_function
from __future__ import division
import article_simulatedata
from article_util import GMM_pre_burnin, sort_thetas
from BayesFlow import GMM
import numpy as np
import scipy.spatial as ss
N_CELLS   = 100000
N_PERSONS = 1
K = 11

y, act_komp, mus, thetas, sigmas, weights, x = article_simulatedata.simulate_data_v2(
														 n_cells = N_CELLS, 
														 n_persons = N_PERSONS,
														 silent = True)





GMM = GMM.mixture(data= y[0], K = K)
GMM.add_noiseclass(5)
GMM_pre_burnin(GMM)


ss_mat =  ss.distance.cdist( np.transpose(mus[:,:,0]),
						   np.array(GMM.mu)
						    )	
col_index = np.zeros((K), dtype = np.int)
for k in range(K):  # @UnusedVariable
	mu_est_index = np.floor(np.argmin(ss_mat)/K)
	mu_index    = np.argmin(ss_mat)%K
	col_index[mu_est_index] = np.int(mu_index)
	ss_mat[mu_est_index,:] = np.inf
	ss_mat[:,mu_index] = np.inf
#print(sort_thetas(np.array(GMM.mu), mus[0] ))

for k in range(K):
	print('mu[{index}] : '.format(index = k), end='')
	for d in range(mus[:,k,0].shape[0]):
		print('{error:+0.2f} '.format(error = mus[d,k,0] - GMM.mu[col_index[k]][d]) ,end='')
	print('')
	#{error:.2}
	
	