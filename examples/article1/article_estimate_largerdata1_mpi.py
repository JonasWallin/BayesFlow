'''
Created on Jul 11, 2014

@author: jonaswallin
'''
from __future__ import print_function
from __future__ import division
import article_simulatedata
from mpi4py import MPI
from article_util import setup_model, burin_1, burin_2, main_run, HGMM_pre_burnin
import numpy as np
import numpy.random as npr
import sys


npr.seed(10)
K = 11
d = 8
if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable 
	save_data = True
	SIM          = 5000
	SIM_burnin_1 = 100#20
	SIM_burnin_2 = 5000
	N_CELLS = 15*10**4
	THIN = 1
	N_PERSONS = 32*6
	data = {'SIM': SIM, 
		    'N_CELLS': N_CELLS, 
		    'THIN': THIN, 
		    'N_PERSONS': N_PERSONS,
		    'SIM_burnin_1': SIM_burnin_1,
		    'SIM_burnin_2': SIM_burnin_2 }
else:
	data = None
	
SAVE_FIG = 0
y = []

data = MPI.COMM_WORLD.bcast(data, root=0) # @UndefinedVariable 
locals().update(data)
	

	



####
# COLLECTING THE DATA
####
if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable
	y, act_komp, mus, thetas, sigmas, weights, x = article_simulatedata.simulate_data_v2(
														 n_cells = N_CELLS, 
														 n_persons = N_PERSONS,
														 silent = True)
						
else:
	y = None
	act_komp = None


#########################
# Setting up model
########################

#prior = BalancedPrior(N_PERSONS, N_CELLS, d=d, K=K)
#prior.latent_cluster_means(t_inf=None, t_ex=0.5, Sk_ex=1e6)
#prior.component_location_variance(nt=0.3, q=1e-3)
#prior.component_shape(nps=0.1, h=1e3)
#prior.set_noise_class(noise_mu=0.5, noise_Sigma=0.5**2, on=False)  # We do not introduce noise class from start
#prior.pop_size()

hier_gmm = setup_model(y, K)
#hier_gmm.set_prior(prior, init=False)


HGMM_pre_burnin(hier_gmm)
hier_gmm.add_noise_class()
burin_1(hier_gmm, sim = SIM_burnin_1 )
burin_2(hier_gmm, sim = SIM_burnin_2, p_act = [0., 0.] )


simulation_result = main_run(hier_gmm, sim = SIM, p_act = [0,0.])
np.set_printoptions(precision=2)
if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable
	print(simulation_result['mus'][0])

	
hier_gmm.print_timing()




mus_sim = hier_gmm.get_mus()

if MPI.COMM_WORLD.Get_rank() == 0: 
	data_ = [act_komp, mus, thetas, sigmas, weights]
	if save_data:
		print('saveing data')
		sys.stdout.flush()
		np.save('simulation_result.npy', simulation_result) 
		np.save('mus_sim.npy', mus_sim)
		np.save('sim_data.npy', data_) 