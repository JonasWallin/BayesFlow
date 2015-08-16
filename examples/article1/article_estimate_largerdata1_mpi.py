'''
Created on Jul 11, 2014

@author: jonaswallin
'''
from __future__ import division
import article_simulatedata
from mpi4py import MPI
import BayesFlow as bf



SIM = 10
N_CELLS = 15000
THIN = 2
N_PERSONS = 10
SAVE_FIG = 0
y = []


####
# COLLECTING THE DATA
####
if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable
	y, act_komp, mus, thetas, sigmas, weights = article_simulatedata.simulate_data_v2(
														 n_cells = N_CELLS, 
														 n_persons = N_PERSONS,
														 silent = True)
																									 
	
else:
	y = None
	act_komp = None
	#npr.seed(123546)


####
# Setting up model
####
hier_gmm = bf.hierarical_mixture_mpi(K = 4)
hier_gmm.set_data(y)
hier_gmm.set_prior_param0()
hier_gmm.update_GMM()
hier_gmm.update_prior()
hier_gmm.toggle_timing()

for i in range(SIM):
	hier_gmm.sample()
	
hier_gmm.print_timing()