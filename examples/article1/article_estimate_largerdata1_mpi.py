'''
Created on Jul 11, 2014

@author: jonaswallin
'''
from __future__ import print_function
from __future__ import division
import article_simulatedata
from mpi4py import MPI
from article_util import setup_model, burin_1, burin_2, main_run
import numpy as np
import numpy.random as npr
import sys



npr.seed(10)
K = 11

if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable 
	save_data = True
	SIM          = 2
	SIM_burnin_1 = 2
	SIM_burnin_2 = 2
	N_CELLS = 100000
	THIN = 1
	N_PERSONS = 96
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

hier_gmm = setup_model(y, K)



burin_1(hier_gmm, sim = SIM_burnin_1 )
burin_2(hier_gmm, sim = SIM_burnin_2, p_act = [0., 0.] )
simulation_result = main_run(hier_gmm, sim = SIM, p_act = [0,0.])
np.set_printoptions(precision=2)
if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable
	print(simulation_result['mus'][0])

	
hier_gmm.print_timing()




mus_sim = hier_gmm.get_mus()

if MPI.COMM_WORLD.Get_rank() == 0: 
	data_ = [y, act_komp, mus, thetas, sigmas, weights]
	if save_data:
		print('saveing data')
		sys.stdout.flush()
		np.save('simulation_result.npy', simulation_result) 
		np.save('mus_sim.npy', mus_sim)
		np.save('sim_data.npy', data_) 