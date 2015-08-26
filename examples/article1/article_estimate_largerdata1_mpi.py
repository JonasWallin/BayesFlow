'''
Created on Jul 11, 2014

@author: jonaswallin
'''
from __future__ import division
import article_simulatedata
from mpi4py import MPI
import BayesFlow as bf
from article_util import setup_model, burin_1, burin_2, main_run
import numpy as np
import numpy.random as npr

npr.seed(0)
K = 11

if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable 
	SIM          = 10**3
	SIM_burnin_1 = 10**3
	SIM_burnin_2 = 10**3
	N_CELLS = 15000
	THIN = 2
	N_PERSONS = 20
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
	y, act_komp, mus, thetas, sigmas, weights = article_simulatedata.simulate_data_v2(
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
burin_2(hier_gmm, sim = SIM_burnin_2 )
simulation_result = main_run(hier_gmm, sim = SIM)
np.set_printoptions(precision=2)
if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable
	print(simulation_result['mus'][0])

	
hier_gmm.print_timing()


from matplotlib.pyplot import cm 
import matplotlib.pyplot as plt
color=cm.rainbow(np.linspace(0,1,K))
f, ax = hier_gmm.plot_mus([0,1,2], colors =color, size_point = 40 )
f, ax = hier_gmm.plot_mus([3,4,5], colors =color, size_point = 40 )

if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable
	plt.show()