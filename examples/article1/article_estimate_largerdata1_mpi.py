'''
run with ex: mpiexec -n 10 python article_simulated_estimate_mpi.py
Created on Jul 11, 2014

@author: jonaswallin
'''
from __future__ import division
import sys
import article_simulatedata
from mpi4py import MPI
import numpy as np
folderFigs = "/Users/jonaswallin/Dropbox/articles/FlowCap/figs/"

sim = 10**2
N_CELLS = 15000
thin = 2
N_PERSONS = 800
save_fig = 0
Y = []


####
# COLLECTING THE DATA
####
if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable
    Y,act_komp, mus, Thetas, Sigmas, P = article_simulatedata.simulate_data_v2(
                                                         n_cells = N_CELLS, 
                                                         n_persons = N_PERSONS,
                                                         silent = False)
                                                                                                     
    
else:
    Y = None
    act_komp = None
    #npr.seed(123546)


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10, 10))
import BayesFlow.plot as bm_plot
print("enerting histnd")  
sys.stdout.flush()  
f_ = bm_plot.histnd(Y[0][:,:5],200,[0, 100],[0,100], f = fig)
f_ = bm_plot.histnd(Y[1][:,:5],200,[0, 100],[0,100], f = fig)