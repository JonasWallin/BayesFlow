'''
Created on Jul 11, 2014

@author: jonaswallin
'''


import time

from mpi4py import MPI
import os
import numpy as np
import BayesFlow as bm
import matplotlib.pyplot as plt

sim = 10
data = []
names = []
if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable
	for file_ in os.listdir("../data/flow_dataset/"):
		if file_.endswith(".dat"):
			data.append(np.ascontiguousarray(np.loadtxt("../data/flow_dataset/" + file_)))
			names.append(file_)
else:
	data = None
hGMM = bm.hierarical_mixture_mpi(K = 10)
hGMM.set_data(data, names)
hGMM.set_prior_param0()
hGMM.update_GMM()
hGMM.update_prior()
tic = MPI.Wtime()  # @UndefinedVariable
t_GMM = 0.
t_rest = 0.
t_load = 0.

for i in range(sim):
	if  hGMM.comm.Get_rank() == 0:
		t0 = time.time()
	for GMM in hGMM.GMMs:
		GMM.sample() 
	
	if  hGMM.comm.Get_rank() == 0:
		t1 = time.time()
		t_GMM += np.double(t1-t0) 
	hGMM.update_prior()
	
	if  hGMM.comm.Get_rank() == 0:
		t2 = time.time()
		t_load += np.double(t2-t1) 
	if hGMM.comm.Get_rank() == 0:
		for k in range(hGMM.K):
			hGMM.normal_p_wisharts[k].sample()
			hGMM.wishart_p_nus[k].sample()
	hGMM.comm.Barrier()
	hGMM.update_GMM()
	if  hGMM.comm.Get_rank() == 0:
		t3 = time.time()
		t_rest += np.double(t3-t2) 


hGMM.set_nuss(10**4)
hGMM.set_nu_mus(10**4)
for i in range(sim):
	hGMM.sample()
if  hGMM.comm.Get_rank() == 0:
	print("theta[1] = %s"%(hGMM.GMMs[1].prior[1]['mu']['theta'].reshape(GMM.d)))
	k = 0

for k in range(hGMM.K):
	for GMM in hGMM.GMMs:
		#print("sigma_mu[1] = %s"%(GMM.prior[1]['mu']['Sigma']))
		#print("nu[1] = %s"%(GMM.prior[1]["sigma"]["nu"]))
	#hGMM.comm.Barrier()	
	#print("sigma[%d] = %s"%(k,GMM.sigma[1]))
	#hGMM.comm.Barrier()	
		print("mu[%d] = %s"%(k,GMM.mu[k]))
#	k +=1
		hGMM.comm.Barrier()
	
if  hGMM.comm.Get_rank() == 0:
	print("hgmm GMM  %.4f sec/sim"%(t_GMM/sim))
	print("hgmm load  %.4f sec/sim"%(t_load/sim))
	print("hgmm rank=0  %.4f sec/sim"%(t_rest/sim))
toc = MPI.Wtime()  # @UndefinedVariable
wct = hGMM.comm.gather(toc-tic, root=0)

if hGMM.comm.Get_rank() == 0:
	for task, time_ in enumerate(wct):
		print('wall clock time: %8.2f seconds (task %d)' % (time_, task))
	def mean(seq): return sum(seq)/len(seq)
	print    ('all tasks, mean: %8.2f seconds' % mean(wct))
	print    ('all tasks, min:  %8.2f seconds' % min(wct))
	print    ('all tasks, max:  %8.2f seconds' % max(wct))
	print    ('all tasks, sum:  %8.2f seconds' % sum(wct))
	print("sim = %d"%sim)
bm.plot.plot_GMM_scatter_all(hGMM,[0, 1])
#plt.show()

for k, GMM in enumerate(hGMM.GMMs):
	print("name[%d] = %s"%(k,GMM.name))