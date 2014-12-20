# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 22:58:48 2014

@author: jonaswallin
"""
from __future__ import division

import scipy.spatial as ss
import article_simulatedata
from mpi4py import MPI
import BayesFlow.plot as bm_plot
import numpy as np
import BayesFlow as bm
import matplotlib.pyplot as plt
import numpy.random as npr
sim0  = 50
nCells = 1000
nPers = 20
save_fig = 1
Y = []
if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable
	Y,act_komp, mus, Thetas, Sigmas, P = np.array(article_simulatedata.simulate_data(nCells = nCells, nPersons = nPers, ratio_P = [1.,1.,1.,0.]))
	
else:
	Y = None
	act_komp = None
	npr.seed(123546)



hGMM = bm.hierarical_mixture_mpi(K = 3)
hGMM.set_data(Y)
hGMM.set_prior_param0()
hGMM.update_GMM()
hGMM.update_prior()
hGMM.set_p_labelswitch(1.)
hGMM.set_prior_actiavation(10)
hGMM.set_nu_MH_param(10,60)
for i,GMM in enumerate(hGMM.GMMs):	
	GMM._label  =i
for i in range(sim0):
	hGMM.sample()

if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable
	col_index = range(hGMM.K)
	cm = plt.get_cmap('gist_rainbow')
	if 1:
		f = bm_plot.histnd(Y[0],50,[0, 100],[0, 100])
		f.savefig("/Users/jonaswallin/Dropbox/talks/flowcym_chalmers/figs/sim_hist_indv.pdf", type="pdf",bbox_inches='tight')
		f = bm_plot.histnd(np.vstack(Y),100,[0, 100],[0, 100])
		f.savefig("/Users/jonaswallin/Dropbox/talks/flowcym_chalmers/figs/sim_hist_joint.pdf", type="pdf",bbox_inches='tight')
		f = plt.figure()
		ax = f.gca(projection='3d')
		for k in range(hGMM.K):
			mu_k = mus[:,k,:].T
			index = np.isnan(mu_k[:,0])==False
			ax.scatter(mu_k[index,0],mu_k[index,1],mu_k[index,2], s=50, edgecolor=cm(col_index[k]/hGMM.K),facecolors='none')	
		ax.view_init(48,22)
		f.savefig("/Users/jonaswallin/Dropbox/talks/flowcym_chalmers/figs/mu0.pdf", type="pdf",bbox_inches='tight')
		

hGMM.set_nuss(10000)
for i in range(min(sim0,2000)):
	hGMM.sample()	
f, ax = hGMM.plot_mus([0,1,2], size_point = 150)
mus_ = hGMM.get_mus()
if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable
	######################
	#ordering mus
	mus_true_mean = []
	mus_mean = []
	for k in range(hGMM.K):
		mus_true_mean.append(np.array(np.ma.masked_invalid(mus[:,k,:]).mean(0)))
		mus_mean.append(np.array(np.ma.masked_invalid(mus_[:,k,:].T).mean(0)))
	mus_true_mean =  np.array(mus_true_mean)
	mus_mean =  np.array(mus_mean)
	ss_mat =  ss.distance.cdist( mus_true_mean, mus_mean, "euclidean")
	#print ss_mat
	
	if True:
		for k in range(hGMM.K):
						mu_k = mus[:,k,:].T
						index = np.isnan(mu_k[:,0])==False
	
						ax.scatter(mu_k[index,0],mu_k[index,1],mu_k[index,2], s=50, edgecolor=cm(col_index[k]/hGMM.K),facecolors='none')
	ax.view_init(48,22)
	
	#f.savefig("/Users/jonaswallin/Dropbox/talks/flowcym_chalmers/figs/sim_high_nu.pdf", type="pdf",bbox_inches='tight')