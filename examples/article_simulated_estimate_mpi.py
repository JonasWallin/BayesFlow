'''
run with ex: mpiexec -n 10 python article_simulated_estimate_mpi.py
Created on Jul 11, 2014

@author: jonaswallin
'''
from __future__ import division

import time
import scipy.spatial as ss
import article_simulatedata
from mpi4py import MPI
import numpy as np
import BayesFlow as bm
import matplotlib
import matplotlib.pyplot as plt
import numpy.random as npr
import BayesFlow.plot as bm_plot
import matplotlib.ticker as ticker
from article_plotfunctions import plotQ_joint, plotQ, plot_theta
folderFigs = "/Users/jonaswallin/Dropbox/articles/FlowCap/figs/"

sim = 10**5
nCells = 15000
thin = 2
nPers = 80
save_fig = 1
Y = []


####
# COLLECTING THE DATA
####
if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable
	Y,act_komp, mus, Thetas, Sigmas, P = np.array(article_simulatedata.simulate_data(nCells = nCells, nPersons = nPers))
	
else:
	Y = None
	act_komp = None
	#npr.seed(123546)


####
# Setting up model
####
hGMM = bm.hierarical_mixture_mpi(K = 4)
hGMM.set_data(Y)
hGMM.set_prior_param0()
hGMM.update_GMM()
hGMM.update_prior()
hGMM.set_p_labelswitch(1.)
hGMM.set_prior_actiavation(10)
hGMM.set_nu_MH_param(10,200)
for i,GMM in enumerate(hGMM.GMMs):	
	GMM._label  =i
for i in range(min(sim,2000)):
	hGMM.sample()
	

np.set_printoptions(precision=3)

#hGMM.reset_prior()
bm.distance_sort_MPI(hGMM)
hGMM.set_p_activation([0.7,0.7])

if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable 
	theta_sim = []
	Q_sim = []
	nu_sim = []
	Y_sim = []
	Y0_sim = []


##############
#			MCMC PART
##############
##############
#			BURN IN
##############
for i in range(min(np.int(np.ceil(0.1*sim)),8000)):#burn in
	hGMM.sample()
	
if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable 
	mus_vec = np.zeros((len(Y), hGMM.K, hGMM.d))
	actkomp_vec = np.zeros((len(Y), hGMM.K))
	count = 0

hGMM.set_p_labelswitch(.4)
for i in range(sim):#
	
	# sampling the thining 
	for k in range(thin):
		# simulating
		hGMM.sample()
		
		##
		# since label switching affects the posterior of mu, and active_komp
		# it needs to be estimated each time
		##
		labels = hGMM.get_labelswitches()
		if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable 
			for j in range(labels.shape[0]):
				if labels[j,0] != -1:
					mus_vec[j,labels[j,0],:], mus_vec[j,labels[j,1],:] = mus_vec[j,labels[j,1],:], mus_vec[j,labels[j,0],:] 
					actkomp_vec[j,labels[j,0]], actkomp_vec[j,labels[j,1]] = actkomp_vec[j,labels[j,1]], actkomp_vec[j,labels[j,0]] 
	###################
	# storing data
	# for post analysis
	###################	
	mus_ = hGMM.get_mus()
	thetas = hGMM.get_thetas()
	Qs = hGMM.get_Qs()
	nus = hGMM.get_nus()
	if sim - i < nCells * nPers:
		Y_sample = hGMM.sampleY()
	active_komp = hGMM.get_activekompontent()


	if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable 
		print "iter =%d"%i
		count += 1
		mus_vec += mus_
		actkomp_vec += active_komp
		

		theta_sim.append(thetas)
		Q_sim.append(Qs/(nus.reshape(nus.shape[0],1,1)- Qs.shape[1]-1)  )
		nu_sim.append(nus)
		
		# storing the samples equal to number to the first indiviual
		if sim - i < nCells:
			Y0_sim.append(hGMM.GMMs[0].simulate_one_obs().reshape(3))
			Y_sim.append(Y_sample)
	
	


if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable
	actkomp_vec /= count
	mus_vec     /= count
	mus_ = mus_vec

hGMM.save_to_file("/Users/jonaswallin/Dropbox/temp/")
##
#	fixing ploting options
##
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
#hGMM.plot_GMM_scatter_all([0, 1])
mus_colors = ['r','b','k','m']
f, ax = hGMM.plot_mus([0,1,2], colors =mus_colors, size_point = 5 )

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
	col_index = []
	for k in range(hGMM.K):
		col_index.append( np.argmin(ss_mat[k,:]))
	#print col_index
	#####################
	######################
	
	
	theta_sim = np.array(theta_sim)
	Q_sim     = np.array(Q_sim)
	nu_sim    = np.array(nu_sim)
	np.set_printoptions(precision=2)
	perc_theta = []
	perc_Q_vec = []
	for k in range(hGMM.K):
		perc_ = np.percentile(theta_sim[:,col_index[k],:] - Thetas[k],[2.5,50,97.5],axis=0)
		perc_theta.append(np.array(perc_).T)
		#print "%d & %s & %s & %s &  \\hline" %(k, np.mean(theta_sim[:,col_index[k],:],0) - Thetas[k],perc_[0],perc_[1])

		perc_Q = np.percentile(Q_sim[:,col_index[k],:] - Sigmas[k],[2.5,50,97.5],axis=0)
		#print "Q = %s"%(np.mean(Q_sim[:,col_index[k],:],0))
		perc_Q_vec.append(perc_Q)
		theta_string = ""
		Q_string = ""
		
		theta_diff = np.mean(theta_sim[:,col_index[k],:],0) - Thetas[k]
		Q_diff = np.mean(Q_sim[:,col_index[k],:] - Sigmas[k] ,0)
		for d in range(hGMM.d):
			theta_string += " %.2f (%.2f, %.2f) &"%(perc_[1][d], perc_[0][d], perc_[2][d]) 
			for dd in range(hGMM.d):
				Q_string += " %.3f (%.3f, %.3f) &"%(perc_Q[1][d,dd],perc_Q[0][d,dd],perc_Q[2][d,dd] )
			Q_string = Q_string[:-1]
			Q_string +="\\\ \n"
		theta_string = theta_string[:-1] 
		print "theta[%d]= \n%s\n"%(k,theta_string)
		
		print "Q[%d]= \n%s "%(k,Q_string)
		perc_nu  = np.percentile(nu_sim[:,col_index[k]] - 100,[2.5,50,97.5],axis=0)
		print "nu = %.2f (%d, %d)"%(perc_nu[1],perc_nu[0],perc_nu[2])
	
	Y_sim = np.array(Y_sim)
	Y0_sim = np.array(Y0_sim)
	
	for k in range(hGMM.K):
				
		k_ = np.where(np.array(col_index)==k)[0][0]
		print("k_ == %s"%k_)
		mu_k = mus[:,k_,:].T
		#print actkomp_vec[:,col_index[k]]
		index = np.isnan(mu_k[:,0])==False
		ax.scatter(mu_k[index,0],mu_k[index,1],mu_k[index,2], s=50, edgecolor=mus_colors[k],facecolors='none')
	ax.view_init(48,22)
	
	fig_nu = plt.figure(figsize=(6,0.5))
	ax_nu = fig_nu.add_subplot(111)
	for k in range(hGMM.K):
		ax_nu.plot(nu_sim[:,col_index[k]])
	
	f_histY  = bm_plot.histnd(Y_sim,  50, [0, 100], [0,100])
	f_histY0 = bm_plot.histnd(Y0_sim, 50, [0, 100], [0,100])
	f_theta  = plot_theta(np.array(perc_theta))
	figs_Q   = plotQ(perc_Q_vec)
	fig_Q_joint   = plotQ_joint(perc_Q_vec)

np.set_printoptions(precision=4, suppress=True)
for i, GMM in enumerate(hGMM.GMMs):
	#print("p[%d,%d] = %s"%(hGMM.comm.Get_rank(),i,GMM.p))
	hGMM.comm.Barrier()



if MPI.COMM_WORLD.Get_rank() == 0 and save_fig:  # @UndefinedVariable
	print col_index
	fig_nu.savefig(folderFigs + "nus_simulated.eps", type="eps",transparent=True,bbox_inches='tight')
	fig_nu.savefig(folderFigs + "nus_simulated.pdf", type="pdf",transparent=True,bbox_inches='tight')
	f.savefig(folderFigs + "dcluster_centers_simulated.eps", type="eps",transparent=True,bbox_inches='tight')
	f.savefig(folderFigs + "dcluster_centers_simulated.pdf", type="pdf",transparent=True,bbox_inches='tight')
	f_histY.savefig(folderFigs + "hist2d_simulated.eps", type="eps",bbox_inches='tight')
	f_histY.savefig(folderFigs + "hist2d_simulated.pdf", type="pdf",bbox_inches='tight')
	f_histY0.savefig(folderFigs + "hist2d_indv_simulated.eps", type="eps",bbox_inches='tight')
	f_histY0.savefig(folderFigs + "hist2d_indv_simulated.pdf", type="pdf",bbox_inches='tight')
	f_theta.savefig(folderFigs + "theta_simulated.pdf", type="pdf",transparent=True,bbox_inches='tight')
	f_theta.savefig(folderFigs + "theta_simulated.eps", type="eps",transparent=True,bbox_inches='tight')
	fig_Q_joint.savefig(folderFigs + "Qjoint_simulated.pdf", type="pdf",transparent=True,bbox_inches='tight')
	fig_Q_joint.savefig(folderFigs + "Qjoint_simulated.eps", type="eps",transparent=True,bbox_inches='tight')
	for i,f_Q in enumerate(figs_Q):
		f_Q.savefig(folderFigs + "Q%d_simulated.pdf"%(i+1), type="pdf",transparent=True,bbox_inches='tight')
		f_Q.savefig(folderFigs + "Q%d_simulated.eps"%(i+1), type="eps",transparent=True,bbox_inches='tight')
else:
	plt.show()