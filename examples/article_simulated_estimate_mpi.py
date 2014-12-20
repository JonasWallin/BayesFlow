'''
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



sim = 10**2
nCells = 10000
thin = 2
nPers = 80
save_fig = 1
Y = []


def plotQ_joint(Qs):
	
	n_Q = len(Qs)
	
	fig = plt.figure(figsize=(6,0.5*n_Q))
	# no space between subplots
	fig.subplots_adjust(hspace=0, wspace=0)
	
	for Q_i,Q_in in enumerate(Qs):
		Q = np.zeros_like(Q_in)
		Q[:] = Q_in[:]
		d = Q[0].shape[0]
		#print Q_i
		ax = fig.add_subplot(411+Q_i)
		Q_min = np.min(Q[0])
		Q_max = np.max(Q[2])
		Q[0] = np.abs(Q[0] - Q[1])
		Q[2] = np.abs(Q[2] - Q[1])

		ra = 0.1
		index = np.triu(np.ones((d,d))) == True
		err = [Q[0][index],Q[2][index]]
		
		ax.errorbar(np.array(range(err[0].shape[0]))*ra,Q[1][index],yerr = err,fmt='.')
		ax.set_xlim([-.01,(err[0].shape[0]-1)*ra+.01])
		ax.plot(np.array([-.01,(err[0].shape[0]-1)*ra+.01]),[0,0],color='r',alpha=0.2)
		ax.xaxis.set_ticks(np.array(range(err[0].shape[0]))*ra)
		a = np.array([range(d),range(d),range(d)])
		b = a.T
		
		a = a[index]
		b = b[index]
		x_tick = []
		if Q_i == n_Q-1 :
			for i in range(len(a)):
				x_tick.append("(%d,%d)"%(a[i] + 1,b[i] + 1))
			
			plt.setp(ax, xticklabels=x_tick)
			ax.xaxis.set_ticks_position('bottom')
		else:
			plt.setp(ax, xticklabels=[])
			
		ax.set_ylabel(r'$\frac{\boldsymbol{\Psi}_%d}{\nu_%d + 3}$'%(n_Q-Q_i, n_Q - Q_i),fontsize=15, rotation='horizontal',ha='right')
		ax.tick_params(labeltop='off', labelright='off')
		ax.xaxis.set_ticks_position('none')
		ax.yaxis.set_ticks_position('none')
		ax.yaxis.tick_right()
		ax.set_ylim([Q_min - 0.1*abs(abs(Q_min)-abs(Q_max)),Q_max + 0.1*abs(abs(Q_min)-abs(Q_max))])
		#print [Q_min - 0.1*max(abs(Q_min),abs(Q_max)),Q_max + 0.1*max(abs(Q_min),abs(Q_max))]
		ax.axes.yaxis.set_ticks(np.linspace(Q_min*0.9,Q_max*0.9,2))
		ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
		print "Q_min,max = (%lf,%lf)"%(Q_min,Q_max)
		print "print Q = %s"%Q[1][index]
		print "print Q_range = %s"%np.linspace(Q_min,Q_max,4)
	return fig

def plotQ(Qs):
	
	figs = []
	for Q_i, Q_in in enumerate(Qs):
		Q = np.zeros_like(Q_in)
		Q[:] = Q_in[:]
		Q_min = np.min(Q[0])
		Q_max = np.max(Q[2])
		Q[0] = np.abs(Q[0] - Q[1])
		Q[2] = np.abs(Q[2] - Q[1])
		
		d = Q[0].shape[0]
		fig = plt.figure(figsize=(4,0.5))
		ax = fig.add_subplot(111)
		ra = 0.1
		index = np.triu(np.ones((d,d))) == True
		err = [Q[0][index],Q[2][index]]
		
		ax.errorbar(np.array(range(err[0].shape[0]))*ra,Q[1][index],yerr = err,fmt='.')
		ax.set_xlim([-.01,(err[0].shape[0]-1)*ra+.01])
		ax.plot(np.array([-.01,(err[0].shape[0]-1)*ra+.01]),[0,0],color='r',alpha=0.2)
		ax.xaxis.set_ticks(np.array(range(err[0].shape[0]))*ra)
		a = np.array([range(d),range(d),range(d)])
		b = a.T
		
		a = a[index]
		b = b[index]
		x_tick = []
		for i in range(len(a)):
			x_tick.append("(%d,%d)"%(a[i] + 1,b[i] + 1))
		plt.setp(ax, xticklabels=x_tick)
		#ax.set_title(r"$\boldsymbol{Q}_{%d}/(\nu_{%d} + %d)$"%(Q_i +1,Q_i +1,d), fontsize = 14)
		ax.tick_params(labeltop='off', labelright='off')
		ax.xaxis.set_ticks_position('none')
		ax.yaxis.set_ticks_position('none')
		ax.yaxis.tick_right()
		
		ax.set_ylim([Q_min - 0.1*abs(abs(Q_min)-abs(Q_max)),Q_max + 0.1*abs(abs(Q_min)-abs(Q_max))])
		ax.axes.yaxis.set_ticks(np.linspace(Q_min,Q_max,2))
		ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
		figs.append(fig)
	return figs

def plot_theta(theta_percentile):
	spines_to_remove = []
	K = theta_percentile.shape[0]
	d = theta_percentile.shape[1]
	y_lim = [np.min(theta_percentile[:,:,0]) - 0.1*np.abs(np.min(theta_percentile[:,:,0])),np.max(theta_percentile[:,:,2])+ 0.1*np.abs(np.max(theta_percentile[:,:,2]))]
	theta_percentile[:,:,0] = np.abs(theta_percentile[:,:,0] - theta_percentile[:,:,1])
	theta_percentile[:,:,2] = np.abs(theta_percentile[:,:,2] - theta_percentile[:,:,1])
	xticklabels  =[]
	
	fig = plt.figure(figsize=(6,0.5))
	fig.subplots_adjust(wspace=0)
	for j in range(K):
		xticklabels = []
		ax = plt.subplot2grid((1,4), (0,j))
		for i in range(d):
			xticklabels.append("%d"%(i + 1))
		
		err =  [theta_percentile[j,:, 0] ,theta_percentile[j,:,2]]
		ra = 0.1	
		ax.errorbar(np.array(range(d))*ra,theta_percentile[j,:,1],yerr = err,fmt='.')
		ax.plot(np.array([-.02,(d-1)*ra+.02]),[0,0],color='r',alpha=0.2)
		ax.xaxis.set_ticks(np.array(range(d))*ra)
		ax.set_xlim([-.02,(d-1)*ra+.02])
		plt.setp(ax, xticklabels=xticklabels)
		ax.tick_params(labeltop='off', labelright='off')
		ax.xaxis.set_ticks_position('none')
		ax.yaxis.set_ticks_position('none')
		ax.set_ylim(y_lim)
		if j < K-1:
			ax.yaxis.set_ticks([])
		else:
			ax.yaxis.tick_right()
			ax.axes.yaxis.set_ticks(np.linspace(y_lim[0],y_lim[1],2))
			ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
		ax.set_xlabel(r"$\boldsymbol{\theta}_{%d}$"%(j + 1), fontsize = 14)
		ax.xaxis.set_label_coords(0.5, -0.75) 
		for spine in spines_to_remove:
			ax.spines[spine].set_visible(False)
	return fig

if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable
	Y,act_komp, mus, Thetas, Sigmas, P = np.array(article_simulatedata.simulate_data(nCells = nCells, nPersons = nPers))
	
else:
	Y = None
	act_komp = None
	#npr.seed(123546)

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
		
		# storing the samples equal to number of observations
		if sim - i < nCells * nPers:
			Y_sim.append(Y_sample)
			theta_sim.append(thetas)
			Q_sim.append(Qs/(nus.reshape(nus.shape[0],1,1)- Qs.shape[1]-1)  )
			nu_sim.append(nus)
		
		# storing the samples equal to number to the first indiviual
		if sim - i < nCells:
			Y0_sim.append(hGMM.GMMs[0].simulate_one_obs().reshape(3))
	
	


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
	
	f_histY  = bm_plot.histnd(Y_sim, 100, [0, 100], [0,100])
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
	fig_nu.savefig("/Users/jonaswallin/Dropbox/articles/FlowCap/figs/nus_simulated.eps", type="eps",transparent=True,bbox_inches='tight')
	fig_nu.savefig("/Users/jonaswallin/Dropbox/articles/FlowCap/figs/nus_simulated.pdf", type="pdf",transparent=True,bbox_inches='tight')
	f.savefig("/Users/jonaswallin/Dropbox/articles/FlowCap/figs/dcluster_centers_simulated.eps", type="eps",transparent=True,bbox_inches='tight')
	f.savefig("/Users/jonaswallin/Dropbox/articles/FlowCap/figs/dcluster_centers_simulated.pdf", type="pdf",transparent=True,bbox_inches='tight')
	f_histY.savefig("/Users/jonaswallin/Dropbox/articles/FlowCap/figs/hist2d_simulated.eps", type="eps",bbox_inches='tight')
	f_histY.savefig("/Users/jonaswallin/Dropbox/articles/FlowCap/figs/hist2d_simulated.pdf", type="pdf",bbox_inches='tight')
	f_histY0.savefig("/Users/jonaswallin/Dropbox/articles/FlowCap/figs/hist2d_indv_simulated.eps", type="eps",bbox_inches='tight')
	f_histY0.savefig("/Users/jonaswallin/Dropbox/articles/FlowCap/figs/hist2d_indv_simulated.pdf", type="pdf",bbox_inches='tight')
	f_theta.savefig("/Users/jonaswallin/Dropbox/articles/FlowCap/figs/theta_simulated.pdf", type="pdf",transparent=True,bbox_inches='tight')
	f_theta.savefig("/Users/jonaswallin/Dropbox/articles/FlowCap/figs/theta_simulated.eps", type="eps",transparent=True,bbox_inches='tight')
	fig_Q_joint.savefig("/Users/jonaswallin/Dropbox/articles/FlowCap/figs/Qjoint_simulated.pdf", type="pdf",transparent=True,bbox_inches='tight')
	fig_Q_joint.savefig("/Users/jonaswallin/Dropbox/articles/FlowCap/figs/Qjoint_simulated.eps", type="eps",transparent=True,bbox_inches='tight')
	for i,f_Q in enumerate(figs_Q):
		f_Q.savefig("/Users/jonaswallin/Dropbox/articles/FlowCap/figs/Q%d_simulated.pdf"%(i+1), type="pdf",transparent=True,bbox_inches='tight')
		f_Q.savefig("/Users/jonaswallin/Dropbox/articles/FlowCap/figs/Q%d_simulated.eps"%(i+1), type="eps",transparent=True,bbox_inches='tight')

if MPI.COMM_WORLD.Get_rank() == 0  and save_fig:	  # @UndefinedVariable
	f_theta  = plt.figure()
	ax_theta = f_theta.add_subplot(111)
	ax_theta.set_title("theta")
	f_Q = plt.figure()
	ax_Q = f_Q.add_subplot(111)
	ax_Q.set_title("Q")
	f_nu = plt.figure()
	ax_nu = f_nu.add_subplot(111)	
	ax_nu.set_title("nu")
	for i in range(Q_sim.shape[1]):
		ax_nu.plot(bm_plot.autocorr(nu_sim[:,i]))
		for ii in range(Q_sim.shape[2]):
			ax_theta.plot(bm_plot.autocorr(theta_sim[:,i,ii]))
			for iii in range(Q_sim.shape[3]):
				ax_Q.plot(bm_plot.autocorr(Q_sim[:,i, ii, iii]))

	f_theta.savefig("/Users/jonaswallin/Dropbox/articles/FlowCap/figs/correlation_theta.pdf", type="pdf",transparent=True,bbox_inches='tight')
	f_Q.savefig("/Users/jonaswallin/Dropbox/articles/FlowCap/figs/correlation_Q.pdf", type="pdf",transparent=True,bbox_inches='tight')
	f_nu.savefig("/Users/jonaswallin/Dropbox/articles/FlowCap/figs/correlation_nu.pdf", type="pdf",transparent=True,bbox_inches='tight')
	
#plt.show()