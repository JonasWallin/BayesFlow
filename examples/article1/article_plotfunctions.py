'''
Plot functions for article
Created on Dec 14, 2014

@author: jonaswallin
'''
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.ticker as ticker


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
		
		ax.errorbar(np.array(range(err[0].shape[0]))*ra,Q[1][index],yerr = err,fmt='.',color='black')
		ax.set_xlim([-.01,(err[0].shape[0]-1)*ra+.01])
		ax.plot(np.array([-.01,(err[0].shape[0]-1)*ra+.01]),[0,0],color='gray',alpha=0.2)
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
		
		ax.errorbar(np.array(range(err[0].shape[0]))*ra,Q[1][index],yerr = err,fmt='.',color='black')
		ax.set_xlim([-.01,(err[0].shape[0]-1)*ra+.01])
		ax.plot(np.array([-.01,(err[0].shape[0]-1)*ra+.01]),[0,0],color='gray',alpha=0.2)
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


def plot_mu_abs(mu_vs_theta):
	'''
		figures for displaying the difference between
		using mus and thetas for cluster centers.
		using theta corresponds to pooling the data 
	'''
	d = mu_vs_theta.shape[1]
	K = mu_vs_theta.shape[0]
	y_lim = [0, np.max(mu_vs_theta)+ 0.1 * np.abs(np.max(mu_vs_theta))]
	fig = plt.figure(figsize=(1.5 * d,1.1 * (int(np.ceil(K/4))+1)))
	fig.subplots_adjust(hspace=1.25, wspace=0)
	ra = 0.1	
	for j in range(K):
		xticklabels = []
		ax = plt.subplot2grid((int(np.ceil(K/4))+1, 4), (int(np.round(j/4)) ,j % 4))
		
		for i in range(d):
			xticklabels.append("%d"%(i + 1))
		
		ax.plot(np.array(range(d))*ra, mu_vs_theta[j,:],'o',color='black')
		ax.set_xlim([-.02,(d-1)*ra+.02])
		plt.setp(ax, xticklabels=xticklabels)
		ax.xaxis.set_ticks(np.array(range(d))*ra)
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
		ax.set_xlabel(r"$\boldsymbol{\mu}_{.,%d}$"%(j + 1), fontsize = 14)
		ax.xaxis.set_label_coords(0.5, -0.65) 
	
	return fig

def plot_theta(theta_percentile, theta_percentile_true = None):
	'''
		To plot one need to add the options:
		matplotlib.rc('text', usetex=True)
		matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
  		
  		*theta_percentile_true* if the true mu is observed we can generate true conf 
	
	'''
	spines_to_remove = []
	K = theta_percentile.shape[0]
	d = theta_percentile.shape[1]
	y_lim = [np.min(theta_percentile[:,:,0]) - 0.1*np.abs(np.min(theta_percentile[:,:,0])),np.max(theta_percentile[:,:,2])+ 0.1*np.abs(np.max(theta_percentile[:,:,2]))]
	theta_percentile[:,:,0] = np.abs(theta_percentile[:,:,0] - theta_percentile[:,:,1])
	theta_percentile[:,:,2] = np.abs(theta_percentile[:,:,2] - theta_percentile[:,:,1])
	
	if theta_percentile_true is not None:
		theta_percentile_true[:,:,0] = np.abs(theta_percentile_true[:,:,0] - theta_percentile_true[:,:,1])
		theta_percentile_true[:,:,2] = np.abs(theta_percentile_true[:,:,2] - theta_percentile_true[:,:,1])
	xticklabels  =[]
	
	fig = plt.figure(figsize=(1.5 * d,1.1 * (int(np.ceil(K/4))+1)))
	fig.subplots_adjust(hspace=1.25, wspace=0)
	for j in range(K):
		xticklabels = []
		ax = plt.subplot2grid((int(np.ceil(K/4))+1, 4), (int(np.round(j/4)) ,j % 4))
		
		for i in range(d):
			xticklabels.append("%d"%(i + 1))
		
		err =  [theta_percentile[j,:, 0] ,theta_percentile[j,:,2]]
		ra = 0.1	
		ax.errorbar(np.array(range(d))*ra,theta_percentile[j,:,1],yerr = err,fmt='.',color='black')
		if theta_percentile_true is not None:
			err_true =  [theta_percentile_true[j,:, 0] ,theta_percentile_true[j,:,2]]
			ax.errorbar(np.array(range(d))*ra + 0.1*ra,
					    theta_percentile_true[j,:,1],
					    yerr = err_true,
					    fmt='.',
					    color='red')
		ax.plot(np.array([-.02,(d-1)*ra+.02]),[0,0],color='gray',alpha=0.2)
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
		ax.xaxis.set_label_coords(0.5, -0.65) 
		for spine in spines_to_remove:
			ax.spines[spine].set_visible(False)
	return fig
