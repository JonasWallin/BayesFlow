# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 14:04:17 2015

@author: jonaswallin
"""
import numpy as np
import scipy.spatial as ss
from article_util import sort_mus, sort_thetas
from article_plotfunctions import  plot_theta, plot_mu_abs
import matplotlib.pyplot as plt
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import cm 
import scipy.stats
q_ = scipy.stats.norm.ppf(0.975)


matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
res=np.load('/Users/jonaswallin/repos/BaysFlow/script/sim_data.npy')
#res=np.load('/Users/jonaswallin/repos/BaysFlow/examples/article1/sim_data_v1.npy')
#res=np.load('/Users/jonaswallin/repos/BaysFlow/examples/article1/sim_data.npy')
theta = res[2]
mus = res[1]
#simulation_result = np.load('simulation_result.npy').item()
#mus_sim = np.load('mus_sim.npy')
simulation_result=np.load('/Users/jonaswallin/repos/BaysFlow/script/simulation_result.npy').item()
mus_sim =  np.load('/Users/jonaswallin/repos/BaysFlow/script/mus_sim.npy')


K = theta.shape[0]
d = theta.shape[1]
color=cm.rainbow(np.linspace(0,1,K))

# index sorting the thetas
col_index = sort_thetas( np.array(simulation_result['theta']), theta)

# get the quantiles of theta
perc_theta = []
theta_optim= []
perc_Q = []
for k in range(K):
		k_pos = np.where(np.array(col_index)==k)
		
		if(len(k_pos[0]) > 0):
			k_ = k_pos[0][0]
			mu_k = mus[:,k,:].T
			#std_theta = np.max(np.abs(np.array(simulation_result['theta'])[:,col_index[k],:]- theta[k]),0)
			std_theta =  np.std(simulation_result['mus'][:,col_index[k],:],0)
			theta_err = (np.array(simulation_result['theta'])[:,col_index[k],:] - theta[k])/std_theta
			perc_ = np.percentile(theta_err,[2.5,50,97.5],axis=0)
			perc_theta.append(np.array(perc_).T)
			temp = [ np.mean(mu_k,0) - theta[k] - q_ * np.std(mu_k,0)/np.sqrt(mu_k.shape[0]),
				     np.mean(mu_k,0) - theta[k],
				     np.mean(mu_k,0) - theta[k] + q_ * np.std(mu_k,0)/np.sqrt(mu_k.shape[0])
			]
			theta_optim.append((np.array(temp)/std_theta).T)

			# Q analysis
			perc_Q_temp = np.zeros((d,3))
			for j in range(d):
				Q_err = np.array(simulation_result['Q'])[:,col_index[k],j,j] - res[3][k,j,j]
				Q_err /= np.std(Q_err)
				perc_ = np.percentile(Q_err,[2.5,50,97.5],axis=0)
				perc_Q_temp[j,:] = perc_
			perc_Q.append(perc_Q_temp)

f_theta  = plot_theta(np.array(perc_theta), np.array(theta_optim))
f_theta.savefig('f_theta.pdf',type='pdf',transparent=True,bbox_inches='tight')

fig_mu_ = plt.figure()
ax_mu_ = fig_mu_.gca(projection='3d')
fig_mu1 = plt.figure()
ax_mu1  = fig_mu1.gca(projection='3d')

fig_mu2 = plt.figure()
ax_mu2  = fig_mu2.gca(projection='3d')

fig_mu3 = plt.figure()
ax_mu3  = fig_mu3.gca(projection='3d')
mu_vs_theta = []
mu_error= []
theta_error = []
for k in range(K):
	
	mu_k = mus[:,k,:]
	mu_k_est =  simulation_result['mus'][:,col_index[k],:]
	act_     =	simulation_result['actkomp'][:,col_index[k]] > 0.9
	mu_k_est = mu_k_est[act_,:]
	
	
	index = np.isnan(mu_k[0,:])==False

	mu_theta_diff = np.abs(mu_k_est - mu_k[:, index].T)/np.abs(mu_k[:, index].T - theta[k])
	mu_error.append(np.abs(mu_k_est - mu_k[:, index].T))
	theta_error.append(np.abs(theta[k] - mu_k[:, index].T))
	mu_vs_theta.append(np.transpose(np.percentile(mu_theta_diff,[2.5,50,97.5],axis=0) ) )
	ax_mu_.scatter((mu_k[0, index]- mu_k_est[:,0])   ,
                (mu_k[1, index] - mu_k_est[:,1])  ,
                ( mu_k[2, index] - mu_k_est[:,2]) , s=1,edgecolor=color[k], facecolors='none')
	ax_mu1.scatter(mu_k[0, index], mu_k[1, index], mu_k[2, index], s=50, edgecolor=color[k], facecolors='none')
	ax_mu1.scatter(mu_k_est[:, 0], mu_k_est[:,1], mu_k_est[:,2], s=1, edgecolor=color[k], facecolors='none')
	
	ax_mu2.scatter(mu_k[3, :], mu_k[4, index], mu_k[5, index], s=50, edgecolor=color[k], facecolors='none')
	ax_mu2.scatter(mu_k_est[:,3], mu_k_est[:,4], mu_k_est[:,5], s=1, edgecolor=color[k], facecolors='none')
	ax_mu3.scatter(mu_k[4, index], mu_k[5, index], mu_k[7, index], s=50, edgecolor=color[k], facecolors='none')
	ax_mu3.scatter(mu_k_est[:,4], mu_k_est[:,5], mu_k_est[:,7], s=1, edgecolor=color[k], facecolors='none')

fig_mu1.savefig('f1_1.pdf', type="pdf",transparent=True,bbox_inches='tight')
fig_mu2.savefig('f1_2.pdf', type="pdf",transparent=True,bbox_inches='tight')
fig_mu3.savefig('f1_3.pdf', type="pdf",transparent=True,bbox_inches='tight')	
fig_mu_.savefig('st1_3.pdf', type="pdf",transparent=True,bbox_inches='tight')	

fig4 = plt.figure()
plt.scatter(np.array(range(1,K+1)),np.array(range(1,K+1)),color=color,s=50)
fig4.savefig('color_reminder.pdf', type="pdf",transparent=True,bbox_inches='tight')
#fig_vs = plot_mu_abs(np.array(mu_vs_theta)[:,:,1])
#fig_vs.savefig('mu_vs_theta.pdf', type="pdf",transparent=True,bbox_inches='tight')