'''
Created on Aug 10, 2014

@author: jonaswallin
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors

def autocorr(x_in, lag=100):
	"""
		returning the autocorrelation of x_in (1D)
		Lazy implimentation
	"""
	x = np.zeros_like(x_in)
	x[:] = x_in[:]
	x -= np.mean(x)
	n = min(len(x),lag)
	res = np.zeros(n)
	for t in range(n):
		res[t] = np.corrcoef(np.array([x[0:len(x)-t], x[t:len(x)]]))[0,1]
	return res

def histnd(dat, bins, quan = [0.5,99.5], quan_plot = [5, 95], f = None):
	
	nv = np.shape(dat)[1]
	count = 0
	
	if f == None:
		f = plt.figure()
	
	gs = gridspec.GridSpec(nv, nv)
	for i in range(nv):
		ax =  f.add_subplot(gs[nv*i + i])
		count += 1	 
		index_i = (dat[:,i] > np.percentile(dat[:,i], quan[0])) * (dat[:,i] < np.percentile(dat[:,i], quan[1]))
		q1 = np.percentile(dat[index_i ,i], quan_plot[0])
		q2 = np.percentile(dat[index_i ,i], quan_plot[1])
		ax.hist(dat[index_i ,i], bins = bins)
		n_bins = ax.hist(dat[index_i ,i], bins = bins,facecolor='black',edgecolor='black')[0]
		ax.axes.xaxis.set_ticks(np.linspace(q1,q2,num=4))
		ax.axes.yaxis.set_ticks(np.ceil(np.linspace(0,np.max(n_bins),num=4)))
		ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
		ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
		ax.tick_params(axis='both', which='major', labelsize=8)
		xlims = np.percentile(dat[:,i], quan)
		ax.set_xlim(xlims[0],xlims[1])
		for j in range(i+1,nv):
			index_j = (dat[:,j] > np.percentile(dat[:,j], quan[0]) ) * (dat[:,j] < np.percentile(dat[:,j], quan[1]))
			
			q1_y = np.percentile(dat[index_j*index_i ,j], quan_plot[0])
			q2_y = np.percentile(dat[index_j*index_i ,j], quan_plot[1])
			q1_x = np.percentile(dat[index_j*index_i ,i], quan_plot[0])
			q2_x = np.percentile(dat[index_j*index_i ,i], quan_plot[1])
			ax = f.add_subplot(gs[nv*i + j])
			count += 1
			dat_j  = dat[index_i*index_j ,j]
			#dat_j[dat_j == 0] = 1.
			dat_i  = dat[index_i*index_j ,i]
			#dat_i[dat_i == 0] = 1.
			
			ax.hist2d(dat_j, dat_i, bins = bins, norm=colors.LogNorm(),vmin=1)
			ax.patch.set_facecolor('white')
			ax.axes.xaxis.set_ticks(np.linspace(q1_y, q2_y,num=4))
			ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
			ax.axes.yaxis.set_ticks(np.linspace(q1_x, q2_x,num=4))
			ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
			ax.tick_params(axis='both', which='major', labelsize=8)
			
		f.subplots_adjust(wspace = .25)
		f.subplots_adjust(hspace = .25)
	
	return f