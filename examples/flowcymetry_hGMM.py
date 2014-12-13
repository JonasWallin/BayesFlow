'''
Created on Jul 10, 2014

@author: jonaswallin
'''
from __future__ import division
import numpy as np
import bayesianmixture as bm
import os
import time
import matplotlib.pyplot as plt
def plot(GMM, ax ,dim):
	data= GMM.data[:,dim]
	x = GMM.x
	cm = plt.get_cmap('gist_rainbow')
	ax.set_color_cycle([cm(1.*i/GMM.K) for i in range(GMM.K)])
	if len(dim) == 2:
		for k in range(GMM.K):
			plt.plot(data[x==k,0],data[x==k,1],'+',label='k = %d'%(k+1))
		

def plot_mu(hGMM, ax, dim):
	
	cm = plt.get_cmap('gist_rainbow')
	ax.set_color_cycle([cm(1.*i/hGMM.K) for i in range(hGMM.K)])
	if len(dim) == 2:
		for k in range(hGMM.K):
			data = np.array([ GMM.mu[k] for GMM in  hGMM.GMMs])
		 	plt.plot(data[:,dim[0]],data[:,dim[1]],'+',label='k = %d'%(k+1))	


def simulate( hGMM, sim):
	for i in range(sim):
		hGMM.sample()
if __name__ == '__main__':
	plt.close('all')
	plt.ion()
	sim  = 200
	sim2 = 500
	data = []
	for file_ in os.listdir("../data/flow_dataset/"):
		if file_.endswith(".dat"):
			data.append(np.ascontiguousarray(np.loadtxt("../data/flow_dataset/" + file_)))
			
	hGMM = bm.hierarical_mixture(K = 8)
	hGMM.set_p_labelswitch(1.)
	hGMM.set_data(data)
	hGMM.set_prior_param0()
	t0 = time.time()
	simulate(hGMM, sim)
	t1 = time.time()

	sim  = sim2
	hGMM.set_nuss(10**4)
	hGMM.set_nu_mus(10**4)
	t0 = time.time()
	for i in range(sim):
		print("i=%d"%i)
		hGMM.sample()
	t1 = time.time()
	print("hgmm sim %.4f sec"%(np.double(t1-t0)/sim))	
	print("hgmm sim %.4f sec"%(np.double(t1-t0)/sim))
	print("theta[1] = %s"%(hGMM.GMMs[0].prior[1]['mu']['theta'].reshape(hGMM.d)))

	for k in range(hGMM.K):
		for GMM in hGMM.GMMs:
			#print("sigma[%d] = %s"%(k,GMM.sigma[1]))
			print("mu[%d] = %s"%(k,GMM.mu[1]))
	if 1:
		if 1:
			for j in range(hGMM.n):
				fig = plt.figure()
				ax = plt.subplot(111)
				plot(hGMM.GMMs[j],ax,[0,1])
				ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
		if 0:
			fig = plt.figure()
			ax = plt.subplot(111)
			plot_mu(hGMM,ax,[0,1])
			ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
		#plt.show(block=False)
