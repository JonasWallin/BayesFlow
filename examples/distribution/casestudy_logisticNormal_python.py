# -*- coding: utf-8 -*-
"""
Testing if the sampler in multinormal_python 
converges towards the correct thing (which it does)

Created on Sun May 24 14:07:19 2015

@author: jonaswallin
"""
from __future__ import division

import numpy as np
#import BayesFlow.PurePython.distribution.logisticMNormal as MMN
from BayesFlow.PurePython.distribution import logisticNormal as MMN
import matplotlib.pyplot as plt
n = 10000
sim = 200
if __name__ == "__main__":
	p = [0.1,0.1,0.5,0.3]
	Y = np.random.multinomial(n, p, size=1)
	MMN_obj = MMN.logisticMNormal()
	MMN_obj.sigma_MCMC = 1
	MMN_obj.set_data(Y)
	A = np.random.randn(len(p) - 1, len(p) - 1)
	MMN_obj.set_prior({"mu":np.zeros(len(p)-1),"Sigma": 1000*np.dot(A.T,A)})
	MMN_obj.set_alpha(np.zeros(len(p)-1))
	MMN_obj.set_AMCMC(batch = 20)
	p_vec = np.zeros((sim,len(p)))
	for i in range(sim):
		MMN_obj.sample()
		p_vec[i,:] = MMN_obj.get_p().reshape(len(p))
		
	print("acc = %d",MMN_obj.accept_mcmc/MMN_obj.count_mcmc)
	plt.plot(p_vec)