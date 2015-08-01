# -*- coding: utf-8 -*-
"""
Testing the speed of various components in the logisticNormal class

Created on Sun May 24 14:07:19 2015


cython run:
    298321 function calls in 0.450 seconds

Pure python version:
%prun main():
    338321 function calls in 0.486 seconds

To examin indivual time use for example:

%lprun -f MMN_obj.update_llik main()


@author: jonaswallin
"""
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import BayesFlow.PurePython.distribution.logisiticNormal as MMN
#import BayesFlow.distribution.logisticNormal as MMN#
n = 10000
sim = 2000
p = [0.1,0.1,0.5,0.3,0.8,0.05,0.01]
p = p / np.sum(p)
Y = np.random.multinomial(n, p, size=1)
MMN_obj = MMN.logisticMNormal()
MMN_obj.sigma_MCMC = 1
MMN_obj.set_data(Y)
A = np.random.randn(len(p) - 1, len(p) - 1)
MMN_obj.set_prior({"mu":np.zeros(len(p)-1),"Sigma": 1000*np.dot(A.T,A)})
MMN_obj.set_alpha(np.zeros(len(p)-1))
MMN_obj.set_AMCMC(batch = 20)
def main():
	p_vec = np.zeros((sim,len(p)))
	for i in range(sim):
		MMN_obj.sample()
		p_vec[i,:] = MMN_obj.get_p()
	return p_vec	
if __name__ == "__main__":
	
	
	
 p_vec = main()
 plt.plot(p_vec)		
	