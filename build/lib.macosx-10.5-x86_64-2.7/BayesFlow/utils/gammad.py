'''
Created on Jul 3, 2014

@author: jonaswallin
'''
from __future__ import division
import scipy.special as sps
import numpy as np
import time
class ln_gamma_d(object):
	"""
		lookup table for ln( \Gamma_d)
	
	"""
	
	def __init__(self, d = None):
		
		self.res = {}
		self.d = d
	
	def __call__(self, x):
		
		if x not in self.res:
			res = sps.multigammaln(x, self.d)
			self.res[x] = res
		else: 
			res = self.res[x]
			
		return res
	

def test_speed(d, sim):
	xs = np.ceil(np.random.rand(sim)*100) + d
	lg = ln_gamma_d(d=d)
	t0 = time.time()
	y = [lg(x) for x in xs]  # @UnusedVariable
	t1 = time.time()
	string = "lookup            = %.4f msec/sim (sim ,d ) = (%d %d) "%(1000*np.double(t1-t0)/sim, sim, d )
	print string
	t0 = time.time()
	yd = [sps.multigammaln(x,d) for x in xs]  # @UnusedVariable
	t1 = time.time()
	string = "sps.multigammaln =  %.4f msec/sim (sim ,d ) = (%d %d) "%(1000*np.double(t1-t0)/sim, sim, d )
	print string
if __name__ == '__main__':
	
	test_speed(2, 10**3)
	test_speed(2, 10**4)
	test_speed(5, 10**3)
	test_speed(5, 10**4)