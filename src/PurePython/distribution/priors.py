'''
Created on Jul 5, 2014

@author: jonaswallin
'''
import numpy as np
import numpy.random as npr
from bayesianmixture.utils.gammad import ln_gamma_d

def f_prior_nu(x, nu  =0.01): 
	"""
		deafult prior for nu which is gamma
		returns : log of prior
	"""
	
	return -nu*x


class nu_class(object):
	'''
		Class object for sampling the prior paramaeter of an wishart distribution
	'''


	def __init__(self, nu0 = None, param = None, prior = None, prior_func = None):
		'''
			param - dict with ['Q'] or better ['detQ'] (which containes log det of Q)
			prior is empty
			prior_func - function representing prior, should return log of prior
						if None use expontial prior with lambda 0.01
		'''
		
		self.log2 = np.log(2)
		self.n = 0
		self.d = 0
		if param != None:
			self.set_param(param)
		
		self.nu = nu0
		self.ln_gamma_d = None
		self.acc = 0.
		self.iter = 0.
		self.calc_lik = False
		
		self.sigma = 5
		self.iterations =  5
		if prior_func == None:
			self.prior = {'nu':0.01}
			self.prior_func = f_prior_nu
	
	def set_val(self, nu ):
		
		self.nu = nu
		self.calc_lik = False
		
			
	def set_d(self,d):
		"""
			Set dimension
		"""	
		
		self.calc_lik = False
		self.d = d
		if self.nu == None:
			self.nu = 2*self.d
		self.ln_gamma_d = ln_gamma_d(self.d)
		
	def set_parameter(self,param):
		"""
			param - dict with ['Q'] or better ['detQ'] (which containes log det of Q)
		"""
		
		#print param['Q']
		if 'detQ' not in param:
			self.logDetQ = np.log(np.linalg.det(param['Q']))
		else:
			self.logDetQ = param['detQ']

		
		self.calc_lik = False
			
	
	def set_prior(self, *args):
		"""
			dont have prior for this class
		"""
		pass
	
	def set_data(self, data = None, det_data = None):
		"""	
			data is a list of covariances
			and det_data is the list of the log determinant of data
		
		"""
		self.calc_lik = False	
		self.logDetSigma = 0
		if det_data == None:
			self.n  = len(data)
			for Sigma in data:
				self.logDetSigma += np.log(np.linalg.det(Sigma))
		else:
			self.n = len(det_data)
			
			for det in det_data:
				self.logDetSigma += det


	def set_MH_param(self, sigma = 5, iterations = 5):
		"""
			setting the parametet for the MH algorithm
			
			sigma     -  the sigma in the MH algorihm on the Natural line
			iteration -  number of time to sample using the MH algortihm  
			
		"""
		self.sigma = sigma
		self.iterations  = iterations
	
	def sample(self):
		"""
			Samples a metropolis hastings random walk proposal
			on N^+
		"""
		
		
		
		for i in range(self.iterations):  # @UnusedVariable
			self.sample_()
			
		return self.nu
			
	def sample_(self):
		nu_star = npr.randint(self.nu - self.sigma, self.nu + self.sigma + 1)
		if nu_star == self.nu:
			self.acc += 1
			self.iter += 1
			return
		
		
		
		if nu_star <= self.d + 1:
			self.acc += 0
			self.iter += 1
			return
		
		
		loglik_star = self.loglik(nu_star)
		loglik = self.__call__()
		#print "***********"
		#print "nu = %d"%self.nu
		#print "nus = %d"%nu_star
		#print "loglik_star = %.2f"%loglik_star
		#print "loglik = %.2f"%loglik
		#print "log[Sigma] = %.2f"%self.logDetSigma 
		#print "n*log[Q] = %.2f"%(self.n * self.logDetQ)  
		#print "***********"
		if np.log(npr.rand()) < loglik_star - loglik:
			self.acc += 1
			self.iter += 1	
			self.loglik_val = loglik_star
			self.nu = nu_star		
		
	def __call__(self):
		if self.calc_lik == False:
			self.loglik_val = self.loglik(self.nu)
			self.calc_lik = True		
		
		return self.loglik_val
		
	def loglik(self, nu):
		nud2 = 0.5 * nu
		loglik =- nud2*self.d*self.n*self.log2
		loglik -= self.n * self.ln_gamma_d(nud2)
		loglik -= nud2 * self.logDetSigma
		loglik += nud2 * self.n * self.logDetQ 
		if self.prior_func != None:
			loglik += self.prior_func(nu,**self.prior)
		return loglik
	
		
		