
#TODO: move inv into into C 

import numpy as np
cimport numpy as np
cimport cython
cdef extern void sum_Y(double* sumY,const double* Y, const long int n, const int d)  nogil
cdef extern void update_mu_Q_sample(double* mu_sample, double *Q_sample, const double* Q_pmu_p, const double* Q,
						const double* Q_p, const double* sumY, const long int n, const int d) nogil
cdef extern void Lt_XpZ(const double* L, double* X, const double* Z, const int d) nogil

import cPickle as pickle

cdef inv(np.ndarray[np.double_t, ndim=2] X):
	"""
		inverse through cholesky of X
	""" 
	cdef int d = X.shape[0] 
	cdef np.ndarray[np.double_t, ndim=2] Xout = np.zeros((d, d))
	inv_c(<double *> &Xout[0,0], <double *> &X[0,0], d)
	return Xout 
	
def rebuild_multivariatenormal(param, prior, data_obj):
	"""
		Used for pickling and unpickling multivariatenormal class
	
	"""
	obj 	    	= multivariatenormal(param, prior)
	obj.n   	    = data_obj['n']
	obj.sumY	    = data_obj['sumY']
	return obj



cdef class  multivariatenormal_regression:
	'''
		Class for sampling posterior distribution of covariates of coeffients in regression.
		The model has the form
		X \sim N( \mu, \Sigma)
		Y_i   \sim N( B_i X, \Sigma_{Y,i}) 
	'''
	
	@cython.boundscheck(False)
	@cython.wraparound(False) 
	def __init__(self, prior = None):
		'''
			prior:
			prior['mu'] = np.array(dim=1)
			prior['Sigma'] = np.array(dim=2)
		'''
		self.n = 0
		if not prior is None:
			self.set_prior(prior)

	def set_prior(self, prior):
		
		self.mu_p = np.empty_like(prior['mu'])
		self.mu_p[:] = prior['mu'][:]
		self.Sigma_p = np.empty_like(prior['Sigma'])
		self.Sigma_p[:] = prior['Sigma'][:]
		
		
		self.Q_p = np.linalg.inv(self.Sigma_p)
		self.Q_pmu_p = np.dot(self.Q_p,self.mu_p)
		self.d = self.Sigma_p.shape[0]
		self.sumY = np.zeros(self.d)
		self.mu_sample = np.empty(self.d)
		self.Q_sample = np.empty((self.d , self.d))		
			
			
			
	@cython.boundscheck(False)
	@cython.wraparound(False)	
	def set_prior0(self, d ):# @DuplicatedSignature
		"""
			Deafult values non informative values
		"""

		prior = {}
		prior['mu'] = np.zeros(d)
		prior['Sigma'] = 10.**6 * np.eye(d)
		self.set_prior(prior)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def set_B(self, np.ndarray[np.double_t, ndim=3, mode='c']  B):
		"""
			sets the regression coeff, typically fixed in regression models
			B       - (d x k x n) numpy vector, the covariates k - dimension of beta 
		"""
		self.B = np.empty_like(B)
		self.B[:] = B[:]
		
	@cython.boundscheck(False)
	@cython.wraparound(False)
	def set_QY(self, np.ndarray[np.double_t, ndim=3, mode='c']  QY):
		"""
			sets the regression coeff, typically fixed in regression models
			QY       - (d x d x n) inverse of the covariance matrix
		"""
		self.QY = np.empty_like(QY)
		self.QY[:] = QY[:]	
		self.SigmaY = None
		
	def compute_QY(self):
		"""
			Computes the inverses of SimgaY
		"""
		
		if self.SigmaY is None:
			raise Exception('SigmaY must exists if QY is tobe used')
	@cython.boundscheck(False)
	@cython.wraparound(False)
	def set_SigmaY(self, np.ndarray[np.double_t, ndim=3, mode='c']  SigmaY):
		"""
			sets the covaraince of the residuals
			sigmaY       - (d x d x n) numpy vector, the covariance of residuals (y-B * X)
		"""
		self.SigmaY = np.empty_like(SigmaY)
		self.SigmaY[:] = SigmaY[:]		
		self.QY = None
		
	@cython.boundscheck(False)
	@cython.wraparound(False)
	def set_Y(self, np.ndarray[np.double_t, ndim=2, mode='c']  Y):
		"""
			Y     - (nxd) numpy vector , the data where n number of observation, d - dimension of data
		"""
		self.Y = np.empty_like(Y)
		self.Y[:] = Y[:]
		self.n  = Y.shape[0]
		
	@cython.boundscheck(False)
	@cython.wraparound(False)		 
	def set_data(self, np.ndarray[np.double_t, ndim=2, mode='c']  Y = None,
				np.ndarray[np.double_t, ndim=3, mode='c']  SigmaY = None,
				np.ndarray[np.double_t, ndim=3, mode='c']  B = None,
				np.ndarray[np.double_t, ndim=3, mode='c']  QY = None):
		"""
			if QY is given SigmaY is not used
			Y       - (nxd) numpy vector , the data where n number of observation, d - dimension of data
			SigmaY  - (d x d x n) numpy vector, the covariance of residuals (y-B * X)
			B       - (d x k x n) numpy vector, the covariates k - dimension of beta 
			QY      - (d x d x n) the inverses of SimgaY
		"""
		if B is not None:
			self.set_B(B)
		if Y is not None:
			self.set_Y(Y)
			
		
		if (SigmaY is not None) and (QY is None):
			self.set_SigmaY(SigmaY)
		
		if QY is not None:
			self.set_QY(QY)
	

	@cython.boundscheck(False)
	@cython.wraparound(False)  
	cdef np.ndarray[np.double_t, ndim=1] _sample(self):
		"""
			internal function sampling X
			return X
		"""
		pass
		
# 		if self.n != 0:
# 			update_mu_Q_sample(<double *>  self.mu_sample.data,
# 							   <double *>  self.Q_sample.data,
# 							   <double *>  self.Q_pmu_p.data,
# 							   <double *>  self.Q.data,
# 							   <double *>  self.Q_p.data, 
# 							   <double*>   self.sumY.data, 
# 							   self.n, 
# 							   self.d)
# 		else:
# 			self.Q_sample = np.linalg.cholesky(self.Q_p)
# 			self.mu_sample =  np.linalg.solve(self.Q_sample, self.Q_pmu_p)
		#cdef np.ndarray[np.double_t, ndim=1] X = np.random.randn(self.d)
		#Lt_XpZ(<double *>  self.Q_sample.data,<double *>  X.data,<double *>  self.mu_sample.data, self.d)
		#return X
	
	@cython.boundscheck(False)
	@cython.wraparound(False)		 
	def sample(self):
		"""
			Sampling from the poserior distribution
			return X
		"""

		return self._sample()
			
			
cdef class  multivariatenormal:
	'''
		Class for sampling from a Multivariate normal distribution on the form
		f(X| Y, \Sigma, \mu_p, \Sigma_p) \propto N(Y; X, \Sigma) N(X; \mu_p, \Sigma_p)
	'''
	cdef public np.ndarray Q_p, Q_pmu_p, Y, mu_p, Sigma, Sigma_p, sumY, Q, mu_sample, Q_sample, Y_outer
	cdef public long int d, n
	
	@cython.boundscheck(False)
	@cython.wraparound(False) 
	def __init__(self, param = None ,prior = None):
		'''
			prior:
			prior['mu'] = np.array(dim=1)
			prior['Sigma'] = np.array(dim=2)
		'''
		self.n = 0
		if not prior is None:
			self.set_prior(prior)
		
		if not param is None:
			self.set_parameter(param)



	def __reduce__(self):
		"""
			method to help pickiling
		"""
		param  = {'Sigma':self.Sigma}
		prior  = {'mu':self.mu_p, 'Sigma':self.Sigma_p}
		data_obj = {'n':self.n, 'sumY':self.sumY}
		return (rebuild_multivariatenormal, (param, prior, data_obj))	
	
	def pickle(self, filename):
		"""
			store object in file
		"""
		f = file(filename, 'wb')
		pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
		f.close()

	@staticmethod
	def unpickle(filename):
		"""
			load object from file
			use:
			
			object = multivariatenormal.unpickle(filename)
		"""
		with file(filename, 'rb') as f:
			return pickle.load(f)	
		
					
	@cython.boundscheck(False)
	@cython.wraparound(False)			 
	def set_parameter(self, parameter):
		"""
			Setting the parameter in the likelihood
		"""
		
		self.Sigma = np.empty_like(parameter['Sigma'])
		self.Sigma[:] = parameter['Sigma'][:]
		self.Q = np.linalg.inv(self.Sigma)
		
	
	@cython.boundscheck(False)
	@cython.wraparound(False) 
	def set_prior(self, prior):
		
		self.mu_p = np.empty_like(prior['mu'])
		self.mu_p[:] = prior['mu'][:]
		self.Sigma_p = np.empty_like(prior['Sigma'])
		self.Sigma_p[:] = prior['Sigma'][:]
		
		
		self.Q_p = np.linalg.inv(self.Sigma_p)
		self.Q_pmu_p = np.dot(self.Q_p,self.mu_p)
		self.d = self.Sigma_p.shape[0]
		self.sumY = np.zeros(self.d)
		self.mu_sample = np.empty(self.d)
		self.Q_sample = np.empty((self.d , self.d))
		
	@cython.boundscheck(False)
	@cython.wraparound(False)	
	def set_prior0(self, d ):# @DuplicatedSignature
		"""
			Deafult values non informative values
		"""

		prior = {}
		prior['mu'] = np.zeros(d)
		prior['Sigma'] = 10.**6 * np.eye(d)
		self.set_prior(prior)
	
	@cython.boundscheck(False)
	@cython.wraparound(False)		 
	def set_data(self, np.ndarray[np.double_t, ndim=2, mode='c']  Y):
		"""
			Y - (nxd) numpy vector
		"""
	
		self.n  = Y.shape[0]
		sum_Y( <double *> self.sumY.data, &Y[0,0], self.n, self.d)
		#self.sumY = self._set_data(Y)
	

	@cython.boundscheck(False)
	@cython.wraparound(False)  
	cdef np.ndarray[np.double_t, ndim=1] _sample(self):
		"""
			return X
		"""
		
		
		if self.n != 0:
			update_mu_Q_sample(<double *> self.mu_sample.data,<double *>  self.Q_sample.data,
							   <double *> self.Q_pmu_p.data,<double *>  self.Q.data,<double *>  self.Q_p.data, 
							   <double*> self.sumY.data ,
							   self.n, 
							   self.d)
		else:
			self.Q_sample = np.linalg.cholesky(self.Q_p)
			self.mu_sample =  np.linalg.solve(self.Q_sample, self.Q_pmu_p)
		cdef np.ndarray[np.double_t, ndim=1] X = np.random.randn(self.d)
		Lt_XpZ(<double *> self.Q_sample.data, <double *> X.data,<double *> self.mu_sample.data, self.d)
		return X
	
	@cython.boundscheck(False)
	@cython.wraparound(False)		 
	def sample(self):
		"""
			return X
		"""

		return self._sample()


cdef extern void outer_Y(double* YYt, const double* Y, const int n, const int d) nogil
cdef extern void wishartrand(double* phi, const int d , double* X_rand, double* X_out) nogil
cdef extern  void inv_c( double *X_inv, const double *X, const int d) nogil


def rebuild_invWishart(param, prior, data_obj):
	"""
		Used for pickling and unpickling invWishart class
	
	"""
	obj 	    	= invWishart(param, prior)
	obj.n   	    = data_obj['n']
	obj.sumY	    = data_obj['sumY']
	obj.Y_outer     = data_obj['Y_outer']
	return obj


cdef class invWishart:
	'''
		Class for sampling from a inverse Wishart where the distribution of 
		f(\Sigma| Y, \theta, Q, \nu)  \propto N(Y; theta, \Sigma) IW(\Sigma; Q, nu)
	'''
	cdef public long int n, nu, d
	cdef public np.ndarray  Q, Q_sample, theta_outer, theta, sumY, Y_outer
	
	@cython.boundscheck(False)
	@cython.wraparound(False)  
	def __init__(self, param = None ,prior = None):  # @DuplicatedSignature
		'''
			prior:
			prior['Q']  = np.array(dim=2)
			prior['nu'] = int
			
			param:
			param['theta'] = np.array(dim = 1)
		'''
		self.d = 0
		self.n = 0 
		if not prior is None:
			self.set_prior(prior)
		
		if not param is None:
			self.set_parameter(param)
		
		
	def __reduce__(self):
		param  = {'theta':self.theta}
		prior  = {'nu':self.nu, 'Q':self.Q}
		data_obj = {'n':self.n, 'sumY':self.sumY, 'Y_outer':self.Y_outer}
		return (rebuild_invWishart, (param, prior, data_obj))	
	
	def pickle(self, filename):
		"""
			store object in file
		"""
		f = file(filename, 'wb')
		pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
		f.close()

	@staticmethod
	def unpickle(filename):
		"""
			load object from file
			use:
			
			object = invWishart.unpickle(filename)
		"""
		with file(filename, 'rb') as f:
			return pickle.load(f)	
				
	@cython.boundscheck(False)
	@cython.wraparound(False)		  
	def set_prior(self,prior):# @DuplicatedSignature
		"""
		
		"""
		self.nu = prior['nu']
		self.Q = np.empty_like(prior['Q'])
		self.Q[:] = prior['Q'][:]
		self.Q_sample = np.empty_like(self.Q)
		self.d = self.Q.shape[0]
		self.sumY = np.empty(self.d)
		self.Y_outer = np.empty((self.d,self.d))


	@cython.boundscheck(False)
	@cython.wraparound(False)	
	def set_prior0(self, d):# @DuplicatedSignature
		"""
			Deafult values non informative values
		"""
		prior = {}
		prior['nu'] = d
		prior['Q'] = (10.**-6)*np.eye(d) 
		self.set_prior(prior)
		
		
	@cython.boundscheck(False)
	@cython.wraparound(False)	
	def set_parameter(self, parameter): # @DuplicatedSignature
		"""
			parameter should be a dict with 'theta'
		
		"""
		self.theta = np.empty_like(parameter['theta'])
		self.theta[:] = parameter['theta'][:]
		self.theta_outer = np.outer(self.theta, self.theta)

	@cython.boundscheck(False)
	@cython.wraparound(False)  
	def set_data(self, np.ndarray[np.double_t, ndim=2, mode='c'] Y, np.ndarray[np.double_t, ndim=1] sumY = None): # @DuplicatedSignature
		"""
			Y - (nxd) numpy vector
		"""
		self.n = Y.shape[0]
		if sumY is None:
			sum_Y( <double *> self.sumY.data, &Y[0,0], self.n, self.d)
		else:
			self.sumY = sumY 
		

		outer_Y(<double *> self.Y_outer.data, <double *>  &Y[0,0], self.n, self.d)
		#self.set_Y_outer(Y)
		
	
	

	
	
	@cython.boundscheck(False)
	@cython.wraparound(False)   
	def sample(self): # @DuplicatedSignature
		return self._sample()
		
	@cython.boundscheck(False)
	@cython.wraparound(False)  
	cdef _sample(self):# @DuplicatedSignature
		"""
			return X
		"""
		self.Q_sample[:] = self.Q[:] 
		
		if self.n != 0:
			self.Q_sample += self.n * self.theta_outer
			Temp = np.outer(self.sumY, self.theta)
			self.Q_sample -= Temp + Temp.T
			self.Q_sample += self.Y_outer
		nu_sample = self.n + self.nu 
		cdef np.ndarray[np.double_t, ndim=2, mode='c'] R  =  np.empty((self.d,self.d))
		inv_c(<double *> &R[0,0], <double *>  self.Q_sample.data ,self.d)
		cdef np.ndarray[np.double_t, ndim=2] X = np.zeros((self.d, self.d))
		
		cdef int i,j
		for i in range(self.d):		
			for j in range(i):
				X[i, j] = np.random.randn(1)
			X[i,i] =  np.sqrt(np.random.chisquare(nu_sample -i ))
			
			
		cdef np.ndarray[np.double_t, ndim=2] Xout = np.empty((self.d,self.d))
		wishartrand(<double *> &R[0,0], self.d,<double *>  &X[0,0],<double *>  &Xout[0,0])
		inv_c(<double *> &R[0,0], <double *>  &Xout[0,0] ,self.d)
		return R
	



def rebuild_Wishart(param, prior, data_obj):
	"""
		Used for pickling and unpickling invWishart class
	
	"""
	obj 	    	= Wishart(param, prior)
	obj.n   	    = data_obj['n']
	obj.Q    	    = data_obj['Q']
	return obj
	
cdef class Wishart:
	
	'''
		Class for sampling from a Wishart where the distribution of 
		f(Q;  \Sigma, \Sigma_s, \nu_s,\nu)  \propto W(\Sigma; Q, \nu) IW(Q; \nu_s, Q_s)
	'''
	cdef public long int n, d, nu, nu_s
	cdef public np.ndarray Q_s, Q  # @DuplicatedSignature

	
	def __init__(self, param = None, prior =None):# @DuplicatedSignature
		
		self.n   = 0
		self.d   = 0
		self.nu  = 0
		self.nu_s = 0
		if not param is None:
			self.set_parameter(param)
		
		if not prior is None:
			self.set_prior(prior)


	def __reduce__(self): # @DuplicatedSignature
		param  = {'nu':self.nu}
		prior  = {'nus':self.nu_s, 'Qs':self.Q_s}
		data_obj = {'n':self.n, 'Q':self.Q}
		return (rebuild_Wishart, (param, prior, data_obj))	
	
	def pickle(self, filename): # @DuplicatedSignature
		"""
			store object in file
		"""
		f = file(filename, 'wb')
		pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
		f.close()

	@staticmethod
	def unpickle(filename): # @DuplicatedSignature
		"""
			load object from file
			use:
			
			object = Wishart.unpickle(filename)
		"""
		with file(filename, 'rb') as f:
			return pickle.load(f)	

			
	@cython.boundscheck(False)
	@cython.wraparound(False)	
	def set_parameter(self, param):# @DuplicatedSignature
		
		self.nu = param['nu']
	
	def set_prior(self, prior):# @DuplicatedSignature
		
		self.nu_s    = prior['nus']
		self.Q_s     = np.empty_like(prior['Qs'])
		self.Q_s[:]  = prior['Qs'][:]
		self.d       = self.Q_s.shape[0]


	@cython.boundscheck(False)
	@cython.wraparound(False)	
	def set_prior0(self, d):# @DuplicatedSignature
		"""
			Deafult values non informative values
		"""
		prior = {}
		prior['nus'] = d
		prior['Qs']  = np.eye(d)*10.**-6 
		self.set_prior(prior)
		
	@cython.boundscheck(False)
	@cython.wraparound(False) 
	def set_data(self, Sigmas = None, Qs =None):# @DuplicatedSignature
		"""
			Sigma is a list containg sigmas
		"""
		
		if Qs is None:
			self.n = len(Sigmas)
			self.Q = np.zeros((self.d, self.d))
			for Sigma in Sigmas:
				self.Q[:] += inv(Sigma)[:]
		else:
			self.n = len(Qs)
			self.Q = np.zeros((self.d, self.d))
			for Q in Qs:
				self.Q[:] += Q[:]



	@cython.boundscheck(False)
	@cython.wraparound(False) 	
	def sample(self): # @DuplicatedSignature
		
		return self._sample()

	@cython.boundscheck(False)
	@cython.wraparound(False) 	
	cdef _sample(self):
		
		cdef np.ndarray[np.double_t, ndim=2] Q_Q_inv  = np.zeros_like(self.Q_s)
		
		if self.n > 0:
			Q_Q_inv[:] += self.Q[:]
			
		Q_Q_inv[:] += self.Q_s[:]
		cdef int nu = self.nu_s + self.n * self.nu
		cdef np.ndarray[np.double_t, ndim=2] X     = np.zeros((self.d,self.d))# np.random.randn(self.d,self.d)  # @DuplicatedSignature
		cdef np.ndarray[np.double_t, ndim=2] Xout = np.empty((self.d,self.d))  # @DuplicatedSignature
		cdef int i  # @DuplicatedSignature
		for i in range(self.d):
			for j in range(i):
				X[i,j] = np.random.randn(1)
			X[i,i] =  np.sqrt(np.random.chisquare(nu -i ))
		cdef  np.ndarray[np.double_t, ndim=2]  iQ_Q_inv = inv(Q_Q_inv)

		wishartrand(<double *> &iQ_Q_inv[0,0], self.d, <double *>  &X[0,0],<double *>  &Xout[0,0])

		return Xout