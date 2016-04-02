
from numpy cimport ndarray as ar
import numpy as np
cimport numpy as np
cimport cython

from libc.stdlib cimport calloc, malloc, free
cdef extern void Lt_XpZ(const double* L, double* X, const double* Z, const int d) nogil
cdef extern  void inv_c( double *X_inv, const double *X, const int d) nogil
cdef extern void chol_c(double* R, const double* X, const int d) nogil
cdef extern void solve_R_c(const double* R, double* X, const int d, const int n) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
def cholesky(np.ndarray[np.double_t, ndim=2] X):
	"""
		cholesky of X
	""" 
	cdef int d = X.shape[0] 
	cdef np.ndarray[np.double_t, ndim=2] R = np.zeros((d, d))
	chol_c(<double *> &R[0,0], <double *> &X[0,0], d)
	return R 

@cython.boundscheck(False)
@cython.wraparound(False)
def log_det(np.ndarray[np.double_t, ndim=2] X):
	"""
		log(|X|)
	""" 
	cdef int d = X.shape[0]  # @DuplicatedSignature
	cdef np.ndarray[np.double_t, ndim=2] R = np.zeros((d, d))  # @DuplicatedSignature
	chol_c(<double *> &R[0,0], <double *> &X[0,0], d)
	cdef double log_det_X = 0.
	for i in range(d):
		log_det_X	+= np.log(R[i, i])
	log_det_X	 *= 2 
	
	return log_det_X

cdef cholesky_cython(np.ndarray[np.double_t, ndim=2] X):
	"""
		cholesky of X
	""" 
	cdef int d = X.shape[0]  # @DuplicatedSignature
	cdef np.ndarray[np.double_t, ndim=2] R = np.zeros((d, d))  # @DuplicatedSignature
	chol_c(<double *> &R[0,0], <double *> &X[0,0], d)
	return R 

@cython.boundscheck(False)
@cython.wraparound(False)
def solve_R(np.ndarray[np.double_t, ndim=2] X, np.ndarray[np.double_t, ndim=2] R):
	"""
		Solving A^-1 X using the cholesky factoriszation of A
	""" 
	cdef int d = R.shape[0]  # @DuplicatedSignature
	cdef int n = X.shape[1]
	cdef np.ndarray[np.double_t, ndim=2] X_out = np.zeros_like(X)
	X_out[:] = X[:]
	solve_R_c(<double *> &R[0,0], <double *> &X_out[0,0], d, n)
	return X_out

cdef solve_R_cython(np.ndarray[np.double_t, ndim=2] X, np.ndarray[np.double_t, ndim=2] R):
	"""
		Solving A^-1 X using the cholesky factoriszation of A
	""" 
	cdef int d = R.shape[0]  # @DuplicatedSignature
	cdef int n = X.shape[1]  # @DuplicatedSignature
	cdef np.ndarray[np.double_t, ndim=2] X_out = np.zeros_like(X)  # @DuplicatedSignature
	X_out[:] = X[:]
	solve_R_c(<double *> &R[0,0], <double *> &X_out[0,0], d, n)
	return X_out

@cython.boundscheck(False)
@cython.wraparound(False)
def likelihood_prior(np.ndarray[np.double_t, ndim=2] mu, np.ndarray[np.double_t, ndim=2] Theta, theta,np.ndarray[np.double_t, ndim=2]  R_S_mu
					,np.ndarray[np.double_t, ndim=2] R_S, int nu, np.ndarray[np.double_t, ndim=2] Q):
	"""
		part of the function likelihood prior moved to cython
	"""

	cdef int d = Theta.shape[0]
	cdef double log_det_Sigma = 0.
	cdef double log_det_Sigma_mu = 0.
	for i in range(d):
		log_det_Sigma	+= np.log(R_S[i, i])
		log_det_Sigma_mu += np.log(R_S_mu[i, i])
	log_det_Sigma	 *= 2 
	log_det_Sigma_mu *= 2 
	
	cdef np.ndarray[np.double_t, ndim=2] mu_theta = mu.reshape((d,1)) - theta
	# N(\mu; \theta[k], \Sigma[k])
	cdef double lik = 0.
	lik += - np.dot(mu_theta.T, solve_R_cython(mu_theta,R_S_mu))  /2.
	lik = lik - 0.5 * (nu + d + 1.) * log_det_Sigma
	lik = lik - 0.5 * log_det_Sigma_mu
	
	cdef double trace_est = 0.
	cdef np.ndarray[np.double_t, ndim=2] Sigma_inv_Q = solve_R_cython( Q,R_S)
	for i in range(d):
		trace_est += Sigma_inv_Q[i, i]
	lik = lik - 0.5 * trace_est
	return lik




cdef inv(np.ndarray[np.double_t, ndim=2] X):
	"""
		inverse through cholesky of X
	""" 
	cdef int d = X.shape[0]  # @DuplicatedSignature
	cdef np.ndarray[np.double_t, ndim=2] Xout = np.zeros((d, d))
	inv_c(<double *> &Xout[0,0], <double *> &X[0,0], d)
	return Xout 


def inv_sigma_c_cython(np.ndarray[np.double_t, ndim=2] Sigma):
	"""
		for debuging inv_sigma_c
	"""
	cdef int d = Sigma.shape[0]  # @DuplicatedSignature
	cdef np.ndarray[np.double_t, ndim=2] SigmaInvOut = np.zeros((d, d))
	
	inv_sigma_c( <double *> &SigmaInvOut[0,0], <double *> &Sigma[0,0], d)
	return SigmaInvOut

def inv_cython(np.ndarray[np.double_t, ndim=2] X):
	"""
		For debuging inv and inv_c
	""" 
	return(inv(X))

# declare the interface to the C code
cdef extern void draw_x_c(long* x,long* x_index, long* x_count,const double* P,const double* U,const int N, const int K)  nogil

@cython.boundscheck(False)
@cython.wraparound(False)
def draw_x(np.ndarray[long, ndim=1, mode="c"] x not None, np.ndarray[long, ndim=2] x_index,np.ndarray[long, ndim=1] x_count , np.ndarray[double, ndim=2, mode="c"] P not None):
	"""
	draws random sample of the vector P

	param: x	  -- a 1-d array of integer where to store the result
	param: P	  -- the probabilility vector

	"""
	cdef int N, K

	N = x.shape[0]
	K = P.shape[1]
	cdef np.ndarray[double, ndim=1, mode="c"]  U = np.random.rand(N)
	draw_x_c(&x[0], &x_index[0,0], &x_count[0], &P[0,0], &U[0], N, K)

	return None


cdef extern void calc_lik_c(double* lik ,const double* X,const double* Rp, const long int N, const int D, const double log_detQ_div)  nogil

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_lik(np.ndarray[double, ndim=1, mode="c"] lik not None,np.ndarray[double, ndim=2, mode="c"] X not None, np.ndarray[double, ndim=2] Sigma not None):
	"""
		Calculates the logliklihood of multivariate normal
	
	
	"""
	cdef int N = X.shape[0]
	cdef int d = Sigma.shape[0] #@UndefinedVariable
	cdef  np.ndarray[np.double_t, ndim=2]  inv_Sigma = inv(Sigma)
	cdef  np.ndarray[np.double_t, ndim=2]  L = np.empty((d, d))
	chol_c(&L[0,0], &inv_Sigma[0,0], d)
	cdef double *Rp = <double*>calloc(d*(d+1)/2 , sizeof(double))
	cdef int i,j,count = 0
	cdef double log_detQ_div = 0.
	for i in range(d):
		Rp[count] = L[ i , i]
		log_detQ_div += 2 * np.log(Rp[count])
		count += 1
		for j in range(i+1,d):
			Rp[count] = L[ i,j]
			count += 1
	calc_lik_c(&lik[0] , &X[0,0], Rp, N, d, log_detQ_div)
	free(Rp)

cdef extern void calc_exp_normalize_c(double* P,const double* prior, const long int N, const int K, const int K_act, const long int* act_komp)  nogil

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_exp_normalize(np.ndarray[double, ndim=2, mode="c"] P not None,np.ndarray[double, ndim=1] prior not None, np.ndarray[dtype=np.int_t, ndim=1] act_index not None):
	calc_exp_normalize_c(&P[0,0], &prior[0], P.shape[0], P.shape[1], act_index.shape[0], < long int *>act_index.data)
	
	
	
cdef extern void inv_sigma_c(double *sigma_inv,const double*sigma, const int d)  nogil
cdef extern void mult_c(double *res,  const double *sigma_inv,const double* theta,const int d, const double beta) nogil
cdef extern void sample_muc(double *sigma_inv, const double *sigma_mu_inv,double* mu_c, double* r, const double n_d, const int d) nogil


@cython.boundscheck(False)
@cython.wraparound(False)	
def sample_mu(np.ndarray[np.double_t, ndim=2] X, np.ndarray[long, ndim=2] z_index, np.ndarray[long, ndim=1] z_count, np.ndarray[np.double_t, ndim=2] Sigma,
			  np.ndarray[np.double_t, ndim=1] theta,np.ndarray[np.double_t, ndim=2] Sigma_mu,long int k):
	"""
		Sampling the posterior mean given:
		(depricated explenation)
		X		 - (nxd)  the data
		z		 - (n)	the beloning of the data
		Sigma	 - (dxd)  the covariance of the data
		theta	 - (dx1)  the prior mean of mu
		Sigma_mu  - (dx1)  the prior cov of mu
		k		 - (int)  the class
	"""
	
	
	cdef long int n = X.shape[0]
	cdef int d = Sigma.shape[0]
	cdef double *x_sum = <double*>malloc(d * sizeof(double))
	
	cdef int i,j	
	for i in range(d):
		x_sum[i] = 0
	
	cdef double n_d = z_count[k] 
	cdef long int n_  = z_count[k] 
	cdef long *z_data =&z_index[0,0] 
	for i in range(n_):
		index = z_data[i + k*n]
		for j in range(d):
			x_sum[j] += X[index, j]
			
	cdef double *inv_sigma = <double*>malloc(d*d * sizeof(double))
	cdef double *inv_sigma_mu = <double*>malloc(d * d* sizeof(double))
	inv_sigma_c(inv_sigma, &Sigma[0,0],d)
	inv_sigma_c(inv_sigma_mu, &Sigma_mu[0,0],d)


	cdef double *mu_sc = <double*>calloc(d, sizeof(double))
	mult_c(mu_sc,  inv_sigma_mu, &theta[0], d, 0)
	mult_c(mu_sc,  inv_sigma, &x_sum[0], d, 1.)
	free(x_sum)

	cdef np.ndarray[double, ndim=1, mode="c"]  n_01 = np.random.randn(d)
	sample_muc(inv_sigma, inv_sigma_mu, mu_sc,&n_01[0], n_d, d)
	cdef np.ndarray[np.double_t, ndim=1] res = np.empty(d)	   
	for i in range(d):
		res[i] = mu_sc[i]
	
	free(mu_sc)   
	free(inv_sigma)
	free(inv_sigma_mu)
	#del Q
	return res


#	@cython.boundscheck(False)
#	@cython.wraparound(False) 
#	cdef mat invwishartrand_precc(self, double nu, mat & phi)  nogil:
#		return inv(self.wishartrand(nu, inv(phi)))

cdef extern int sigma_update(double *Qstar,const double* Q,const double* X,const long *z_index,const long* z_count, const int k, const int d, const long int n)  nogil
cdef extern void wishartrand(double* phi, const int d , double* X_rand, double* X_out) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
def sample_mix_sigma_zero_mean(np.ndarray[np.double_t, ndim=2]  X,
							   np.ndarray[long, ndim=2] z_index, np.ndarray[long, ndim=1] z_count, int k, np.ndarray[np.double_t, ndim=2]  Q,long int nu):
	"""
		Sampling the posterior covariance given:
		X   - (nxd) the data
		z   - (n)   the class belnoning
		k   - (int) which class
		Q   -(dxd) the prior 
		nu  -(int)  the prior 
	"""
	cdef int d = X.shape[1]
	cdef np.ndarray[np.double_t, ndim=2, mode='c']  Q_star = np.zeros((d, d))
	#cdef np.ndarray[np.double_t, ndim=2]  Q_star = Q +  np.dot(X.transpose(), X)


	cdef int n_count = sigma_update(&Q_star[0,0],&Q[0,0],&X[0,0],&z_index[0,0],&z_count[0],k,d,X.shape[0])
	cdef int nu_star = nu + n_count
	cdef  np.ndarray[np.double_t, ndim=2, mode='c']  inv_Q_star = inv(Q_star)
	cdef np.ndarray[np.double_t, ndim=2, mode='c'] rand_mat = np.zeros((d, d))
		
	cdef int i,j
	for i in range(d):		
		for j in range(i):
			rand_mat[i, j] = np.random.randn(1)
		rand_mat[i,i] =  np.sqrt(np.random.chisquare(nu_star -i ))
	cdef np.ndarray[np.double_t, ndim=2] Xout = np.zeros((d,d))
	cdef np.ndarray[np.double_t, ndim=2] Xout2 = np.zeros((d,d))
	#print('inv_Q_star = {}'.format(inv_Q_star))
	wishartrand( &inv_Q_star[0,0], d, &rand_mat[0,0], &Xout[0,0])
	inv_c( &Xout2[0,0],   &Xout[0,0] ,d)
	#print(Xout2)
	return Xout2