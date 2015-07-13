from __future__ import division
from sklearn import mixture as skmixture
from BayesFlow.utils import dat_util
from BayesFlow.utils import mpiutil
import numpy as np
import scipy.stats as stats

class LazyProperty(object):

    def __init__(self, func):
        self._func = func
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__

    def __get__(self, obj, cls=None):
        if obj is None: 
            return None
        result = obj.__dict__[self.__name__] = self._func(obj)
        return result

class WeightsMPI(object):

	def __init__(self,comm,weights):
		self.comm = comm
		self.rank = comm.Get_rank()
		self.weights = weights

	@LazyProperty
	def ws_loc(self):
		return [sum(w) for w in self.weights]

	@LazyProperty
	def W_loc(self):
		return sum([sum(w) for w in self.weights])

	@LazyProperty
	def W_locs(self):
		return self.comm.gather(self.W_loc)

	@LazyProperty
	def W(self):
		return sum(self.comm.bcast(self.W_locs))

	def sample_from_all(self,N):
		W = self.W
		W_locs = self.W_locs
		if self.rank == 0:
			alpha = np.sort(W*np.random.random(N))
			ind_W = [0]+list(np.searchsorted(alpha,np.cumsum(W_locs)))
			W_cum = 0.
			for i in range(self.comm.Get_size()):
				alpha_i = alpha[ind_W[i]:ind_W[i+1]] - W_cum
				W_cum += W_locs[i]
				self.comm.send(alpha_i,i)
		alpha_loc = self.comm.recv()
		#print "alpha_loc retrieved"
		ind_w = [0]+list(np.searchsorted(alpha_loc,np.cumsum(self.ws_loc)))
		w_cum = 0.
		indices = []
		for i,weights_samp in enumerate(self.weights):
			alpha_samp = alpha_loc[ind_w[i]:ind_w[i+1]] - w_cum
			w_cum += self.ws_loc[i]
			indices.append(self.retrieve_ind_at_alpha(weights_samp,alpha_samp))
		return indices

	@staticmethod
	def retrieve_ind_at_alpha(weights,alpha):
		i = 0
		j = 0
		indices = np.empty(len(alpha),dtype='i')
		if len(alpha) == 0:
			return indices
		cumweight = weights[0]
		while i < len(alpha):
			if alpha[i] <= cumweight:
				indices[i] = j
				i += 1
			else:
				j += 1
				cumweight += weights[j]
		return indices

class DataMPI(object):

	def __init__(self,comm,data=None):
		self.comm = comm
		self.rank = comm.Get_rank()
		if not data is None:
			self.data = data
		else:
			self.data = []

	@property
	def J_loc(self):
		return len(self.data)

	@property
	def J(self):
		return(sum(self.comm.bcast(self.comm.gather(self.J_loc))))

	@property
	def d(self):
		return self.data[0].shape[1]

	@property
	def n_j_loc(self):
		return [dat.shape[0] for dat in self.data]

	@property
	def N_loc(self):
		return sum(self.n_j_loc)

	@property
	def n_j(self):
		n_j = self.comm.gather(self.n_j_loc)
		if self.rank == 0:
			n_j = [nj for nj_w in n_j for nj in nj_w]
		return self.comm.bcast(n_j)

	@property
	def N_locs(self):
		return self.comm.gather(self.N_loc)

	def load(self, sampnames, scale='percentilescale', q=(1,99), **kw):
		self.data = dat_util.load_fcdata(sampnames,scale,(1,99),self.comm,**kw)

	# def subsample_to_root(self,N):
	# 	if self.rank == 0:
	# 		ps = self.N_locs
	# 		P = sum(ps)
	# 		ps = [p/N for p in ps]
	# 		N_samp = np.random.multinomial(N,ps)
	# 	else:
	# 		N_samp = None
	# 	N_samp_loc = self.comm.scatter(N_samp)

	def subsample_from_each_to_root(self,N):
		data_subsamp_loc = np.vstack([dat[np.random.choice(dat.shape[0],N,replace=False),:] for dat in self.data])
		return mpiutil.collect_data(data_subsamp_loc,self.d,np.float,MPI.DOUBLE,self.comm)

	def subsample_weighted_to_root(self,weights,N):
		weightsMPI = WeightsMPI(self.comm,weights)
		print "tot weight for subsampling {}".format(weightsMPI.W)
		indices = weightsMPI.sample_from_all(N)
		data_loc = np.vstack([self.data[j][ind,:] for j,ind in enumerate(indices)])
		return mpiutil.collect_data(data_loc,self.d,np.float,MPI.DOUBLE,self.comm)

	def subsample_weighted(self,weights,N):
		weightsMPI = WeightsMPI(self.comm,weights)
		print "tot weight for subsampling {}".format(weightsMPI.W)
		indices = weightsMPI.sample_from_all(N)
		return [self.data[j][ind,:] for j,ind in enumerate(indices)]

class EmptyClusterError(Exception):
	pass

def E_step_pooled(comm,data,weights):
	weights_mpi = WeightsMPI(comm,weights)
	if weights_mpi.W == 0:
		raise EmptyClusterError
	#print "tot weight of cluster = {}".format(weights_mpi.W)
	mu_loc = sum([np.sum(dat*weights[j].reshape(-1,1),axis=0) for j,dat in enumerate(data)])
	mu = sum(comm.bcast(comm.gather(mu_loc)))/weights_mpi.W
	if weights_mpi.W < data[0].shape[1]:
		Sigma = np.eye(data[0].shape[1])
	else:
		wXXT_loc = np.zeros((data[0].shape[1],data[0].shape[1]))
		for j,dat in enumerate(data):
			for i in range(dat.shape[0]):
				x = dat[i,:].reshape(-1,1)
				wXXT_loc += weights[j][i]*x.dot(x.T)
		wXXT = sum(comm.bcast(comm.gather(wXXT_loc)))/weights_mpi.W
		Sigma = wXXT - mu.reshape(-1,1).dot(mu.reshape(1,-1))

	return mu,Sigma,weights_mpi.W

def M_step_pooled(comm,data,mus,Sigmas,pis):
	K = len(mus)
	#print "mus = {}".format(mus)
	#print "Sigmas = {}".format(Sigmas)
	weights = [np.empty((dat.shape[0],K)) for dat in data]
	for j,dat in enumerate(data):
		for k in range(K):
			weights[j][:,k] = stats.multivariate_normal.pdf(dat,mus[k],Sigmas[k])
	for weight in weights:
		weight *= pis
		weight /= np.sum(weight,axis=1).reshape(-1,1)
	return weights


def normalize_pi(p,k_fixed):
	p[np.isnan(p)] = 1./len(p)
	p_fixed = sum(p[k] for k in k_fixed)
	W = sum([p_k for k,p_k in enumerate(p) if not k in k_fixed])
	for k,p_k in enumerate(p):
		if not k in k_fixed:
			p[k] *= (1-p_fixed)/W
	return p

def EM_pooled(comm,data,K,n_iter=10,n_init=5,mus_fixed=[],Sigmas_fixed=[],pis_fixed=[]):
	"""
		Fitting GMM with EM algorithm with fixed components

		comm 			- MPI communicator.
		data 			- list of data sets.
		K 				- number of components.
		n_iter			- number of EM steps.
		n_init 			- number of random initializations.
		mus_fixed 		- mu values for fixed components.
		Sigmas_fixed	- Sigma values for fixed components.
		pis_fixed		- pi values for fixed components. Proportions will be fixed if
						  values are not nan, but not fixed when pi is nan.
				
	"""
	mu0,Sigma0,_ = E_step_pooled(comm,data,[np.array([1./K for i in range(dat.shape[0])]).reshape(-1,1) for dat in data])
	d = data[0].shape[1]
	max_log_lik = -np.inf
	K_fix = len(mus_fixed)
	K -= K_fix
	k_pi_fixed = [K+k for k,pi_k in enumerate(pis_fixed) if not np.isnan(pi_k)]

	for init in range(n_init):
		mus = stats.multivariate_normal.rvs(mu0,Sigma0,size=K).reshape(-1,d).tolist()+mus_fixed
		#mus = [mus_array[i,:] for i in range(mus_array.shape[0])]+mus_fixed
		Sigmas = [Sigma0 for k in range(K)]+Sigmas_fixed#[np.eye(data[0].shape[1]) for k in range(K)]+Sigmas_fixed
		pis = np.array([np.nan for k in range(K)]+pis_fixed)
		pis = normalize_pi(pis,k_pi_fixed)
		for it in range(n_iter):
			weights = M_step_pooled(comm,data,mus,Sigmas,pis)
			for k in range(K):
				mus[k],Sigmas[k],pis[k] = E_step_pooled(comm,data,[weight[:,k] for weight in weights])
			for k_fix in range(K,K+K_fix):
				if not k_fix in k_pi_fixed:
					pis[k] = WeightsMPI(comm,[weight[:,k] for weight in weights]).W
			pis = normalize_pi(pis,k_pi_fixed)
		#weights = M_step_pooled(comm,data,mus,Sigmas,pis)
		log_lik_loc = np.sum([np.log(np.sum(weight,axis=1)) for weight in weights])
		log_lik = sum(comm.bcast(comm.gather(log_lik_loc)))
		if log_lik > max_log_lik:
			best_mus,best_Sigmas,best_pis = mus,Sigmas,pis
	return best_mus,best_Sigmas,best_pis

def EM_weighted_iterated_subsampling(comm,data,K,noise_class,N,iterations=10,iter_final=2):
	K_it = int(np.ceil(K/iterations))
	data_mpi = DataMPI(comm,data)
	noise_pi = 0.001

	if noise_class:
		mus_fixed = [0.5*np.ones(data_mpi.d)]
		Sigmas_fixed = [0.5**2*np.eye(data_mpi.d)]
		pis_fixed = [noise_pi]
	else:
		mus_fixed = []
		Sigmas_fixed = []
		pis_fixed = []
	K_fix = len(mus_fixed)

	data_subsamp = data

	while K_fix < K+int(noise_class):
		mus,Sigmas,pis = EM_pooled(comm,data_subsamp,K,mus_fixed=mus_fixed,Sigmas_fixed=Sigmas_fixed,pis_fixed=pis_fixed)
		k_fixed = np.argpartition(-pis[:len(pis)-K_fix],K_it-1)[:K_it]
		weights = M_step_pooled(comm,data,mus,Sigmas,pis)
		weights_subsamp = [1-(np.sum(weight[:,weight.shape[1]-K_fix:],axis=1)+sum([weight[:,k] for k in k_fixed])) for weight in weights]
		data_subsamp = data_mpi.subsample_weighted(weights_subsamp,N)

		mus_fixed += [mus[k] for k in k_fixed]
		Sigmas_fixed += [Sigmas[k] for k in k_fixed]
		pis_fixed += [np.nan for k in k_fixed]
		K_fix = len(mus_fixed)

	# Extra EM it
	mus,Sigmas = mus_fixed,Sigmas_fixed
	pis = np.array([1./K for k in range(K)])
	if noise_class:
		pis.append(noise_pi)
		pi = normalize_pi(pi,K)

	for it in range(iter_final):
		weights = M_step_pooled(comm,data,mus,Sigmas,pis)
		for k in range(int(noise_class),weights[0].shape[1]):
			mus[k],Sigmas[k],pis[k] = E_step_pooled(comm,data,[weight[:,k] for weight in weights])
		if not noise_class:
			pis = normalize_pi(pis,[])
		else:
			pis = normalize_pi(pis,K)
	return mus,Sigmas,pis

if __name__ == '__main__':

	class MixMod(object):

		def __init__(self,mus,Sigmas,pis):
			self.mus = mus
			self.Sigmas = Sigmas
			self.pis = pis
			self.d = mus[0].shape[0]

		def sample(self,N):
			n_ks = np.random.multinomial(N,pis)
			data = np.zeros((N,self.d))
			i = 0
			for k,n_k in enumerate(n_ks):
				data[i:i+n_k,:] = np.random.multivariate_normal(mus[k],Sigmas[k],size=n_k)
				i += n_k
			return data

	from mpi4py import MPI
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()

	if 0:
		data = DataMPI(MPI.COMM_WORLD,[np.eye(3) for k in range(3)])
		weights = [range(3) for k in range(3)]
		print "data.J_loc = {}".format(data.J_loc)
		print "data.J = {}".format(data.J)
		print "data.n_j = {}".format(data.n_j)
		print "data.subsample_from_each_to_root(2) = {}".format(data.subsample_from_each_to_root(2))
		print "data.subsample_weighted_to_root(weights,20) = {}".format(data.subsample_weighted_to_root(weights,20))

	if 0:
		pi = np.array([1,3,5,0.1,0.2])
		k_fixed = [3,4]
		print "normalize_pi(pi,k_fixed) = {}".format(normalize_pi(pi,k_fixed))
		print "sum(normalize_pi(pi,k_fixed)) = {}".format(sum(normalize_pi(pi,k_fixed)))

		pi = np.array([1,3,5,0.1,0.2,np.nan])
		k_fixed = [3,4]
		print "normalize_pi(pi,k_fixed) = {}".format(normalize_pi(pi,k_fixed))
		print "sum(normalize_pi(pi,k_fixed)) = {}".format(sum(normalize_pi(pi,k_fixed)))
	if 1:
		d = 2
		mus = [m*np.ones(d)+np.random.normal(0,0.1) for m in [0,1,10]]
		Sigmas = [m*np.eye(d) for m in [1,0.5,0.1]]
		pis = np.array([10000,10000,100])
		pis = pis/np.sum(pis)
		N = 10000
		data = [MixMod(mus,Sigmas,pis).sample(N)]
		import matplotlib.pyplot as plt
		K = 5
		mus_fitted,Sigmas_fitted,pis_fitted = EM_weighted_iterated_subsampling(comm,data,K,False,N/10,iter_final=0)
		# print "mus_fitted = {}".format(mus_fitted)
		# print "Sigmas_fitted = {}".format(Sigmas_fitted)
		# print "pis_fitted = {}".format(pis_fitted)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.scatter(data[0][:,0],data[0][:,1])
		from BayesFlow.plot import component_plot
		component_plot(mus_fitted,Sigmas_fitted,[0,1],ax,colors=[(1,0,0)]*len(mus_fitted),lw=2)
		plt.show()


