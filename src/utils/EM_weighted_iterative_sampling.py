from __future__ import division
from sklearn import mixture as skmixture
from BayesFlow.utils import dat_util
from BayesFlow.utils import mpiutil
import numpy as np
import scipy.stats as stats

class WeightsMPI(object):

	def __init__(self,comm,weights):
		self.comm = comm
		self.rank = comm.Get_rank()
		self.weights = weights

	@property
	def ws_loc(self):
		return [sum(w) for w in self.weights]

	@property
	def W_loc(self):
		return sum([sum(w) for w in self.weights])

	@property
	def W_locs(self):
		return self.comm.gather(self.W_loc)

	@property
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
		print "alpha_loc retrieved"
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
		self.data = dat_util.load_fcdata(sampnames,scale,(1,99),comm,**kw)

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
		indices = weightsMPI.sample_from_all(N)
		data_loc = np.vstack([self.data[j][ind,:] for j,ind in enumerate(indices)])
		return mpiutil.collect_data(data_loc,self.d,np.float,MPI.DOUBLE,self.comm)

	def subsample_weighted(self,weights,N):
		weightsMPI = WeightsMPI(self.comm,weights)
		indices = weightsMPI.sample_from_all(N)
		return [self.data[j][ind,:] for j,ind in enumerate(indices)]

def E_step_pooled(comm,data,weights):
	weights_mpi = WeightsMPI(comm,weights)
	mu_loc = sum([np.sum(dat*weights[j],axis=0) for j,dat in enumerate(data)])
	mu = comm.bcast(sum(comm.gather(center_loc)))/weights_mpi.W
	
	wXXT_loc = np.zeros((data[0].shape[1],data[0].shape[1]))
	for j,dat in enumerate(data):
		for i in range(dat.shape[0]):
			x = self.data[i,:].reshape(1,-1)
			wXXT_loc += weights[j][i]*x.T.dot(x)
	wXXT = sum(comm.bcast(comm.gather(wXXT_loc)))/weights_mpi.W
	Sigma = wXXT - mu.T.dot(mu)

	return mu,Sigma,weights_mpi.W

def M_step_pooled(comm,data,mus,Sigmas,pis):
	K = len(mus)
	weights = [np.empty((dat.shape[0],K)) for dat in data]
	for k in range(K):
		for j,dat in enumerate(data):
			weights[j][:,k] = stats.multivariate_normal.pdf(data,mus[k],Sigmas[k])
	for weight in weights:
		weight *= pis
		weight /= np.sum(weight,axis=1)
	return weights


def normalize_pi(p,k_fixed):
	p_fixed = sum(p[k] for k in k_fixed)
	W = sum([p_k for k,p_k in enumerate(p) if not k in k_fixed])
	for k,p_k in enumerate(p):
		if not k in k_fixed:
			p[k] *= (1-p_fixed)/W
	return p

def EM_pooled(comm,data,K,n_iter=100,n_init=5,mus_fixed=[],Sigmas_fixed=[],pis_fixed=[]):
	mu0,Sigma0,_ = E_step_pooled(comm,data,[1./dat.shape[0] for dat in data])
	max_log_lik = -np.inf
	K_fix = len(mus_fixed)
	K -= K_fix
	k_pi_fixed = [K+k for k,pi_k in enumerate(pis_fixed) if not np.isnan(pi_k)] 
	for init in range(n_init):
		mus = stats.multivariate_normal.rvs(mu0,Sigma0,size=K).tolist()+mus_fixed
		Sigmas = [np.eye(data[0].shape[1]) for k in range(K)]+Sigmas_fixed
		pis = np.array([1./K for k in range(K)]+pis_fixed)
		pis[np.isnan(pis)] == 1./K
		pis = normalize_pi(pis,k_pi_fixed)
		for it in range(n_iter):
			weights = M_step_pooled(comm,data,mus,Sigmas,pis)
			for k in range(K):
				mus[k],Sigmas[k],pis[k] = E_step_pooled(comm,data,[weight[:,k] for weight in weights])
			for k_fix in range(K,K+K_fix):
				if not k_fix in k_pi_fixed:
					pis[k] = WeightsMPI([weight[:,k] for weight in Weights]).W
			pis = normalize_pi(pi,k_pi_fixed)
		weights = M_step_pooled(comm,data,mus,Sigmas,pis)
		log_lik_loc = np.sum([np.log(np.sum(weight,axis=1)) for weight in weights])
		log_lik = sum(comm.bcast(comm.gather(log_lik_loc)))
		if log_lik > max_log_lik:
			best_mus,best_Sigmas,best_pis,best_weights = mus,Sigmas,pis,weights
	return best_mus,best_Sigmas,best_pis,best_weights

def EM_weighted_iterated_subsampling(comm,data,K,noise_class,N,iterations=10,iter_final=10):
	K_it = int(np.ceil(K/iterations))
	data_mpi = DataMPI(data)

	if noise_class:
		mus_fixed = [0.5*np.ones(data_mpi.d)]
		Sigmas_fixed = [0.5**2*np.eye(data_mpi.d)]
		pis_fixed = [0.001]
	else:
		mus_fixed = []
		Sigmas_fixed = []
		pis_fixed = []
	K_fixed = len(mus_fixed)

	mus,Sigmas,pis,weights = EM_pooled(comm,data,K,mus_fixed,Sigmas_fixed,pis_fixed)
	k_fixed = np.argpartition(-pis[:-K_fixed],K_it-1)[:K_it]
	mus_fixed += [mus[k] for k in k_fixed]
	Sigmas_fixed += [Sigmas[k] for k in k_fixed]
	pis_fixed += [np.nan for k in k_fixed]
	K_fix = len(mus_fix)

	while K_fix < K+1:
		weights_subsamp = [1-np.sum(weight[-K_fix:],axis=1) for weight in weights]
		data_subsamp = data_mpi.subsample_weighted(weights,N)
		mus,Sigmas,pis = EM_pooled(comm,data_subsamp,K,mus_fixed,Sigmas_fixed,pis_fixed)
		k_fixed = np.argpartition(-pis[:-K_fixed],K_it-1)[:K_it]
		mus_fixed += [mus[k] for k in k_fixed]
		Sigmas_fixed += [Sigmas[k] for k in k_fixed]
		pis_fixed += [np.nan for k in k_fixed]
		K_fix = len(mus_fix)

	# Extra EM it
	mus,Sigmas,pis = mus_fixed,Sigmas_fixed,pis_fixed
	for it in range(iter_final):
		weights = M_step_pooled(comm,data,mus,Sigmas,pis)
		for k in range(int(noise_class):weights.shape[1]):
			mus[k],Sigmas[k],pis[k] = E_step_pooled(comm,data,weights[:,k])
	return mus,Sigmas,pis	

if 0:
	from mpi4py import MPI
	data = DataMPI(MPI.COMM_WORLD,[np.eye(3) for k in range(3)])
	weights = [range(3) for k in range(3)]
	print "data.J_loc = {}".format(data.J_loc)
	print "data.J = {}".format(data.J)
	print "data.n_j = {}".format(data.n_j)
	print "data.subsample_from_each_to_root(2) = {}".format(data.subsample_from_each_to_root(2))
	print "data.subsample_weighted_to_root(weights,20) = {}".format(data.subsample_weighted_to_root(weights,20))

if 0:
	pi = [1,3,5,0.1,0.2]
	k_fixed = [3,4]
	print "normalize_pi(pi,k_fixed) = {}".format(normalize_pi(pi,k_fixed))
	print "sum(normalize_pi(pi,k_fixed)) = {}".format(sum(normalize_pi(pi,k_fixed)))