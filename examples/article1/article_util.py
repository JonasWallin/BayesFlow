'''
Functions helping with running the estimation procedure
Created on Aug 25, 2015

@author: jonaswallin
'''
from __future__ import print_function
import BayesFlow as bf
import sys
from mpi4py import MPI
import numpy as np
import scipy.spatial as ss
import copy as cp


def HGMM_pre_burnin(Hier_GMM, init_iter = 40, prec = 1, iteration = 10, mutate_iteration = 20, burst_iteration = 20, local_iter = 5, silent = False):
	"""	
		A mutation type algorithm to find good starting lcoation for MCMC
		*prec* precentage of data counting as outlier
		*iteration*         number of iteration of mutation burst mutation
		*mutate_iteration*  number of gibbs samples to run before testing if the mutation improved
		*burst_iteration*  number of gibbs samples to run before testing if the mutation improved
	"""
	
	if  (MPI.COMM_WORLD.Get_rank() == 0):   # @UndefinedVariable 
		silent = silent
	else:
		silent = True
	
	for GMM in Hier_GMM.GMMs:
		
		for i in range(init_iter):  # @UnusedVariable
			GMM.sample()
			
		GMM_pre_burnin(GMM 			    = GMM, 
					   prec 			= prec,
					   iteration 		= iteration,
					   mutate_iteration = mutate_iteration,
					   burst_iteration  = burst_iteration,
					   silent           = silent,
					   local_iter       = local_iter)
	Hier_GMM.comm.Barrier()

def GMM_pre_burnin(GMM, prec = 1, iteration = 10, mutate_iteration = 20, burst_iteration = 20, local_iter = 5, silent= False):
	"""	
		A mutation type algorithm to find good starting lcoation for MCMC
		*prec* precentage of data counting as outlier
		*iteration*         number of iteration of mutation burst mutation
		*mutate_iteration*  number of gibbs samples to run before testing if the mutation improved
		*burst_iteration*  number of gibbs samples to run before testing if the mutation improved
	"""
	for j in range(iteration):  # @UnusedVariable
		if not silent:
			print('pre burnin iteration {j}'.format(j = j))
		mutate(GMM, prec, iteration = mutate_iteration, silent = silent)
		mutate(GMM, prec, iteration = mutate_iteration, silent = silent, rand_class =True)
		burst(GMM,         iteration = burst_iteration, silent = silent)
		for k in range(local_iter):  # @UnusedVariable
			GMM.sample()

def draw_outlier_point(GMM, prec = 0.1):
	"""
		draws a random outlier point (outlier defined through likelihood)
		*prec* - [0,1] lower quantile what is defined as outlier 
	"""
	GMM.compute_ProbX(norm=False)
	l  =np.max(GMM.prob_X,1)
	index  = l < np.percentile(l,0.1)
	index_p = np.random.randint(0.1*GMM.data.shape[0])
	point_ = GMM.data[index[index_p],:]
	return(point_)


def store_param(GMM, lik = None):
	'''
		Stores the likelihood components
	'''
	if lik is None:
		lik = GMM.calc_lik()
	
	res = {'lik':    lik,
		   'p':      cp.deepcopy(GMM.p),
		    'mu':    cp.deepcopy(GMM.mu), 
		    'sigma': cp.deepcopy(GMM.sigma)}
	return(res)

def restore_param(GMM, param):
	'''
		reset the GMM from the parameters
	'''
	GMM.p = param['p']
	GMM.set_mu(param['mu'])
	GMM.set_sigma(param['sigma'])




def mutate(GMM, prec = 0.1, iteration = 10, silent = True, rand_class = False):
	'''
		mutate by setting a random class to outiler class
		*prec*      - [0,100] lower quantile what is defined as outlier (precentage)
		*iter*      - number of iteration in the Gibbs sampler 
  *rand_class* - draw the class at random (else always take the smallest) 
	'''
	
	param0 = store_param(GMM)
	point_ = draw_outlier_point(GMM, prec)
	if rand_class:
		k  = np.random.randint(GMM.K)
	else:
		k  = np.argmin(GMM.p[:GMM.K])
	set_mutated(GMM, k, point_)
	
	for i in range(iteration):
		GMM.sample()
		
	lik = GMM.calc_lik()
	if param0['lik'] < lik:
		if silent is False:
			if rand_class:
				print('random mutation %.2f < %.2f'%(param0['lik'], lik))
			else:
				print('min mutation %.2f < %.2f'%(param0['lik'], lik))
		return
	
	restore_param(GMM, param0)


def burst(GMM, iteration = 10, silent = True):
	'''
		trying to burst two classes (k, k2)
	'''
	param0 = store_param(GMM)
	
	k  = np.random.randint(GMM.K)
	d_ = np.random.randint(GMM.d)
	dist = np.abs((np.array(GMM.mu)[k, d_]- np.array(GMM.mu)[:, d_]))
	dist[k] = np.Inf
	k2 = np.argmin(dist)
	index = (GMM.x == k) + (GMM.x == k2)
	y_index = GMM.data[index,:]
	if y_index.shape[0] < 3:
		return
	index_points = np.random.choice(y_index.shape[0], 2, replace=False)
	points = y_index[index_points,:]
	set_mutated(GMM, k, points[0,:],  update = False)
	set_mutated(GMM, k2, points[1,:], update = True)


	for i in range(iteration):
		GMM.sample()

	lik = GMM.calc_lik()
	if param0['lik'] < lik:
		if silent is False:
			print('burst mutation %.2f < %.2f'%(param0['lik'], lik))
		return
	
	restore_param(GMM, param0)

def set_mutated(GMM, k, point_, update = True):
	'''
		sets the mutated point
	'''
	
	GMM.mu[k][:] = point_[:]
	GMM.sigma[k] = 50 * np.diag(np.diag(GMM.sigma[k]))
	if update:
		GMM.updata_mudata()
		GMM.sample_x()



def setup_model(y, K, active_prior = 10., time_it = True, silent=False):
	'''
		Setting up the basic model
		*y*            - the data 
		*K*            - the number of classes
		*active_prior* - prior that supresses number of active components
		*time_it*      - time the sampling
	'''
	if not silent and (MPI.COMM_WORLD.Get_rank() == 0):   # @UndefinedVariable 
		print('seting up model, ', end='')
		sys.stdout.flush()
	
	#########################
	# Setting up model
	########################
	hier_gmm = bf.hierarical_mixture_mpi(K = K)
	hier_gmm.set_data(y)
	hier_gmm.set_prior_param0()
	hier_gmm.update_GMM()
	hier_gmm.update_prior()
	hier_gmm.set_prior_actiavation(active_prior)
	if time_it:
		hier_gmm.toggle_timing()

	if not silent and (MPI.COMM_WORLD.Get_rank() == 0):   # @UndefinedVariable 
		print('done ')
		sys.stdout.flush()
			
	return hier_gmm



def burin_1(hGMM, sim = 2000, p_label = 1., silent = False):
	'''
		First burnin period, letting the mixture classes tune in before trying to switch them off
		
		*hGMM*     - the hier GMM object
		*sim*      - number of simulation to run in the first burin face
		*p_label*  - how often to try to switch the labels
	
	'''
	
	if not silent and (MPI.COMM_WORLD.Get_rank() == 0):   # @UndefinedVariable 
		print('burnin face one: ', end='')
		sys.stdout.flush()
	
	
	hGMM.set_p_labelswitch(p_label)
	hGMM.set_nu_MH_param(10,200)
	for i,GMM in enumerate(hGMM.GMMs):	
		GMM._label  =i
	for i in range(sim):
		
		if not silent and (MPI.COMM_WORLD.Get_rank() == 0):   # @UndefinedVariable 
			if i % 100 == 0:
				print('*', end='')
				sys.stdout.flush()
		hGMM.sample()
		#if i % 200 == 0:
		#	bf.distance_sort_MPI(hGMM)

	if not silent and (MPI.COMM_WORLD.Get_rank() == 0):   # @UndefinedVariable 
		print(', done')
		sys.stdout.flush()
		
def burin_2(hGMM, sim = 8000, p_label = 1., p_act = [0.7, 0.7], silent= False):
	"""
		second burnin, here we allow turning of cluster and try to sort the clusters
		assumes burin_1 has run
		
		*hGMM*     - the hier GMM object
		*sim*      - number of simulation to run in the second burin face
		*p_label*  - how often to try to switch the labels
		*p_act*    - probability of trying to switch on / switch off a cluster for a class
	"""
	
	if not silent and (MPI.COMM_WORLD.Get_rank() == 0):   # @UndefinedVariable 
		print('burnin face two: ', end='')
		sys.stdout.flush()
	
	hGMM.set_p_labelswitch(p_label)
	bf.distance_sort_MPI(hGMM)
	hGMM.set_p_activation(p_act)
	
	for i in range(sim):#burn in
		
		if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable 
			if not silent:
				if i % 100 == 0:
					print('*', end='')
					sys.stdout.flush()
		hGMM.sample()
	
	
		
	#bf.distance_sort_MPI(hGMM) 
		
	if not silent and (MPI.COMM_WORLD.Get_rank() == 0): # @UndefinedVariable 
		print(', done')
		sys.stdout.flush()
		
		
def main_run(hGMM, sim = 16000, thin = 1, p_label = .4, p_act = [0.4, 0.4], silent= False):
	"""
		After burnin run main face
		
		*hGMM*     - the hier GMM object
		*sim*      - number of simulation to run in the main
		*thin*     - thin the samples
		*p_label*  - how often to try to switch the labels
		*p_act*    - probability of trying to switch on / switch off a cluster for a class
	"""
	
	
	
	if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable 
		count = 0
		data_out = {'mus'   : np.zeros((hGMM.n_all, hGMM.K, hGMM.d)),
				   'actkomp': np.zeros((hGMM.n_all, hGMM.K)),
				    'Q'     :[], 
				    'nu'    :[],
				    'Y'     :[],
				    'Y_0'   :[],
				    'theta' :[]}
	else:
		data_out = {}

	
	if not silent and (MPI.COMM_WORLD.Get_rank() == 0):   # @UndefinedVariable 
		print('main face: ', end='')
		sys.stdout.flush()
	
	hGMM.set_p_labelswitch(p_label)
	hGMM.set_p_activation(p_act)
	
	for i in range(sim):
		for k in range(thin): # @UndefinedVariable 
			hGMM.sample()
			
			# sorting the vector after a label switch 
			labels = hGMM.get_labelswitches()
			if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable 
				label_sort(labels, data_out['mus'], data_out['actkomp'])	

		#################################################
		###################
		# storing data
		# for post analysis
		###################	
		################################################
		mus_ = hGMM.get_mus()
		thetas = hGMM.get_thetas()
		Qs = hGMM.get_Qs()
		nus = hGMM.get_nus()
		active_komp = hGMM.get_activekompontent()
		Y_sample = hGMM.sampleY()
	
		if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable 
			if not silent:
				if i % 100 == 0:
					print('*', end='')
					sys.stdout.flush()
					
			count += 1
			data_out['mus']     += mus_
			data_out['actkomp'] += active_komp
			
			data_out['theta'].append(thetas)
			data_out['Q'].append(Qs/(nus.reshape(nus.shape[0],1,1)- Qs.shape[1]-1)  )
			data_out['nu'].append(nus)
			
	
			data_out['Y_0'].append(hGMM.GMMs[0].simulate_one_obs().flatten())
			data_out['Y'].append(Y_sample)
		
	
	
	
	if MPI.COMM_WORLD.Get_rank() == 0:  # @UndefinedVariable
		data_out['actkomp'] /= count
		data_out['mus']     /= count
		
	if not silent and (MPI.COMM_WORLD.Get_rank() == 0): # @UndefinedVariable
		print(', done')
		sys.stdout.flush()
	
	
	
	return data_out

def label_sort(labels, *args):
	"""
		When using label switch stored data must also be reodered
		args must numpy arguemnts
	"""
	for j in range(labels.shape[0]):
		if labels[j,0] != -1:	
			label_flip = np.fliplr(np.atleast_2d(labels[j,:]))[0]
			for thing in args:
				if len(thing.shape) > 2:
					thing[j,labels[j,:],:] = thing[j,label_flip,:]
				else:
					thing[j,labels[j,:]] = thing[j,label_flip]
					
	
def sort_mus(mus, mus_true):
	'''
		Takes a true mu vector (mus_true) and tries to find the cloest mus
		to that vector through distances
		returns the index ''converting'' mus to mus_true
	'''
	
	K = mus.shape[1] #number of classes
	mus_true_mean = []
	mus_mean = []
	for k in range(K):
		mus_true_mean.append(np.array(np.ma.masked_invalid( mus_true[:,k,:]).mean(0)))
		mus_mean.append(     np.array(np.ma.masked_invalid( mus[:,k,:].T   ).mean(0)))
		
		
	mus_true_mean =  np.array(mus_true_mean)
	mus_mean =  np.array(mus_mean)
	
	ss_mat =  ss.distance.cdist( mus_true_mean, mus_mean, "euclidean")
	mu_true_index = np.array([i for i in range(K)])
	col_index = []
	for k in range(K):
		k_ = np.argmin(ss_mat[k,:])
		ss_mat = np.delete(ss_mat, (k_), axis=1)
		
		col_index.append(mu_true_index[k_])	
		mu_true_index = np.delete(mu_true_index, (k_))
	
	return col_index

def sort_thetas(thetas_est, thetas_true):	
	
	
	ss_mat =  ss.distance.cdist(thetas_true,
							    np.mean(thetas_est,0)
							    )
	
	K = thetas_true.shape[0]
	col_index = np.zeros((K), dtype = np.int)
	for k in range(K):  # @UnusedVariable
		theta_est_index = np.floor(np.argmin(ss_mat)/K)
		theta_index    = np.argmin(ss_mat)%K
		col_index[theta_est_index] = np.int(theta_index)
		ss_mat[theta_est_index,:] = np.inf
		ss_mat[:,theta_index] = np.inf
	
	return col_index
 