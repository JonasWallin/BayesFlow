from __future__ import division
'''
Created on Jul 10, 2014

@author: jonaswallin
'''
from mpi4py import MPI
import GMM
import numpy as np
from BayesFlow.distribution import normal_p_wishart, Wishart_p_nu
from BayesFlow.GMM import mixture
import HMlog
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.random as npr
import os
import glob
import time
import warnings

#TODO: change to geomtrical median instead of mean!!
def distance_sort(hGMM):
	"""
		sorts the clusters after the geometrical mean of the means
		UGLY SOLUTION BUT OPTIMALILITY IS NOT NEEDED
	"""
	size = hGMM.comm.Get_size()  # @UndefinedVariable
	
	for com_obj in range(size):
		if hGMM.comm.Get_rank() == com_obj:
			if hGMM.comm.Get_rank() == 0:
				mus  = [hGMM.GMMs[0].mu[k].reshape((1, hGMM.d)) for k in range(hGMM.K)]
				GMMs = hGMM.GMMs[1:]
			else:
				GMMs = hGMM.GMMs
				mus = hGMM.comm.recv(source=0)['mus']
			
			for GMM in GMMs:
				index_k = np.argsort(GMM.p)[::-1] #sort by probabilility
				mus_t = np.array([np.mean(mu,axis=0) for mu in  mus])
				list_temp = [None for k in range(GMM.K)]
				for index in index_k:
					#print mus_t
					#print mus_t - GMM.mu[index]
					dist = np.linalg.norm(mus_t - GMM.mu[index],axis=1)
					i = np.argsort(dist)[0]
					mus_t[i,:] = np.inf
					list_temp[index] = i 
				list_temp = np.argsort(np.array(list_temp))
				mus = [np.vstack((mu,GMM.mu[list_temp[i]])) for i,mu in enumerate(mus) ]
				GMM.mu = [GMM.mu[i] for i in list_temp]
				GMM.sigma = [GMM.sigma[i] for i in list_temp]
				GMM.p = np.array([GMM.p[i] for i in list_temp])
			if hGMM.comm.Get_rank() != 0:
				hGMM.comm.send({'mus':mus}, dest=0)
				
		if com_obj != 0:
			if hGMM.comm.Get_rank() == 0:
				hGMM.comm.send({'mus':mus}, dest=com_obj)
				mus = hGMM.comm.recv(source=com_obj)['mus']
		hGMM.comm.Barrier()
	
	hGMM.update_prior()
	hGMM.comm.Barrier()
	[GMM.updata_mudata() for GMM in hGMM.GMMs]
	return mus


def load_hGMM(dirname):
	'''
		Load hGMM from a directory
	'''
	if dirname[-1] != '/':
		dirname += '/'
	K = len(glob.glob(dirname + 'normal_p_wishart*'))
	hGMM = hierarical_mixture_mpi(K)
	hGMM.load_from_file(dirname)
	return hGMM

class SimulationError(Exception):
	def __init__(self,msg,name='',it=0):
		self.msg = msg
		self.name = name
		self.iteration = it

#class NoClusterError(Exception):
#	def __init__(self,msg):
#		self.msg = msg
		
class hierarical_mixture_mpi(object):
	"""	
		Comment about noise_class either all mpi hier class are noise class or none!
		either incosistency can occuer when loading
	"""
	def __init__(self, K = None, data = None, sampnames = None, prior = None, sim_param = None ):
		"""
			starting up the class and defning number of classes
			
		"""
		if K is None:
			K = prior.K
		self.K = K
		self.d = 0
		self.n = 0
		self.n_all = -1
		self.noise_class = 0 
		self.GMMs = []
		self.comm = MPI.COMM_WORLD  # @UndefinedVariable
		rank = self.comm.Get_rank()  # @UndefinedVariable

		#master
		if rank == 0:
			self.normal_p_wisharts = [ normal_p_wishart() for k in range(self.K)]  # @UnusedVariable
			self.wishart_p_nus	 = [Wishart_p_nu() for k in range(self.K) ]  # @UnusedVariable

		else:
			self.normal_p_wisharts = None 
			self.wishart_p_nus	 = None
			
		self.set_data(data,sampnames)
		if not prior is None:
			self.set_prior(prior,init=True)
		if not sim_param is None:
			self.set_simulation_param(sim_param)
	
	def save_prior_to_file(self,dirname):
		"""
			Saves the prior to files
			
		"""
		
		rank = self.comm.Get_rank()  # @UndefinedVariable
		if rank == 0:
			if dirname.endswith("/") == False:
				dirname += "/"
			[self.normal_p_wisharts[k].pickle("%snormal_p_wishart_%d.pkl"%(dirname,k)) for k in range(self.K)]
			[self.wishart_p_nus[k].pickle("%sWishart_p_nu_%d.pkl"%(dirname,k)) for k in range(self.K)]
	
	def load_prior_from_file(self,dirname):
		"""
			load the prior to files
			
		"""
		rank = self.comm.Get_rank()  # @UndefinedVariable
		if rank == 0:
			if dirname.endswith("/") == False:
				dirname += "/"		
			self.normal_p_wisharts = [ normal_p_wishart.unpickle("%snormal_p_wishart_%d.pkl"%(dirname,k)) for k in range(self.K)] 
			self.Wishart_p_nu	  = [ Wishart_p_nu.unpickle("%sWishart_p_nu_%d.pkl"%(dirname,k)) for k in range(self.K)] 
		
	def save_to_file(self,dirname):
		"""
			Stores the entire hgmm object to a directory which can be loaded by load_to_file
		
		"""
		
		self.save_prior_to_file(dirname)
		self.save_GMMS_to_file(dirname)
		
		rank = self.comm.Get_rank()  # @UndefinedVariable
		
		if rank == 0:
			if dirname.endswith("/") == False:
				dirname += "/"
			f = open(dirname+'noise_class.txt', 'w')
			f.write('%d' % self.noise_class)
			f.close()		
		

	def load_from_file(self,dirname):
		"""
			Loads the Hgmm from file
		"""
		
		self.load_prior_from_file(dirname)
		self.load_GMMS_from_file(dirname)		

		rank = self.comm.Get_rank()  # @UndefinedVariable
		
		if rank == 0:
			if dirname.endswith("/") == False:
				dirname += "/"
				
				
			with open(dirname+'noise_class.txt') as f:
				line = f.readline()
				noise_class =  np.array([int(line)])
				f.close()
		else:
			noise_class = np.array([-1])
			
		self.comm.Bcast([noise_class, MPI.INT],root=0)  # @UndefinedVariable
		self.noise_class = noise_class[0]
		self.comm.Barrier()
		
				

	
	def load_GMMS_from_file(self,dirname):
		
		rank = self.comm.Get_rank()  # @UndefinedVariable
		
		if rank == 0:
			if dirname.endswith("/") == False:
				dirname += "/"
			size = self.comm.Get_size()  # @UndefinedVariable
			gmms_name_ =  [name for name in os.listdir(dirname) if name.endswith(".pkl") and name.startswith("gmm")]
			self.n_all = len(gmms_name_)
			gmms_name =  [gmms_name_[i::size] for i in range(size)]  #split into sublists
			del gmms_name_
			
			
			gmms = [mixture.unpickle("%s%s"%(dirname,gmm_name)) for gmm_name in gmms_name[0]]
			self.counts = np.array([len(gmm)*gmms[0].K for gmm in gmms_name],dtype='i') 
			self.GMMs = gmms
			self.d = self.GMMs[0].d
			self.n = len(self.GMMs)			
			for i in range(1,size):
				gmms = [mixture.unpickle("%s%s"%(dirname,gmm_name)) for gmm_name in gmms_name[i]]
				self.comm.send({"gmms":gmms}, dest=i, tag=11)
		else:
			self.GMMs  = self.comm.recv(source=0, tag=11)['gmms']
			self.d = self.GMMs[0].d
			self.n = len(self.GMMs)
			#mixture.unpickle(name)
		# TODO: Send counts. Send names?
			
	def save_GMMS_to_file(self,dirname):
		"""
			moving the GMMs to rank1 and storing them!
		"""
		
		rank = self.comm.Get_rank()  # @UndefinedVariable
		
		
		if rank == 0:
			if dirname.endswith("/") == False:
				dirname += "/"
			size = self.comm.Get_size()  # @UndefinedVariable
			count = 0
			for gmm in self.GMMs:
				gmm.pickle("%s/gmm_%d.pkl"%(dirname, count))
				count += 1
			
			for i in range(1,size):
				data = self.comm.recv(source=i, tag=11)
				for gmm in data['GMM']:
					gmm.pickle("%s/gmm_%d.pkl"%(dirname, count))
					count += 1
					
		else:
			data = {'GMM':self.GMMs}
			self.comm.send(data, dest=0, tag=11)

		
	
	def set_nu_MH_param(self, sigma = 5, iteration = 5, sigma_nuprop = None):
		"""
			setting the parametet for the MH algorithm
			
			sigma	 -  the sigma in the MH algorihm on the Natural line
			iteration -  number of time to sample using the MH algortihm
			sigma_nuprop - if provided, the proportion of nu that will be set as sigma
			
		"""
		if self.comm.Get_rank() == 0:  # @UndefinedVariable
			for k,wpn in enumerate(self.wishart_p_nus):
				if sigma_nuprop is None:
					if not isinstance(sigma, list):
						wpn.set_MH_param( sigma , iteration)
					else:
						wpn.set_MH_param( sigma[k] , iteration)
				else:
					wpn.set_MH_param( max(np.ceil(sigma_nuprop*wpn.param['nu']),self.d) , iteration)
		
	def add_noise_class(self,Sigma_scale = 5.,mu=None,Sigma=None):
		
		[GMM.add_noiseclass(Sigma_scale,mu,Sigma) for GMM in self.GMMs ]
		self.noise_class = 1
		
		
	def set_prior_param0(self):
		
		rank =self.comm.Get_rank()  # @UndefinedVariable

		#master
		if rank == 0:
			if self.d == 0:
				raise ValueError('have not set d need for prior0')
			
			for npw in self.normal_p_wisharts:
				npw.set_prior_param0(self.d)
	
			for wpn in self.wishart_p_nus:
				wpn.set_prior_param0(self.d)
	
	
	def reset_nus(self, nu, Q = None):
		"""
		reseting the values of the latent parameters of the covariance 
			run update_GMM after otherwise dont know what happens
		
		"""
		rank = self.comm.Get_rank()  # @UndefinedVariable
		if rank == 0:
			for wpn in self.wishart_p_nus:
				Q_ = np.zeros_like(wpn.param['Q'].shape[0])
				if Q == None:
					Q_ = 10**-10*np.eye(wpn.param['Q'].shape[0]) 
				else:
					Q_  = Q[:]
				param = {'nu':nu, 'Q': Q_}
				wpn.set_val(param)
	
	
	def reset_Sigma_theta(self, Sigma = None):
		"""
		reseting the values of the latent parameters of the mean
			run update_GMM after otherwise dont know what happens
		
		"""
		rank = self.comm.Get_rank()  # @UndefinedVariable
		if rank == 0:
			for npw in self.normal_p_wisharts:
				if Sigma == None:
					npw.param['Sigma']  = 10**10*np.eye(npw.param['Sigma'].shape[0])
				else:
					npw.param['Sigma'][:]  = Sigma[:]
		
		
	def reset_prior(self,nu = 10):
		"""
			reseting the values for the latent layer
		"""
		
		self.reset_nus(nu)
		self.reset_Sigma_theta()	
		self.update_GMM()
			
	def update_GMM(self):
		"""
			Transforms the data from the priors to the GMM
		"""
		rank = self.comm.Get_rank()  # @UndefinedVariable
		if rank == 0:
			mu_theta = np.array([npw.param['theta'] for npw in self.normal_p_wisharts],dtype='d')
			mu_sigma = np.array([npw.param['Sigma'] for npw in self.normal_p_wisharts],dtype='d')
			sigma_nu = np.array([wpn.param['nu'] for wpn in self.wishart_p_nus],dtype='i')
			sigma_Q = np.array([wpn.param['Q'] for wpn in self.wishart_p_nus],dtype='d')
		else:
			mu_theta = np.empty((self.K,self.d),dtype='d')
			mu_sigma = np.empty((self.K,self.d,self.d),dtype='d')
			sigma_nu = np.empty(self.K,dtype='i')
			sigma_Q  = np.empty((self.K,self.d,self.d),dtype='d')
			
		self.comm.Bcast([mu_theta, MPI.DOUBLE])  # @UndefinedVariable
		self.comm.Bcast([mu_sigma, MPI.DOUBLE])  # @UndefinedVariable
		self.comm.Bcast([sigma_nu, MPI.INT])  # @UndefinedVariable
		self.comm.Bcast([sigma_Q, MPI.DOUBLE])  # @UndefinedVariable
		self.comm.Barrier()

		for i in range(self.n):
			self.GMMs[i].set_prior_mu_np(mu_theta, mu_sigma)
			self.GMMs[i].set_prior_sigma_np(sigma_nu, sigma_Q)
	
	def update_prior(self):
		"""
			transforms the data from the GMM to the prior
		
		"""
		
		rank = self.comm.Get_rank()  # @UndefinedVariable
		
		if rank == 0:
			recv_obj = np.empty((self.n_all, self.K, self.d * (self.d + 1)),dtype='d')
		else:
			recv_obj = None
		
		
		send_obj = np.array([[np.hstack([GMM.mu[k].flatten(),GMM.sigma[k].flatten()]) for k in range(self.K) ]  for GMM in self.GMMs ],dtype='d')
		self.comm.Gatherv(sendbuf=[send_obj, MPI.DOUBLE], recvbuf=[recv_obj, (self.counts * self.d * (self.d+1), None), MPI.DOUBLE],  root=0)  # @UndefinedVariable

		cl_off = np.array([0],dtype='i')
		if rank == 0:
			mu_k = np.empty((self.n_all,self.d)) 
			Sigma_k = np.empty((self.n_all,self.d,self.d)) 
			#print recv_obj[0,:,:self.d]
			for k in range(self.K):
				mu_k[:] = recv_obj[:,k,:self.d]
				index = np.isnan(mu_k[:,0])==False
				if np.sum(index) == 0:
					cl_off = np.array([1],dtype='i')
				#if k <2:
				#	print("mu[%d] = %s"%(k, mu_k))
				Sigma_k[:] = recv_obj[:,k,self.d:].reshape((self.n_all,self.d, self.d))
				self.normal_p_wisharts[k].set_data(mu_k[index,:])
				self.wishart_p_nus[k].set_data(Sigma_k[index,:,:])

		self.comm.Bcast([cl_off, MPI.INT])
		if cl_off[0]:
			warnings.warn('One cluster turned off in all samples')

	def set_simulation_param(self,sim_par):
		self.set_p_labelswitch(sim_par['p_sw'])
		if not sim_par['nu_MH_par'] is None:
			self.set_nu_MH_param(**sim_par['nu_MH_par'])
		self.set_p_activation(sim_par['p_on_off'])


	def set_p_activation(self, p):
		
		for GMM in self.GMMs:
			GMM.p_act   = p[0]
			GMM.p_inact = p[1]
	
	
	def set_prior_actiavation(self, komp_prior):
		
		for GMM in self.GMMs:
			GMM.komp_prior = komp_prior
	
	def set_p_labelswitch(self,p):
		"""
			setting the label switch parameter
		"""
		
		for GMM in self.GMMs:
			GMM.p_switch   = p
	
	def set_nuss(self, nu):
		"""
			increase to force the mean to move together
		"""
		if self.comm.Get_rank() == 0:
			for k in range(self.K):
				self.normal_p_wisharts[k].Sigma_class.nu = nu
			
	def set_nu_mus(self, nu):
		"""
			increase to force the covariance to move together
		
		"""
		if self.comm.Get_rank() == 0:
			for k in range(self.K):
				self.wishart_p_nus[k].Q_class.nu_s = nu
		
	def set_data(self, data, names = None):
		"""
			List of np.arrays
		
		"""
		rank = self.comm.Get_rank()  # @UndefinedVariable
		if rank == 0:
			if data is None:
				nodata = np.array(1,dtype="i")
			else:
				nodata = np.array(0,dtype="i")
		else:
			nodata = np.array(0,dtype="i")
		self.comm.Bcast([nodata, MPI.INT],root=0)  # @UndefinedVariable
		if nodata:
			return				
		
		if rank == 0:
			d = np.array(data[0].shape[1],dtype="i")
			self.n_all = len(data)
			size = self.comm.Get_size()  # @UndefinedVariable
			data = np.array(data)
			send_data = np.array_split(data,size)
			self.counts = np.empty(size,dtype='i') 
			if names == None:
				names = range(self.n_all)
			send_name = np.array_split( np.array(names),size)
		else:
			d  =np.array(0,dtype="i")
			self.counts = 0
			send_data = None
			send_name = None
			
		self.comm.Bcast([d, MPI.INT],root=0)  # @UndefinedVariable
		self.d = d[()]
		dat = self.comm.scatter(send_data, root= 0)  # @UndefinedVariable
		names_dat = self.comm.scatter(send_name, root= 0)  # @UndefinedVariable
		self.n = len(dat)
		for Y, name in zip(dat,names_dat):
			if self.d != Y.shape[1]:
				raise ValueError('dimension missmatch in thet data')
			self.GMMs.append(GMM.mixture(data= Y, K = self.K, name = name))
		#print "mpi = %d, len(GMMs) = %d"%(MPI.COMM_WORLD.rank, len(self.GMMs))  # @UndefinedVariable
		
		#storing the size of the data used later when sending data
		self.comm.Gather(sendbuf=[np.array(self.n * self.K,dtype='i'), MPI.INT], recvbuf=[self.counts, MPI.INT], root=0)  # @UndefinedVariable

	def set_prior(self, prior, init = False):
		
		self.set_prior_param0()
		if prior.noise_class:
			self.add_noise_class(mu = prior.noise_mu, Sigma = prior.noise_Sigma)

		self.set_location_prior(prior)
		self.set_var_prior(prior)

		for GMM in self.GMMs:
			GMM.alpha_vec = prior.a
			
		if init:
			self.set_latent_init(prior)
			self.set_GMM_init()

	def set_location_prior(self,prior):
		if self.comm.Get_rank() == 0:
			for k in range(self.K):
				thetaprior = {}
				thetaprior['mu'] = prior.t[k,:]
				thetaprior['Sigma'] = np.diag(prior.S[k,:])
				self.normal_p_wisharts[k].theta_class.set_prior(thetaprior)
				
	def set_var_prior(self,prior):
		self.set_Sigma_theta_prior(prior.Q,prior.n_theta)
		self.set_Psi_prior(prior.H,prior.n_Psi)

	def set_Sigma_theta_prior(self,Q,n_theta):
		if self.comm.Get_rank() == 0:
			for k in range(self.K):
				Sigmathprior = {}
				Sigmathprior['Q'] = Q[k]*np.eye(self.d)
				Sigmathprior['nu'] = n_theta[k]
				self.normal_p_wisharts[k].Sigma_class.set_prior(Sigmathprior)
				
	def set_Psi_prior(self,H,n_Psi):
		if self.comm.Get_rank() == 0:
			for k in range(self.K):	            
				Psiprior = {}
				Psiprior['Qs'] = 1/H[k]*np.eye(self.d)
				Psiprior['nus'] = n_Psi[k]
				self.wishart_p_nus[k].Q_class.set_prior(Psiprior)
				
	def set_latent_init(self,prior):
		rank = self.comm.Get_rank()
		if rank == 0:
			for k in range(self.K):
				npw = self.normal_p_wisharts[k]
				npwparam = {}
				npwparam['theta'] = npw.theta_class.mu_p
				npwparam['theta'] = npwparam['theta'] + np.random.normal(0,.3,self.d)
				if k >= prior.K_inf:
					npwparam['theta'] = npwparam['theta'] + np.random.normal(0,.3,self.d)
				npwparam['Sigma'] = npw.Sigma_class.Q/(npw.Sigma_class.nu-self.d-1)
				npw.set_parameter(npwparam)
	
				wpn = self.wishart_p_nus[k]
				wpnparam = {}
				wpnparam['Q'] = np.linalg.inv(wpn.Q_class.Q_s)*wpn.Q_class.nu_s
				wpnparam['nu'] = wpn.Q_class.nu_s
				wpn.set_parameter(wpnparam)
				wpn.nu_class.set_val(wpn.Q_class.nu_s)

	def set_GMM_init(self):
		param = [None]*self.K
		for k in range(self.K):
			param[k] = {}
			param[k]['mu'] = self.GMMs[0].prior[k]['mu']['theta'].reshape(self.d)
			param[k]['sigma'] = self.GMMs[0].prior[k]['sigma']['Q']/(self.GMMs[0].prior[k]['sigma']['nu']-self.d-1)
		for GMM in self.GMMs:
			GMM.set_param(param)
			GMM.p = GMM.alpha_vec/sum(GMM.alpha_vec)
			GMM.active_komp = np.ones(self.K+self.noise_class,dtype='bool')

	def deactivate_outlying_components(self):
		any_deactivated = 0
		for GMM in self.GMMs:
			any_deactivated = max(any_deactivated,GMM.deactivate_outlying_components())
			
		if self.comm.Get_rank() == 0:
			any_deactivated_all = np.empty(self.comm.Get_size(),dtype = 'i')
		else:
			any_deactivated_all = 0
		self.comm.Gather(sendbuf=[np.array(any_deactivated,dtype='i'), MPI.INT], recvbuf=[any_deactivated_all, MPI.INT], root=0)
		if self.comm.Get_rank() == 0:
			any_deactivated = np.array([max(any_deactivated_all)])
		else:
			any_deactivated = np.array([any_deactivated])
		self.comm.Bcast([any_deactivated, MPI.INT],root=0)
		
		return any_deactivated

	def get_thetas(self):
		"""
			collecting all latent parameters from the classes
		"""
		
		rank = self.comm.Get_rank()  # @UndefinedVariable
		thetas = None	
		
		
		if rank == 0:
			thetas = np.array([npw.param['theta'] for npw in self.normal_p_wisharts])
			
		return thetas

	def get_Qs(self):
		
		
		rank = self.comm.Get_rank()  # @UndefinedVariable
		Qs = None	
		
		
		if rank == 0:
			Qs = np.array([wpn.param['Q'] for wpn in self.wishart_p_nus])
			
		return Qs	
		
	def get_nus(self):
		
		
		rank = self.comm.Get_rank()  # @UndefinedVariable
		nus = None	
		
		
		if rank == 0:
			nus = np.array([wpn.param['nu'] for wpn in self.wishart_p_nus])
			
		return nus
				
	def get_mus(self):
		"""
			Collects all mu and sends them to rank ==0
			returns:
			
			mu  - (NxKxd) N - the number of persons
							   K - the number of classes
							   d - the dimension
							   [i,j,:] - gives the i:th person j:th class mean covariate
		"""
		rank = self.comm.Get_rank()  # @UndefinedVariable
		
		if rank == 0:
			recv_obj = np.empty((self.n_all, self.K, self.d ),dtype='d')
		else:
			recv_obj = None
		
		
		send_obj = np.array([[GMM.mu[k].flatten() for k in range(self.K) ]  for GMM in self.GMMs ],dtype='d')
		self.comm.Gatherv(sendbuf=[send_obj, MPI.DOUBLE], recvbuf=[recv_obj, (self.counts * self.d , None), MPI.DOUBLE],  root=0)  # @UndefinedVariable
		
			
		return recv_obj
	
	def get_labelswitches(self):
		"""
			Collects all the label switches made in the previous iteration
		"""
		rank = self.comm.Get_rank()  # @UndefinedVariable
		
		if rank == 0:
			recv_obj = np.empty((self.n_all,  2 ),dtype='d')
		else:
			recv_obj = None
		
		
		send_obj = np.array([GMM.lab.flatten()   for GMM in self.GMMs ],dtype='d')
		self.comm.Gatherv(sendbuf=[send_obj, MPI.DOUBLE], recvbuf=[recv_obj, ((self.counts * 2 )/self.K , None), MPI.DOUBLE],  root=0)  # @UndefinedVariable

		return recv_obj
	
	def get_activekompontent(self):
		"""
			returning the vector over all active components
		"""
		rank = self.comm.Get_rank()  # @UndefinedVariable
		
		if rank == 0:
			recv_obj = np.empty((self.n_all, self.K ),dtype='d')
		else:
			recv_obj = None
		
		
		send_obj = np.array([GMM.active_komp.flatten()  for GMM in self.GMMs ],dtype='d')
		self.comm.Gatherv(sendbuf=[send_obj, MPI.DOUBLE], recvbuf=[recv_obj, (self.counts  , None), MPI.DOUBLE],  root=0)  # @UndefinedVariable

		return recv_obj		
		
		
	
	def get_ps(self):
		"""
			Collects all p and sends them to rank ==0
			returns:
			
			p  - (NxK) 		   N - the number of persons
							   K - the number of classes
							  
		"""
		rank = self.comm.Get_rank()  # @UndefinedVariable
		
		if rank == 0:
			recv_obj = np.empty((self.n_all, self.K  + self.noise_class),dtype='d')
		else:
			recv_obj = None
		
		
		send_obj = np.array([GMM.p.flatten()   for GMM in self.GMMs ],dtype='d')
		self.comm.Gatherv(sendbuf=[send_obj, MPI.DOUBLE], recvbuf=[recv_obj, (self.counts * (self.K + self.noise_class)/self.K , None), MPI.DOUBLE],  root=0)  # @UndefinedVariable
		
			
		return recv_obj

	
	def get_Sigmas(self):
		"""
			Collects all Sigmas and sends them to rank ==0
			
			returns:
			
			Sigma  - (NxKxdxd) N - the number of persons
							   K - the number of classes
							   d - the dimension
							   [i,j,:,:] - gives the i:th person j:th class covariance matrix	
		"""
		rank = self.comm.Get_rank()  # @UndefinedVariable
		
		if rank == 0:
			recv_obj = np.empty((self.n_all, self.K, self.d ,self.d),dtype='d')
		else:
			recv_obj = None
		
		# self.counts is number of classes times the number of data
		send_obj = np.array([[GMM.sigma[k].flatten() for k in range(self.K) ]  for GMM in self.GMMs ],dtype='d')
		self.comm.Gatherv(sendbuf=[send_obj, MPI.DOUBLE], recvbuf=[recv_obj, (self.counts * self.d * self.d , None), MPI.DOUBLE],  root=0)  # @UndefinedVariable
		
			
		return recv_obj

	def sampleY(self):
		"""
			draws a sample from the joint distribution of all persons
		"""
		rank = self.comm.Get_rank()  # @UndefinedVariable
		
		if rank == 0:
			recv_obj = np.empty((self.n_all, self.d + 1),dtype='d')
		else:
			recv_obj = None
		
		send_obj = np.array([np.hstack((GMM.simulate_one_obs().flatten(),GMM.n))  for GMM in self.GMMs])
		
		# self.counts is number of classes times the number of data, thus self.countsself.K is only the number of data vectors for each mpi object
		self.comm.Gatherv(sendbuf=[send_obj, MPI.DOUBLE], recvbuf=[recv_obj, (self.counts/self.K * (self.d + 1) , None), MPI.DOUBLE],  root=0)  # @UndefinedVariable
		
		
		if rank == 0:
			prob = recv_obj[:,-1]
			Y = recv_obj[npr.choice(range(self.n_all), p = prob/np.sum(prob)),:-1]
		else:
			Y = None
		return Y

		
	def plot_mus(self,dim, ax = None, cm = plt.get_cmap('gist_rainbow'), size_point = 1, colors = None):
		"""
			plots all the posteriror mu's dimension dim into ax
		
		"""
		

		
		mus = self.get_mus()
		
		
		if self.comm.Get_rank() == 0:
			
			if colors != None:
				if len(colors) != self.K:
					print "in hier_GMM_MPI.plot_mus: can't use colors aurgmen with length not equal to K"
					return
		
			
			if ax == None:
				f = plt.figure()
				if len(dim) < 3:
					ax = f.add_subplot(111)
				elif len(dim) == 3:
					ax = f.gca(projection='3d')
			else:
				f = None
			
			if len(dim) == 1:
				
				print("one dimension not implimented yet")
				pass
			
			elif len(dim) == 2:
				
				
				
				
				
				for k in range(self.K):
					mu_k = np.empty((self.n_all,self.d)) 
					mu_k[:] = mus[:,k,:]
					index = np.isnan(mu_k[:,0])==False
					if colors != None:
						ax.plot(mu_k[index,dim[0]],mu_k[index,dim[1]],'.',color=cm(k/self.K), s = size_point)
					else:
						ax.plot(mu_k[index,dim[0]],mu_k[index,dim[1]],'.',color=colors[k], s = size_point)
				return f, ax
				
			elif len(dim) == 3:
				
				cm = plt.get_cmap('gist_rainbow')
				for k in range(self.K):
					mu_k = np.empty((self.n_all,self.d)) 
					mu_k[:] = mus[:,k,:]
					index = np.isnan(mu_k[:,0])==False
					if colors == None:
						ax.scatter(mu_k[index,dim[0]],mu_k[index,dim[1]],mu_k[index,dim[2]],marker = '.',color=cm(k/self.K),edgecolor=cm(k/self.K), s=size_point)
					else:
						ax.scatter(mu_k[index,dim[0]],mu_k[index,dim[1]],mu_k[index,dim[2]],marker = '.',color=colors[k],edgecolor=colors[k], s=size_point)
				return f, ax	
		
			else:
				print("more then three dimensions thats magic!")
		
		return None, None
	
	
	def sample(self):
		
		for GMM in self.GMMs:
			GMM.sample() 
		self.update_prior()
		if self.comm.Get_rank() == 0:
			for k in range(self.K):
				self.normal_p_wisharts[k].sample()
				self.wishart_p_nus[k].sample()
		self.comm.Barrier()
		self.update_GMM()
		

	def simulate(self,simpar,name='simulation',printfrq=100,timed=False,stop_if_cl_off=True):
		if stop_if_cl_off:
			warnings.filterwarnings("error",'One cluster turned off in all samples')
		else:
			warnings.filterwarnings("default",'One cluster turned off in all samples')			
		if timed:
			t_GMM = 0
			t_load = 0
			t_rest = 0
			t_save = 0
		iterations = np.int(simpar['iterations'])
		self.set_simulation_param(simpar)
		hmlog = getattr(HMlog,simpar['logtype'])(self,iterations,**simpar['logpar'])
		for i in range(iterations):
			if i%printfrq == 0 and self.comm.Get_rank() == 0:
				print "{} iteration = {}".format(name,i)
				if timed:
					print "{} iteration = {}".format(name,i)
					print("hgmm GMM  %.4f sec/sim"%(t_GMM/(i+1)))
					print("hgmm load  %.4f sec/sim"%(t_load/(i+1)))
					print("hgmm rank=0  %.4f sec/sim"%(t_rest/(i+1)))
					print("save  %.4f sec/sim"%(t_save/(i+1)))					
			try:
				if not timed:
					self.sample()
					hmlog.savesim(self)
				else:
					if  self.comm.Get_rank() == 0:
						t0 = time.time()
		        		for GMM in self.GMMs:
		        			GMM.sample() 
													
					if  self.comm.Get_rank() == 0:
						t1 = time.time()
						t_GMM += np.double(t1-t0) 
		        		self.update_prior()
												
					if  self.comm.Get_rank() == 0:
						t2 = time.time()
						t_load += np.double(t2-t1) 
		        		if self.comm.Get_rank() == 0:
		        			for k in range(self.K):
		        				self.normal_p_wisharts[k].sample()
		        				self.wishart_p_nus[k].sample()
		        		self.comm.Barrier()
		        		self.update_GMM()
					if  self.comm.Get_rank() == 0:
						t3 = time.time()
						t_rest += np.double(t3-t2) 
		  
					hmlog.savesim(self)
					if  self.comm.Get_rank() == 0:
						t4 = time.time()
						t_save += np.double(t4-t3)

			except UserWarning as w:
				raise SimulationError(str(w),name=name,it=i)
		hmlog.postproc()
		
		return hmlog

	def simulate_timed(self,simpar,name='simulation',printfrq=100):
		t_GMM = 0
		t_load = 0
		t_rest = 0
		t_save = 0
		debug = False
		iterations = np.int(simpar['iterations'])
		self.set_simulation_param(simpar)
		hmlog = getattr(HMlog,simpar['logtype'])(self,iterations,**simpar['logpar'])
		for i in range(iterations):
			if i%printfrq == 0 and self.comm.Get_rank() == 0:
				print "{} iteration = {}".format(name,i)
				print("hgmm GMM  %.4f sec/sim"%(t_GMM/(i+1)))
				print("hgmm load  %.4f sec/sim"%(t_load/(i+1)))
				print("hgmm rank=0  %.4f sec/sim"%(t_rest/(i+1)))
				print("save  %.4f sec/sim"%(t_save/(i+1)))
			if  self.comm.Get_rank() == 0:
				t0 = time.time()
        		for GMM in self.GMMs:
        			GMM.sample() 
											
			if  self.comm.Get_rank() == 0:
				t1 = time.time()
				t_GMM += np.double(t1-t0) 
        		self.update_prior()
										
			if  self.comm.Get_rank() == 0:
				t2 = time.time()
				t_load += np.double(t2-t1) 
        		if self.comm.Get_rank() == 0:
        			for k in range(self.K):
        				self.normal_p_wisharts[k].sample()
        				self.wishart_p_nus[k].sample()
        		self.comm.Barrier()
        		self.update_GMM()
			if  self.comm.Get_rank() == 0:
				t3 = time.time()
				t_rest += np.double(t3-t2) 
  
			hmlog.savesim(self)
			if  self.comm.Get_rank() == 0:
				t4 = time.time()
				t_save += np.double(t4-t3) 
		if debug:
			print "Iterations done at rank {}".format(MPI.COMM_WORLD.Get_rank())
		hmlog.postproc()
		if debug:
			print "Postproc done"
		
		return hmlog