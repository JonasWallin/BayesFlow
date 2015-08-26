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
			for thing in args:
				if len(thing.shape) > 2:
					thing[j,labels[j,0],:], thing[j,labels[j,1],:] = thing[j,labels[j,1],:], thing[j,labels[j,0],:] 
				else:
					thing[j,labels[j,0]], thing[j,labels[j,1]] = thing[j,labels[j,1]], thing[j,labels[j,0]]  
				