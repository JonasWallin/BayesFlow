
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 20:02:37 2014

@author: johnsson
"""
from __future__ import division
import numpy as np
import copy

class SimulationParam(object):
	
    def __init__(self,pri):
        """
            Define experiment parameters
        """
        self.expname = 'test'
        self.rond = 1

        self.loadinit = False
        self.loadrond = None

        self.seed = 15
        
        nbriter = 200000
        self.nbriter = nbriter
        self.printfrq = 100
        
        qburn = 0.5
        qprod = 1 - qburn      
        self.qburn = qburn
        self.qprod = qprod        
        
        nbrsaveit = min(1000,nbriter)
        nbrsimy = min(20000,nbriter*qprod)
        
        qB1 = 0.1
        qB2 = 1 - qB1
     
        
        self.phaseB1 = {}
        self.phaseB1['iterations'] =  np.int(nbriter*qburn*qB1) # nbr iterations
        self.phaseB1['p_sw'] = 0.1 # probability of label switch
        self.phaseB1['p_on_off'] = [0, 0] #probability of activating/deactivating clusters
        self.phaseB1['nu_MH_par'] = {
            'sigma': [max(pri.d,np.ceil(.1*pri.n_Psi[k])) for k in range(pri.K)], # MH proposal step
            'iteration': 5 # Nbr of MH iterations
        }
        self.phaseB1['logtype'] = 'HMlogB'
        self.phaseB1['logpar'] = {'nbrsave': np.int(nbrsaveit*qburn*qB1)}
        
        
        self.phaseB2 = copy.deepcopy(self.phaseB1)
        self.phaseB2['iterations'] = np.int(nbriter*qburn*qB2)
        self.phaseB2['nu_MH_par'] = {'sigma_nuprop': 0.1}
        self.phaseB2['logpar']['nbrsave'] = np.int(nbrsaveit*qburn*qB2)

        self.phaseP = {}
        self.phaseP['iterations'] = np.int(nbriter*qprod) 
        self.phaseP['p_sw'] = 0
        self.phaseP['p_on_off'] = [0, 0]
        self.phaseP['nu_MH_par'] = None
        self.phaseP['logtype'] = 'HMElog'
        self.phaseP['logpar'] = {
            'nbrsave':nbrsaveit*qprod,
            'nbrsavey': nbrsimy,
            'savesamp': [0,1]
        }
								
class PostProcParam(object):

    def __init__(self):
        '''
            Define merging method and parameters
        '''
        self.postproc = True # Should postprocessing be done after MCMC iterations. Note that this is done on a single core.
        self.mergemeth = 'bhat_hier_dip'
        self.mergekws = {'thr': 0.47, 'lowthr': 0.08, 'dipthr': 0.28}

class Prior(object):
    
    
    def __init__(self):
        
        self.K = 17
        self.noise_class = 1
        self.d = 4
        
        """
            Define priors on latent cluster means
        """
        pos = 0.85
        neg = 1 - pos
        posneg = 0.05**2
        notpn = 100**2
        
        ts = np.array([[0.5, 0.5, neg, pos],
                    [pos, neg, pos, neg],
                    [neg, pos, pos, neg],
                    [neg, neg, pos, neg],
                    [0.5, 0.5, neg, neg]])

        Sks = posneg*np.ones(ts.shape)
        Sks[ts == 0.5] = notpn
        if ts.shape[0] < self.K:
            self.K_inf = ts.shape[0]
            ts = np.vstack([ts,0.5*np.ones((self.K-self.K_inf,self.d))])
            Sks  = np.vstack([Sks,10**6*np.ones((self.K-self.K_inf,self.d))])
        else:
            self.K_inf = 0

        self.t = ts
        self.S = Sks
        
        """
            Define priors on component mean spread
        """
        self.n_theta = 1000*np.ones(self.K)
        self.Q = (self.n_theta-self.d-1)*10**-4*np.ones(self.K)
        
        """
            Define priors on component covariance shape
        """
        self.n_Psi = 50*np.ones(self.K)
        self.H = (self.n_Psi-self.d-1)/self.n_Psi*(1/3)**2
        
        
        """
            Define prior on population sizes
        """
        self.a = np.ones(self.K + self.noise_class)

        """
            Define noise class parameters							
        """
        self.noise_mu = 0.5*np.ones(self.d)
        self.noise_Sigma = 0.5**2*np.eye(self.d)




