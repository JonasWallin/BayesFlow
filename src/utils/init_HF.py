
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 20:02:37 2014

@author: johnsson

Functions for initializing hGMM for healthyFlowData.
"""
from __future__ import division
import numpy as np
from mpi4py import MPI
#import re

def add_noise_class(hGMM):
    hGMM.add_noise_class(1)
    hGMM.noise_class = 1
    for GMM in hGMM.GMMs:
        GMM.noise_mean = 0.5*np.ones(hGMM.d)
        GMM.noise_sigma = 0.5**2*np.eye(hGMM.d)
        GMM.update_noiseclass()

def set_init(hGMM,randthall,nbrnonrandth=None):

    set_latent_init(hGMM,randthall,nbrnonrandth)            
    hGMM.update_GMM()

    K = hGMM.K
    d = hGMM.d

    param = [None]*K
    for k in range(K):
        param[k] = {}
        param[k]['mu'] = hGMM.GMMs[0].prior[k]['mu']['theta'].reshape(d)
        param[k]['sigma'] = hGMM.GMMs[0].prior[k]['sigma']['Q']/(hGMM.GMMs[0].prior[k]['sigma']['nu']-d-1)
    for GMM in hGMM.GMMs:
        GMM.set_param(param)
        GMM.p = GMM.alpha_vec/sum(GMM.alpha_vec)

    #hGMM.update_prior() # Is done during sampling, before sampling latent layer


def set_latent_init(hGMM,randthall,nbrnonrandth=None,hGMMprev=None):

    K = hGMM.K
    d = hGMM.d

    if MPI.COMM_WORLD.Get_rank() == 0:
        for k in range(K):
            npw = hGMM.normal_p_wisharts[k]
            if hGMMprev is None:
                npwparam = {}
                npwparam['theta'] = npw.theta_class.mu_p
                if randthall:
                    npwparam['theta'] = npwparam['theta'] + np.random.normal(0,.3,d)
                elif not nbrnonrandth is None:
                    if k >= nbrnonrandth:
                        npwparam['theta'] = npwparam['theta'] + np.random.normal(0,.3,d)
                npwparam['Sigma'] = npw.Sigma_class.Q/(npw.Sigma_class.nu-d-1)
            else:
                npwparam = hGMMprev.normal_p_wisharts[k].param
            npw.set_parameter(npwparam)
            
            wpn = hGMM.wishart_p_nus[k]
            if hGMMprev is None:
                wpnparam = {}
                wpnparam['Q'] = np.linalg.inv(wpn.Q_class.Q_s)*wpn.Q_class.nu_s
                wpnparam['nu'] = wpn.Q_class.nu_s
            else:
                wpnparam = hGMMprev.wishart_p_nus[k].param
            wpn.set_parameter(wpnparam)
            
            if hGMMprev is None:
                wpn.nu_class.set_val(wpn.Q_class.nu_s)
            else:
                wpn.nu_class.set_val(hGMMprev.wishart_p_nus[k].Q_class.nu_s)

def set_prior(hGMM,pri):

    K = hGMM.K
    d = hGMM.d
    if hGMM.K != pri.K:
        raise ValueError, 'inconsistent number of components'
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        for k in range(K):
            thetaprior = {}
            thetaprior['mu'] = pri.t[k,:]
            thetaprior['Sigma'] = np.diag(pri.S[k,:])
            hGMM.normal_p_wisharts[k].theta_class.set_prior(thetaprior)
            
            Sigmathprior = {}
            Sigmathprior['Q'] = pri.Q[k]*np.eye(d)
            Sigmathprior['nu'] = pri.n_theta[k]
            hGMM.normal_p_wisharts[k].Sigma_class.set_prior(Sigmathprior)
            
            Psiprior = {}
            Psiprior['Qs'] = 1/pri.H[k]*np.eye(4)
            Psiprior['nus'] = pri.n_Psi[k]
            hGMM.wishart_p_nus[k].Q_class.set_prior(Psiprior)
            hGMM.wishart_p_nus[k].nu_class.sigma = pri.sigma_nu[k]
    #hGMM.set_nu_MH_param(min(hfp.sigma_nu), hfp.iter_nu)
           
    for GMM in hGMM.GMMs:
        GMM.alpha_vec = pri.a

    if hasattr(pri, 'p_sw'):        
        hGMM.set_p_labelswitch(pri.p_sw)

    if hasattr(pri, 'p_sw_mod'):
        for GMM in hGMM.GMMs:
            GMM.p_switch_mod = pri.p_sw_mod
            GMM.nu_sw = pri.nu_sw
            GMM.Sigma_mu_sw = pri.Sigma_th_sw    

    if hasattr(pri, 'p_on_off'):
        for GMM in hGMM.GMMs:
            GMM.set_p_activation(pri.p_on_off)
      




#def copy_object(obj,objprev,nocopy):
#
#    attr_list = objprev.__dict__.keys()
#    #print "Original attributes: {}".format(attr_list)
#    rm_attr = nocopy
#    for rm in rm_attr:
#        if attr_list.count(rm) > 0:
#            attr_list.remove(rm)
#    #print "Copied attributes: {}".format(attr_list)
#    for attr in attr_list:
#        setattr(obj,attr,getattr(objprev,attr))


