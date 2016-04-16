# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 23:33:57 2014

@author: jonaswallin
"""

import numpy as np
import numpy.random as npr

#import PurePython.GMM
from .PurePython import GMM as PPGMM
#import BayesFlow.mixture_util.GMM_util as GMM_util
from .mixture_util import GMM_util
from .plot import component_plot
#import bayesianmixture.distributions.rng_cython as rng_cython
from logisticnormal import LogisticMNormal
class mixture(PPGMM.mixture):
    """
        The main Gaussian mixture model
    """
    def __init__(self, data, K,  prior = None, high_memory=True , name = None):
        super(mixture,self).__init__(None,K = K, prior = prior, high_memory = high_memory, name = name)
        #self.rng = rng_cython.rng_class()
        self.set_data(data)
        self.x_count = np.empty(K,dtype=np.int)
        self.logisticNormal  = LogisticMNormal({'mu':np.zeros(K-1),'Sigma':2*10**2*np.eye(K - 1)})
        
        self.logisticNormal.set_alpha(np.zeros(K-1))
        self.alpha_vec = None
        
    def load_param(self, params):    
        
        super(mixture,self).load_param(params)
        if self.noise_class == 1:
            self.x_index = np.empty((self.n, self.K + 1),dtype=np.int) 
            self.x_count = np.empty(self.K + 1,dtype=np.int)
            
        
        
    def add_noiseclass(self, Sigma_scale = 5., mu = None, Sigma = None):
        """
            adds a class that does not update and cant be deactiveted or label switch
            the data need to be loaded first!
            
            Sigma_scale  - (double)  the scaling constants time the covariance matrix
        """
        super(mixture,self).add_noiseclass(Sigma_scale,mu,Sigma)
        self.x_index = np.empty((self.n, self.K + 1),dtype=np.int) 
        self.x_count = np.empty(self.K + 1,dtype=np.int)
        
    def set_data(self, data):
        if not data is None:
            super(mixture,self).set_data(data)
            self.x_index = np.empty((self.n, self.K),dtype=np.int)
        
    def sample_mu(self):
        """
            Draws the mean parameters
            self.mu[k] - (d,) np.array
        """   
        
        
        
        for k in range(self.K):  
            if self.active_komp[k] == True:
                theta = self.prior[k]['mu']['theta'].reshape(self.d)
                self.mu[k] = GMM_util.sample_mu(self.data, self.x_index, self.x_count,
                                            self.sigma[k], 
                                            theta,
                                            self.prior[k]['mu']['Sigma'],
                                            np.int(k))
            else:
                self.mu[k] = np.NAN * np.ones(self.d)
            
        self.updata_mudata()
        
    def sample_labelswitch(self):
        """
            Tries to switch two random labels
        """    
    
        if npr.rand() < self.p_switch:
                if self.K < 2:
                    return np.array([-1, -1])
                labels = npr.choice(self.K,2,replace=False)
                if np.sum(self.active_komp[labels]) == 0:
                        return np.array([-1,-1])
                    
                lik_old, R_S_mu0, log_det_Q0, R_S0  = self.likelihood_prior(self.mu[labels[0]],self.sigma[labels[0]], labels[0], switchprior = True)
                lik_oldt, R_S_mu1, log_det_Q1, R_S1 = self.likelihood_prior(self.mu[labels[1]],self.sigma[labels[1]], labels[1], switchprior = True)
                
                #updated added alpha contribution
                alpha_    = np.zeros_like(self.logisticNormal.alpha)
                alpha_[:] = self.logisticNormal.alpha[:]
                llik_alpha, _, __ = self.logisticNormal.get_lprior_grad_hess(alpha_)
                
                self.p[labels[0]], self.p[labels[1]] = self.p[labels[1]], self.p[labels[0]]
                self.logisticNormal.set_alpha_p(self.p)
                lliks_alpha, _, __ = self.logisticNormal.get_lprior_grad_hess()
                
                lik_old += lik_oldt + llik_alpha
                lik_star = self.likelihood_prior(self.mu[labels[1]],self.sigma[labels[1]], labels[0], R_S_mu0, log_det_Q0, R_S1, switchprior = True)[0]
                lik_star += self.likelihood_prior(self.mu[labels[0]],self.sigma[labels[0]], labels[1], R_S_mu1,log_det_Q1, R_S0, switchprior = True)[0]
                lik_star += lliks_alpha
                if np.log(npr.rand()) < lik_star - lik_old:
                        self.active_komp[labels[0]], self.active_komp[labels[1]] = self.active_komp[labels[1]], self.active_komp[labels[0]]
                        self.mu[labels[0]], self.mu[labels[1]] = self.mu[labels[1]], self.mu[labels[0]]
                        self.sigma[labels[0]], self.sigma[labels[1]] = self.sigma[labels[1]], self.sigma[labels[0]]
                        self.p[labels[0]], self.p[labels[1]] = self.p[labels[1]], self.p[labels[0]]
                        self.updata_mudata()
                        return labels
                self.logisticNormal.set_alpha(alpha_)
                self.p = self.logisticNormal.get_p()
        
        return np.array([-1,-1])        
        
    def likelihood_prior(self, mu, Sigma, k, R_S_mu = None, log_det_Q = None, R_S = None, switchprior = False):
            """
                    Computes the prior that is 
                    \pi( \mu | \theta[k], \Sigma[k]) \pi(\Sigma| Q[k], \nu[k]) = 
                    N(\mu; \theta[k], \Sigma[k]) IW(\Sigma; Q[k], \nu[k]) 
                    added:
                    logistic normal contribution -> - (\alpha_l - \mu_l)^T Q_l (\alpha_l - \mu_l)/2

                    If switchprior = True, special values of nu and Sigma_mu
                    are used if the parameters nu_sw and Sigma_mu_sw are set
                    respectively. This enables use of "relaxed" priors
                    facilitating label switch. NB! This makes the kernel
                    non-symmetric, hence it cannot be used in a stationary state.
            """

            if switchprior:            
                try:
                    nu = self.nu_sw
                except:
                    nu = self.prior[k]['sigma']['nu']
                try:
                    Sigma_mu = self.Sigma_mu_sw
                except:
                    Sigma_mu = self.prior[k]['mu']['Sigma']
                Q = self.prior[k]['sigma']['Q']*nu/self.prior[k]['sigma']['nu']
            else:
                nu = self.prior[k]['sigma']['nu']
                Sigma_mu = self.prior[k]['mu']['Sigma']
                Q = self.prior[k]['sigma']['Q']
            
            if np.isnan(mu[0]) == 1:
                    return 0, None, None, None
            
            if R_S_mu is None:
                    R_S_mu = GMM_util.cholesky(Sigma_mu)
                    #R_S_mu = sla.cho_factor(Sigma_mu,check_finite = False)
                    
            
            
            if log_det_Q is None:
                    log_det_Q = GMM_util.log_det(Q)
            
            if R_S is None:
                    R_S = GMM_util.cholesky(Sigma)
                    #R_S = sla.cho_factor(Sigma,check_finite = False)
            
            
            
            lik = GMM_util.likelihood_prior(mu.reshape((self.d,1)),  self.prior[k]['mu']['theta'],  self.prior[k]['mu']['theta'], R_S_mu, R_S, nu,
                                        Q)
            lik = lik +  (nu * 0.5) * log_det_Q
            lik = lik - self.ln_gamma_d(0.5 * nu) - 0.5 * np.log(2) * (nu * self.d)
            
            return lik, R_S_mu, log_det_Q, R_S
        
                        
    def sample_sigma(self):
        """
            Draws the covariance parameters
        
        """
        
        if self.high_memory == True:
            
            for k in range(self.K):  
                if self.active_komp[k] == True: 
                    self.sigma[k] =  GMM_util.sample_mix_sigma_zero_mean(self.data_mu[k],self.x_index,self.x_count, k,
                     self.prior[k]["sigma"]["Q"],
                     self.prior[k]["sigma"]["nu"])
                    
                else:
                    self.sigma[k] = np.NAN * np.ones((self.d, self.d))
        else:
            for k in range(self.K):  
                if self.active_komp[k] == True: 
                    X_mu = self.data - self.mu[k]
                    self.sigma[k] =  GMM_util.sample_mix_sigma_zero_mean(X_mu,self.x_index,self.x_count, k,  # @UndefinedVariable
                     self.prior[k]["sigma"]["Q"],
                     self.prior[k]["sigma"]["nu"])
                else:
                    self.sigma[k] = np.NAN * np.ones((self.d, self.d))
                
    def sample_x(self):
        """
            Draws the label of the observations
        
        """
        self.compute_ProbX()
        GMM_util.draw_x(self.x,self.x_index,self.x_count, self.prob_X)  # @UndefinedVariable


    def set_prior_alpha(self, prior): 
        """
            updates the prior for the logistic regression parameters
            should contain ['mu'], ['Sigma'],
        """ 
        self.logisticNormal.set_prior(prior)

    def sample_p(self):
        """
            samples the posterior distribution of p, given a logistic transformation
            (warning not defined is component in active, or noise component
        """
        
        #if self.noise_class:
        #    raise Exception('self.noise_class not implimented with logistic regression')
        #TODO add option with our without missing comp
        if np.sum(self.active_komp == 0):
            raise Exception('self.active_komp not implimented with logistic regression')
        
        
        n = np.zeros(self.K + self.noise_class)
        for k in range(self.K + self.noise_class):
            if self.active_komp[k]:
                n[k] += np.sum(self.x==k)
        self.logisticNormal.set_data(n[:self.K])
        self.logisticNormal.sample()
        self.p = self.logisticNormal.get_p()
      
    def calc_lik(self, mu = None, sigma = None, p = None, active_komp = None):
        
        return np.sum(np.log(self.calc_lik_vec(mu, sigma, p, active_komp)))

    def calc_lik_vec(self, mu = None, sigma = None, p = None, active_komp = None):
        
        self.compute_ProbX(norm=False, mu = mu, sigma = sigma,p = p, active_komp =  active_komp)
        if p is None:
            p = self.p
        
        if active_komp is None:
            active_komp = self.active_komp
            
        
        l = np.zeros(self.n)
        for k in range(self.K + self.noise_class): 
            if active_komp[k] == True:
                l += np.exp(self.prob_X[:,k])*p[k]
        
        return l
    
            
    def compute_ProbX(self,norm =True, mu = None, sigma = None, p =None, active_komp = None):
        """
            Computes the E[x=i|\mu,\Sigma,p,Y] 
        """
        if mu is None:
            mu = self.mu
            sigma = self.sigma
            high_memory = self.high_memory
            p = self.p
        else:
            high_memory = False
            
        if active_komp is None:
            active_komp = self.active_komp
        l = np.empty(self.n, order='C' )
        for k in range(self.K):
            
            if active_komp[k] == True:
                if high_memory == True:
                    GMM_util.calc_lik(l, self.data_mu[k], self.sigma[k])
                    
                else:
                    X_mu = self.data - mu[k].reshape(self.d)
                    
                    GMM_util.calc_lik(l,   X_mu, sigma[k])
                
                self.prob_X[:,k] = l
            else:
                self.prob_X[:,k] = 0.
        
        
        if self.noise_class:
            self.prob_X[:,self.K] = self.l_noise
        
        if norm==True:
            GMM_util.calc_exp_normalize(self.prob_X, p, np.array(range(self.K + self.noise_class), dtype = np.int )[active_komp])

    def plot_components(self, dim, ax, colors=None, lw=2):
        return component_plot(self.mu, self.sigma, dim, ax, colors, lw)
    
    
