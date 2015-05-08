# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 10:25:01 2015

@author: johnsson
"""
from __future__ import division
import numpy as np
from mpi4py import MPI
import os
import inspect

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def setup_sim(expdir,seed,setupfile=None,**kws):#tightfac=1,i_th=None):
    '''
        Define save and load directories, create save directories and copy experiment setup
    '''
    if rank == 0:
        runarr = np.array([1],dtype='i')
        while os.path.exists(expdir+'run'+str(runarr[0])+'/'):
            runarr += 1
    else:
        runarr = np.array([0],dtype='i')
    comm.Bcast([runarr, MPI.INT],root=0)
    run = runarr[0]

    savedir = expdir +'run'+str(run)+'/'
    #if simpar.loadinit:
    #    loaddir = expdir + 'run'+str(simpar.loadrond)+'/hGMM/burn/'

    if rank == 0:
        for dr in [savedir,savedir+'hGMM/burn/',savedir+'hGMM/prod/']:
            if not os.path.exists(dr):
                os.makedirs(dr)
        simfile = inspect.getouterframes(inspect.currentframe())[1][1]
        os.system("cp \""+simfile+"\" "+savedir)
        if not setupfile is None:
            os.system("cp \""+setupfile+"\" "+savedir)
        for kw in kws:
            if not kw is None:
                with open(savedir+kw+'.dat','w') as f:
                    f.write(str(kws[kw]))
    '''
        Set seed
    '''
    if rank == 0:
        with open(savedir+'seed.dat','w') as f:
            f.write(str(seed))        
    comm.Barrier()
    with open(savedir+'seed.dat','r') as f:
        seed = np.int(f.readline())
    np.random.seed(seed)
    print "seed set to {} at rank {}".format(seed,rank)

    return savedir,run

class Prior(object):
    
    def __init__(self,J,n_J,d,K):
        """
            J   -   number of flow cytometry samples
            n_J -   number of events per flow cytometry sample
            d   -   dimension of a sample
        """      
        self.J = J
        self.n_J = n_J
        self.d = d
        self.K = K
        self.noise_class = 0
    
    def latent_cluster_means(self,t_inf=None,t_ex=np.nan,Sk_inf=None,Sk_ex=np.nan):
        """
            Priors on latent cluster means

            t_inf   -   (K_inf x d) matrix. Expected values for components with informative priors.
            t_ex    -   scalar or (1 x d) matrix. Expected value for components with non-informative priors.
            Sk_inf  -   (K_inf x d) matrix. Variance in each dimension for components with informative priors.
            Sk_ex   -   scalar or (1 x d) matrix. Variance for components with non-informataive priors.
        """
        if rank == 0:
            if t_inf is None:
                ts = np.zeros((0,self.d))
                Sks = np.zeros((0,self.d))
            else:
                ts = t_inf
                Sks = Sk_inf
            
            if ts.shape[0] < self.K:
                self.K_inf = ts.shape[0]
                ts = np.vstack([ts,t_ex*np.ones((self.K-self.K_inf,self.d))])
                Sks  = np.vstack([Sks,Sk_ex*np.ones((self.K-self.K_inf,self.d))])
            else:
                self.K_inf = 0

            self.t = ts
            self.S = Sks       

    def component_location_variance(self,nt=None,q=None,n_theta=None,Q=None):
        """
            Prior on covariance of sample component locations within latent component

            nt      -   scalar. If n_theta is not provided, n_theta will be determined based
                        on n_J and K with nt as a scaling factor.
            q       -   scalar. If Q is not provided, Q will be determined based on n_theta,
                        d, and J, with q as a scaling factor.
            n_theta -   scalar. Degrees of freedom.
            Q       -   scalar. The scale matrix will be diagonal with Q as its diagonal 
                        elements.

        """
        if rank == 0:
            if n_theta is None:
                n_theta = int(nt*self.n_J/self.K)
            self.n_theta = n_theta*np.ones(self.K,dtype='i')

            if Q is None:
                Q = q*(self.n_theta-self.d-1)*self.J
            self.Q = Q*np.ones(self.K)
     
    def component_shape(self,nps=None,min_n_Psi=12,h=None,n_Psi=None,H=None):   
        """
            Prior on component covariance shape

            nps         -   scalar. If n_Psi is not provided, n_Psi will be determined based on
                            n_J and K with nps as a scaling constant. The value will not be smaller 
                            than min_n_Psi.
            min_n_Psi   -   minimal value for n_Psi if set through scaling constant.
            h           -   scalar. If H is not provided, H will be determined based on n_Psi, d
                            and J, with h as a scaling constant.
            n_Psi       -   scalar. Degrees of freedom.
            H           -   scalar. The scale matrix will be diagonal with H as its diagonal
                            elements.
        """
        if rank == 0:
            if n_Psi is None:
                n_Psi = max([int(nps*self.n_J/self.K),min_n_Psi])
            self.n_Psi = n_Psi*np.ones(self.K,dtype='i')
       
            if H is None:
                H = h*(self.n_Psi-self.d-1)/self.n_Psi/self.J
            self.H = H

    def set_noise_class(self,noise_mu,noise_Sigma,on=True):
        """
            Noise class parameters

            noise_mu    -   scalar or (1 x d) matrix. Expected value for noise component.
            noise_Sigma -   scalar or (1 x d) matrix. Variance for noise component.    
            on          -   boolean. Should noise class be activated from start?                    
        """
        if on:
            self.noise_class = 1
        self.noise_mu = noise_mu*np.ones(self.d)
        self.noise_Sigma = noise_Sigma*np.eye(self.d)

    def pop_size(self,a=None):  
        """
            Define prior on population sizes. If noise class is used it has to be 
            set first.

            a   -   (1 x (K+noise_class)) matrix. Dirichlet distribution parameter.
        """
        if a is None:
            self.a = np.ones(self.K + self.noise_class)
        else:
            self.a = a

    def resize_var_priors(self,c):
        """
            Resizing Sigma_theta and Psi priors with factor c.
            With c >> 1 used to get similar sample components.
        """
        self.resize_Sigma_theta_prior(c)
        self.resize_Psi_prior(c)

    def resize_Sigma_theta_prior(self,c):
        """
            Resize Sigma_theta prior with factor c.
            With c >> 1 this can be used to force sample component locations together.
        """
        if rank == 0:
            n_theta_old = self.n_theta
            self.n_theta = c*self.n_theta
            self.Q = (self.n_theta-self.d-1)/(n_theta_old-self.d-1)*self.Q

    def resize_Psi_prior(self,c):
        """
            Resize Psi prior with factor c.
            With c >> 1 this can be used to force sample components to same shape.
        """
        if rank == 0:
            n_Psi_old = self.n_Psi
            self.n_Psi = c*self.n_Psi
            self.H = (self.n_Psi-self.d-1)/self.n_Psi * (n_Psi_old-self.d-1)/n_Psi_old

class PostProcPar(object):

    def __init__(self,postproc,mergemeth=None,**kw):
        '''
            Define merging method and parameters in post processing.
            NB! Merging is done on a single core.

            postproc    -   Boolean. Should postprocessing (merging) be done after MCMC interations?
        '''
        self.postproc = postproc
        self.mergemeth = mergemeth
        self.mergekws = kw

class SimPar(object):    

    def __init__(self,nbriter,qburn,tightinit=1,simsamp='all',nbrsaveit=1000,nbrsimy=20000,printfrq=100):
        """
            Simulation parameters

            nbriter     -   Total number of iterations (burnin AND production)
            qburn       -   Proportion of iterations that are burnin iterations.
            tightinit   -   Tightening factor that can be used during initial burnin phase to   
                            force components to be similar then.
            simsamp     -   'all' or list. List of flow cytometry sample indices for which synthetic data
                            should be generated during production and fail phase. If 'all', synthetic data
                            is generated for all samples.
            nbrsaveit   -   Number of iterations that should be saved for trace plots.
            nbrsimy     -   Number of synthetic data points that should be gerated for each sample 
                            specified by simsamp.
            printfrq    -   how often should progress be printed.
        """
        
        self.nbriter = nbriter
        self.printfrq = printfrq
        
        self.qburn = qburn
        self.qprod = 1 - qburn            
        
        self.nbrsaveit = min(nbrsaveit,nbriter)
        self.nbrsimy = min(nbrsimy,int(np.round(nbriter*self.qprod)))    

        self.tightinitfac = tightinit

        self.simsampnames = simsamp

        self.phases = {}

    def set(self,name,**kw):
        """
            Set switch and reversible jump parameters.

            p_sw        -   probability to propose a label swich.
            p_on_off    -   list size 2. p_on_off[0] is probability to propose to turn on a component.
                            p_on_off[1] is probability to propose to turn off a component.
        """
        self.phases[name].update(kw)

    def set_nu_MH(self,name,**kw):
        """
            Set parameters for Metropolis Hastings sampling of nu during a certain phase.

            name            -   Phase name.
            sigma           -   A new nu will sampled in interval [nu-sigma,nu+sigma].
            iteration       -   number of MH iterations per Gibbs iteration.
            sigma_nuprop    -   sigma will be set to sigma_nuprop times current value of nu
                                (individually for each latent component).   
        """
        self.phases[name]['nu_MH_par'] = kw
    
    def new_phase(self,q):
        phase = {}
        phase['iterations'] = int(np.round(self.nbriter*q))
        phase['logpar'] = {'nbrsave': int(np.round(self.nbrsaveit*q))}
        phase['nu_MH_par'] = None # Implies no change to nu MH par. Has to be reset for first phase
        return phase
        
    def new_burnphase(self,q,name):
        phase = self.new_phase(q*self.qburn)
        phase['logtype'] = 'HMlogB'
        self.phases[name] = phase
        
    def new_prodphase(self,q=1,name='P',last_burnphase=None):
        phase = self.new_phase(q*self.qprod)
        phase['logtype'] = 'HMElog'
        phase['logpar']['nbrsavey'] = self.nbrsimy
        if not last_burnphase is None:
            phase['p_sw'] = self.phases[last_burnphase]['p_sw']
            phase['p_on_off'] = self.phases[last_burnphase]['p_on_off']
        phase['logpar']['savesampnames'] = self.simsampnames
        print "Prodphase has simulation param {}".format(phase)
        self.phases[name] = phase

    def new_failphase(self,nbrit,name='F'):
        phase = {}
        phase['iterations'] = nbrit
        phase['logtype'] = 'HMElog'
        phase['logpar'] = {'nbrsave':nbrit,'nbrsavey':nbrit}
        phase['p_sw'] = 0
        phase['p_on_off'] = [0, 0]
        phase['nu_MH_par'] = None
        phase['logpar']['savesampnames'] = self.simsampnames
        self.phases[name] = phase
        
    def new_trialphase(self,nbrit,name='T'):
        phase = {}
        phase['iterations'] = nbrit
        phase['logtype'] = 'HMElog'
        phase['logpar'] = {'nbrsave':nbrit}
        phase['p_sw'] = 0
        phase['p_on_off'] = [0, 0]
        phase['nu_MH_par'] = None
        self.phases[name] = phase
        
    def new_trialphase_q(self,q,name):
        self.new_trialphase(q*self.nbrsaveit,name)