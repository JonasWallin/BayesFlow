from __future__ import division
'''
Created on Jul 10, 2014

@author: jonaswallin
'''
from mpi4py import MPI
import numpy as np
import numpy.random as npr
import os
import glob
import time
import warnings
import traceback
import sys
import json
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

from . import GMM
from .distribution import normal_p_wishart, Wishart_p_nu
from .GMM import mixture
from .utils.timer import Timer
from .utils.initialization.EM import EM_pooled
from .utils import load_fcdata
from .utils.jsonutil import ObjJsonEncoder, class_decoder 
from .exceptions import SimulationError
from . import HMlog


#TODO: change to geomtrical median instead of mean!!
#TODO: if a cluster is turned off use the latent mean insead of mu
#		and then it will work!
def distance_sort(hGMM):
    """
        sorts the clusters after the geometrical mean of the means
        UGLY SOLUTION BUT OPTIMALILITY IS NOT NEEDED
    """
    size = hGMM.comm.Get_size()  # @UndefinedVariable
    
    for com_obj in range(size):
        if hGMM.comm.Get_rank() == com_obj:
            if hGMM.comm.Get_rank() == 0:
                mu_param = [npw.param for npw in hGMM.normal_p_wisharts]
                mus = []
                for k in range(hGMM.K):
                    if np.isnan(hGMM.GMMs[0].mu[k][0])==False:
                        mus.append(hGMM.GMMs[0].mu[k].reshape((1, hGMM.d)))
                    else:
                        mus.append(mu_param[k]['theta'].reshape((1, hGMM.d)))
                GMMs = hGMM.GMMs[1:]
            else:
                GMMs = hGMM.GMMs
                mus = hGMM.comm.recv(source=0)['mus']
            
            for GMM in GMMs:
                index_k = np.argsort(GMM.p[:hGMM.K])[::-1] #sort by probabilility
                mus_t = np.array([np.mean(mu,axis=0) for mu in  mus])
                list_temp = [None for k in range(GMM.K)]
                for index in index_k:
                    #print mus_t
                    #print mus_t - GMM.mu[index]
                    if np.isnan(GMM.mu[index][0]) == False:
                        dist = np.linalg.norm(mus_t - GMM.mu[index],axis=1)
                    else:
                        dist = np.linalg.norm(mus_t - GMM.prior[index]['mu']['theta'].transpose(),axis=1)
                    i = np.argsort(dist)[0]
                    mus_t[i,:] = np.inf
                    list_temp[index] = i 
                list_temp = np.argsort(np.array(list_temp))
                mus = [np.vstack((mu,GMM.mu[list_temp[i]])) for i,mu in enumerate(mus) ]
                GMM.mu = [GMM.mu[i] for i in list_temp]
                GMM.sigma = [GMM.sigma[i] for i in list_temp]
                GMM.p[:hGMM.K] = np.array([GMM.p[i] for i in list_temp])[:]
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


# def load_hGMM(dirname):
#     '''
#         Load hGMM from a directory
#     '''
#     if dirname[-1] != '/':
#         dirname += '/'
#     K = len(glob.glob(dirname + 'normal_p_wishart*'))
#     hGMM = hierarical_mixture_mpi(K)
#     hGMM.load_from_file(dirname)
#     return hGMM


class hierarical_mixture_mpi(object):
    """    
        Comment about noise_class either all mpi hier class are noise class or none!
        either inconsistency can occur when loading
    """
    def __init__(self, K=None, data=None, sampnames=None, prior=None,
                 thetas=None, expSigmas=None, high_memory=True, timing=False,
                 AMCMC=False, comm=MPI.COMM_WORLD, init=True):
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
        self.comm = comm
        self.rank = self.comm.Get_rank()  # @UndefinedVariable
        #master
        if self.rank == 0:
            self.normal_p_wisharts = [ normal_p_wishart() for k in range(self.K)]  # @UnusedVariable
            self.wishart_p_nus     = [Wishart_p_nu(AMCMC=AMCMC) for k in range(self.K) ]  # @UnusedVariable

        else:
            self.normal_p_wisharts = None 
            self.wishart_p_nus     = None

        self.high_memory = high_memory

        set_data_out = self.set_data(data, sampnames)
        print "prior = {}".format(prior)
        if not prior is None:
            if set_data_out == 0:
                self.set_prior(prior, init=init, thetas=thetas, expSigmas=expSigmas)
            else:
                self.prior = prior

        self.timing = timing

    def mpiexceptabort(self, type_in, value, tb):
        traceback.print_exception(type_in, value, tb)
        self.comm.Abort(1)

    def encode_json(self):
        jsondict = {'__type__': 'hierarical_mixture_mpi'}
        for arg in self.__dict__.keys():
            if arg in ['GMMs', 'normal_p_wisharts', 'wishart_p_nus',
                       'comm', 'rank']:
                continue
            jsondict[arg] = getattr(self, arg)
        return jsondict

    def save(self, dirname):
        if self.rank == 0:
            for dname in [dirname, os.path.join(dirname, 'GMMs'),
                          os.path.join(dirname, 'latent')]:
                if not os.path.exists(dname):
                    os.mkdir(dname)
        self.comm.Barrier()
        if self.rank == 0:
            with open(os.path.join(dirname, 'hGMM.json'), 'w') as f:
                json.dump(self, f, cls=ObjJsonEncoder)
            self.save_prior_to_file(os.path.join(dirname, 'latent'))
        for gmm in self.GMMs:
            gmm.save_param_to_file(os.path.join(dirname, 'GMMs'))

    @classmethod
    def load(cls, dirname, comm, names, prior_class=None, **data_kws):
        with open(os.path.join(dirname, 'hGMM.json'), 'r') as f:
            hgmm = json.load(f, object_hook=lambda obj:
                             class_decoder(obj, {'hierarical_mixture_mpi': cls, 'Prior': prior_class},
                                           comm=comm, init=False))
        hgmm.load_data(names, **data_kws)
        hgmm.load_prior_from_file(os.path.join(dirname, 'latent'))
        hgmm.update_GMM()
        for gmm in hgmm.GMMs:
            gmm.load_param_from_file(os.path.join(dirname, 'GMMs'))
        return hgmm

    def save_prior_to_file(self, dirname):
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
            self.wishart_p_nus      = [ Wishart_p_nu.unpickle("%sWishart_p_nu_%d.pkl"%(dirname,k)) for k in range(self.K)]
        
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
                #    print("mu[%d] = %s"%(k, mu_k))
                Sigma_k[:] = recv_obj[:,k,self.d:].reshape((self.n_all,self.d, self.d))
                self.normal_p_wisharts[k].set_data(mu_k[index,:])
                self.wishart_p_nus[k].set_data(Sigma_k[index,:,:])

        self.comm.Bcast([cl_off, MPI.INT])  # @UndefinedVariable

        if cl_off[0]:
            warnings.warn('One cluster turned off in all samples')

    def set_simulation_param(self, sim_par):
        self.set_p_labelswitch(sim_par['p_sw'])
        if not sim_par['nu_MH_par'] is None:
            self.set_nu_MH_param(**sim_par['nu_MH_par'])
        self.set_p_activation(sim_par['p_on_off'])

    def set_nu_MH_param(self, sigma=5, iteration=5, sigma_nuprop=None):
        """
            setting the parameters for the MH algorithm

            sigma           - the sigma in the MH algorihm on the
                              natural line
            iteration       - number of time to sample using the MH algortihm
            sigma_nuprop    - if provided, the proportion of nu that will
                              be set as sigma

        """
        if self.comm.Get_rank() == 0:  # @UndefinedVariable
            for k, wpn in enumerate(self.wishart_p_nus):
                if sigma_nuprop is None:
                    if not isinstance(sigma, list):
                        wpn.set_MH_param(sigma, iteration)
                    else:
                        wpn.set_MH_param(sigma[k], iteration)
                else:
                    wpn.set_MH_param(max(np.ceil(sigma_nuprop*wpn.param['nu']),
                                         self.d), iteration)

    def set_p_activation(self, p):
        '''
            *p*   - probability of trying to switch on / switch off a cluster for a class
        '''
        for GMM in self.GMMs:
            GMM.p_act   = p[0]
            GMM.p_inact = p[1]

    def set_prior_actiavation(self, komp_prior):
        '''
            setting the expontial covariates on likelihood on that a component is active
            exp(- sum(active_kmp) * komp_prior) * sum(active_kmp)
            *komp_prior* - expontial prior? gamma prior?
        '''
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

    def load_data(self, sampnames, datadir, ext, loadfilef, **kw):
        """
            Load data corresponding to sampnames directly onto worker.
            When called multiple times, new data is appended after the
            old data.
        """
        data = load_fcdata(datadir, ext, loadfilef, comm=self.comm,
                           sampnames=sampnames, **kw)
        rank = self.comm.Get_rank()

        if hasattr(self, 'hasdata') and self.hasdata:
            self.n += len(data)
        else:
            self.n = len(data)
        ns = self.comm.gather(self.n)
        if rank == 0:
            self.n_all = np.sum(ns)
            self.counts = np.array([n*self.K for n in ns], dtype='i')
        else:
            self.counts = 0
        print "self.counts at rank {} = {}".format(rank, self.counts)

        if not hasattr(self, 'hasdata') or not self.hasdata:
            if rank == 0:
                self.d = data[0].shape[1]
            else:
                self.d = 0
            self.d = self.comm.bcast(self.d)

        for Y, name in zip(data, sampnames):
            if self.d != Y.shape[1]:
                raise ValueError('dimension mismatch in the data')
            self.GMMs.append(GMM.mixture(data=Y, K=self.K, name=name, 
                             high_memory=self.high_memory))
        self.hasdata = True

    def set_data(self, data, names=None):
        """
            List of np.arrays
            Three possible inputs:
                - data is None at all ranks => no data is set
                - data is None at all ranks except 0 => data is
                scattered from rank 0.
                - data is list of np.arrays at all ranks.
        """

        nodata = data is None
        nodata_at_0 = self.comm.bcast(nodata)
        if nodata_at_0:
            return 1

        nodata_at_any = self.comm.gather(nodata)
        if self.rank == 0:
            nodata_at_any = True in nodata_at_any
        nodata_at_any = self.comm.bcast(nodata_at_any)

        if not nodata_at_any:
            dat, names_dat = data, names
            self.d = data[0].shape[1]
            self.d = self.comm.bcast(self.d)
        else:
            if self.rank == 0:
                d = np.array(data[0].shape[1],dtype="i")
                self.n_all = len(data)
                size = self.comm.Get_size()  # @UndefinedVariable
                data = np.array(data)
                send_data = np.array_split(data,size)
                self.counts = np.empty(size,dtype='i') 
                if names is None:
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
                raise ValueError('dimension mismatch in the data: self.d = {}, Y.shape[1] = {}'.format(self.d, Y.shape[1]))
            self.GMMs.append(GMM.mixture(data= Y, K = self.K, name = name,high_memory = self.high_memory))
        #print "mpi = %d, len(GMMs) = %d"%(MPI.COMM_WORLD.rank, len(self.GMMs))  # @UndefinedVariable

        #storing the size of the data used later when sending data
        if not nodata_at_any:
            self.n_all = self.comm.gather(len(data))
            self.counts = np.array(self.comm.gather(self.n*self.K))
            if self.rank == 0:
                self.n_all = sum(self.n_all)
            else:
                self.n_all = 0
                self.counts = 0
        else:
            self.comm.Gather(sendbuf=[np.array(self.n * self.K,dtype='i'), MPI.INT], recvbuf=[self.counts, MPI.INT], root=0)  # @UndefinedVariable
        return 0

    def set_prior(self, prior, init=False, thetas=None, expSigmas=None):

        self.set_prior_param0()
        self.set_prior_actiavation(prior.lamb)
        if prior.noise_class:
            self.add_noise_class(mu=prior.noise_mu, Sigma=prior.noise_Sigma)

        self.set_location_prior(prior)
        self.set_var_prior(prior)

        if hasattr(prior, 'nu_sw'):
            for gmm in self.GMMs:
                gmm.nu_sw = prior.nu_sw

        if hasattr(prior, 'Sigma_mu_sw'):
            for gmm in self.GMMs:
                gmm.Sigma_mu_sw = prior.Sigma_mu_sw

        for gmm in self.GMMs:
            gmm.alpha_vec = prior.a

        if init:
            self.set_latent_init(prior, thetas, expSigmas)
            self.set_GMM_init()

        self.prior = prior

    def set_location_prior(self, prior):
        if self.comm.Get_rank() == 0:
            for k in range(self.K):
                thetaprior = {}
                thetaprior['mu'] = prior.t[k, :]
                thetaprior['Sigma'] = np.diag(prior.S[k, :])
                self.normal_p_wisharts[k].theta_class.set_prior(thetaprior)

    def set_var_prior(self, prior=None):
        if prior is None:
            prior = self.prior
        if self.comm.Get_rank() == 0:
            self.set_Sigma_theta_prior(prior.Q, prior.n_theta)
            self.set_Psi_prior(prior.H, prior.n_Psi)

    def set_Sigma_theta_prior(self, Q, n_theta):
        if self.comm.Get_rank() == 0:
            for k in range(self.K):
                Sigmathprior = {}
                Sigmathprior['Q'] = Q[k]*np.eye(self.d)
                Sigmathprior['nu'] = n_theta[k]
                self.normal_p_wisharts[k].Sigma_class.set_prior(Sigmathprior)

    def set_Psi_prior(self, H, n_Psi):
        if self.comm.Get_rank() == 0:
            for k in range(self.K):
                Psiprior = {}
                Psiprior['Qs'] = H[k]*np.eye(self.d)
                Psiprior['nus'] = n_Psi[k]
                self.wishart_p_nus[k].Q_class.set_prior(Psiprior)

    def resize_var_priors(self, c):
        self.prior.resize_var_priors(c)
        self.set_var_prior()

    def set_init(self, prior=None, thetas=None, expSigmas=None, method='random', **kw):
        self.set_latent_init(prior, thetas=thetas, expSigmas=expSigmas,
                             method=method, **kw)
        self.set_GMM_init()

    def set_latent_init(self, prior=None, thetas=None, expSigmas=None,
                        method='random', **initkw):

        if thetas is None:
            thetas = []
        if expSigmas is None:
            expSigmas = []

        rank = self.comm.Get_rank()

        if method == 'EM_pooled':
            data = [gmm.data for gmm in self.GMMs]
            thetas_EM, expSigmas_EM, _ = EM_pooled(self.comm, data, self.K-len(thetas), **initkw)
            thetas += thetas_EM
            expSigmas += expSigmas_EM

        if rank == 0:
            for k in range(self.K):
                # Prior for mu_jk
                npw = self.normal_p_wisharts[k]
                npwparam = {}
                if len(thetas) == 0:
                    npwparam['theta'] = npw.theta_class.mu_p
                    if k >= prior.K_inf:
                        npwparam['theta'] = (npwparam['theta']
                                             + np.random.normal(0, .3, self.d))
                else:
                    npwparam['theta'] = thetas[k].reshape(-1)
                npwparam['Sigma'] = (npw.Sigma_class.Q /
                                     (npw.Sigma_class.nu-self.d-1))
                print "npwparam = {}".format(npwparam)
                npw.set_parameter(npwparam)

                # Prior for Sigma_jk
                wpn = self.wishart_p_nus[k]
                wpnparam = {}
                wpnparam['nu'] = wpn.Q_class.nu_s
                if expSigmas is None:
                    wpnparam['Q'] = wpn.Q_class.Q_s * wpn.Q_class.nu_s
                else:
                    wpnparam['Q'] = expSigmas[k] * (wpnparam['nu']-self.d-1)
                wpn.set_parameter(wpnparam)
                wpn.nu_class.set_val(wpn.Q_class.nu_s)
        self.update_GMM()

    def set_GMM_init(self):
        self.set_GMMs_mu_Sigma_from_prior()
        for gmm in self.GMMs:
            gmm.p = gmm.alpha_vec/sum(gmm.alpha_vec)
            gmm.active_komp = np.ones(self.K+self.noise_class, dtype='bool')

    def set_GMMs_mu_Sigma_from_prior(self):
        param = [None]*self.K
        for k in range(self.K):
            param[k] = {}
            param[k]['mu'] = self.GMMs[0].prior[k]['mu']['theta'].reshape(self.d)
            param[k]['sigma'] = self.GMMs[0].prior[k]['sigma']['Q']/(self.GMMs[0].prior[k]['sigma']['nu']-self.d-1)
        for gmm in self.GMMs:
            gmm.set_param(param, active_only=True)

    def set_init_from_matched_previous(self, prior, match_comp):

        self.set_latent_init(prior, thetas=match_comp.latent.mus,
                             expSigmas=match_comp.latent.Sigmas)
        for gmm in self.GMMs:
            j = match_comp.names.index(gmm.name)
            comp = match_comp.samp_comps[j]
            print "*"*30
            print "gmm {} has matched comp {}".format(gmm.name, comp.ks)
            param = [None]*prior.K
            for k in range(prior.K):
                param[k] = {}
                param[k]['mu'] = comp.get_mu(k)
                param[k]['sigma'] = comp.get_Sigma(k)
                if k > 17:
                    if k in comp.ks:
                        print "param[{}] = {}".format(k, param[k])
            gmm.set_param(param)
            gmm.p = np.array([comp.get_p(k) for k in range(prior.K)])
            gmm.active_komp = np.array([k in comp.ks for k in range(prior.K)]
                                       + [True]*prior.noise_class, dtype='bool')

    def deactivate_outlying_components(self, aquitted=None, bhat_distance=False):
        any_deactivated_loc = 0
        for gmm in self.GMMs:
            any_deactivated_loc = max(
                any_deactivated_loc, gmm.deactivate_outlying_components(aquitted, bhat_distance))

        any_deactivated_all = self.comm.gather(any_deactivated_loc)
        if self.comm.Get_rank() == 0:
            any_deactivated = max(any_deactivated_all)
        else:
            any_deactivated = None
        return self.comm.bcast(any_deactivated)

        # if self.comm.Get_rank() == 0:
        #     any_deactivated_all = np.empty(self.comm.Get_size(),dtype = 'i')
        # else:
        #     any_deactivated_all = 0
        # self.comm.Gather(sendbuf=[np.array(any_deactivated,dtype='i'), MPI.INT], recvbuf=[any_deactivated_all, MPI.INT], root=0)
        # if self.comm.Get_rank() == 0:
        #     any_deactivated = np.array([max(any_deactivated_all)])
        # else:
        #     any_deactivated = np.array([any_deactivated])
        # self.comm.Bcast([any_deactivated, MPI.INT],root=0)
        # return any_deactivated

    def set_theta_to_median(self):
        mus = self.get_mus()
        if self.comm.Get_rank() == 0:
            medians = np.median(mus, axis=0)
            for k in range(self.K):
                npw = self.normal_p_wisharts[k]
                npwparam = {}
                npwparam['theta'] = medians[k, :]
                npwparam['Sigma'] = npw.param['Sigma']
                npw.set_parameter(npwparam)
        self.comm.Barrier()
        self.update_GMM()

    def get_thetas(self):
        """
            collecting all latent parameters from the classes
        """
        
        rank = self.comm.Get_rank()  # @UndefinedVariable
        thetas = None    
        
        
        if rank == 0:
            thetas = np.array([npw.param['theta'] for npw in self.normal_p_wisharts])
            
        return thetas

    def get_Sigma_mus(self):

        rank = self.comm.Get_rank()  # @UndefinedVariable
        Sigma_mus = None    
        
        
        if rank == 0:
            Sigma_mus = np.array([npw.param['Sigma'] for npw in self.normal_p_wisharts])
            
        return Sigma_mus
        
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

    def get_sigma_nus(self):
        sigma_nus = None

        if self.comm.Get_rank() == 0:
            sigma_nus = np.array([wpn.nu_class.sigma for wpn in self.wishart_p_nus])

        return sigma_nus

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
            recv_obj = np.empty((self.n_all,  2 ),dtype='int')
        else:
            recv_obj = None
        
        
        send_obj = np.array([GMM.lab.flatten()   for GMM in self.GMMs ],dtype='int')
        self.comm.Gatherv(sendbuf=[send_obj, MPI.DOUBLE], recvbuf=[recv_obj, ((self.counts * 2 )/self.K , None), MPI.DOUBLE],  root=0)  # @UndefinedVariable

        return recv_obj
    
    def get_activekompontent(self):
        """
            returning the vector over all active components
        """
        rank = self.comm.Get_rank()  # @UndefinedVariable
        
        if rank == 0:
            recv_obj = np.empty((self.n_all, self.K  + self.noise_class),dtype='d')
            send_size = np.zeros_like(self.counts)
            send_size[:] = self.counts[:]
            send_size[send_size>0] += self.noise_class
        else:
            recv_obj = None
            send_size = 0
        
        
        
        send_obj = np.array([GMM.active_komp.flatten()  for GMM in self.GMMs ],dtype='d')
        self.comm.Gatherv(sendbuf=[send_obj, MPI.DOUBLE], recvbuf=[recv_obj, (send_size  , None), MPI.DOUBLE],  root=0)  # @UndefinedVariable

        return recv_obj        
        
        
    
    def get_ps(self):
        """
            Collects all p and sends them to rank ==0
            returns:
            
            p  - (NxK)            N - the number of persons
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

    
        
        
    def plot_mus(self, dim, ax = None, cm = plt.get_cmap('gist_rainbow'), size_point = 1, colors = None):
        """
            plots all the posteriror mu's dimension dim into ax
        
        """
        

        
        mus = self.get_mus()
        
        
        if self.comm.Get_rank() == 0:
            
            if colors is None:
                if len(colors) != self.K:
                    print "in hier_GMM_MPI.plot_mus: can't use colors aurgmen with length not equal to K"
                    return
        
            
            if ax is None:
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
                    if colors is None:
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
                    if colors is None:
                        ax.scatter(mu_k[index,dim[0]],mu_k[index,dim[1]],mu_k[index,dim[2]],marker = '.',color=cm(k/self.K),edgecolor=cm(k/self.K), s=size_point)
                    else:
                        ax.scatter(mu_k[index,dim[0]],mu_k[index,dim[1]],mu_k[index,dim[2]],marker = '.',color=colors[k],edgecolor=colors[k], s=size_point)
                return f, ax    
        
            else:
                print("more then three dimensions thats magic!")
        
        return None, None
    
    
    def sample(self):
        """
            generate sample of all parameters and auxilleraly variable
        
        """
        
        if (self.comm.Get_rank() == 0) and self.timing:
            self.simulation_times['iteration'] += 1.
            self.simulation_times['GMM']       -= time.time()
            
        
        for GMM in self.GMMs:
            GMM.sample() 
        
        
        if (self.comm.Get_rank() == 0) and self.timing:
            self.simulation_times['GMM']          += time.time()    
            self.simulation_times['update_prior'] -= time.time()    
        
        self.update_prior()
        
        if self.comm.Get_rank() == 0:
            
            if self.timing:
                self.simulation_times['update_prior'] += time.time()    
                self.simulation_times['sample_prior'] -= time.time()    
                
            
            for k in range(self.K):
                self.normal_p_wisharts[k].sample()
                self.wishart_p_nus[k].sample()
                
            
            if self.timing:
                self.simulation_times['sample_prior'] += time.time()    
                self.simulation_times['update_GMM']   -= time.time()    
                
        self.comm.Barrier()
        self.update_GMM()
        
        if (self.comm.Get_rank() == 0) and self.timing:
            self.simulation_times['update_GMM']   += time.time()    
            
            
            
    
    def print_timing(self):
        """
            priting timing results
        """
        if self.comm.Get_rank() == 0:
            
            if self.timing:
                
                iteration = self.simulation_times['iteration']
                
                if iteration == 0:
                    print('zero iteration so for')
                    return
                
                print('for {iteration} iterations the average times where:'.format(iteration = iteration))
                for key in self.simulation_times.keys():
                    if key not in ['iteration']:
                        print('{name:12} : {time:.2e} sec/sim'.format(name = key,
                                                                  time = self.simulation_times[key] / iteration))
                
                
                if len(self.GMMs) > 0:
                    print('for GMMs[0]:')
                    self.GMMs[0].print_timing()
                
            else:
                print("timing is turned off")
            
        
    def toggle_timing(self, timing=True):
        """
            turning on alternative off timer function
            *timing* if true turn on, else turn off
        """
        
        if timing:
            self.timing = True
            
            self.simulation_times = {'GMM':        0., 
                                    'update_prior':0.,
                                    'sample_prior':0.,
                                    'update_GMM':  0., 
                                    'iteration' :  0.}
            if self.comm.Get_rank() == 0:
                if len(self.GMMs) > 0:
                    self.GMMs[0].toggle_timing()
            
        else:
            self.timing = False

    def simulate(self, simpar, name='simulation', printfrq=100,
                 stop_if_cl_off=True, plotting=False, plotdim=None):

        if self.comm.Get_size() > 1:
            sys.excepthook = self.mpiexceptabort
        if stop_if_cl_off:
            warnings.filterwarnings("error", 'One cluster turned off in all samples')
        else:
            warnings.filterwarnings("default", 'One cluster turned off in all samples')

        iterations = np.int(simpar['iterations'])
        self.set_simulation_param(simpar)
        hmlog = getattr(HMlog, simpar['logtype'])(self, iterations,
                                                  **simpar['logpar'])

        if self.timing:
            timer = Timer(self.comm)

        try:
            for i in range(iterations):
                if i % printfrq == 0:
                    if plotting:
                        mus = self.get_mus()
                    if self.comm.Get_rank() == 0:
                        print "{} iteration = {}".format(name, i)
                        if self.timing:
                            timer.print_timepoints(iter=i)
                        if plotting:
                            # fig, axs = plt.subplots(len(plotdim), len(self.GMMs),
                            #                         sharex=True, sharey=True,
                            #                         squeeze=False)
                            # for j, gmm in enumerate(self.GMMs):
                            #     for k, dim in enumerate(plotdim):
                            #         gmm.plot_components(dim, axs[k, j])
                            fig, axs = plt.subplots(self.K, figsize=(3, 12))
                            for k, ax in enumerate(axs):
                                for j in range(mus.shape[0]):
                                    ax.plot(range(self.d), mus[j, k, :])
                                    ax.plot([0, self.d-1], [.5, .5], color='grey')
                                    ax.set_ylim(-.1, 1.1)

                if not self.timing:
                    self.sample()
                    hmlog.savesim(self)
                else:

                    for gmm in self.GMMs:
                        gmm.sample()
                    timer.timepoint('GMM')

                    self.update_prior()
                    timer.timepoint('load')

                    if self.comm.Get_rank() == 0:
                        for k in range(self.K):
                            self.normal_p_wisharts[k].sample()
                            self.wishart_p_nus[k].sample()
                    self.comm.Barrier()
                    self.update_GMM()
                    timer.timepoint('hGMM rank=0')

                    hmlog.savesim(self)
                    timer.timepoint('save')

        except UserWarning as w:
            raise SimulationError(str(w), name=name, it=i)
        hmlog.postproc()
        if simpar['logtype'] == 'HMlogB':
            try:
                self.blog = self.blog.cat(hmlog)
            except:
                self.blog = hmlog
        else:
            self.log = hmlog

        if self.timing:
            timer.print_timepoints(iterations)

        return hmlog

    def save_burnlog(self, savedir):
        self.blog.save(savedir)
        del self.blog

    def save_log(self, savedir):
        self.log.save(savedir)
        del self.log
