from __future__ import division
from mpi4py import MPI
import numpy as np
import collections
import warnings
from scipy import io, sparse
import os
import cPickle as pickle
import json

from .utils import mpiutil
from .utils.jsonutil import ObjJsonEncoder, class_decoder

warnings.filterwarnings('ignore',message='using a non-integer number.*',category= DeprecationWarning)

class HMlogB(object):
    '''
        Class for saving burn-in iterations from sampling of posterior distribution
    '''
    def __init__(self,hGMM,sim,nbrsave=None,savefrq=None,comm=MPI.COMM_WORLD):
        self.comm = comm
        self.rank = comm.Get_rank()
        if not savefrq is None:
            self.savefrq = savefrq
        else:
            if nbrsave is None:
                nbrsave = sim
            self.savefrq = max(int(sim/nbrsave),1)
        self.nbrsave = int(np.ceil(sim/self.savefrq))
        self.sim = sim
        self.K = hGMM.K
        self.d = hGMM.d
        self.noise_class = hGMM.noise_class
        self.names_loc = [GMM.name for GMM in hGMM.GMMs]
        self.set_names(hGMM)
        self.active_komp_loc = np.zeros((len(hGMM.GMMs),self.K+self.noise_class),dtype = 'i')
        self.active_komp_curr_loc = np.ones((len(hGMM.GMMs),self.K+self.noise_class),dtype = 'i')
        self.tmp_active_comp_curr_loc = np.zeros((len(hGMM.GMMs),self.K+self.noise_class),dtype = 'i')
        self.lab_sw_loc = []
        self.i = -1
        if self.rank == 0:
            self.theta_sim = np.empty((self.nbrsave,self.K,self.d))
            self.nu_sim = np.empty((self.nbrsave,self.K))
            print "log nbrsave = {}".format(self.nbrsave)
            print "log savefrq = {}".format(self.savefrq)
            print "log iterations = {}".format(self.sim)

    def savesim(self,hGMM):
        '''
            Save burn-in iteration
        '''
        self.nextsim()
        debug = False
        nus = hGMM.get_nus()
        if self.i % self.savefrq == 0:
            #print "self.i/self.savefrq = {}".format(self.i/self.savefrq)
            #print "self.i = {}".format(self.i)
            #print "self.savefrq = {}".format(self.savefrq)
            thetas = hGMM.get_thetas()       
            if self.rank == 0:
                self.append_theta(thetas)
                self.append_nu(nus)
            
        for j,GMM in enumerate(hGMM.GMMs):
            self.tmp_active_comp_curr_loc[j,:] = GMM.active_komp         
            if np.amax(GMM.lab) > -1:
                self.lab_sw_loc.append([GMM.lab])
                print "Label switch iteration {}, sample {} at rank {}: {}".format(self.i,j,self.rank, GMM.lab)
        self.active_komp_loc += self.tmp_active_comp_curr_loc
        on_or_off = np.nonzero(self.tmp_active_comp_curr_loc - self.active_komp_curr_loc)
        if len(on_or_off[0]) > 0:
            print "Components switched on or off at iteration {} rank {}, samples: {}, components {}".format(self.i,self.rank,on_or_off[0],on_or_off[1])
        self.active_komp_curr_loc = np.copy(self.tmp_active_comp_curr_loc)
        if debug:
            print "savesim ok at iter {} at rank {}".format(self.i,self.rank)

        return nus
        
    def cat(self,hmlog):
        if self.rank == 0:
            if self.noise_class and not hmlog.noise_class:
                hmlog.active_komp = np.hstack([hmlog.active_komp,np.zeros((hmlog.active_komp.shape[0],1))])
                hmlog.noise_class = 1
            if not self.noise_class and hmlog.noise_class:
                self.active_komp = np.hstack([self.active_komp,np.zeros((self.active_komp.shape[0],1))])
                self.noise_class = 1
        if self.savefrq != hmlog.savefrq:
            warnings.warn('Savefrq not compatible: {} vs {}'.format(self.savefrq,hmlog.savefrq))
        if self.rank == 0:
            self.theta_sim = np.vstack([self.theta_sim,hmlog.theta_sim])
            self.nu_sim = np.vstack([self.nu_sim,hmlog.nu_sim])
            self.lab_sw = np.vstack([self.lab_sw,hmlog.lab_sw])
            qself = self.sim/(self.sim + hmlog.sim)
            self.active_komp = qself*self.active_komp + (1-qself)*hmlog.active_komp
            self.nbrsave = self.nbrsave + hmlog.nbrsave
            self.sim = self.sim + hmlog.sim
        return self
    
    def postproc(self):
        '''
           Post-processing burn-in iterations
        '''
        debug = False        
        self.set_active_komp()
        if debug:
            print "active komp set"
        self.set_lab_sw()
        if debug:
            print "lab switch set"
    
    def nextsim(self):
        self.i += 1
        
    def append_theta(self,theta):
        self.theta_sim[int(self.i/self.savefrq),:,:] = theta
        
    def append_nu(self,nus):
        self.nu_sim[int(self.i/self.savefrq),:] = nus
        
    def get_last_nus(self):
        return self.nu_sim[int(self.i/self.savefrq),:]

    def set_lab_sw(self):
        lab_sw_all = mpiutil.collect_data(self.lab_sw_loc,2,'i',MPI.INT)
        if self.rank == 0:
            self.lab_sw = lab_sw_all
        del self.lab_sw_loc
         
    def set_active_komp(self):
        active_komp_all = mpiutil.collect_data(self.active_komp_loc,self.K+self.noise_class,'i',MPI.INT)
        if self.rank == 0:
            self.active_komp = active_komp_all/self.sim
        del self.active_komp_loc

    def set_names(self,hGMM):
        names_list_of_lists = self.comm.gather([GMM.name for GMM in hGMM.GMMs])
        if self.rank == 0:
            self.names = [name for lst in names_list_of_lists for name in lst]
#        name_data = [np.array([ch for ch in GMM.name]) for GMM in hGMM.GMMs]
#        name_all = mpiutil.collect_arrays(name_data,1,'S',MPI.UNSIGNED_CHAR,self.comm)
#        if self.rank == 0:
#            self.names = [''.join(nam.reshape((-1,))) for nam in name_all]

    def encode_json(self):
        jsondict = {'__type__':'HMlogB'}
        for arg in ['savefrq','nbrsave','sim','K','d','noise_class','names',
                    'active_komp','lab_sw']:
            jsondict.update({arg:getattr(self,arg)})
        #print "jsondict= {}".format(jsondict)
        #with open('jsondump.pkl','w') as f:
        #    pickle.dump(jsondict,f,-1)
        return jsondict    

    def save(self,savedir,logname='blog'):
        if self.rank == 0:
            if not savedir[-1] == '/':
                savedir += '/'
            with open(savedir+logname+'.json','w') as f:
                json.dump(self,f,cls=ObjJsonEncoder)
            with open(savedir+logname+'_theta_sim.npy','w') as f:
                np.save(f,self.theta_sim)
            with open(savedir+logname+'_nu_sim.npy','w') as f:
                np.save(f,self.nu_sim)

    @classmethod
    def load(cls,savedir,logname='blog',comm=MPI.COMM_WORLD):
        if not savedir[-1] == '/':
            savedir += '/'
        with open(savedir+logname+'.json','r') as f:
            hmlog = json.load(f,object_hook=lambda obj: class_decoder(obj,cls,comm=comm))
        print "load burnlog json"
        if comm.Get_rank() == 0:
            with open(savedir+logname+'_theta_sim.npy','r') as f:
                hmlog.theta_sim = np.load(f)
            with open(savedir+logname+'_nu_sim.npy','r') as f:
                hmlog.nu_sim = np.load(f)
        return hmlog

class HMlog(HMlogB):
    '''
        Class for saving results from sampling of posterior distribution
        NB! Does not save classification frequencies. If this is needed, use class HMElog below.
    '''
    
    def __init__(self,hGMM,sim,savesamp=None,savesampnames=None,nbrsave=None,
                 savefrq=None,nbrsavey=None,savefrqy=None,comm=MPI.COMM_WORLD):
        super(HMlog,self).__init__(hGMM,sim,nbrsave,savefrq,comm)
        
        if not savefrqy is None:
            self.savefrqy = savefrqy
        else:
            if nbrsavey is None:
                nbrsavey = sim
            self.savefrqy = max(np.floor((sim/nbrsavey)),1)
        self.nbrsavey = np.ceil(sim/self.savefrqy)
        self.J_loc = len(hGMM.GMMs)
        self.set_J(hGMM)
        
        if savesamp is None:
            self.savesamp_loc = []
            if not savesampnames is None:
                if savesampnames == 'all':
                    self.savesamp_loc = range(len(hGMM.GMMs))
                else:
                    for j,GMM in enumerate(hGMM.GMMs):
                        if GMM.name in savesampnames:
                            self.savesamp_loc.append(j)
            print "savesamp_loc = {} at rank {}".format(self.savesamp_loc,self.rank)
            print "len(hGMM.GMMs) = {} at rank {}".format(len(hGMM.GMMs),self.rank)
            self.savesampnames_loc = [hGMM.GMMs[samp].name for samp in self.savesamp_loc]
            if self.rank == 0:
                self.savesampnames = savesampnames
        else:
            if self.rank == 0:
                self.savesamp_loc = savesamp
                self.savesampnames_loc = [hGMM.GMMs[samp].name for samp in self.savesamp_loc]
            else:
                self.savesamp_loc = []
                self.savesampnames_loc = []
        self.Y_sim_loc = [np.empty((np.ceil(sim/self.savefrqy),self.d)) for i in range(len(self.savesamp_loc))]

                
        if self.rank == 0:
            self.Y_pooled_sim = np.empty((np.ceil(sim/self.savefrqy),self.d))
            self.theta_sim_mean = np.zeros((self.K,self.d))
            self.Sigma_mu_sim_mean = np.zeros((self.K,self.d,self.d))
            self.Sigmaexp_sim_mean = np.zeros((self.K,self.d,self.d))
            self.mupers_sim_mean = np.zeros((self.J,self.K,self.d))
            self.Sigmapers_sim_mean = np.zeros((self.J,self.K,self.d,self.d))
            self.prob_sim_mean = np.zeros((self.J,self.K+self.noise_class))

        
    def savesim(self,hGMM):
        '''
            Save production iteration
        '''
        super(HMlog,self).savesim(hGMM)
        thetas = hGMM.get_thetas()
        Sigma_mus = hGMM.get_Sigma_mus()
        Qs = hGMM.get_Qs()
        mus = hGMM.get_mus()
        Sigmas = hGMM.get_Sigmas()
        ps = hGMM.get_ps()

        if self.i % self.savefrqy == 0:
            for j in range(len(self.savesamp_loc)):
                self.append_Y(hGMM.GMMs[self.savesamp_loc[j]].simulate_one_obs(),j)        
            Y_pooled = hGMM.sampleY()
            
        if self.rank == 0:
            if self.i % self.savefrqy == 0:
                self.append_pooled_Y(Y_pooled)
            self.add_theta(thetas)
            self.add_Sigma_mu(Sigma_mus)
            self.add_Sigmaexp(Qs,self.get_last_nus())
            self.add_mupers(mus)
            self.add_Sigmapers(Sigmas)
            self.add_prob(ps)            

    def cat(self):
        print "cat not implemented yet for this object type"

    def postproc(self,high_memory = False):
        '''
            Post-processing production iterations

            high_memory     - set to True if root can handle data from all workers
        '''
        super(HMlog,self).postproc()
        if self.rank == 0:
            self.theta_sim_mean /= self.sim
            self.Sigma_mu_sim_mean /= self.sim
            self.Sigmaexp_sim_mean /= self.sim
            self.mupers_sim_mean /= self.sim
            self.Sigmapers_sim_mean /= self.sim
            self.prob_sim_mean /= self.sim

            if self.noise_class:
                nonnoise_active_komp = self.active_komp[:,:-1]
            else:
                nonnoise_active_komp = self.active_komp
            for dd in range(self.mupers_sim_mean.shape[2]):
                self.mupers_sim_mean[:,:,dd] /= nonnoise_active_komp
                self.mupers_sim_mean[~nonnoise_active_komp.astype('bool'),dd] = np.nan
                for ddd in range(self.mupers_sim_mean.shape[2]):
                    self.Sigmapers_sim_mean[~nonnoise_active_komp.astype('bool'),dd,ddd] = np.nan
                    self.Sigmapers_sim_mean[:,:,dd,ddd] /= nonnoise_active_komp
            self.prob_sim_mean[~self.active_komp.astype('bool')] = np.nan
            self.prob_sim_mean /= self.active_komp
            self.prob_sim_mean[np.isnan(self.prob_sim_mean)] = 0
        
        if high_memory:    
            self.set_savesampnames()
            self.set_savesamp()
            self.set_Y_sim()
         
    def append_Y(self,Y,j):
        self.Y_sim_loc[j][int(self.i/self.savefrqy),:] = Y
        
    def append_pooled_Y(self,Y):
        self.Y_pooled_sim[int(self.i/self.savefrqy),:] = Y

    def add_theta(self,thetas):
        self.theta_sim_mean += thetas

    def add_Sigma_mu(self,Sigma_mus):
        for k in range(self.K):
            self.Sigma_mu_sim_mean[k,:,:] += Sigma_mus[k]
        
    def add_Sigmaexp(self,Psis,nus):
        for k in range(self.K):
            self.Sigmaexp_sim_mean[k,:,:] += Psis[k]/(nus[k]-self.d-1)
        
    def add_mupers(self,mus):
        mus[np.isnan(mus)] = 0
        self.mupers_sim_mean += mus
        
    def add_Sigmapers(self,Sigmas):
        Sigmas[np.isnan(Sigmas)] = 0
        self.Sigmapers_sim_mean += Sigmas
        
    def add_prob(self,prob):
        prob[np.isnan(prob)] = 0
        self.prob_sim_mean += prob

    def set_J(self,hGMM):
        self.J_loc = np.array(self.J_loc,dtype = 'i')
        if self.rank == 0:
            J_locs = np.empty(self.comm.Get_size(),dtype='i')
        else:
            J_locs = None
        self.comm.Gather(sendbuf=[self.J_loc,MPI.INT],recvbuf=[J_locs,MPI.INT],root=0)
        if self.rank == 0:
            self.J = sum(J_locs)

    def set_Y_sim(self):
        self.Y_sim = mpiutil.collect_arrays(self.Y_sim_loc,self.d,'d',MPI.DOUBLE)

    def set_savesampnames(self):
        #print "self.savesampnames_loc at rank {}: {}".format(self.rank,self.savesampnames_loc)
        name_data = [np.array([ch for ch in name]) for name in self.savesampnames_loc]
        #print "name_data at rank {}".format(name_data,self.rank)
        name_all = mpiutil.collect_arrays(name_data,1,'S',MPI.UNSIGNED_CHAR)
        if self.rank == 0:
            self.savesampnames = [''.join(nam.reshape((-1,))) for nam in name_all]
            #print "self.savesampnames at rank 0: {}".format(self.savesampnames)
         
    def set_savesamp(self):
        if self.rank == 0:
            self.savesamp = [self.names.index(name) for name in self.savesampnames]

    def set_syndata_dir(self,dirname):
        if not dirname[-1] == '/':
            dirname += '/'
        self.syndata_dir = dirname

    def encode_json(self):
        jsondict = super(HMlog,self).encode_json()
        jsondict['__type__'] = 'HMlog'
        for arg in ['theta_sim_mean','Sigma_mu_sim_mean', 'Sigmaexp_sim_mean',
                    'mupers_sim_mean','Sigmapers_sim_mean','prob_sim_mean',
                    'J','savesampnames']:
            jsondict.update({arg:getattr(self,arg)})
        try:
            jsondict['syndata_dir'] = self.syndata_dir
        except:
            pass
        return jsondict

    def save(self, savedir):
        if not savedir[-1] == '/':
            savedir += '/'
        self.syndata_dir = savedir + 'syndata/'
        if self.rank == 0:
            if not os.path.exists(self.syndata_dir):
                os.mkdir(self.syndata_dir)
        self.comm.Barrier()

        for j, name in enumerate(self.savesampnames_loc):
            with open(self.syndata_dir+name+'_MODEL.pkl', 'w') as f:
                pickle.dump(self.Y_sim_loc[j], f, -1)

        if self.rank == 0:
            with open(self.syndata_dir+'pooled_MODEL.pkl', 'w') as f:
                pickle.dump(self.Y_pooled_sim, f, -1)
            if not hasattr(self, 'savesampnames'):
                self.set_savesampnames()
        super(HMlog, self).save(savedir, logname='log')

    @classmethod
    def load(cls,savedir,comm=MPI.COMM_WORLD):
        if not savedir[-1] == '/':
            savedir += '/'
        hmlog = super(HMlog,cls).load(savedir,logname='log',comm=comm)
        try:
            syndata_dir = hmlog.syndata_dir
        except:
            syndata_dir = savedir+'syndata/'

        # TODO! Load dat to all cores instead
        if comm.Get_rank() == 0:
            hmlog.Y_sim = []
            nofiles = []
            for j,name in enumerate(hmlog.savesampnames):
                try:
                    with open(syndata_dir+name+'_MODEL.pkl','r') as f:
                        hmlog.Y_sim.append(pickle.load(f))
                except IOError as e:
                    print e
                    nofiles.append(name)
            hmlog.savesampnames = [name for name in hmlog.savesampnames 
                                    if name not in nofiles]
        if comm.Get_rank() == 0:
            with open(syndata_dir+'pooled_MODEL.pkl','r') as f:
                hmlog.Y_pooled_sim = pickle.load(f) 
        return hmlog       


class HMElog(HMlog):
    '''
        Class for saving results from sampling of posterior distribution
        NB! Does save classification frequencies and
        thus makes it possible to create Clustering object.
    '''
    
    def __init__(self,hGMM,sim,savesamp=None,savesampnames=None,nbrsave=None,savefrq=None,nbrsavey=None,savefrqy=None,
                 high_memory=False, comm=MPI.COMM_WORLD):
        super(HMElog,self).__init__(hGMM,sim,savesamp,savesampnames,nbrsave,
                                    savefrq,nbrsavey,savefrqy,comm)
        self.batch = 1000
        self.ii = -1
        self.init_classif(hGMM)
        self.set_ns()
        if self.rank == 0 and high_memory:
            self.classif_freq_all = np.zeros((sum(self.ns),self.K+self.noise_class))

    def init_classif(self,hGMM):
        self.classif = [-np.ones((GMM.data.shape[0],self.batch),dtype = 'i') for GMM in hGMM.GMMs]
        
    def savesim(self,hGMM):
        '''
           Saving production iteration
        '''
        super(HMElog,self).savesim(hGMM)
        self.ii += 1
        if self.ii == self.batch:
            self.add_classif_fr()
            self.init_classif(hGMM)
            self.ii = 0
        for j,GMM in enumerate(hGMM.GMMs):
            self.classif[j][:,self.ii] = GMM.x[:]

    def postproc(self,high_memory=False):
        '''
            Post-processing production iterations
        '''
        super(HMElog,self).postproc(high_memory)
        if not self.ii == self.batch:
            self.add_classif_fr()
        del self.classif
        if self.rank == 0 and high_memory:
            self.classif_freq = mpiutil.collect_arrays(self.classif_freq_loc)     
            
    # def postproc_old(self):
    #     '''
    #         Post-processing production iterations
    #     '''
    #     super(HMElog,self).postproc()
    #     if not self.ii == self.batch:
    #         self.add_classif_fr()
    #     del self.classif
    #     if rank == 0:
    #         self.classif_freq = np.split(self.classif_freq_all, np.cumsum(self.ns[0:-1]))
    #         del self.classif_freq_all        

    def add_classif_fr(self):
        try:
            classif_freq_loc = self.classif_freq_loc
        except:
            classif_freq_loc= [np.zeros((cl.shape[0],self.K + self.noise_class),dtype = 'i') for cl in self.classif]
        for j,cl in enumerate(self.classif):
            for ii in range(cl.shape[0]):
                cnt = collections.Counter(cl[ii,:])
                del cnt[-1]
                classif_freq_loc[j][ii,cnt.keys()] += cnt.values()
        self.classif_freq_loc = classif_freq_loc

    # def add_classif_fr_old(self):
    #     classif_freqs_loc = [np.zeros((cl.shape[0],self.K + self.noise_class),dtype = 'i') for cl in self.classif]
    #     for j,cl in enumerate(self.classif):
    #         for ii in range(cl.shape[0]):
    #             cnt = collections.Counter(cl[ii,:])
    #             del cnt[-1]
    #             classif_freqs_loc[j][ii,cnt.keys()] = cnt.values()
    #     classif_freq_loc = np.array(np.vstack(classif_freqs_loc),dtype = 'i')
    #     #classif_freq_loc = classif_freq_loc[0:100,:]
    #     if rank == 0:
    #         counts = np.empty(comm.Get_size(),dtype = 'i')
    #     else:
    #         counts = 0
    #     comm.Gather(sendbuf=[np.array(classif_freq_loc.shape[0] * classif_freq_loc.shape[1],dtype='i'), MPI.INT], recvbuf=[counts, MPI.INT], root=0)
        
    #     if rank == 0:
    #         classif_all = np.empty((sum(counts)/(self.K+self.noise_class),self.K+self.noise_class),dtype = 'i')
    #         #print "classif_all.shape = {}".format(classif_all.shape)
    #     else:
    #         classif_all = None
    #     comm.Gatherv(sendbuf=[classif_freq_loc,MPI.INT],recvbuf=[classif_all,(counts,None),MPI.INT],root=0)
    #    if rank == 0:
    #        self.classif_freq_all += classif_all

    def set_ns(self):
        ns_loc = np.array([cl.shape[0] for cl in self.classif],dtype='i')
        if self.rank == 0:
            counts = np.empty(self.comm.Get_size(),dtype = 'i')
        else:
            counts = 0
        self.comm.Gather(sendbuf=[np.array(self.J_loc,dtype='i'), MPI.INT], recvbuf=[counts, MPI.INT], root=0)
        
        if self.rank == 0:
            ns = np.empty(self.J,dtype = 'i')
        #print "ns.shape = {}".format(ns.shape)
        else:
            ns = None
        self.comm.Gatherv(sendbuf=[ns_loc,MPI.INT],recvbuf=[ns,(counts,None),MPI.INT],root=0)
        if self.rank == 0:
            self.ns = ns

    def encode_json(self):
        jsondict = super(HMElog,self).encode_json()
        jsondict['__type__'] = 'HMElog'
        try:
            jsondict['classif_freq_dir'] = self.classif_freq_dir
        except:
            pass
        return jsondict

    def save(self, savedir):
        if not savedir[-1] == '/':
            savedir += '/'
        self.classif_freq_dir = savedir+'classif_freq/'
        if self.rank == 0:
            if not os.path.exists(self.classif_freq_dir):
                os.mkdir(self.classif_freq_dir)
        self.comm.Barrier()
        # try:
        #     print "names_loc at rank {}: {}".format(self.rank,self.names_loc)
        # except AttributeError as e:
        #     print e
        #     if self.rank == 0:
        #         for j,name in enumerate(self.names):
        #             io.mmwrite(self.classif_freq_dir+name+'_CLASSIF_FREQ.mtx',sparse.coo_matrix(self.classif_freq[j]))
        # else:
        for j, name in enumerate(self.names_loc):
            io.mmwrite(self.classif_freq_dir+name+'_CLASSIF_FREQ.mtx', sparse.coo_matrix(self.classif_freq_loc[j]))
        super(HMElog, self).save(savedir)

    @classmethod
    def load(cls,savedir,comm=MPI.COMM_WORLD):
        if not savedir[-1] == '/':
            savedir += '/'
        hmlog = super(HMElog,cls).load(savedir,comm)
        try:
            classif_freq_dir = hmlog.classif_freq_dir
        except:
            classif_freq_dir = savedir+'classif_freq/'

        # TODO! Load classif freq to all cores
        if comm.Get_rank() == 0:
            hmlog.classif_freq = []
            for j,name in enumerate(hmlog.names):
                hmlog.classif_freq.append(io.mmread(classif_freq_dir+name+'_CLASSIF_FREQ.mtx'))
        return hmlog       

#if 0:
#    homedir = '/Users/johnsson/'
#else:
#    homedir = '/home/johnsson/'
#expdir = homedir+'Forskning/Experiments/FlowCytometry/BHM/FCI/StemCell/'
#expname = 'CYTO'
#run = 19
#loaddirres = expdir+expname+'/' + 'run'+str(run)+'/'
#HMElog.load(loaddirres)


