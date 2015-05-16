from __future__ import division
from mpi4py import MPI
import numpy as np
import collections
import warnings

class HMlogB(object):
    '''
        Class for saving burn-in iterations from sampling of posterior distribution
    '''
    def __init__(self,hGMM,sim,nbrsave=None):
        if nbrsave is None:
            nbrsave = sim
        self.nbrsave = nbrsave
        self.sim = sim
        self.savefrq = max(np.floor(sim/(nbrsave-1)),1)
        self.K = hGMM.K
        self.noise_class = hGMM.noise_class
        self.set_names(hGMM)
        self.active_komp_loc = np.zeros((len(hGMM.GMMs),self.K+self.noise_class),dtype = 'i')
        self.lab_sw = []
        self.i = -1
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.d = hGMM.d
            self.theta_sim = np.empty((np.ceil(sim/self.savefrq),self.K,self.d))
            self.nu_sim = np.empty((np.ceil(sim/self.savefrq),self.K))

    def savesim(self,hGMM):
        '''
            Save burn-in iteration
        '''
        self.nextsim()
        nus = hGMM.get_nus() 
        if self.i % self.savefrq == 0:
            thetas = hGMM.get_thetas()       
            if MPI.COMM_WORLD.Get_rank() == 0:
                self.append_theta(thetas)
                self.append_nu(nus)
        for j,GMM in enumerate(hGMM.GMMs):
            self.active_komp_loc[j,:] += GMM.active_komp
            if np.amax(GMM.lab) > -1:
                self.lab_sw.append(GMM.lab)
                print "Label switch: {}".format(GMM.lab)
        return nus
        
    def cat(self,hmlog):
        if self.savefrq != hmlog.savefrq:
            warnings.warn('Savefrq not compatible: {} vs {}'.format(self.savefrq,hmlog.savefrq))
        if MPI.COMM_WORLD.Get_rank() == 0:
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
        self.set_active_komp() 
        self.set_lab_sw()
    
    def nextsim(self):
        self.i += 1
        
    def append_theta(self,theta):
        self.theta_sim[self.i/self.savefrq,:,:] = theta
        
    def append_nu(self,nus):
        self.nu_sim[self.i/self.savefrq,:] = nus
        
    def get_last_nus(self):
        return self.nu_sim[self.i/self.savefrq,:]
        
    def set_lab_sw(self):
        comm = MPI.COMM_WORLD
        if len(self.lab_sw) > 0:
            lab_sw_loc = np.array(np.vstack(self.lab_sw),dtype = 'i')
        else:
            lab_sw_loc = np.empty((0,2))    
            
        if comm.Get_rank() == 0:
            counts = np.empty(comm.Get_size(),dtype = 'i')
        else:
            counts = 0
        comm.Gather(sendbuf=[np.array(lab_sw_loc.shape[0] * lab_sw_loc.shape[1],dtype='i'), MPI.INT], recvbuf=[counts, MPI.INT], root=0)
        
        if comm.Get_rank() == 0:
            lab_sw_all = np.empty((sum(counts)/lab_sw_loc.shape[1],lab_sw_loc.shape[1]),dtype = 'i')
        else:
            lab_sw_all = None
        comm.Gatherv(sendbuf=[lab_sw_loc,MPI.INT],recvbuf=[lab_sw_all,(counts,None),MPI.INT],root=0)
        if comm.Get_rank() == 0:
            self.lab_sw = lab_sw_all  

    def set_active_komp(self):
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            counts = np.empty(comm.Get_size(),dtype = 'i')
        else:
            counts = 0
        comm.Gather(sendbuf=[np.array(self.active_komp_loc.shape[0] * self.active_komp_loc.shape[1],dtype='i'), MPI.INT], recvbuf=[counts, MPI.INT], root=0)

        if comm.Get_rank() == 0:
            active_komp_all = np.empty((sum(counts)/(self.K+self.noise_class),self.K+self.noise_class),dtype = 'i')
        else:
            active_komp_all = None
        comm.Gatherv(sendbuf=[self.active_komp_loc,MPI.INT],recvbuf=[active_komp_all,(counts,None),MPI.INT],root=0)
        if comm.Get_rank() == 0:
            self.active_komp = active_komp_all/self.sim
            
    def set_names(self,hGMM):
        comm = MPI.COMM_WORLD
        names_loc = ":".join([GMM.name for GMM in hGMM.GMMs])
        names_loc += ':'
        if comm.Get_rank() == 0:
            counts = np.empty(comm.Get_size(),dtype = 'i')
        else:
            counts = 0
        comm.Gather(sendbuf=[np.array(len(names_loc),dtype='i'), MPI.INT], recvbuf=[counts, MPI.INT], root=0)

        if comm.Get_rank() == 0:
            names_all = np.chararray(sum(counts))
        else:
            names_all = None
        comm.Gatherv(sendbuf=[np.array(list(names_loc)),MPI.CHAR],recvbuf=[names_all,(counts,None),MPI.CHAR],root=0)
        if comm.Get_rank() == 0:
            names_all = "".join(names_all)
            self.names = names_all.split(':')[0:-1]

class HMlog(HMlogB):

    '''
        Class for saving results from sampling of posterior distribution
        NB! Does not save classification frequencies or sample names. If this is needed, use class BMEresult below.
    '''
    
    def __init__(self,hGMM,sim,savesamp = [],nbrsave=None,nbrsavey=None):
        super(HMlog,self).__init__(hGMM,sim,nbrsave)
        if nbrsavey is None:
            nbrsavey = sim
        self.savefrqy = max(np.floor((sim/nbrsavey)),1)
        self.J_loc = len(hGMM.GMMs)
        self.set_J(hGMM)
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.Y_sim = [np.empty((np.ceil(sim/self.savefrqy),self.d)) for i in range(len(savesamp))]
            self.Y_pooled_sim = np.empty((np.ceil(sim/self.savefrqy),self.d))
            self.theta_sim_mean = np.zeros((self.K,self.d))
            self.Sigmaexp_sim_mean = np.zeros((self.K,self.d,self.d))
            self.mupers_sim_mean = np.zeros((self.J,self.K,self.d))
            self.Sigmapers_sim_mean = np.zeros((self.J,self.K,self.d,self.d))
            self.savesamp = savesamp
            self.savesampnames = [hGMM.GMMs[savesamp[j]].name for j in self.savesamp]
            self.prob_sim_mean = np.zeros((self.J,self.K+self.noise_class))

        
    def savesim(self,hGMM):
        '''
            Save production iteration
        '''
        super(HMlog,self).savesim(hGMM)
        thetas = hGMM.get_thetas()
        Qs = hGMM.get_Qs()
        mus = hGMM.get_mus()
        Sigmas = hGMM.get_Sigmas()
        ps = hGMM.get_ps()
        Y_pooled = hGMM.sampleY()        
        if MPI.COMM_WORLD.Get_rank() == 0:
            if self.i % self.savefrqy == 0:
                for j in range(len(self.savesamp)):
                    self.append_Y(hGMM.GMMs[self.savesamp[j]].simulate_one_obs(),j)
                self.append_pooled_Y(Y_pooled)
            self.add_theta(thetas)
            self.add_Sigmaexp(Qs,self.get_last_nus())
            self.add_mupers(mus)
            self.add_Sigmapers(Sigmas)
            self.add_prob(ps)            

        def cat(self):
            print "cat not implemented yet for this object type"

    def postproc(self):
        '''
            Post-processing production iterations
        '''
        super(HMlog,self).postproc()
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.theta_sim_mean /= self.sim
            self.Sigmaexp_sim_mean /= self.sim
            self.mupers_sim_mean /= self.sim
            self.Sigmapers_sim_mean /= self.sim
            self.prob_sim_mean /= self.sim

            nonnoise_active_komp = self.active_komp[:,0:-1]
            for dd in range(self.mupers_sim_mean.shape[2]):
                self.mupers_sim_mean[:,:,dd] /= nonnoise_active_komp
                for ddd in range(self.mupers_sim_mean.shape[2]):
                    self.Sigmapers_sim_mean[:,:,dd,ddd] /= nonnoise_active_komp
            self.prob_sim_mean /= self.active_komp
            self.prob_sim_mean[np.isnan(self.prob_sim_mean)] = 0
             

    def set_J(self,hGMM):
        self.J_loc = np.array(self.J_loc,dtype = 'i')
        if MPI.COMM_WORLD.Get_rank() == 0:
            J_locs = np.empty(MPI.COMM_WORLD.Get_size(),dtype='i')
        else:
            J_locs = None
        MPI.COMM_WORLD.Gather(sendbuf=[self.J_loc,MPI.INT],recvbuf=[J_locs,MPI.INT],root=0)
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.J = sum(J_locs)
         
    def append_Y(self,Y,j):
        self.Y_sim[j][self.i/self.savefrqy,:] = Y
        
    def append_pooled_Y(self,Y):
        self.Y_pooled_sim[self.i/self.savefrqy,:] = Y

    def add_theta(self,thetas):
        self.theta_sim_mean += thetas
        
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


class HMElog(HMlog):

    '''
        Class for saving results from sampling of posterior distribution
        NB! Does save classification frequencies and sample names and
        thus makes it possible to create Clustering object.
    '''
    
    def __init__(self,hGMM,sim,savesamp = [],nbrsave=None,nbrsavey=None):
        super(HMElog,self).__init__(hGMM,sim,savesamp,nbrsave,nbrsavey)
        self.batch = 1000
        self.ii = -1
        self.init_classif(hGMM)
        self.set_ns()
        if MPI.COMM_WORLD.Get_rank() == 0:
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
            
    def postproc(self):
        '''
            Post-processing production iterations
        '''
        super(HMElog,self).postproc()
        self.add_classif_fr()
        del self.classif
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.classif_freq = np.split(self.classif_freq_all, np.cumsum(self.ns[0:-1]))
            del self.classif_freq_all
    
        
    def add_classif_fr(self):
        comm = MPI.COMM_WORLD
        classif_freqs_loc = [np.zeros((cl.shape[0],self.K + self.noise_class),dtype = 'i') for cl in self.classif]
        for j,cl in enumerate(self.classif):
            for ii in range(cl.shape[0]):
                cnt = collections.Counter(cl[ii,:])
                del cnt[-1]
                classif_freqs_loc[j][ii,cnt.keys()] = cnt.values()
        classif_freq_loc = np.array(np.vstack(classif_freqs_loc),dtype = 'i')
        #classif_freq_loc = classif_freq_loc[0:100,:]
        if comm.Get_rank() == 0:
            counts = np.empty(comm.Get_size(),dtype = 'i')
        else:
            counts = 0
        comm.Gather(sendbuf=[np.array(classif_freq_loc.shape[0] * classif_freq_loc.shape[1],dtype='i'), MPI.INT], recvbuf=[counts, MPI.INT], root=0)
        
        if comm.Get_rank() == 0:
            classif_all = np.empty((sum(counts)/(self.K+self.noise_class),self.K+self.noise_class),dtype = 'i')
            #print "classif_all.shape = {}".format(classif_all.shape)
        else:
            classif_all = None
        comm.Gatherv(sendbuf=[classif_freq_loc,MPI.INT],recvbuf=[classif_all,(counts,None),MPI.INT],root=0)
        if comm.Get_rank() == 0:
            self.classif_freq_all += classif_all

    def set_ns(self):
        comm = MPI.COMM_WORLD
        ns_loc = np.array([cl.shape[0] for cl in self.classif],dtype='i')
        if comm.Get_rank() == 0:
            counts = np.empty(comm.Get_size(),dtype = 'i')
        else:
            counts = 0
        comm.Gather(sendbuf=[np.array(self.J_loc,dtype='i'), MPI.INT], recvbuf=[counts, MPI.INT], root=0)
        
        if comm.Get_rank() == 0:
            ns = np.empty(self.J,dtype = 'i')
        #print "ns.shape = {}".format(ns.shape)
        else:
            ns = None
        comm.Gatherv(sendbuf=[ns_loc,MPI.INT],recvbuf=[ns,(counts,None),MPI.INT],root=0)
        if comm.Get_rank() == 0:
            self.ns = ns






