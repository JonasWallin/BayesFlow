#from __future__ import division
from mpi4py import MPI
import numpy as np
import collections
import warnings

warnings.filterwarnings('ignore',message='using a non-integer number.*',category= DeprecationWarning)

def collect_arrays(data,nbr_col,nptype,mpitype):

    nbr_row = collect_data_1d([dat.shape[0] for dat in data],'i',MPI.INT)
    
    if len(data) > 0:
        data_stack = np.vstack([np.array(dat,dtype=nptype).reshape((-1,nbr_col)) for dat in data])
    else:
        data_stack = np.empty((0,nbr_col),dtype=nptype)

    data_all_stack = collect_array_(data_stack,nbr_col,nptype,mpitype)
    
    if not data_all_stack is None:
        if len(nbr_row) > 0:
            data_all = np.split(data_all_stack, np.cumsum(nbr_row[0:-1]))
        else:
            data_all = data_all_stack
    else:
        data_all = None
        
    return data_all

def collect_data(data,nbr_col,nptype,mpitype):

    if len(data) > 0:
        data_array = np.array(data,dtype=nptype).reshape((-1,nbr_col))
    else:
        data_array = np.empty((0,nbr_col),dtype=nptype)
        
    data_all = collect_array_(data_array,nbr_col,nptype,mpitype)
    return data_all

def collect_data_1d(data,nptype,mpitype):

    data_all = collect_data(data,1,nptype,mpitype)
    if not data_all is None:
        data_all = data_all.reshape((-1,))    
    return data_all

def collect_array_(data_array,nbr_col,nptype,mpitype):
    comm = MPI.COMM_WORLD
    #rank = comm.Get_rank()

    if comm.Get_rank() == 0:
        nbr_row = np.empty(comm.Get_size(),dtype = 'i')
    else:
        nbr_row = 0
    comm.Gather(sendbuf=[np.array(data_array.shape[0],dtype='i'), MPI.INT], recvbuf=[nbr_row, MPI.INT], root=0)
    counts = nbr_row*nbr_col
    #print "counts = {} at rank {}".format(counts,rank)    
    
    if comm.Get_rank() == 0:
        data_all_array = np.empty((sum(nbr_row),nbr_col),dtype = nptype)
    else:
        data_all_array = None
    comm.Gatherv(sendbuf=[data_array,mpitype],recvbuf=[data_all_array,(counts,None),mpitype],root=0)
    return data_all_array

class HMlogB(object):
    '''
        Class for saving burn-in iterations from sampling of posterior distribution
    '''
    def __init__(self,hGMM,sim,nbrsave=None,savefrq=None):
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
        self.set_names(hGMM)
        self.active_komp_loc = np.zeros((len(hGMM.GMMs),self.K+self.noise_class),dtype = 'i')
        self.active_komp_curr_loc = np.ones((len(hGMM.GMMs),self.K+self.noise_class),dtype = 'i')
        self.tmp_active_comp_curr_loc = np.zeros((len(hGMM.GMMs),self.K+self.noise_class),dtype = 'i')
        self.lab_sw_loc = []
        self.i = -1
        if MPI.COMM_WORLD.Get_rank() == 0:
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
        rank = MPI.COMM_WORLD.Get_rank()
        debug = False
        nus = hGMM.get_nus()
        if self.i % self.savefrq == 0:
            #print "self.i/self.savefrq = {}".format(self.i/self.savefrq)
            #print "self.i = {}".format(self.i)
            #print "self.savefrq = {}".format(self.savefrq)
            thetas = hGMM.get_thetas()       
            if MPI.COMM_WORLD.Get_rank() == 0:
                self.append_theta(thetas)
                self.append_nu(nus)
            
        for j,GMM in enumerate(hGMM.GMMs):
            self.tmp_active_comp_curr_loc[j,:] = GMM.active_komp         
            if np.amax(GMM.lab) > -1:
                self.lab_sw_loc.append([GMM.lab])
                print "Label switch iteration {}, sample {} at rank {}: {}".format(self.i,j,MPI.COMM_WORLD.Get_rank(), GMM.lab)
        self.active_komp_loc += self.tmp_active_comp_curr_loc
        on_or_off = np.nonzero(self.tmp_active_comp_curr_loc - self.active_komp_curr_loc)
        if len(on_or_off[0]) > 0:
            print "Components switched on or off at iteration {} rank {}, samples: {}, components {}".format(self.i,MPI.COMM_WORLD.Get_rank(),on_or_off[0],on_or_off[1])
        self.active_komp_curr_loc = np.copy(self.tmp_active_comp_curr_loc)
        if debug:
            print "savesim ok at iter {} at rank {}".format(self.i,rank)


        return nus
        
    def cat(self,hmlog):
        if MPI.COMM_WORLD.Get_rank() == 0:
            if self.noise_class and not hmlog.noise_class:
                hmlog.active_komp = np.hstack([hmlog.active_komp,np.zeros((hmlog.active_komp.shape[0],1))])
                hmlog.noise_class = 1
            if not self.noise_class and hmlog.noise_class:
                self.active_komp = np.hstack([self.active_komp,np.zeros((self.active_komp.shape[0],1))])
                self.noise_class = 1
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
        lab_sw_all = collect_data(self.lab_sw_loc,2,'i',MPI.INT)
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.lab_sw = lab_sw_all
        del self.lab_sw_loc
        
#    def set_lab_sw_old(self):
#        comm = MPI.COMM_WORLD
#        print "self.lab_sw = {}".format(self.lab_sw)
#        if len(self.lab_sw) > 0:
#            lab_sw_loc = np.array(np.vstack(self.lab_sw),dtype = 'i')
#        else:
#            lab_sw_loc = np.empty((0,2))    
#            
#        if comm.Get_rank() == 0:
#            counts = np.empty(comm.Get_size(),dtype = 'i')
#        else:
#            counts = 0
#        comm.Gather(sendbuf=[np.array(lab_sw_loc.shape[0] * lab_sw_loc.shape[1],dtype='i'), MPI.INT], recvbuf=[counts, MPI.INT], root=0)
#        
#        if comm.Get_rank() == 0:
#            lab_sw_all = np.empty((sum(counts)/lab_sw_loc.shape[1],lab_sw_loc.shape[1]),dtype = 'i')
#        else:
#            lab_sw_all = None
#        comm.Gatherv(sendbuf=[lab_sw_loc,MPI.INT],recvbuf=[lab_sw_all,(counts,None),MPI.INT],root=0)
#        if comm.Get_rank() == 0:
#            self.lab_sw = lab_sw_all  

    def set_active_komp(self):
        active_komp_all = collect_data(self.active_komp_loc,self.K+self.noise_class,'i',MPI.INT)
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.active_komp = active_komp_all/self.sim
        del self.active_komp_loc

#    def set_active_komp_old(self):
#        comm = MPI.COMM_WORLD
#        if comm.Get_rank() == 0:
#            counts = np.empty(comm.Get_size(),dtype = 'i')
#        else:
#            counts = 0
#        print "self.active_komp_loc at rank {}: \n {}".format(comm.Get_rank(),self.active_komp_loc)
#        comm.Gather(sendbuf=[np.array(self.active_komp_loc.shape[0] * self.active_komp_loc.shape[1],dtype='i'), MPI.INT], recvbuf=[counts, MPI.INT], root=0)
#
#        if comm.Get_rank() == 0:
#            active_komp_all = np.empty((sum(counts)/(self.K+self.noise_class),self.K+self.noise_class),dtype = 'i')
#        else:
#            active_komp_all = None
#        comm.Gatherv(sendbuf=[self.active_komp_loc,MPI.INT],recvbuf=[active_komp_all,(counts,None),MPI.INT],root=0)
#        if comm.Get_rank() == 0:
#            self.active_komp = active_komp_all/self.sim

    def set_names(self,hGMM):
        name_data = [np.array([ch for ch in GMM.name]) for GMM in hGMM.GMMs]
        name_all = collect_arrays(name_data,1,'S',MPI.UNSIGNED_CHAR)
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.names = [''.join(nam.reshape((-1,))) for nam in name_all]
            
#    def set_names_old(self,hGMM):
#        comm = MPI.COMM_WORLD
#        names_loc = ":".join([GMM.name for GMM in hGMM.GMMs])
#        names_loc += ':'
#        if comm.Get_rank() == 0:
#            counts = np.empty(comm.Get_size(),dtype = 'i')
#        else:
#            counts = 0
#        comm.Gather(sendbuf=[np.array(len(names_loc),dtype='i'), MPI.INT], recvbuf=[counts, MPI.INT], root=0)
#
#        if comm.Get_rank() == 0:
#            names_all = np.chararray(sum(counts))
#        else:
#            names_all = None
#        comm.Gatherv(sendbuf=[np.array(list(names_loc)),MPI.CHAR],recvbuf=[names_all,(counts,None),MPI.CHAR],root=0)
#        if comm.Get_rank() == 0:
#            names_all = "".join(names_all)
#            self.names = names_all.split(':')[0:-1]

class HMlog(HMlogB):

    '''
        Class for saving results from sampling of posterior distribution
        NB! Does not save classification frequencies. If this is needed, use class HMElog below.
    '''
    
    def __init__(self,hGMM,sim,savesamp=None,savesampnames=None,nbrsave=None,savefrq=None,nbrsavey=None,savefrqy=None):
        super(HMlog,self).__init__(hGMM,sim,nbrsave,savefrq)
        
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
            print "savesamp_loc = {} at rank {}".format(self.savesamp_loc,MPI.COMM_WORLD.Get_rank())
            print "len(hGMM.GMMs) = {} at rank {}".format(len(hGMM.GMMs),MPI.COMM_WORLD.Get_rank())
            self.savesampnames_loc = [hGMM.GMMs[samp].name for samp in self.savesamp_loc]
        else:
            if MPI.COMM_WORLD.Get_rank() == 0:
                self.savesamp_loc = savesamp
                self.savesampnames_loc = [hGMM.GMMs[samp].name for samp in self.savesamp_loc]
            else:
                self.savesamp_loc = []
                self.savesampnames_loc = []
        self.Y_sim_loc = [np.empty((np.ceil(sim/self.savefrqy),self.d)) for i in range(len(self.savesamp_loc))]

                
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.Y_pooled_sim = np.empty((np.ceil(sim/self.savefrqy),self.d))
            self.theta_sim_mean = np.zeros((self.K,self.d))
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
        Qs = hGMM.get_Qs()
        mus = hGMM.get_mus()
        Sigmas = hGMM.get_Sigmas()
        ps = hGMM.get_ps()

        if self.i % self.savefrqy == 0:
            for j in range(len(self.savesamp_loc)):
                self.append_Y(hGMM.GMMs[self.savesamp_loc[j]].simulate_one_obs(),j)        
            Y_pooled = hGMM.sampleY()
            
        if MPI.COMM_WORLD.Get_rank() == 0:
            if self.i % self.savefrqy == 0:
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
            
        self.set_savesampnames()
        self.set_savesamp()
        self.set_Y_sim()
         
    def append_Y(self,Y,j):
        self.Y_sim_loc[j][int(self.i/self.savefrqy),:] = Y
        
    def append_pooled_Y(self,Y):
        self.Y_pooled_sim[int(self.i/self.savefrqy),:] = Y

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

    def set_J(self,hGMM):
        self.J_loc = np.array(self.J_loc,dtype = 'i')
        if MPI.COMM_WORLD.Get_rank() == 0:
            J_locs = np.empty(MPI.COMM_WORLD.Get_size(),dtype='i')
        else:
            J_locs = None
        MPI.COMM_WORLD.Gather(sendbuf=[self.J_loc,MPI.INT],recvbuf=[J_locs,MPI.INT],root=0)
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.J = sum(J_locs)

    def set_Y_sim(self):
        self.Y_sim = collect_arrays(self.Y_sim_loc,self.d,'d',MPI.DOUBLE)

    def set_savesampnames(self):
        #print "self.savesampnames_loc at rank {}: {}".format(MPI.COMM_WORLD.Get_rank(),self.savesampnames_loc)
        name_data = [np.array([ch for ch in name]) for name in self.savesampnames_loc]
        #print "name_data at rank {}".format(name_data,MPI.COMM_WORLD.Get_rank())
        name_all = collect_arrays(name_data,1,'S',MPI.UNSIGNED_CHAR)
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.savesampnames = [''.join(nam.reshape((-1,))) for nam in name_all]
            #print "self.savesampnames at rank 0: {}".format(self.savesampnames)
         
    def set_savesamp(self):
        if MPI.COMM_WORLD == 0:
            self.savesamp = [self.names.index(name) for name in self.savesampnames]

#    def set_Y_sim_old(self):
#        comm = MPI.COMM_WORLD
#        self.set_save_ns()
#        if len(self.Y_sim_loc) > 0:
#            Y_sim_loc_array = np.array(np.vstack(self.Y_sim_loc),dtype = 'i')
#        else:
#            Y_sim_loc_array = np.empty((0,self.d),dtype='i')
#        #classif_freq_loc = classif_freq_loc[0:100,:]
#        if comm.Get_rank() == 0:
#            counts = np.empty(comm.Get_size(),dtype = 'i')
#        else:
#            counts = 0
#        comm.Gather(sendbuf=[np.array(Y_sim_loc_array.shape[0] * Y_sim_loc_array.shape[1],dtype='i'), MPI.INT], recvbuf=[counts, MPI.INT], root=0)
#        
#        if comm.Get_rank() == 0:
#            Y_sim_all = np.empty((sum(counts)/(self.d),self.d),dtype = 'd')
#            #print "classif_all.shape = {}".format(classif_all.shape)
#        else:
#            Y_sim_all = None
#        comm.Gatherv(sendbuf=[Y_sim_loc_array,MPI.DOUBLE],recvbuf=[Y_sim_all,(counts,None),MPI.DOUBLE],root=0)
#        if comm.Get_rank() == 0:
#            self.Y_sim = np.split(self.Y_sim_all, np.cumsum(self.save_ns[0:-1]))
#            
#    def set_savesampnames_old(self):
#        comm = MPI.COMM_WORLD
#        if comm.Get_rank() == 0:
#            counts = np.empty(comm.Get_size(),dtype='i')
#        else:
#            counts = 0
#        comm.Gather(sendbuf=[np.array(sum([len(name) for name in self.savesampnames_loc])),MPI.INT],recvbuf=[counts,MPI.INT],root=0)
#
#        if len(self.savesampnames_loc) > 0:
#            namearray = np.vstack(self.savesampnames_loc)
#        else:
#            namearray = np.array([])
#            
#        if comm.Get_rank() == 0:
#            savesampnames_ = np.empty(sum(counts),dtype='S')
#        else:
#            savesampnames_ = None
#        comm.Gatherv(sendbuf=[namearray,MPI.UNSIGNED_CHAR],recvbuf=[savesampnames_,(counts,None),MPI.UNSIGNED_CHAR],root=0)
#        if comm.Get_rank() == 0:
#            savesampnames = np.split(savesampnames_,np.cumsum(counts[:-1]))
#            self.savesampnames = [''.join(name) for name in savesampnames]
#         
#    def set_save_ns(self):
#        '''
#            Obsolete
#        '''
#        comm = MPI.COMM_WORLD
#        ns_loc = np.array([ysim.shape[0] for ysim in self.Y_sim_loc],dtype='i')
#        if comm.Get_rank() == 0:
#            counts = np.empty(comm.Get_size(),dtype = 'i')
#        else:
#            counts = 0
#        comm.Gather(sendbuf=[np.array(len(ns_loc),dtype='i'), MPI.INT], recvbuf=[counts, MPI.INT], root=0)
#        
#        if comm.Get_rank() == 0:
#            ns = np.empty(sum(ns_loc),dtype = 'i')
#        else:
#            ns = None
#        comm.Gatherv(sendbuf=[ns_loc,MPI.INT],recvbuf=[ns,(counts,None),MPI.INT],root=0)
#        if comm.Get_rank() == 0:
#            self.save_ns = ns

class HMElog(HMlog):

    '''
        Class for saving results from sampling of posterior distribution
        NB! Does save classification frequencies and
        thus makes it possible to create Clustering object.
    '''
    
    def __init__(self,hGMM,sim,savesamp=None,savesampnames=None,nbrsave=None,savefrq=None,nbrsavey=None,savefrqy=None):
        super(HMElog,self).__init__(hGMM,sim,savesamp,savesampnames,nbrsave,savefrq,nbrsavey,savefrqy)
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
        if not self.ii == self.batch:
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






