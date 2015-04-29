# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 18:35:14 2014

@author: johnsson
"""
from __future__ import division
from mpi4py import MPI
import BMplot as bmp
from .utils.results import Mres, Traces, MimicSample, Components

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class HMres(Mres):
    """
        Class for processing results from MCMC simulation, e.g. for merging and computing dip test
    """

    def __init__(self,bmlog,bmlog_burn,data,meta_data):
        if rank == 0:
            self.noise_class = bmlog.noise_class
            if self.noise_class:
                self.p_noise = bmlog.prob_sim_mean[:,bmlog.K] 
            else:
                self.p_noise = None
            super(HMres,self).__init__(bmlog.d,bmlog.K,bmlog.prob_sim_mean[:,:bmlog.K],
            bmlog.classif_freq,data,meta_data,self.p_noise)

            self.sim = bmlog.sim

            self.active_komp = bmlog.active_komp

            self.meta_data.sort(bmlog.names)
            self.data = [self.data[j] for j in self.meta_data.order]

            self.traces = Traces(bmlog_burn,bmlog)
            self.mimics = {}
            print "bmlog.savesmapnames = {}".format(bmlog.savesampnames)
            for i,name in enumerate(bmlog.savesampnames):
                j = bmlog.names.index(name)
                self.mimics[name] = MimicSample(self.data[j],name,bmlog.Y_sim[i],'BHM_MCMC')
            self.mimics['pooled'] = MimicSample(self.datapooled,'pooled',bmlog.Y_pooled_sim,'BHM_MCMC')
                
            self.components = Components(bmlog,self.p)                   
            
            self.plot = bmp.BMplot(self,self.meta_data.marker_lab)
                
    def merge(self,method,thr,**mmfArgs):
        if rank == 0:
            if method == 'bhat':
                super(HMres,self).greedy_merge(self.components.get_median_bh_dist,thr,**mmfArgs)
                super(HMres,self).self.gclean()
            else:
                super(HMres,self).merge(method,thr,**mmfArgs)
            self.components.mergeind = self.mergeind
            self.plot = bmp.BMplot(self,self.meta_data.marker_lab)
            for i in range(size):
                comm.send(self.mergeind,dest=i,tag=2)
        else:
            self.mergeind = comm.recv(source=0,tag=2)

                













