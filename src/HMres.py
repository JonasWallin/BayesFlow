# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 18:35:14 2014

@author: johnsson
"""
from __future__ import division
from mpi4py import MPI
import numpy as np

from .HMplot import HMplot
from .utils.results_mem_efficient import Mres, Traces, MimicSample, Components, MetaData


class HMres(Mres):
    """
        Class for processing results from MCMC simulation, e.g. for merging and computing dip test
    """

    def __init__(self, hmlog, hmlog_burn, data, meta_data, comm=MPI.COMM_WORLD):
        self.comm = comm
        self.rank = comm.Get_rank()
        if self.rank == 0:
            self.noise_class = hmlog.noise_class
            if self.noise_class:
                self.p_noise = hmlog.prob_sim_mean[:, hmlog.K]
            else:
                self.p_noise = None

            self.meta_data = MetaData(meta_data)
            self.meta_data.sort(hmlog.names)
            self.data = [data[j] for j in self.meta_data.order]

            super(HMres, self).__init__(hmlog.d, hmlog.K, hmlog.prob_sim_mean[:, :hmlog.K],
                                        hmlog.classif_freq, self.p_noise, hmlog.sim)

            self.sim = hmlog.sim

            self.active_komp = hmlog.active_komp

            self.traces = Traces(hmlog_burn, hmlog)
            self.mimics = {}
            print "hmlog.savesampnames = {}".format(hmlog.savesampnames)
            for i, name in enumerate(hmlog.savesampnames):
                j = hmlog.names.index(name)
                self.mimics[name] = MimicSample(self.data[j], name, hmlog.Y_sim[i], 'BHM_MCMC')
            datapooled = np.vstack(data)[np.random.choice(range(sum([dat.shape[0] for dat in data])),
                                                          hmlog.Y_pooled_sim.shape[0], replace=False), :]
            self.mimics['pooled'] = MimicSample(datapooled, 'pooled', hmlog.Y_pooled_sim, 'BHM_MCMC')

            self.components = Components(hmlog, self.p)

            self.plot = HMplot(self, self.meta_data.marker_lab)

    def merge(self, method, thr, **mmfArgs):
        if self.rank == 0:
            if method == 'bhat':
                super(HMres, self).greedy_merge(self.components.get_median_bh_overlap,
                                                thr, **mmfArgs)
                super(HMres, self).self.gclean()
            else:
                super(HMres, self).merge(method, thr, **mmfArgs)
            self.components.mergeind = self.mergeind
            self.plot = HMplot(self, self.meta_data.marker_lab)
            for i in range(self.comm.Get_size()):
                self.comm.send(self.mergeind, dest=i, tag=2)
        else:
            self.mergeind = self.comm.recv(source=0, tag=2)
