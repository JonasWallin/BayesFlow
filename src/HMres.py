# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 18:35:14 2014

@author: johnsson
"""
from __future__ import division
from mpi4py import MPI
import numpy as np

from .HMplot import HMplot
from .PurePython.GMM import mixture
from .utils.results_mem_efficient import Mres, Traces, MimicSample, Components, MetaData
from .utils.initialization.distributed_data import DataMPI
from .utils.initialization.EM import EMD_to_generated_from_model


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

    def get_bh_distance_to_own_latent(self):
        return self.components.get_bh_distance_to_own_latent()

    def get_center_dist(self):
        return self.components.get_center_dist()

    def get_mix(self, j):
        active = self.active_komp[j, :] > 0.05
        mus = [self.components.mupers[j, k, :] for k in range(self.K) if active[k]]
        Sigmas = [self.components.Sigmapers[j, k, :, :] for k in range(self.K) if active[k]]
        ps = self.components.p[j, active]
        return mus, Sigmas, ps

    def generate_from_mix(self, j, N):
        mus, Sigmas, ps = self.get_mix(j)
        return mixture.simulate_mixture(mus, Sigmas, ps, N)

    @property
    def K_active(self):
        return np.sum(np.sum(self.active_komp > 0.05, axis=0) > 0)

    def earth_movers_distance_to_generated(self):
        emds = []
        for j, dat in enumerate(self.data):
            mus, Sigmas, ps = self.get_mix(j)
            N_synsamp = int(dat.shape[0]//10)
            emds.append(EMD_to_generated_from_model(
                DataMPI(MPI.COMM_SELF, [dat]), mus, Sigmas, ps, N_synsamp, gamma=1, nbins=50)/N_synsamp)
            print "\r EMD computed for {} components".format(j+1),
        print "\r ",
        print ""
        return np.vstack(emds)
