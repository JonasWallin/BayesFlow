# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 18:35:14 2014

@author: johnsson
"""
from __future__ import division
import os
from mpi4py import MPI
import numpy as np
import json
import matplotlib.pyplot as plt

from .HMplot import HMplot
from .HMlog import HMlogB, HMElog
from .PurePython.GMM import mixture
from .utils.dat_util import load_fcdata
from .utils.results_mem_efficient import Mres, Traces, MimicSample, Components, MetaData
from .utils.initialization.distributed_data import DataMPI
from .utils.initialization.EM import EMD_to_generated_from_model
from .exceptions import BadQualityError


class HMres(Mres):
    """
        Class for processing results from MCMC simulation, e.g. for merging and computing dip test
    """

    def __init__(self, hmlog, hmlog_burn, data, meta_data,
                 comm=MPI.COMM_WORLD, maxnbrsucocol=8):
        self.comm = comm
        self.rank = comm.Get_rank()
        if self.rank == 0:
            self.noise_class = hmlog.noise_class
            if self.noise_class:
                self.p_noise = hmlog.prob_sim_mean[:, hmlog.K]
                self.noise_mu = hmlog.noise_mu
                self.noise_sigma = hmlog.noise_sigma
            else:
                self.p_noise = None

            self.meta_data = MetaData(meta_data)
            self.meta_data.sort(hmlog.names)
            self.names = self.meta_data.samp['names']
            self.data = [data[j] for j in self.meta_data.order]

            super(HMres, self).__init__(hmlog.d, hmlog.K, hmlog.prob_sim_mean[:, :hmlog.K],
                                        hmlog.classif_freq, self.p_noise, hmlog.sim,
                                        maxnbrsucocol)

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

            self.plot = HMplot(self)

            self.quality = {}
            self.mergeind = self.mergeind  # make all attributes have same mergeind

    @classmethod
    def load(cls, savedir, data_kws, comm=MPI.COMM_SELF, no_postproc=False):
        hmlog_burn = HMlogB.load(savedir, comm=comm)
        hmlog = HMElog.load(savedir, comm=comm)
        hmlog.prob_sim_mean *= hmlog.active_komp  # compensate for reweighting in HMlog.postproc
        metadata = data_kws.copy()
        marker_lab = metadata.pop('marker_lab')
        data = load_fcdata(hmlog.names, comm=comm, **metadata)
        metadata.update(marker_lab=marker_lab)
        metadata.update(samp={'names': hmlog.names})
        print "metadata = {}".format(metadata)
        res = cls(hmlog, hmlog_burn, data, metadata, comm=comm)
        if not no_postproc:
            try:
                print "savedir = {}".format(savedir)
                with open(os.path.join(savedir, 'postproc_results.json'), 'r') as f:
                    postproc_res = json.load(f)
            except IOError:
                pass
            else:
                for attr in ['_earth_movers_distance_to_generated', 'pdiplist']:
                    try:
                        setattr(res, attr, np.load(os.path.join(savedir, attr+'.npy')))
                    except IOError:
                        pass
                for attr in postproc_res:
                    setattr(res, attr, postproc_res[attr])
        return res

    def save(self, savedir):
        if self.rank == 0:
            savedict = {}
            for attr in ['mergeind', 'postproc_par', '_emd_dims',
                         'quality', 'merged']:
                try:
                    savedict[attr] = getattr(self, attr)
                except AttributeError:
                    pass
            with open(os.path.join(savedir, 'postproc_results.json'), 'w') as f:
                json.dump(savedict, f)
            for attr in ['_earth_movers_distance_to_generated', 'pdiplist']:
                try:
                    np.save(os.path.join(savedir, attr+'.npy'), getattr(self, attr))
                except AttributeError:
                    pass

    @Mres.mergeind.setter
    def mergeind(self, mergeind):
        Mres.mergeind.fset(self, mergeind)
        if hasattr(self, 'components'):
            self.components.mergeind = self._mergeind
            self.components.suco_ord = self.suco_ord
            self.components.comp_ord = self.comp_ord
            self.components.suco_colors = self.suco_colors
            self.components.comp_colors = self.comp_colors
        if hasattr(self, 'traces'):
            self.traces.comp_ord = self.comp_ord

    # def check_active_komp(self):
    #     if ((self.active_komp > 0.05)*(self.active_komp < 0.95)).any():
    #         self.quality['ok_active_komp'] = False
    #         raise BadQualityError('Active components not in ok range')
    #     else:
    #         self.quality['ok_active_komp'] = True

    def check_convergence(self):
        if 'convergence' in self.quality:
            if self.quality['convergence'] == 'no':
                raise BadQualityError('No convergence')
            else:
                return
        self.traces.plot.all(fig=plt.figure(figsize=(18, 4)), yscale=True)
        self.traces.plot.nu()
        self.traces.plot.nu_sigma()
        plt.show()
        print "Are trace plots ok? (y/n)"
        while 1:
            ans = raw_input()
            if ans.lower() == 'y':
                self.quality['convergence'] = 'yes'
                break
            if ans.lower() == 'n':
                self.quality['convergence'] = 'no'
                raise BadQualityError('Trace plots not ok')
            print "Bad answer. Are trace plots ok? (y/n)"

    def check_noise(self, noise_lim=0.01):
        if self.noise_class:
            self.quality['max_p_noise'] = np.max(self.p_noise)
            if (self.p_noise > noise_lim).any():
                raise BadQualityError('Too high noise level')

    def check_outliers(self):
        bh_out = np.sum(self.components.get_latent_bhattacharyya_overlap_quotient() < 1)
        eu_out = np.sum(self.components.get_center_distance_quotient() < 1)
        self.quality['outliers'] = {'bhat': bh_out, 'eucl_loc': eu_out}
        if bh_out+eu_out > 0:
            raise BadQualityError('Not closest to own latent component, bhat: {}, eu: {}'.format(
                                  bh_out, eu_out))

    def check_dip(self, savedir=None):
        self.get_pdip()
        fig_dip = self.plot.pdip()
        fig_dip_summary = self.plot.pdip_summary()
        if not savedir is None:
            fig_dip.savefig(os.path.join(savedir, 'dip.pdf'), type='pdf',
                            transparent=False, bbox_inches='tight')
            fig_dip_summary.savefig(os.path.join(savedir, 'dip_summary.pdf'),
                                    type='pdf', transparent=False, bbox_inches='tight')
        else:
            plt.show()
        if (self.get_pdip_summary(suco=True)['25th percentile'] < 0.28).any():
            raise BadQualityError('25th percentile of pdip not ok')

    def check_emd(self, N=5, savedir=None):
        emd, emd_dim = self.earth_movers_distance_to_generated()
        self.quality['emd'] = {'min': np.min(emd), 'max': np.max(emd), 'median': np.median(emd)}
        fig, ax = plt.subplots(figsize=(15, 4))
        im = ax.imshow(emd.T, interpolation='None')
        plt.colorbar(im, orientation='horizontal')
        top_N = zip(*np.unravel_index(np.argpartition(-emd.ravel(), N)[:N], emd.shape))
        fig_fit, axs = plt.subplots(N, 4, figsize=(15, 15))
        for i, (j, i_dim) in enumerate(top_N):
            self.plot.component_fit([emd_dim[i_dim]], name=self.names[j], axs=axs[i, :].reshape(1, -1))
        if not savedir is None:
            fig.savefig(os.path.join(savedir, 'emd.pdf'), type='pdf',
                        transparent=False, bbox_inches='tight')
            fig_fit.savefig(os.path.join(savedir, 'fit_max_emd.pdf'), type='pdf',
                            transparent=False, bbox_inches='tight')
        else:
            plt.show()

    def check_quality(self, savedir=None, N_emd=5, noise_lim=0.01):
        #self.check_active_komp()
        self.check_noise(noise_lim)
        #self.check_convergence()
        self.check_outliers()
        self.check_dip(savedir)
        self.check_emd(N_emd, savedir)

    def merge(self, method, thr, **mmfArgs):
        if self.rank == 0:
            if method == 'bhat':
                super(HMres, self).greedy_merge(self.components.get_median_bh_overlap,
                                                thr, **mmfArgs)
                super(HMres, self).self.gclean()
            else:
                super(HMres, self).merge(method, thr, **mmfArgs)
            #self.components.mergeind = self.mergeind
            #self.plot = HMplot(self, self.meta_data.marker_lab)
            for i in range(self.comm.Get_size()):
                self.comm.send(self.mergeind, dest=i, tag=2)
        else:
            self.mergeind = self.comm.recv(source=0, tag=2)
        self.postproc_par = {'method': method, 'thr': thr, 'mmfArgs': mmfArgs}

    def get_bh_distance_to_own_latent(self):
        return self.components.get_bh_distance_to_own_latent()

    def get_center_dist(self):
        return self.components.get_center_dist()

    def get_mix(self, j):
        active = self.active_komp[j, :] > 0.05
        mus = [self.components.mupers[j, k, :] for k in range(self.K) if active[k]]
        Sigmas = [self.components.Sigmapers[j, k, :, :] for k in range(self.K) if active[k]]
        ps = [self.components.p[j, k] for k in range(self.K) if active[k]]
        if self.noise_class:
            mus.append(self.noise_mu)
            Sigmas.append(self.noise_sigma)
            ps.append(self.p_noise[j])
        print "np.sum(ps) = {}".format(np.sum(ps))
        ps /= np.sum(ps)  # renormalizing
        return mus, Sigmas, np.array(ps)

    def generate_from_mix(self, j, N):
        mus, Sigmas, ps = self.get_mix(j)
        return mixture.simulate_mixture(mus, Sigmas, ps, N)

    @property
    def K_active(self):
        return np.sum(np.sum(self.active_komp > 0.05, axis=0) > 0)

    def earth_movers_distance_to_generated(self, gamma=1):
        if hasattr(self, '_earth_movers_distance_to_generated'):
            return self._earth_movers_distance_to_generated, self._emd_dims
        emds = []
        dims = [(i, j) for i in range(self.d) for j in range(i+1, self.d)]
        for j, dat in enumerate(self.data):
            mus, Sigmas, ps = self.get_mix(j)
            N_synsamp = int(dat.shape[0]//10)
            emds.append(
                np.array(EMD_to_generated_from_model(
                    DataMPI(MPI.COMM_SELF, [dat]), mus, Sigmas, ps, N_synsamp,
                    gamma=gamma, nbins=50, dims=dims))
                * (1./N_synsamp))
            print "\r EMD computed for {} samples".format(j+1),
        print "\r ",
        print ""
        self._earth_movers_distance_to_generated = np.vstack(emds)
        self._emd_dims = dims
        print "dims = {}".format(dims)
        return self._earth_movers_distance_to_generated, self._emd_dims
