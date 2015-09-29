from __future__ import division
import os
import time
import tempfile
import imp
import shutil
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

from .. import setup_sim, hierarical_mixture_mpi, HMlogB, HMElog, HMres
from .exceptions import BadQualityError
from ..utils import load_fcdata
from ..utils.initialization.EM import (EMD_to_generated_from_model,
                                       data_log_likelihood)
from ..utils.initialization.distributed_data import DataMPI
from ..PurePython.GMM import mixture
from ..exceptions import NoOtherClusterError
from ..PurePython.distribution.wishart import invwishartrand


def some_small_clusters_some_empty(K, q):
    p = np.ones(K)
    p[np.random.randint(K, size=int(K*q))] = 0.05
    p[np.random.randint(K, size=int(K*q))] = 0
    return p/np.sum(p)


def one_small_rare_cluster(K):
    p = np.ones(K)
    if np.random.rand() < 1:#0.2:
        p[0] = 0.002
    else:
        p[0] = 0
    return p/np.sum(p)


p_fun_dict = {
    'A': lambda K: np.ones(K)/K,
    'B': lambda K: some_small_clusters_some_empty(K, 0.25),
    'C': lambda K: one_small_rare_cluster(K)
}


class SynSamp(object):

    def __init__(self, j, n_obs, d, C, ver='A'):
        self.d = d
        self.C = C  # Number of clusters.
        self.n_obs = n_obs
        self.name = str(j)
        self.ver = ver

    def generate_data(self, savedir):
        '''
            Generate data and save
        '''
        #Y = mixture.simulate_mixture(self.mu, self.sigma, self.p, self.n_obs)
        np.savetxt(os.path.join(savedir, self.name+'.txt'), self.data)

    @property
    def data(self):
        return mixture.simulate_mixture(self.mu, self.sigma, self.p, self.n_obs)

    @property
    def p(self):
        return p_fun_dict[self.ver](self.C)


class SynSample(SynSamp):
    def __init__(self, j, n_obs, d=3, C=4, ver='A'):
        super(SynSample, self).__init__(j, n_obs, d, C, ver)
        self.mu = [np.repeat(k+0.1*(j % 5), self.d) for k in range(self.C)]
        self.sigma = [invwishartrand(self.d+1+5*(k+1)**2, np.eye(self.d)*0.01*5*(k+1)**2) for k in range(self.C)]


class SynSample2(SynSamp):
    def __init__(self, j, n_obs, d=2, C=None, ver='A'):
        super(SynSample2, self).__init__(j, n_obs, d, C, ver)
        self.mu = [tuple(int(i) for i in "{0:0{width}b}".format(k, width=d))
                   for k in range(2**d)]  # all corners of d-dimensional cube
        self.mu = [np.array(mu)+0.1*(j % 3) for mu in self.mu]
        self.C = len(self.mu)  # Number of clusters.
        self.sigma = [np.eye(self.d)*0.05 for k in range(self.C)]


class Pipeline(object):

    def __init__(self, J, K, N, d, C=None, data_class=SynSample, ver='A',
                 par_file='src/tests/param/0.py', run=None,
                 copy_data=False, comm=MPI.COMM_WORLD):
        self.comm = comm
        self.rank = comm.Get_rank()

        self.J = J
        self.K = K  # Number of components.
        self.N = N
        self.savedir = 'blah'  # tempfile.mkdtemp()
        print "savedir = {}".format(self.savedir)
        self.datadir = os.path.join(self.savedir, 'data')
        if self.rank == 0:
            js = np.array_split(np.arange(J), comm.Get_size())
        else:
            js = None
        js = self.comm.scatter(js)
        self.synsamples = [data_class(j, N, d, C, ver=ver) for j in js]
        self.d = self.synsamples[0].d

        self.parfile = par_file
        self.data_kws = {'scale': 'percentilescale',
                         'loadfilef': lambda filename: np.loadtxt(filename),
                         'ext': '.txt', 'datadir': self.datadir,
                         'overwrite_eventind': True}
        self.metadata = {'marker_lab': [str(i+1) for i in range(self.d)]}
        self.logdata = {}
        if not run is None:
            self.run_nbr = run
            self.rundir = os.path.join(self.savedir, 'run'+str(self.run_nbr))
        self.copy_data = copy_data

    def generate_data(self):
        if self.rank == 0:
            if not os.path.exists(self.datadir):
                os.mkdir(self.datadir)
        self.comm.Barrier()
        self.names = []
        for synsamp in self.synsamples:
            synsamp.generate_data(self.datadir)
            self.names.append(synsamp.name)

    def setup_run(self):
        if not hasattr(self, 'run_nbr'):
            self.rundir, self.run_nbr = setup_sim(self.savedir, setupfile=self.parfile, comm=self.comm)
            if self.copy_data:
                datadir = self.data_kws['datadir']
                run_datadir = os.path.join(self.rundir, 'data')
                if self.rank == 0:
                    if not os.path.exists(run_datadir):
                        os.mkdir(run_datadir)
                self.comm.Barrier()
                for name in self.names:
                    shutil.copy(os.path.join(datadir,
                                name+self.data_kws['ext']), run_datadir)
                if self.rank == 0:
                    eventind_dir = os.path.join(datadir, 'eventinds')
                    shutil.copytree(eventind_dir, os.path.join(run_datadir, 'eventinds'))
        try:
            self.bf_setup = imp.load_source('src.tests.param.setup', self.parfile).setup
        except IOError as e:
            print "Setupfile {} does not exist".format(self.parfile)
            print "Setupdir has files: {}".format(os.listdir(os.path.split(self.parfile)[0]))
            raise e
        self.Nevent = np.mean([synsamp.n_obs for synsamp in self.synsamples])
        self.prior, self.simpar, self.postpar = self.bf_setup(
            self.comm, self.J, self.Nevent, self.d, K=self.K)

    def init_hGMM(self, method='EM_pooled', WIS=False, rho=2, n_iter=20,
                  n_init=10, plotting=False, selection='likelihood', gamma=2):
        '''
            Load prior, initialize hGMM, load data
        '''
        self.hGMM = hierarical_mixture_mpi(K=self.prior.K, AMCMC=self.simpar.AMCMC,
                                           comm=self.comm)
        #sampnames = sampnames_scattered(self.comm, self.datadir, self.data_kws['ext'])
        #print "sampnames before load: {}".format(sampnames)
        self.hGMM.load_data(self.names, **self.data_kws)
        print "after load data"
        self.hGMM.set_prior(prior=self.prior, init=False)
        t0 = time.time()
        self.hGMM.set_init(self.prior, method=method, WIS=WIS,
                           N=int(self.Nevent*self.J/100), rho=rho, n_iter=n_iter, n_init=n_init,
                           plotting=plotting, selection=selection, gamma=gamma)
        t1 = time.time()
        self.logdata['t_init'] = t1-t0
        self.hGMM.toggle_timing()

    def MCMC(self, plot_sim=False):
        '''
            Burn-in iterations
        '''
        printfrq = 100
        sim_settings = {'printfrq': printfrq, 'stop_if_cl_off': False,
                        'plotting': plot_sim, 'plotdim': [[0, 1]]}

        t0 = time.time()

        self.hGMM.resize_var_priors(self.simpar.tightinitfac)
        self.hGMM.simulate(self.simpar.phases['B1a'], 'Burnin phase 1a', **sim_settings)
        self.hGMM.resize_var_priors(1./self.simpar.tightinitfac)
        self.hGMM.simulate(self.simpar.phases['B1b'], 'Burnin phase 1b', **sim_settings)
        self.hGMM.set_theta_to_median()
        #print "any deactivated at rank {}: {}".format(self.comm.Get_rank(), self.hGMM.deactivate_outlying_components())
        self.hGMM.set_GMMs_mu_Sigma_from_prior()
        self.comm.Barrier()
        self.hGMM.simulate(self.simpar.phases['B2a'], 'Burnin phase 2a', **sim_settings)
        self.hGMM.simulate(self.simpar.phases['B2b'], 'Burnin phase 2b', **sim_settings)
        self.hGMM.simulate(self.simpar.phases['B3'], 'Burnin phase 3', **sim_settings)
        self.hGMM.save_burnlog(self.rundir)

        t1 = time.time()

        '''
                Production iterations
        '''
        self.hGMM.simulate(self.simpar.phases['P'], 'Production phase', **sim_settings)
        self.hGMM.save_log(self.rundir)

        t2 = time.time()

        print 'burnin iterations ({}) and postproc: {} s'.format(
            self.simpar.nbriter*self.simpar.qburn, t1-t0)
        print 'production iterations ({}) and postproc: {} s'.format(
            self.simpar.nbriter*self.simpar.qprod, t2-t1)

        del self.hGMM

    def load_res(self, comm=MPI.COMM_SELF):
        blog = HMlogB.load(self.rundir, comm=comm)
        self.logdata['lab_sw'] = blog.lab_sw
        log = HMElog.load(self.rundir, comm=comm)
        if self.copy_data:
            data_kws = self.data_kws.copy()
            data_kws['datadir'] = os.path.join(self.rundir, 'data')
        else:
            data_kws = self.data_kws
        data = load_fcdata(log.names, comm=comm, **data_kws)
        self.metadata['samp'] = {'names': log.names}

        self.res = HMres(log, blog, data, self.metadata, comm=comm)

    def postproc(self, comm=MPI.COMM_SELF):
        self.load_res(comm)
        self.res.merge(self.postpar.mergemeth, **self.postpar.mergekws)

    @property
    def number_label_switches(self):
        return len(self.logdata['lab_sw'])

    def quality_check(self):
        print "self.res.active_komp = {}".format(self.res.active_komp)

        self.res.traces.plot.all(fig=plt.figure(figsize=(18, 4)), yscale=True)
        self.res.traces.plot.nu()
        self.res.traces.plot.nu_sigma()
        plt.show()
        print "Are trace plots ok? (y/n)"
        while 1:
            ans = raw_input()
            if ans.lower() == 'y':
                break
            if ans.lower() == 'n':
                raise BadQualityError('Trace plots not ok')
            print "Bad answer. Are trace plots ok? (y/n)"

        fig = plt.figure(figsize=(9, 4))
        try:
            self.res.components.plot.center_distance_quotient(fig=fig, totplots=2, plotnbr=1)
            self.res.components.plot.bhattacharyya_overlap_quotient(fig=fig, totplots=2, plotnbr=2)
        except NoOtherClusterError:
            print "Only one super component, cannot plot center distance \
                   and bhattacharyya overlap quotients."
            pass

        fig = plt.figure(figsize=(4, 4))
        self.res.components.plot.cov_dist(fig=fig)

        plt.show()
        print "Are distances to latent components ok? (y/n)"
        while 1:
            ans = raw_input()
            if ans.lower() == 'y':
                break
            if ans.lower() == 'n':
                raise BadQualityError('Distance to latent components not ok')
            print "Bad answer. Are distance to latent components ok? (y/n)"

        emd = np.empty(self.J)
        log_lik = np.empty(self.J)
        data_mpi = [DataMPI(MPI.COMM_SELF, [dat]) for dat in self.res.data]
        for j, dat_mpi in enumerate(data_mpi):
            mus, Sigmas, pis = self.res.get_mix(j)
            emd[j] = EMD_to_generated_from_model(dat_mpi, mus, Sigmas, pis,
                                                 int(self.N/10))
            log_lik[j] = data_log_likelihood(dat_mpi, mus, Sigmas, pis)

        return emd, log_lik

    def plot(self):

        plotdim = [[i, j] for i in range(self.d) for j in range(i+1, self.d)]

        fig_m = plt.figure(figsize=(9, 6))
        self.res.components.plot.center(yscale=False, fig=fig_m, totplots=4,
                                        plotnbr=1, alpha=0.3)
        self.res.plot.prob(fig=fig_m, totplots=4, plotnbr=2)
        self.res.components.plot.center(suco=False, yscale=False, fig=fig_m,
                                        totplots=4, plotnbr=3, alpha=0.3)
        self.res.plot.prob(suco=False, fig=fig_m, totplots=4, plotnbr=4)

        mimicnames = self.res.mimics.keys()
        self.res.plot.component_fit(plotdim, name=mimicnames[-1], fig=plt.figure(figsize=(18, 25)))
        self.res.plot.component_fit(plotdim, name='pooled', fig=plt.figure(figsize=(18, 25)))

    def clean_up(self):
        print "removing savedir {} ...".format(self.savedir)
        try:
            pass
            #shutil.rmtree(self.savedir)
        except Exception as e:
            print "Could not remove savedir {}: {}".format(self.savedir, e)
        else:
            print "removing savedir {} done".format(self.savedir)

    def run(self, init_method='EM_pooled', WIS=False, rho=2, init_n_iter=20,
            n_init=10, init_plotting=False, init_selection='likelihood', gamma=2,
            plot_sim=False):
        if 1:
            self.generate_data()
            self.setup_run()
            self.init_hGMM(method=init_method, WIS=WIS, rho=rho, n_iter=init_n_iter,
                           n_init=n_init, plotting=init_plotting, selection=init_selection, gamma=gamma)
            print "prior vals: {}".format(self.hGMM.prior.__dict__)
            self.MCMC(plot_sim)
            if self.rank == 0:
                self.postproc(MPI.COMM_SELF)
        # except Exception as e:
        #     self.clean_up()
        #     raise e
        # else:
        #     self.clean_up()

if __name__ == '__main__':
    pipeline = Pipeline(J=6, K=8, N=1000, d=3, data_class=SynSample, ver='A',
                        parfile='src/tests/param/0.py',)
    pipeline.run(plot_sim=True)
    if pipeline.rank == 0:
        pipeline.quality_check()
        pipeline.plot()
        plt.show()
