import os
import time
import tempfile
import imp
import shutil
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

from .. import setup_sim, hierarical_mixture_mpi, HMlogB, HMElog, HMres
from ..utils import load_fcdata
from ..utils.dat_util import sampnames_mpi
from ..PurePython.GMM import mixture
from ..exceptions import NoOtherClusterError


class SynSample(object):
    def __init__(self, j):
        self.d = 3
        self.K = 4
        self.n_obs = 1000
        self.mu = [np.repeat(k+0.01*j, self.d) for k in range(self.K)]
        self.sigma = [np.eye(self.d)*0.01 for k in range(self.K)]
        self.p = np.ones(self.K)/self.K
        self.name = str(j)

    def generate_data(self, savedir):
        '''
            Generate data and save
        '''
        Y = mixture.simulate_mixture(self.mu, self.sigma, self.p, self.n_obs)
        np.savetxt(os.path.join(savedir, self.name+'.txt'), Y)


class SynSample2(object):
    def __init__(self, j):
        self.d = 2
        self.n_obs = 1000
        self.mu = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.mu = [np.array(mu)+0.01*j for mu in self.mu]
        self.K = len(self.mu)
        #[np.repeat(k+0.01*j, self.d) for k in range(self.K)]
        self.sigma = [np.eye(self.d)*0.01 for k in range(self.K)]
        self.p = np.ones(self.K)/self.K
        self.name = str(j)

    def generate_data(self, savedir):
        '''
            Generate data and save
        '''
        Y = mixture.simulate_mixture(self.mu, self.sigma, self.p, self.n_obs)
        np.savetxt(os.path.join(savedir, self.name+'.txt'), Y)


class Pipeline(object):

    def __init__(self, comm=MPI.COMM_WORLD):
        self.comm = comm

        self.J = 5
        self.savedir = 'blah'#tempfile.mkdtemp()
        print "savedir = {}".format(self.savedir)
        self.datadir = os.path.join(self.savedir, 'data')
        self.synsamples = [SynSample2(j) for j in range(self.J)]
        self.d = self.synsamples[0].d

        self.parfile = "src/tests/param/0_no_onoff.py"
        self.data_kws = {'scale': 'percentilescale',
                         'loadfilef': lambda filename: np.loadtxt(filename),
                         'ext': '.txt', 'datadir': self.datadir}
        self.metadata = {'marker_lab': [str(i+1) for i in range(self.d)]}

    def generate_data(self):
        if not os.path.exists(self.datadir):
            os.mkdir(self.datadir)
        for synsamp in self.synsamples:
            synsamp.generate_data(self.datadir)

    def setup_run(self):
        self.rundir, self.run = setup_sim(self.savedir, setupfile=self.parfile)
        try:
            self.bf_setup = imp.load_source('src.tests.param.setup', self.parfile).setup
        except IOError as e:
            print "Setupfile {} does not exist".format(self.parfile)
            print "Setupdir has files: {}".format(os.listdir(os.path.split(self.parfile)[0]))
            raise e

    def init_hGMM(self):
        '''
            Load prior, initialize hGMM, load data
        '''
        Nevent = np.mean([synsamp.n_obs for synsamp in self.synsamples])
        self.prior, self.simpar, self.postpar = self.bf_setup(
            self.comm, self.J, Nevent, self.d, K=8)
        self.hGMM = hierarical_mixture_mpi(K=self.prior.K, comm=self.comm)
        sampnames = sampnames_mpi(self.comm, self.datadir, self.data_kws['ext'])
        print "sampnames before load: {}".format(sampnames)
        self.hGMM.load_data(sampnames, **self.data_kws)
        print "after load data"
        self.hGMM.set_prior(prior=self.prior, init=False)
        self.hGMM.set_init(self.prior, method='EM_pooled', WIS=False,
                           N=int(Nevent*self.J/100), rho=2, n_iter=20, n_init=100,
                           plotting=True, selection='likelihood', gamma=5)
        self.hGMM.toggle_timing()

    def MCMC(self):
        '''
            Burn-in iterations
        '''
        printfrq = 10

        t0 = time.time()

        self.hGMM.resize_var_priors(self.simpar.tightinitfac)
        self.hGMM.simulate(self.simpar.phases['B1a'], 'Burnin phase 1a', printfrq=printfrq, stop_if_cl_off=False)
        print "prior vals: {}".format(self.hGMM.prior.__dict__)
        self.hGMM.resize_var_priors(1./self.simpar.tightinitfac)
        print "prior vals: {}".format(self.hGMM.prior.__dict__)
        self.hGMM.simulate(self.simpar.phases['B1b'], 'Burnin phase 1b', printfrq=printfrq, stop_if_cl_off=False)
        self.hGMM.set_theta_to_median()
        #hGMM.deactivate_outlying_components()
        self.hGMM.set_GMMs_mu_Sigma_from_prior()
        self.comm.Barrier()
        self.hGMM.simulate(self.simpar.phases['B2a'], 'Burnin phase 2a', stop_if_cl_off=False, printfrq=printfrq)
        self.hGMM.simulate(self.simpar.phases['B2b'], 'Burnin phase 2a', stop_if_cl_off=False, printfrq=printfrq)
        self.hGMM.simulate(self.simpar.phases['B3'], 'Burnin phase 3', stop_if_cl_off=False)
        self.hGMM.save_burnlog(self.rundir)

        t1 = time.time()

        '''
                Production iterations
        '''
        self.hGMM.simulate(self.simpar.phases['P'], 'Production phase', stop_if_cl_off=False, printfrq=printfrq)
        self.hGMM.save_log(self.rundir)

        t2 = time.time()

        print 'burnin iterations ({}) and postproc: {} s'.format(self.simpar.nbriter*self.simpar.qburn, t1-t0)
        print 'production iterations ({}) and postproc: {} s'.format(self.simpar.nbriter*self.simpar.qprod, t2-t1)

        del self.hGMM

    def postproc(self):
        blog = HMlogB.load(self.rundir)
        log = HMElog.load(self.rundir)
        data = load_fcdata(log.names, **self.data_kws)
        self.metadata['samp'] = {'names': log.names}

        self.res = HMres(log, blog, data, self.metadata)
        self.res.merge(self.postpar.mergemeth, **self.postpar.mergekws)

    def quality_check(self):
        print "self.res.active_komp = {}".format(self.res.active_komp)

    def plot(self):

        plotdim = [[i, j] for i in range(self.d) for j in range(i+1, self.d)]

        fig = plt.figure(figsize=(18, 9))
        try:
            self.res.components.plot.center_distance_quotient(fig=fig, totplots=2, plotnbr=1)
            self.res.components.plot.bhattacharyya_overlap_quotient(fig=fig, totplots=2, plotnbr=2)
        except NoOtherClusterError:
            pass

        fig = plt.figure(figsize=(9, 9))
        self.res.components.plot.cov_dist(fig=fig)

        self.res.traces.plot.all(fig=plt.figure(figsize=(18, 4)), yscale=False)
        self.res.traces.plot.nu()

        fig_m = plt.figure(figsize=(18, 12))
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

    def run(self):
        if 1:
            self.generate_data()
            self.setup_run()
            self.init_hGMM()
            print "prior vals: {}".format(self.hGMM.prior.__dict__)
            self.MCMC()
            self.postproc()
            self.quality_check()
            self.plot()
            if 1:
                plt.show()
        # except Exception as e:
        #     self.clean_up()
        #     raise e
        # else:
        #     self.clean_up()

if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.run()
