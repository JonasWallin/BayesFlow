import os
import time
import tempfile
import imp
from mpi4py import MPI
import numpy as np

import BayesFlow as bf
from BayesFlow.PurePython.GMM import mixture


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
        mix = mixture(K=self.K)
        mix.mu = self.mu
        mix.sigma = self.sigma
        mix.p = self.p
        mix.d = self.d
        Y = mix.simulate_data(self.n_obs)
        np.savetxt(os.path.join(savedir, self.name+'.txt'), Y)


class Pipeline(object):

    def __init__(self, comm=MPI.COMM_WORLD):
        self.comm = comm

        self.J = 20
        self.savedir = tempfile.mkdtemp()
        print "savedir = {}".format(self.savedir)
        self.datadir = os.path.join(self.savedir, 'data')
        self.synsamples = [SynSample(j) for j in range(self.J)]
        self.d = self.synsamples[0].d

        self.parfile = "param/0.py"
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
        self.rundir, self.run = bf.setup_sim(self.savedir)
        try:
            self.bf_setup = imp.load_source('', self.parfile).setup
        except IOError as e:
            print "Setupfile {} does not exist".format(self.parfile)
            print "Setupdir has files: {}".format(os.listdir(os.path.split(self.parfile)[0]))
            raise e

    def init_hGMM(self):
        '''
            Load prior, initialize hGMM, load data
        '''
        Nevent = np.mean([synsamp.n_obs for synsamp in self.synsamples])
        self.prior, self.simpar, self.postpar = self.bf_setup(self.J, Nevent,
                                                              self.d, K=4)
        self.hGMM = bf.hierarical_mixture_mpi(K=self.prior.K, comm=self.comm)
        sampnames = bf.utils.dat_util.sampnames_mpi(self.comm, self.datadir,
                                                    self.data_kws['ext'])
        self.hGMM.load_data(sampnames, **self.data_kws)
        self.hGMM.set_prior(prior=self.prior, init=True)

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
        self.hGMM.save_burnlog(self.savedir)

        t1 = time.time()

        '''
                Production iterations
        '''
        self.hGMM.simulate(self.simpar.phases['P'], 'Production phase', stop_if_cl_off=False, printfrq=printfrq)
        self.hGMM.save_log(self.savedir)

        t2 = time.time()

        print 'burnin iterations ({}) and postproc: {} s'.format(self.simpar.nbriter*self.simpar.qburn, t1-t0)
        print 'production iterations ({}) and postproc: {} s'.format(self.simpar.nbriter*self.simpar.qprod, t2-t1)

        del self.hGMM

    def postproc(self):
        blog = bf.HMlogB.load(self.savedir)
        log = bf.HMElog.load(self.savedir)
        data = bf.utils.dat_util.load_fcdata(log.names, **self.data_kws)
        self.metadata['samp'] = {'names': log.names}
                    
        self.res = bf.HMres(log, blog, data, self.metadata)
        self.res.merge(self.postpar.mergemeth, **self.postpar.mergekws)

    def quality_check(self):
        pass

    def plot(self):
        pass

    def clean_up(self):
        print "remove savedir {}".format(self.savedir)
        #shutil.rmtree(self.savedir)
    
    def run(self):
        try:
            self.generate_data()
            self.setup_run()
            self.init_hGMM()
            print "prior vals: {}".format(self.hGMM.prior.__dict__)
            self.MCMC()
            self.postproc()
        except:
            self.clean_up()
            raise
        else:
            self.clean_up()
    
if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.run()
