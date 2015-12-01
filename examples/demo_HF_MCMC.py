# -*- coding: utf-8 -*-
"""
Created on Fri May  8 16:04:29 2015

@author: johnsson
"""
from __future__ import division

import BayesFlow as bf
from BayesFlow.utils import Timer
from BayesFlow.utils.dat_util import sampnames_scattered

from example_util import retrieve_healthyFlowData, load_setup_HF, get_J

timer = Timer()

retrieve_healthyFlowData(data_kws['datadir'])
metadata = {'marker_lab': data_kws.pop('marker_lab'),
            'samp': {'names': sampnames_scattered(comm, data_kws['datadir'], data_kws['ext'])}}
print "metadata['samp']['names'] = {}".format(metadata['samp']['names'])
timer.timepoint('retrieve data')


'''
    Initialization
'''

setupfile, setup = load_setup_HF(setupdir, setupno)
savedir, run = bf.setup_sim(expdir, seed, setupfile)  # copies experiment setup and set seed
prior, simpar, postpar = setup(get_J(metadata['samp']['names']), Nevent, testrun)

if rank == 0:
    print "n_theta set to {}, n_Psi set to {}".format(prior.n_theta[0], prior.n_Psi[0])
    print "Q set to {}, H set to {}".format(prior.Q[0], prior.H[0])

hGMM = bf.hierarical_mixture_mpi(K=prior.K, AMCMC=simpar.AMCMC, comm=comm)
print "data_kws = {}".format(data_kws)
hGMM.load_data(metadata['samp']['names'], **data_kws)
#hGMM.set_prior(prior = prior, init=True, thetas = thetas_init)
hGMM.set_prior(prior)
hGMM.set_init(prior, method='EM_pooled', n_iter=50, n_init=1)

for j, GMM in enumerate(hGMM.GMMs):
    print "rank {} sample {} has name {}".format(rank, j, GMM.name)

timer.timepoint('initialization')
timer.print_timepoints()

'''
    MCMC sampling
'''

'''
        Burn in iterations
'''

if hGMMtiming:
    hGMM.toggle_timing(on=True)

print "simpar.phases['B'] = {}".format(simpar.phases['B'])
burnlog = hGMM.simulate(simpar.phases['B'], 'Burnin phase')
hGMM.save_burnlog(savedir)


timer.timepoint('burnin iterations ({}) and postproc'.format(simpar.nbriter*simpar.qburn))

'''
        Production iterations
'''
prodlog = hGMM.simulate(simpar.phases['P'], 'Production phase', stop_if_cl_off=False)
hGMM.save_log(savedir)
hGMM.save(savedir)

timer.timepoint('production iterations ({}) and postproc'.format(simpar.nbriter*simpar.qprod))
timer.print_timepoints()
