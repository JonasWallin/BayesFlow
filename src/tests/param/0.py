
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 20:02:37 2014

@author: johnsson
"""
from __future__ import division
import numpy as np
from ... import SimPar, PostProcPar, Prior


def setup(comm, J, n_J, d, K):

    rank = comm.Get_rank()
    if rank == 0:
        print "setup with J = {} and n_J = {}".format(J, n_J)

    #print "Informative prior locations: {}".format(t_inf)

    prior = Prior(J, n_J, d=d, K=K)
    prior.latent_cluster_means(t_inf=None, t_ex=0.5, Sk_inf=1, Sk_ex=1e6)
    prior.component_location_variance(nt=0.3, q=1e-3)
    prior.component_shape(nps=0.1, h=1e3)
    prior.set_noise_class(noise_mu=0.5, noise_Sigma=0.5**2, on=False)  # We do not introduce noise class from start
    prior.pop_size()

    qB1a = 0.20
    qB1b = 0.20
    qB2a = 0.05
    qB2b = 0.25
    qB3 = 0.30

    nbriter = 1000

    simpar = SimPar(nbriter=nbriter, qburn=0.8, tightinit=100, simsamp=['1', '2', '3'])
    simpar.new_burnphase(qB1a, 'B1a')
    simpar.set('B1a', p_sw=0, p_on_off=[0, 0])
    if rank == 0:
        simpar.set_nu_MH('B1a', sigma=[max(prior.d, np.ceil(.1*prior.n_Psi[k]))
                                       for k in range(prior.K)], iteration=5)
    simpar.new_burnphase(qB1b, 'B1b')
    simpar.set('B1b', p_sw=0, p_on_off=[0, 0])
    simpar.new_burnphase(qB2a, 'B2a')
    simpar.set('B2a', p_sw=0.1, p_on_off=[0, 0])
    simpar.new_burnphase(qB2b, 'B2b')
    simpar.set('B2b', p_sw=0.1, p_on_off=[0.1, 0.1])
    simpar.new_trialphase(100)
    simpar.new_burnphase(qB3, 'B3')
    simpar.set('B3', p_sw=0, p_on_off=[0.1, 0.1])
    simpar.set_nu_MH('B3', sigma_nuprop=0.1)
    simpar.new_prodphase(last_burnphase='B3')

    postproc = PostProcPar(True, 'bhat_hier_dip', thr=0.47, lowthr=0.08, dipthr=0.28)

    return prior, simpar, postproc
