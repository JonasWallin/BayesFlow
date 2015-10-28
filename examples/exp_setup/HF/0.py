
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 20:02:37 2014

@author: johnsson
"""
from __future__ import division
import numpy as np
from BayesFlow import SimPar, PostProcPar, Prior
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def setup(J, n_J, testrun=False):

    if rank == 0:
        print "setup with J = {} and n_J = {}".format(J, n_J)

    t_inf = np.loadtxt('exp_setup/HF/t_inf_1.txt')
    Sk_inf = np.loadtxt('exp_setup/HF/Sk_inf_1.txt')
    if rank == 0:
        print "Informative prior locations: {}".format(t_inf)
        print "Informative prior variances: {}".format(Sk_inf)

    prior = Prior(J, n_J, d=4, K=17)
    prior.latent_cluster_means(t_inf=t_inf, t_ex=0.5, Sk_inf=Sk_inf, Sk_ex=1e6)
    prior.component_location_variance(nt=1, q=5e-6)
    prior.component_shape(nps=0.05, h=2)
    prior.set_noise_class(noise_mu=0.5, noise_Sigma=0.5**2)
    prior.pop_size()
    prior.lamb = 10  # activation prior

    if testrun:
        nbriter = 100
    else:
        nbriter = 3000
    simpar = SimPar(nbriter=nbriter, qburn=0.5, simsamp=['sample3', 'sample6'])
    simpar.new_burnphase(1, 'B')
    simpar.set('B', p_sw=0, p_on_off=[0, 0])
    if rank == 0:
        simpar.set_nu_MH('B', sigma=[max(prior.d, np.ceil(.1*prior.n_Psi[k])) for k in range(prior.K)],
                         iteration=5)
    simpar.new_prodphase(last_burnphase='B')

    postproc = setup_postproc()

    return prior, simpar, postproc


def setup_postproc():
    return PostProcPar(True, 'bhat_hier_dip', thr=0.47, lowthr=0.08, dipthr=0.28)