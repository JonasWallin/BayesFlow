'''
Functions helping with running the estimation procedure
Created on Aug 25, 2015

@author: jonaswallin
'''
from __future__ import print_function
from mpi4py import MPI
import numpy as np
import copy as cp


def HGMM_pre_burnin(Hier_GMM, init_iter=40, prec=1, iteration=10, mutate_iteration=20,
                    burst_iteration=20, local_iter=5, silent=False):
    """
        A mutation type algorithm to find good starting lcoation for MCMC
        *prec* precentage of data counting as outlier
        *iteration*         number of iteration of mutation burst mutation
        *mutate_iteration*  number of gibbs samples to run before testing if the mutation improved
        *burst_iteration*  number of gibbs samples to run before testing if the mutation improved
    """

    if (MPI.COMM_WORLD.Get_rank() == 0):
        silent = silent
    else:
        silent = True

    for GMM in Hier_GMM.GMMs:

        for i in range(init_iter):
            GMM.sample()

        GMM_pre_burnin(GMM=GMM, prec=prec, iteration=iteration,
                       mutate_iteration=mutate_iteration,
                       burst_iteration=burst_iteration,
                       silent=silent,
                       local_iter=local_iter)
    Hier_GMM.comm.Barrier()


def GMM_pre_burnin(GMM, prec=1, iteration=10, mutate_iteration=20,
                   burst_iteration=20, local_iter=5, silent=False):
    """
        A mutation type algorithm to find good starting lcoation for MCMC
        *prec* precentage of data counting as outlier
        *iteration*         number of iteration of mutation burst mutation
        *mutate_iteration*  number of gibbs samples to run before testing if the mutation improved
        *burst_iteration*  number of gibbs samples to run before testing if the mutation improved
    """
    for j in range(iteration):
        if not silent:
            print('pre burnin iteration {j}'.format(j=j))
        mutate(GMM, prec, iteration=mutate_iteration, silent=silent)
        mutate(GMM, prec, iteration=mutate_iteration, silent=silent, rand_class=True)
        burst(GMM, iteration=burst_iteration, silent=silent)
        for k in range(local_iter):
            GMM.sample()


def draw_outlier_point(GMM, prec=0.1):
    """
        draws a random outlier point (outlier defined through likelihood)
        *prec* - [0,1] lower quantile what is defined as outlier
    """
    GMM.compute_ProbX(norm=False)
    l = np.max(GMM.prob_X, 1)
    index = l < np.percentile(l, prec)
    index_p = np.random.randint(prec*GMM.data.shape[0])
    point_ = GMM.data[index[index_p], :]
    return(point_)


def store_param(GMM, lik=None):
    '''
        Stores the likelihood components
    '''
    if lik is None:
        lik = GMM.calc_lik()

    res = {'lik':    lik,
           'p':      cp.deepcopy(GMM.p),
           'mu':    cp.deepcopy(GMM.mu),
           'sigma': cp.deepcopy(GMM.sigma)}
    return(res)


def restore_param(GMM, param):
    '''
        reset the GMM from the parameters
    '''
    GMM.p = param['p']
    GMM.set_mu(param['mu'])
    GMM.set_sigma(param['sigma'])


def mutate(GMM, prec=0.1, iteration=10, silent=True, rand_class=False):
    '''
        mutate by setting a random class to outiler class
        *prec*      - [0,100] lower quantile what is defined as outlier (precentage)
        *iter*      - number of iteration in the Gibbs sampler
        *rand_class* - draw the class at random (else always take the smallest)
    '''

    param0 = store_param(GMM)
    point_ = draw_outlier_point(GMM, prec)
    if rand_class:
        k = np.random.randint(GMM.K)
    else:
        k = np.argmin(GMM.p[:, :GMM.K])
    set_mutated(GMM, k, point_)

    for i in range(iteration):
        GMM.sample()

    lik = GMM.calc_lik()
    if param0['lik'] < lik:
        if silent is False:
            if rand_class:
                print('random mutation %.2f < %.2f' % (param0['lik'], lik))
            else:
                print('min mutation %.2f < %.2f' % (param0['lik'], lik))
        return

    restore_param(GMM, param0)


def burst(GMM, iteration=10, silent=True):
    '''
        trying to burst two classes (k, k2)
        k is drawn at random, k2 is class closest to k2 in dimension
        d_ (which is also drawn at random).
    '''
    param0 = store_param(GMM)

    k = np.random.randint(GMM.K)
    d_ = np.random.randint(GMM.d)
    dist = np.abs((np.array(GMM.mu)[k, d_]-np.array(GMM.mu)[:, d_]))
    dist[k] = np.Inf
    k2 = np.argmin(dist)
    index = (GMM.x == k) + (GMM.x == k2)
    y_index = GMM.data[index, :]
    if y_index.shape[0] < 3:
        return
    index_points = np.random.choice(y_index.shape[0], 2, replace=False)
    points = y_index[index_points, :]
    set_mutated(GMM, k, points[0, :],  update=False)
    set_mutated(GMM, k2, points[1, :], update=True)

    for i in range(iteration):
        GMM.sample()

    lik = GMM.calc_lik()
    if param0['lik'] < lik:
        if silent is False:
            print('burst mutation %.2f < %.2f' % (param0['lik'], lik))
        return

    restore_param(GMM, param0)


def set_mutated(GMM, k, point_, update=True):
    '''
        sets the mutated point
    '''

    GMM.mu[k][:] = point_[:]
    GMM.sigma[k] = 50 * np.diag(np.diag(GMM.sigma[k]))
    if update:
        GMM.updata_mudata()
        GMM.sample_x()
